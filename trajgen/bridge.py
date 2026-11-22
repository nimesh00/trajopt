"""Optimiser torques -> position-servo commands.

The executed plant has no torque input. Each actuator is a force-limited servo

    force = clamp(kp * (ctrl - q) - kd * qdot, +-forcerange)

and MuJoCo additionally applies joint damping (-b * qdot) and frictionloss.
For a unit joint transmission, the command that produces the optimiser's joint
torque tau at reference state (q_ref, v_ref) while also cancelling the damper is

    ctrl = q_ref + (tau + (kd + b) * v_ref) / kp

The command is held for a whole control interval while the state moves, so the
reference is taken at the interval MIDPOINT (manifold interpolation), which makes
the interval-average servo torque match tau to first order. This is a model of
the servo, not an equivalence: tracking is measured by executing the 10-substep
response, never assumed. Frictionloss (0.2 N m) is not compensated; its sign
depends on the executed velocity.

Optional feedback uses the solver's Riccati gains, bounded:
    tau = u_k - clip(K_k * diff(x_k, x_measured), +-limit)

GEARED AND TENDON PLANTS. With motor coordinates m = J q, gear g and
gains kp/g^2, the joint-side servo is tau = Kp (q* - q) - Kd qdot with
Kp = kp J^T J, Kd = kd J^T J and ctrl = g * (J q*) (trajgen/plantmap.py). The same
inversion then reads

    q* = q_ref + Kp^-1 (tau + (Kd + B) v_ref),     ctrl = g * (J q*)

which is the formula above when J = I and g = 1. A unit-gear plant takes the scalar line
verbatim, so its commands are bit-identical to before.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import pinocchio as pin

from . import plantmap
from .adapter import Adapter

BRIDGE_VERSION = 1


@dataclass
class BridgeConfig:
    midpoint: bool = True
    compensate_damping: bool = True
    feedback: str = "none"            # "none" | "riccati"
    feedback_limit: float = 5.0       # N m per joint
    rate_limit: float | None = None   # max |ctrl change| per control step

    def as_dict(self) -> dict:
        return {"version": BRIDGE_VERSION, **self.__dict__}


class ServoBridge:
    def __init__(self, ad: Adapter, kp: float, kd: float, config: BridgeConfig | None = None):
        self.ad = ad
        self.kp, self.kd = float(kp), float(kd)
        self.config = config or BridgeConfig()
        mj = ad.mj
        index = {name: k for k, name in enumerate(ad.joint_names)}
        # Indexed by each actuator's PRIMARY joint: for a tendon actuator
        # actuator_trnid is a tendon id, and read as a joint id it names the free joint.
        # Over the SERVO actuators (plantmap): every actuator on a unit-gear or geared plant; a
        # wheel-legged plant's held wheels are excluded here and given their constant target by `project`.
        self.tr = plantmap.transmission(mj)
        ns = len(self.tr.servo)
        self.pin_q = np.zeros(ns, int)
        self.pin_v = np.zeros(ns, int)
        self.damping = np.zeros(ns)
        self.lo = np.zeros(ns)
        self.hi = np.zeros(ns)
        if not self.tr.identity:
            self.Kp = self.tr.stiffness(self.kp)
            self.Kd = self.tr.stiffness(self.kd)
        for s, a in enumerate(self.tr.servo):
            j = int(self.tr.joints[s])
            name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name not in index:
                raise ValueError(f"actuator {a} drives unmapped joint {name}")
            k = index[name]
            self.pin_q[s], self.pin_v[s] = ad.pin_qadr[k], ad.pin_vadr[k]
            self.damping[s] = mj.dof_damping[mj.jnt_dofadr[j]]
            # The compiled plant stores ctrlrange = joint range but leaves
            # ctrllimited off; the trainer's command update clamps to it instead, so
            # the bridge clamps here and MuJoCo is never relied on to do it.
            lo, hi = mj.actuator_ctrlrange[a]
            if not lo < hi:
                raise ValueError(f"actuator {a}: empty ctrlrange")
            self.lo[s], self.hi[s] = lo, hi
        self.state = None
        self.previous = None

    def start_from(self, command):
        """Seed the rate limiter with the command already in effect (a full MuJoCo ctrl)."""
        self.previous = np.asarray(self.tr.servo_part(command), float).copy()

    def reference(self, x_k: np.ndarray, x_next: np.ndarray):
        pm = self.ad.pin
        q_k, v_k = x_k[:pm.nq], x_k[pm.nq:]
        if not self.config.midpoint:
            return q_k, v_k
        q_n, v_n = x_next[:pm.nq], x_next[pm.nq:]
        return pin.interpolate(pm, q_k, q_n, 0.5), 0.5 * (v_k + v_n)

    def command(self, x_k, x_next, u_k, K_k=None, x_measured=None):
        """ctrl[nu] in MuJoCo actuator order, plus diagnostics."""
        tau = np.asarray(u_k, float).copy()          # Pinocchio joint order
        feedback = np.zeros_like(tau)
        if self.config.feedback == "riccati":
            if K_k is None or x_measured is None:
                raise ValueError("riccati feedback needs K and the measured state")
            if self.state is None:
                import crocoddyl
                self.state = crocoddyl.StateMultibody(self.ad.pin)
            dx = self.state.diff(np.asarray(x_k, float), np.asarray(x_measured, float))
            feedback = np.clip(-np.asarray(K_k) @ dx, -self.config.feedback_limit,
                               self.config.feedback_limit)
            tau = tau + feedback
        elif self.config.feedback != "none":
            raise ValueError(f"unknown feedback mode {self.config.feedback}")

        q_ref, v_ref = self.reference(np.asarray(x_k, float), np.asarray(x_next, float))
        tau_a = tau[self.pin_v - 6]
        v_a = v_ref[self.pin_v]
        if self.tr.identity:
            gain = self.kd + (self.damping if self.config.compensate_damping else 0.0)
            raw = q_ref[self.pin_q] + (tau_a + gain * v_a) / self.kp
        else:
            damper = self.damping * v_a if self.config.compensate_damping else 0.0
            q_cmd = q_ref[self.pin_q] + np.linalg.solve(self.Kp, tau_a + self.Kd @ v_a + damper)
            raw = self.tr.ctrl(q_cmd)
        ctrl, info = self.project(raw)
        info["feedback_max"] = float(np.max(np.abs(feedback))) if feedback.size else 0.0
        return ctrl, info

    def project(self, raw):
        """Clip a MuJoCo-order SERVO command to ctrlrange and to the consumer's slew.

        Split out of command() so a planner that already works in command space
        can export through the same projection without going near the
        torque-to-position inversion above. What the consumer can express is a
        property of the consumer, not of how the command was arrived at.
        """
        raw = np.asarray(raw, float)
        ctrl = np.clip(raw, self.lo, self.hi)
        rate = self.config.rate_limit
        limited = 0
        if rate is not None and self.previous is not None:
            # The consumer's command law moves by at most `rate` per step, so a
            # command it could not express is projected rather than exported.
            # `rate` is in SERVO-POSITION units (ctrl / gear), so on a geared plant
            # the ctrl slew is rate * gear: 0.6 rad of motor target is 4.8 ctrl
            # units on an 8:1 geared hip, not 0.6.
            if not self.tr.identity:
                rate = rate * self.tr.gear
            projected = np.clip(ctrl, self.previous - rate, self.previous + rate)
            limited = int(np.sum(projected != ctrl))
            ctrl = projected
        self.previous = ctrl.copy()
        # A full MuJoCo ctrl: held actuators (a wheel-legged plant's wheels) at target 0 (braked). Without
        # held actuators this is `ctrl` itself.
        return self.tr.full(ctrl), {"saturated": int(np.sum(ctrl != raw)), "rate_limited": limited,
                                    "feedback_max": 0.0}

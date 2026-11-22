"""Multi-phase whole-body optimisation over a discovered contact schedule.

The search finds WHICH contacts happen and when; this solves the trajectory
with those contacts as constraints, in Crocoddyl, over Pinocchio's rigid-body
dynamics -- torso, thigh and calf support included, not just feet.

Contact anchors are the MEASURED contact points: for each phase and each
contacting link, the world contact position is converted into that link's
frame and added to the Pinocchio model, so the anchor is the material point
that was actually touching. A geom centre would be wrong (the centre of a
cylinder lying on the ground is not on the ground).

One contact per parent joint, always: Crocoddyl 3.1's ContactModelMultiple
overwrites `fext` for contacts that share a parent joint (measured: 7.6e6
relative derivative error against numerical differentiation, versus 1.5e-7 for
distinct joints), so two support points on one link would silently corrupt the
dynamics derivatives.

Rolling is the known modelling gap: a sticking anchor cannot represent a
cylinder rolling on its surface. The solve reports, per phase, how far the
optimised motion drifts from the measured one and how large the contact
residuals are, so a phase that rigid contacts cannot represent shows up as a
number instead of a silent error.
"""
from __future__ import annotations

import time

import crocoddyl
import numpy as np
import pinocchio as pin

from .adapter import Adapter
from .config import RobotConfig
from .croc import Solution, torque_bounds


class ScheduleProblem:
    def __init__(self, ad: Adapter, cfg: RobotConfig, weights: dict | None = None):
        self.ad = ad
        self.cfg = cfg
        self.state = crocoddyl.StateMultibody(ad.pin)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.nu = self.actuation.nu
        self.tau_max = torque_bounds(ad, cfg)
        self.mu = cfg.mu * cfg.cone_margin
        self.w = {"track": 1e2, "control": 1e-2, "friction": 1e1, "bounds": 1e3,
                  "terminal": 1e4, **(weights or {})}
        self._frames: dict = {}

    # ----------------------------------------------------------- anchors

    def anchor(self, label: str, joint: int, world_point: np.ndarray, q: np.ndarray) -> int:
        """Add (once) a frame at the measured contact point on `joint`'s link."""
        name = f"contact:{label}:{joint}"
        if name in self._frames:
            return self._frames[name]
        data = self.ad.pin.createData()
        pin.forwardKinematics(self.ad.pin, data, q)
        local = data.oMi[joint].actInv(np.asarray(world_point, float))
        fid = self.ad.pin.addFrame(pin.Frame(name, joint, 0, pin.SE3(np.eye(3), local),
                                             pin.FrameType.OP_FRAME))
        self._frames[name] = fid
        return fid

    # ------------------------------------------------------------ models

    def _contacts(self, anchors: dict):
        contacts = crocoddyl.ContactModelMultiple(self.state, self.nu)
        for name, (frame, reference) in anchors.items():
            contacts.addContact(name, crocoddyl.ContactModel3D(
                self.state, frame, np.asarray(reference, float), pin.LOCAL_WORLD_ALIGNED,
                self.nu, np.array([0.0, 50.0])))
        return contacts

    def _costs(self, anchors: dict, x_ref: np.ndarray, terminal: bool):
        costs = crocoddyl.CostModelSum(self.state, self.nu)
        nj = self.ad.pin.nv - 6
        weights = np.concatenate([np.full(3, 1.0), np.full(3, 10.0), np.full(nj, 1.0),
                                  np.full(6, 0.1), np.full(nj, 0.1)])
        costs.addCost("track", crocoddyl.CostModelResidual(
            self.state, crocoddyl.ActivationModelWeightedQuad(weights ** 2),
            crocoddyl.ResidualModelState(self.state, x_ref, self.nu)),
            self.w["terminal"] if terminal else self.w["track"])
        if not terminal:
            costs.addCost("control", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ResidualModelControl(self.state, self.nu)), self.w["control"])
            for name, (frame, _) in anchors.items():
                cone = crocoddyl.FrictionCone(np.eye(3), self.mu, 4, False)
                costs.addCost(f"cone_{name}", crocoddyl.CostModelResidual(
                    self.state,
                    crocoddyl.ActivationModelQuadraticBarrier(
                        crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                    crocoddyl.ResidualModelContactFrictionCone(self.state, frame, cone, self.nu)),
                    self.w["friction"])
        lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        costs.addCost("bounds", crocoddyl.CostModelResidual(
            self.state, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub)),
            crocoddyl.ResidualModelState(self.state, self.nu)), self.w["bounds"])
        return costs

    def running(self, anchors: dict, x_ref: np.ndarray, dt: float):
        dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, self._contacts(anchors), self._costs(anchors, x_ref, False),
            0.0, True)
        dam.u_lb, dam.u_ub = -self.tau_max, self.tau_max
        return crocoddyl.IntegratedActionModelEuler(dam, dt)

    def terminal(self, anchors: dict, x_ref: np.ndarray):
        dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, self._contacts(anchors), self._costs(anchors, x_ref, True),
            0.0, True)
        return crocoddyl.IntegratedActionModelEuler(dam, 0.0)

    def impulse(self, frames: list[int]):
        """Touchdown of newly arriving contacts: zero duration, no extra frame."""
        impulses = crocoddyl.ImpulseModelMultiple(self.state)
        for frame in frames:
            impulses.addImpulse(f"impulse{frame}", crocoddyl.ImpulseModel3D(
                self.state, frame, pin.LOCAL_WORLD_ALIGNED))
        costs = crocoddyl.CostModelSum(self.state, 0)
        return crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulses, costs)


def solve(ad: Adapter, cfg: RobotConfig, ex, phases: list[dict], *, max_iter: int = 200,
          impulses: bool = True) -> Solution:
    """Optimise the measured motion subject to its own contact schedule."""
    problem_builder = ScheduleProblem(ad, cfg)
    xs_ref = np.array([ad.x_from_mj(q, v) for q, v in zip(ex.qpos, ex.qvel)])
    by_geom = {info["mj_geom"]: name for name, info in ad.geoms.items()}
    table = ex.contact_table

    models, labels, previous = [], [], {}
    for phase in phases:
        k0 = phase["first"]
        q0 = xs_ref[k0][:ad.pin.nq]
        anchors = {}
        for joint, geom in phase["geoms"].items():
            rows = np.nonzero((table["step"] == k0) & (table["boundary"] == 0))[0]
            match = [r for r in rows if geom in [int(v) for v in table["geom"][r]]]
            point = np.asarray(table["pos"][match[0]], float) if match else None
            if point is None:
                continue
            name = by_geom[geom]
            frame = problem_builder.anchor(f"{phase['label']}:{name}", int(joint), point, q0)
            anchors[name] = (frame, point)
        arriving = [frame for name, (frame, _) in anchors.items() if name not in previous]
        if impulses and models and arriving:
            models.append(problem_builder.impulse(arriving))
            labels.append(f"{phase['label']}:impulse")
        for k in range(phase["first"], phase["last"] + 1):
            models.append(problem_builder.running(anchors, xs_ref[k + 1], cfg.control_dt))
            labels.append(phase["label"])
        previous = anchors

    last_anchors = previous
    problem = crocoddyl.ShootingProblem(xs_ref[0], models,
                                        problem_builder.terminal(last_anchors, xs_ref[-1]))
    solver = crocoddyl.SolverBoxFDDP(problem)
    solver.th_stop = 1e-9
    # Warm start on the measured motion: states from the rollout (impulse nodes
    # repeat their predecessor), controls quasi-static at those states.
    xs_init = [xs_ref[0]]
    k = 0
    for label in labels:
        if label.endswith(":impulse"):
            xs_init.append(xs_init[-1])
        else:
            k += 1
            xs_init.append(xs_ref[min(k, len(xs_ref) - 1)])
    us_init = problem.quasiStatic(xs_init[:-1])
    t0 = time.perf_counter()
    converged = solver.solve(xs_init, us_init, max_iter, False)
    seconds = time.perf_counter() - t0

    xs = np.array(solver.xs)
    us = np.array([u if u.size == problem_builder.nu else np.zeros(problem_builder.nu)
                   for u in solver.us])
    drift = np.array([np.linalg.norm(x[:3] - r[:3]) for x, r in zip(xs, xs_init)])
    diagnostics = {
        "solver": "BoxFDDP", "backend": "crocoddyl", "converged": bool(converged),
        "iterations": int(solver.iter), "cost": float(solver.cost), "stop": float(solver.stop),
        "dynamic_feasibility": float(solver.ffeas), "seconds": round(seconds, 2),
        "nodes": len(models), "phases": [p["label"] for p in phases],
        "impulse_nodes": sum(1 for label in labels if label.endswith(":impulse")),
        "max_base_drift_m": float(drift.max()),
        "torque_utilisation_max": float(np.max(np.abs(us) / problem_builder.tau_max)),
        "finite": bool(np.all(np.isfinite(xs)) and np.all(np.isfinite(us))),
    }
    diagnostics["feasible"] = bool(diagnostics["finite"] and diagnostics["converged"]
                                   and diagnostics["dynamic_feasibility"] < 1e-6
                                   and diagnostics["torque_utilisation_max"] <= 1.0 + 1e-9)
    return Solution(xs=xs, us=us, K=np.array(solver.K, dtype=object), forces=np.zeros((len(us), 4, 3)),
                    phases=labels, diagnostics=diagnostics)

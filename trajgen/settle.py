"""Solve landing-to-stance as a trajectory optimisation on the plant itself.

A clip that has landed is not finished: the robot arrives with real momentum and has to
reach stance. Doing that by truncating the clip at touchdown and blending the command to
stance does not work, and the measurement says why - a backflip touches down descending at
2.23 m/s and still pitching at 9.97 rad/s. A blend absorbs none of that, so the robot skids
and over-rotates (drift -0.220 -> -0.534 m, turn -357 -> -377 deg). Leaving the clip's own
post-landing commands in place is no better: those were what flung the robot around, lifting
the rear feet 0.119 m and pitching the base to 36 deg after it was already standing.

So the ending is solved. Dynamics are `mjaction.ActionModelMuJoCo` - the plant itself -
so the solved commands replay exactly rather than approximately.

Two things are load-bearing and were each got wrong first:

  * Weight joint posture like it matters. At 4 against base orientation 300 the solve
    bought base pose by giving up posture and stood on two feet with the front feet 11 mm
    up. Stance posture IS four feet down.
  * Start the horizon AT touchdown. Given airborne nodes, the upright-orientation cost
    fights the flip's remaining rotation and lands it 30 degrees short.
"""
from __future__ import annotations

import crocoddyl
import mujoco
import numpy as np

from . import mjaction
from .adapter import Adapter
from .plant import Plant


class SettleOCP:
    """Landing state -> settled stance, as a BoxFDDP problem on the true plant."""

    def __init__(self, plant: Plant, ad: Adapter, *, horizon: int = 110,
                 w_base_xy: float = 4., w_base_z: float = 25., w_rot: float = 150.,
                 w_joint: float = 30., w_lin: float = 12., w_ang: float = 30.,
                 w_jvel: float = 1., w_ctrl: float = 1.5, terminal_scale: float = 25.,
                 margin: float = .02):
        self.plant = plant
        self.ad = ad
        self.horizon = int(horizon)
        self.state = crocoddyl.StateMultibody(ad.pin)
        self.nj = self.state.nv - 6
        m = plant.m
        # u is in Pinocchio actuated order; MuJoCo wants its own actuator order.
        self.slot = np.array([int(ad.pin_qadr[k]) - (self.state.nq - self.nj)
                              for k in range(m.nu)])
        joints = m.actuator_trnid[:, 0]
        self.joints = joints
        self.u_lb = np.empty(self.nj)
        self.u_ub = np.empty(self.nj)
        for k in range(m.nu):
            i = int(ad.pin_qadr[k]) - (self.state.nq - self.nj)
            self.u_lb[i] = m.jnt_range[joints[k], 0] + margin
            self.u_ub[i] = m.jnt_range[joints[k], 1] - margin
        # Tangent ordering: [base_pos(3) base_rot(3) joints base_linvel(3) base_angvel(3)
        #                    joint_vel]
        self.q_diag = np.concatenate([
            [w_base_xy, w_base_xy, w_base_z], np.full(3, w_rot), np.full(self.nj, w_joint),
            np.full(3, w_lin), np.full(3, w_ang), np.full(self.nj, w_jvel)])
        self.w_ctrl = float(w_ctrl)
        self.terminal_scale = float(terminal_scale)

    def target(self, stance_state, landing_qpos):
        """Stance posture, at the place and HEADING the robot arrived with.

        Only roll, pitch and height are corrected. Taking the stance pose wholesale instead
        - including its yaw and its x,y - asks a robot whose feet are already planted to
        translate and rotate, and the only way to serve that is to scrub the feet. On a
        getup, which rolls over and ends on a different heading, that produced 76 frames of
        knee contact and left the robot 7 cm short of stance with joint rms 0.176. On a
        backflip it was invisible, because a backflip ends facing where it started.
        """
        d = self.plant.data_at(stance_state, [])
        q = d.qpos.copy()
        q[:2] = np.asarray(landing_qpos, float)[:2]
        w, x, y, z = np.asarray(landing_qpos, float)[3:7]
        yaw = np.arctan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))
        q[3:7] = [np.cos(yaw / 2.), 0., 0., np.sin(yaw / 2.)]
        return self.ad.x_from_mj(q, np.zeros(self.plant.m.nv)), q

    def solve(self, start, cmds, cut, stance_state, stance_cmd, *, iters: int = 250,
              warm=None, verbose: bool = True):
        """Solve from the state reached by `cmds[:cut]`. Returns MuJoCo-order commands."""
        plant, ad = self.plant, self.ad
        d0 = plant.data_at(start, cmds, cut)
        qpos0, qvel0 = d0.qpos.copy(), d0.qvel.copy()
        x0 = ad.x_from_mj(qpos0, qvel0)
        x_ref, q_stand = self.target(stance_state, qpos0)
        u_ref = np.empty(self.nj)
        u_ref[self.slot] = stance_cmd

        Q = np.diag(self.q_diag)
        R = np.diag(np.full(self.nj, self.w_ctrl))
        Qf = np.diag(self.q_diag * self.terminal_scale)
        N = self.horizon
        dc = getattr(plant, "dc", None)
        running = [mjaction.ActionModelMuJoCo(self.state, ad, plant.m, x_ref, u_ref, Q, R,
                                              substeps=plant.substeps,
                                              u_lb=self.u_lb, u_ub=self.u_ub, dc=dc)
                   for _ in range(N)]
        terminal = mjaction.ActionModelMuJoCo(self.state, ad, plant.m, x_ref, u_ref, Qf, R,
                                              substeps=plant.substeps, terminal=True, dc=dc)
        problem = crocoddyl.ShootingProblem(np.array(x0), running, terminal)
        solver = crocoddyl.SolverBoxFDDP(problem)
        solver.th_stop = 1e-7

        # Warm start. `problem.quasiStatic` is unavailable without an actuation model, and
        # seeding from the clip's own post-touchdown commands is wrong whenever those were
        # never evaluated - it put the rollout on its back, cost 79109 against 11107, and
        # the solve never recovered. Stance at least stands.
        us = []
        for k in range(N):
            u = np.empty(self.nj)
            u[self.slot] = warm[min(k, len(warm) - 1)] if warm is not None else stance_cmd
            us.append(u)
        xs = [np.array(x0)]
        qp, qv = qpos0.copy(), qvel0.copy()
        d = mujoco.MjData(plant.m)
        for k in range(N):
            d.qpos[:] = qp
            d.qvel[:] = qv
            d.qacc_warmstart[:] = 0.
            d.ctrl[:] = np.asarray(us[k])[self.slot]
            mujoco.mj_forward(plant.m, d)
            for _ in range(plant.substeps):
                mujoco.mj_step(plant.m, d)
            qp, qv = d.qpos.copy(), d.qvel.copy()
            xs.append(ad.x_from_mj(qp, qv))

        ok = solver.solve(xs, us, iters, False)
        if verbose:
            print("  settle: converged=%s iters=%d cost %.1f stop %.3e"
                  % (ok, solver.iter, solver.cost, solver.stoppingCriteria()), flush=True)
        out = np.array([np.asarray(u, float)[self.slot] for u in solver.us])
        return np.clip(out, plant.m.jnt_range[self.joints, 0],
                       plant.m.jnt_range[self.joints, 1])

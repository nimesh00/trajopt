"""A Crocoddyl action model whose dynamics ARE the MuJoCo plant.

Why: a Pinocchio-based OCP and this plant do not agree. Measured, stepping MuJoCo from the
plan's own state under the plan's own command lands 0.108 rad away in the joints per
control interval, where a feasible plan needs ~1e-3. Over a 104-step backflip that
compounds into the whole motion - a 350 degree plan replays as 1 degree - and no feedback
can fix it, because feedback corrects deviations from a trajectory the plant can follow
rather than manufacturing one it cannot.

The disagreements are not integrator type. Crocoddyl's IntegratedActionModelEuler is
already semi-implicit. They are force-model differences the substep sweep leaves intact:

    dof_damping      MuJoCo 0.1     Pinocchio 0
    dof_frictionloss MuJoCo 0.2     Pinocchio 0
    actuator clamp   MuJoCo flat +/-40 / +/-80, no speed dependence
    contact          MuJoCo compliant solver vs rigid Baumgarte constraints

Reimplementing all of that inside Pinocchio is more work than wrapping the plant, and
wrapping it makes the agreement exact by construction rather than by approximation. It
also removes the contact schedule from the formulation entirely: there are no
ContactModel3D, friction-cone or impulse models here, because MuJoCo decides what is in
contact. The phase structure survives only as cost references.

State geometry stays Pinocchio's StateMultibody so the free joint's quaternion is handled
on the manifold, and so existing cost/residual machinery still applies where wanted.
"""
from __future__ import annotations

import crocoddyl
import mujoco
import numpy as np


class ActionModelMuJoCo(crocoddyl.ActionModelAbstract):
    """One control interval = `substeps` mj_step calls under a HELD position command.

    Cost is a quadratic tracking term about a reference state and command:

        l = 1/2 dx' Q dx + 1/2 (u - u_ref)' R (u - u_ref),   dx = state.diff(x_ref, x)

    Dynamics derivatives are central finite differences of the whole wrapper, taken in the
    tangent space through Crocoddyl's integrate/diff, so the coordinate conversions and
    the actuator clamping are differentiated along with the physics.
    """

    def __init__(self, state, adapter, model, x_ref, u_ref, Q, R, substeps=10,
                 u_lb=None, u_ub=None, terminal=False, eps=1e-6, dc=None):
        crocoddyl.ActionModelAbstract.__init__(self, state, int(len(u_ref)), 1)
        self.ad = adapter
        self.m = model
        self.substeps = int(substeps)
        # THE SAME MOTOR THE SEARCH USED. The settle is solved through this model
        # and its result is glued onto a flip that was searched under the DC
        # torque-speed envelope; stepping a flat-clamped servo here plans a tail
        # the real actuator cannot execute. Measured: a search that converged to
        # a recovered -371.4 deg flip assembled into a clip that finished at
        # -0.8 deg, plane 150.6, flat on its back, and was rejected.
        self.dc = dc
        self.x_ref = np.asarray(x_ref, float)
        self.u_ref = np.asarray(u_ref, float)
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.terminal = bool(terminal)
        self.eps = float(eps)
        self.nq = state.nq
        self.nv = state.nv
        # u is in PINOCCHIO actuated order; MuJoCo wants its own actuator order.
        self.slot = np.array([int(adapter.pin_qadr[k]) - (state.nq - self.nu)
                              for k in range(model.nu)])
        if u_lb is not None:
            self.u_lb = np.asarray(u_lb, float)
        if u_ub is not None:
            self.u_ub = np.asarray(u_ub, float)
        self._scratch = mujoco.MjData(model)

    # ----------------------------------------------------------------- plant

    def _roll(self, x, u, data=None):
        """One control interval of the true plant. Returns the next state, pin coords."""
        d = data if data is not None else self._scratch
        q_pin = np.asarray(x[: self.nq], float)
        v_pin = np.asarray(x[self.nq:], float)
        d.qpos[:] = self.ad.qpos_from_pin(q_pin)
        d.qvel[:] = self.ad.qvel_from_pin(q_pin, v_pin)
        # Warm-start state is part of MuJoCo's state for an incompletely converged solve.
        # Zeroing it here and in execution makes planning and replay agree deterministically
        # rather than depending on which node or line-search trial ran before.
        d.qacc_warmstart[:] = 0.0
        ctrl = np.asarray(u, float)[self.slot]
        d.ctrl[:] = ctrl
        d.time = 0.0
        mujoco.mj_forward(self.m, d)
        for _ in range(self.substeps):
            if self.dc is not None:
                self.dc.apply(d)
            mujoco.mj_step(self.m, d)
        return self.ad.x_from_mj(d.qpos, d.qvel)

    # ------------------------------------------------------------------ cost

    def _cost(self, x, u):
        dx = self.state.diff(self.x_ref, np.asarray(x, float))
        c = .5 * dx @ (self.Q @ dx)
        if not self.terminal:
            du = np.asarray(u, float) - self.u_ref
            c += .5 * du @ (self.R @ du)
        return c, dx

    def calc(self, data, x, u=None):
        x = np.asarray(x, float)
        if self.terminal or u is None:
            data.cost, _ = self._cost(x, self.u_ref)
            data.xnext[:] = x
            return
        data.xnext[:] = self._roll(x, u, self._scratch)
        data.cost, _ = self._cost(x, u)

    def calcDiff(self, data, x, u=None):
        x = np.asarray(x, float)
        ndx, nu = self.state.ndx, self.nu
        _, dx = self._cost(x, self.u_ref if u is None else u)
        # d(dx)/dx is the Jacobian of state.diff in its SECOND argument, which is not the
        # identity on the quaternion. Using identity makes Lx inconsistent with the cost it
        # is supposed to differentiate, and FDDP's line search then fails almost at once -
        # it quit after 7 to 13 iterations of the 250 requested. Gauss-Newton with the real
        # Jacobian: l = 1/2 dx' Q dx, Lx = J' Q dx, Lxx = J' Q J.
        # Gauss-Newton with the identity Jacobian, as Crocoddyl's own ResidualModelState
        # does. Substituting state.Jdiff here was tried and made the line search fail
        # sooner (6 iterations against 13), so it is not simply a missing Jacobian.
        data.Lx[:] = self.Q @ dx
        data.Lxx[:, :] = self.Q
        if self.terminal or u is None:
            return
        du = np.asarray(u, float) - self.u_ref
        data.Lu[:] = self.R @ du
        data.Luu[:, :] = self.R
        data.Lxu[:, :] = 0.
        # Central differences of the FULL wrapper, in the tangent space.
        base = self._roll(x, u, self._scratch)
        eps = self.eps
        for j in range(ndx):
            e = np.zeros(ndx); e[j] = eps
            xp = self.state.integrate(x, e)
            xm = self.state.integrate(x, -e)
            fp = self._roll(xp, u, self._scratch)
            fm = self._roll(xm, u, self._scratch)
            data.Fx[:, j] = self.state.diff(fm, fp) / (2 * eps)
        for j in range(nu):
            du_ = np.zeros(nu); du_[j] = eps
            fp = self._roll(x, np.asarray(u, float) + du_, self._scratch)
            fm = self._roll(x, np.asarray(u, float) - du_, self._scratch)
            data.Fu[:, j] = self.state.diff(fm, fp) / (2 * eps)
        data.xnext[:] = base

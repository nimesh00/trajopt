"""Optimal control in the POSITION-COMMAND space the plant actually accepts.

The plant's actuators are MuJoCo position servos:

    tau = clip(kp * (q_cmd - q_j) - kd * v_j, -tau_max, tau_max)

Planning in torque and inverting that law per step - q_cmd = q_ref + tau/kp - does not
reproduce the planned dynamics. Substituting the inversion into the servo law gives

    tau_applied = tau_plan + kp * (q_ref - q) + kd * (v_ref - v)

so the robot runs the optimised controller PLUS a joint PD term that was never in the
model it was optimised against. Measured on the vertical jump: 0.291 rad of joint
tracking error, i.e. 11.6 N m of unmodelled torque against a plan whose peak is 35.5.
It also explains why widening the Riccati feedback limit made tracking worse rather
than better - more feedback authority means more deviation, which means more unmodelled
PD torque.

Here the control IS the joint command, the servo law is part of the dynamics, and the
Riccati gain comes out in radians of command per unit state error. There is no bridge
and nothing to invert.
"""
from __future__ import annotations

import crocoddyl
import numpy as np
import pinocchio as pin
from mujoco import mj_id2name
from mujoco import mjtObj

mjOBJ_JOINT = mjtObj.mjOBJ_JOINT


class ServoActuation(crocoddyl.ActuationModelAbstract):
    """u = joint position command; tau = clip(kp (u - q_j) - kd v_j, +/- tau_max(v_j)).

    The bound is the DC torque-speed envelope, not a flat box:

        tau_max(v) = clip(saturation * (1 - v / v_nl), -effort, effort)   [flat, then droop]

    A stall-torque box lets the optimiser plan torque the motor cannot deliver at speed -
    the calf stalls at 68 N m but can only motor 19.4 N m at 8 rad/s - and the executed
    motion then fails on exactly the fast phases a jump depends on. croc.OCP applies this
    envelope as an OUTER iteration, re-solving against bounds evaluated at the previous
    pass's speeds. Here the velocity is already an argument of the dynamics, so the law
    goes inside the model and is differentiated exactly, including the saturated branch
    where tau depends on v through the bound itself.
    """

    def __init__(self, state, kp: float, kd: float, effort, saturation, v_nl, nj: int,
                 Kp=None, Kd=None, damping=None):
        """`Kp`, `Kd` (nj x nj, u order) replace the scalar gains on a COUPLED plant.

        `damping` (nj, u order) is the plant's passive joint damping b, N m s/rad. MuJoCo
        applies -b v outside the actuator, so it is not clipped by the motor envelope and is
        added after it. Leaving it out plans against a robot without it: Go2's Menagerie
        joints carry b = 2.0 against the servo's kd = 0.5, and its trot cruised at 0.17 m/s
        of a 0.25 m/s command with peak torque at 0.99 of the limit; modelled, 0.225 m/s at
        0.64. A plant with b = 0.1 barely notices (walk travel 1.795 -> 1.798 m).

        A geared, belt-coupled plant's servos act on motor coordinates m = J q through gears, which in joint
        space is tau = Kp (u - q) - Kd v with Kp = kp J^T J, Kd = kd J^T J
        (trajgen/plantmap.py); u stays a JOINT-space command. Left None, the law is
        the scalar one below, unchanged to the bit for a unit-gear plant.
        """
        crocoddyl.ActuationModelAbstract.__init__(self, state, nj)
        self.kp = float(kp)
        self.kd = float(kd)
        self.nj = int(nj)
        self.nv = state.nv
        self.nq = state.nq
        self.base = self.nv - self.nj
        self.effort = np.asarray(effort, float).reshape(nj)
        self.saturation = np.asarray(saturation, float).reshape(nj)
        self.v_nl = np.asarray(v_nl, float).reshape(nj)
        if (Kp is None) != (Kd is None):
            raise ValueError("Kp and Kd go together")
        self.Kp = None if Kp is None else np.asarray(Kp, float).reshape(nj, nj)
        self.Kd = None if Kd is None else np.asarray(Kd, float).reshape(nj, nj)
        self.damping = np.zeros(nj) if damping is None else np.asarray(damping, float).reshape(nj)

    def demand(self, q_j, v_j, u):
        """Servo torque before the envelope clip, per joint."""
        if self.Kp is None:
            return self.kp * (np.asarray(u, float) - q_j) - self.kd * v_j
        return self.Kp @ (np.asarray(u, float) - q_j) - self.Kd @ v_j

    def demand_jacobians(self):
        """(d demand/dq_j, d demand/dv_j, d demand/du), each nj x nj."""
        if self.Kp is None:
            eye = np.eye(self.nj)
            return -self.kp * eye, -self.kd * eye, self.kp * eye
        return -self.Kp, -self.Kd, self.Kp

    def _joint_state(self, x):
        q = np.asarray(x[: self.nq], float)
        v = np.asarray(x[self.nq:], float)
        return q[self.nq - self.nj:], v[self.nv - self.nj:]

    def _envelope(self, v_j):
        """Same physical motor as DcClamp; no fictitious planning derating.

        By default the envelope is floored at zero (upper bound >= 0, lower bound <= 0),
        so the motor is never forced to brake past no-load speed. Set
        TRAJGEN_SERVO_BRAKE=1 for the physical law, with forced braking.
        """
        import os

        from .dcmotor import envelope
        up, lo, d_up, d_lo = envelope(v_j, self.saturation, self.v_nl, self.effort)
        if os.environ.get("TRAJGEN_SERVO_BRAKE"):
            return up, lo, d_up, d_lo
        keep_up = up > 0.
        keep_lo = lo < 0.
        return (np.clip(up, 0., None), np.clip(lo, None, 0.),
                np.where(keep_up, d_up, 0.), np.where(keep_lo, d_lo, 0.))

    def calc(self, data, x, u):
        q_j, v_j = self._joint_state(x)
        raw = self.demand(q_j, v_j, u)
        up, lo, _, _ = self._envelope(v_j)
        data.tau[:] = 0.
        data.tau[self.base:] = np.clip(raw, lo, up) - self.damping * v_j

    def calcDiff(self, data, x, u):
        q_j, v_j = self._joint_state(x)
        raw = self.demand(q_j, v_j, u)
        up, lo, d_up, d_lo = self._envelope(v_j)
        data.dtau_dx[:] = 0.
        data.dtau_du[:] = 0.
        b, nv, nj = self.base, self.nv, self.nj
        for i in range(self.nj):
            row = self.base + i
            if lo[i] < raw[i] < up[i]:                   # on the servo law
                if self.Kp is None:
                    data.dtau_dx[row, self.base + i] = -self.kp
                    data.dtau_dx[row, self.nv + self.base + i] = -self.kd
                    data.dtau_du[row, i] = self.kp
                else:                                    # coupled: a whole row per joint
                    data.dtau_dx[row, b:b + nj] = -self.Kp[i]
                    data.dtau_dx[row, nv + b:nv + b + nj] = -self.Kd[i]
                    data.dtau_du[row, :] = self.Kp[i]
            else:                                        # riding the envelope: tau = bound(v)
                data.dtau_dx[row, self.nv + self.base + i] = (
                    d_up[i] if raw[i] >= up[i] else d_lo[i])
            data.dtau_dx[row, self.nv + self.base + i] -= self.damping[i]

    def commands(self, data, x, tau):
        q_j, v_j = self._joint_state(x)
        if self.Kp is None:
            data.u[:] = q_j + (np.asarray(tau, float)[self.base:] + (self.kd + self.damping) * v_j) / self.kp
        else:
            data.u[:] = q_j + np.linalg.solve(self.Kp, np.asarray(tau, float)[self.base:]
                                              + self.Kd @ v_j + self.damping * v_j)

    def torqueTransform(self, data, x, u):
        data.Mtau[:] = 0.
        if self.Kp is None:
            for i in range(self.nj):
                data.Mtau[i, self.base + i] = 1. / self.kp
        else:
            data.Mtau[:, self.base:] = np.linalg.inv(self.Kp)


class TorqueReserveResidual(crocoddyl.ResidualModelAbstract):
    """Soft reserve on servo DEMAND, with the physical motor left unchanged.

    Penalise violations of a contracted DC interval, normalised by rated Nm.
    The contraction is towards zero (or the nearest feasible braking torque
    above no-load). Unlike a cost on clipped torque, this retains a control
    gradient when the servo is saturated. A finite cost is not a hard bound.
    """

    def __init__(self, state, actuation, fraction):
        if not 0. < fraction <= 1.:
            raise ValueError("torque reserve fraction must be in (0, 1]")
        super().__init__(state, 2 * actuation.nj, actuation.nj, True, True, True)
        self.motor = actuation
        self.fraction = float(fraction)

    def _terms(self, x, u):
        a = self.motor
        q, v = a._joint_state(x)
        up, lo, dup, dlo = a._envelope(v)
        centre = np.clip(np.zeros(a.nj), lo, up)
        dc = np.where(lo > 0., dlo, np.where(up < 0., dup, 0.))
        f = self.fraction
        safe_up, safe_lo = f * up + (1. - f) * centre, f * lo + (1. - f) * centre
        d_up, d_lo = f * dup + (1. - f) * dc, f * dlo + (1. - f) * dc
        raw = a.demand(q, v, u)
        return raw - safe_up, safe_lo - raw, d_up, d_lo

    def calc(self, data, x, u=None):
        if u is None:
            data.r[:] = 0.
            return
        hi, lo, _, _ = self._terms(x, u)
        data.r[:] = np.maximum(np.r_[hi, lo], 0.) / np.tile(self.motor.effort, 2)

    def calcDiff(self, data, x, u=None):
        data.Rx[:] = 0.
        data.Ru[:] = 0.
        if u is None:
            return
        a = self.motor
        hi, lo, dup, dlo = self._terms(x, u)
        if a.Kp is None:
            for j in range(a.nj):
                for row, active, dq, dv, du in (
                        (j, hi[j] > 0., -a.kp, -a.kd - dup[j], a.kp),
                        (a.nj + j, lo[j] > 0., a.kp, a.kd + dlo[j], -a.kp)):
                    if active:
                        data.Rx[row, a.base + j] = dq / a.effort[j]
                        data.Rx[row, a.nv + a.base + j] = dv / a.effort[j]
                        data.Ru[row, j] = du / a.effort[j]
            return
        # Coupled servo: the demand of joint j depends on every joint of its leg.
        Dq, Dv, Du = a.demand_jacobians()
        b, nv, nj = a.base, a.nv, a.nj
        for j in range(nj):
            for row, active, sign, d_env in ((j, hi[j] > 0., 1., dup[j]),
                                             (nj + j, lo[j] > 0., -1., dlo[j])):
                if active:
                    dv = Dv[j].copy()
                    dv[j] -= d_env
                    data.Rx[row, b:b + nj] = sign * Dq[j] / a.effort[j]
                    data.Rx[row, nv + b:nv + b + nj] = sign * dv / a.effort[j]
                    data.Ru[row, :] = sign * Du[j] / a.effort[j]


class ServoOCP:
    """croc.OCP with the control redefined as the joint position command.

    Everything structural - contacts, friction cones, impulse, cost terms - is reused
    from croc.OCP. Three things change, and all three follow from u no longer being a
    torque:

      * the actuation model is the servo law above, so the plan is expressed in the
        variable the plant accepts;
      * the control bounds are JOINT COMMAND limits, not torque limits (torque is
        bounded inside the actuation);
      * the control regularisation is measured from the standing command rather than
        from zero, since penalising |q_cmd|^2 would pull every joint toward the origin
        instead of toward a neutral effort.
    """

    def __init__(self, base_ocp, stand_command, margin: float = .02,
                 attitude_sigma: float = .002, yaw_sigma: float | None = None,
                 torque_fraction: float = .8, torque_reserve_weight: float = 0.):
        """`stand_command` is in MUJOCO ACTUATOR ORDER -- the order the plant's
        ctrl vector and stand.Start.ctrl use. It is converted to u order here, so
        no caller has to know that u indexes Pinocchio's actuated block."""
        self.b = base_ocp
        self.state = base_ocp.state
        self.ad = base_ocp.ad
        self.cfg = base_ocp.cfg
        self.dt = base_ocp.dt
        self.w = base_ocp.w
        self.torque_fraction = float(torque_fraction)
        self.torque_reserve_weight = float(torque_reserve_weight)
        if not 0. < self.torque_fraction <= 1. or self.torque_reserve_weight < 0.:
            raise ValueError("invalid torque reserve fraction or weight")
        nj = self.state.nv - 6
        self.nu = nj
        ad = base_ocp.ad
        mj = ad.mj
        # u is in PINOCCHIO actuated-joint order (it indexes the last nj rows of tau),
        # while ad.joint_names and the MuJoCo ranges are in MuJoCo actuator order.
        # pin_qadr[k] is the pinocchio q index of MuJoCo actuator k; the actuated block
        # starts at 7, so its position within u is pin_qadr[k] - 7.
        # Actuator k's PRIMARY joint (trajgen/plantmap.py): for a unit-gear plant exactly
        # actuator_trnid[k, 0]; for a belt actuator, the calf, not the tendon id.
        tr = ad.transmission
        joints = tr.joints
        row = {name: k for k, name in enumerate(ad.joint_names)}
        jrow = [row[mj_id2name(mj, mjOBJ_JOINT, int(j))] for j in joints]
        lo = np.empty(nj)
        hi = np.empty(nj)
        ns = len(tr.servo)          # mj.nu without held actuators (plantmap)
        for k in range(ns):
            slot = int(ad.pin_qadr[jrow[k]]) - (self.state.nq - nj)
            lo[slot] = mj.jnt_range[joints[k], 0] + margin
            hi[slot] = mj.jnt_range[joints[k], 1] - margin
        # TOLERANCE ON BASE ORIENTATION, in radians, and the one state term that
        # has to be set rather than inherited. croc.OCP squares its weights twice,
        # so orientation there lands at 500^2 * w.state = 2.5e6; restating it
        # honestly at 1/sigma^2 with sigma = .10 gives 1e3, which is 2500x less
        # attitude authority against unchanged com and swing costs. The feet keep
        # tracking and the trunk stops being held: measured at 0.45 m/s, yaw
        # standard deviation over cruise went 0.069 -> 1.348 deg and yaw
        # peak-to-peak 0.56 -> 4.65 deg against the torque-space plan.
        #
        # The default is the sigma that reproduces croc.OCP's ORIENTATION weight
        # exactly -- 1/.002^2 * 1e1 = 2.5e6 = 500^2 * 1e1 -- so the trunk is held
        # as firmly as it was before, while every other state term keeps the
        # honest single-squared scaling. Measured there: yaw sd 0.048 deg and
        # yaw p-p 0.255 deg, both better than the torque-space plan, with peak
        # landing force 0.78 xBW against its 0.95.
        self.attitude_sigma = float(attitude_sigma)
        # ROLL/PITCH ARE REGULATION; YAW IS TRACKING. Roll and pitch have one
        # correct value and the cost exists to hold them at it, so it is set as
        # tight as the motor allows. Yaw follows a command that MOVES, and the
        # same stiffness applied to a moving reference asks for the whole
        # heading change instantly: at 2.5e6, a 0.6 rad/s turn put 0.36 rad of
        # error on the horizon, cost ~3e5 a node, and the solver answered it
        # with 37 rad/s of joint speed and fell over. Defaults to
        # attitude_sigma, which is the forward-walk behaviour unchanged.
        self.yaw_sigma = float(attitude_sigma if yaw_sigma is None else yaw_sigma)
        self.u_lb, self.u_ub = lo, hi
        # u_slot[k] is the position in u of MuJoCo actuator k, so u[u_slot] is a
        # MuJoCo-ordered ctrl vector and the inverse scatter builds u from one.
        self.u_slot = np.array([int(ad.pin_qadr[jrow[k]]) - (self.state.nq - nj)
                                for k in range(ns)], int)
        self.transmission = tr
        self.u_ref = self.u_from_mj(stand_command)
        # Match the plant's physical clip. Derating this clip would make the
        # same position command deliver different torque on the real plant.
        # Reserve belongs in a demand cost, not in a fictitious weaker motor.
        effort = np.empty(nj); saturation = np.empty(nj); vnl = np.empty(nj); damping = np.empty(nj)
        by_joint = {mj_id2name(mj, mjOBJ_JOINT, int(joints[a])): a for a in range(ns)}
        for k, name in enumerate(ad.joint_names):
            sat, v_nl = self.cfg.motor[name.split("_")[1]]
            i = int(ad.pin_vadr[k]) - 6
            effort[i] = float(mj.jnt_actfrcrange[joints[by_joint[name]], 1])
            saturation[i] = float(sat)
            vnl[i] = float(v_nl)
            damping[i] = float(mj.dof_damping[mj.jnt_dofadr[joints[by_joint[name]]]])
        Kp = Kd = None
        if not tr.identity:
            # The coupled joint-space gains, scattered from actuator order into u order.
            Kp, Kd = np.zeros((nj, nj)), np.zeros((nj, nj))
            Kp[np.ix_(self.u_slot, self.u_slot)] = tr.stiffness(self.cfg.kp)
            Kd[np.ix_(self.u_slot, self.u_slot)] = tr.stiffness(self.cfg.kd)
        self.actuation = ServoActuation(self.state, self.cfg.kp, self.cfg.kd,
                                        effort, saturation, vnl, nj, Kp=Kp, Kd=Kd, damping=damping)

    def mj_order(self, values):
        """Any per-joint u-ordered vector (command, torque) -> MuJoCo actuator order."""
        return np.asarray(values, float)[self.u_slot]

    def mj_from_u(self, u):
        """u (joint command, Pinocchio actuated order) -> ctrl (MuJoCo actuator order).

        ctrl = g * (J q*) through the transmission; for a unit-gear plant (J = I, g = 1) this is the
        reordering alone. Torques are NOT commands: reorder those with mj_order."""
        return self.transmission.ctrl(self.mj_order(u))

    def u_from_mj(self, ctrl):
        """ctrl (MuJoCo actuator order; a full ctrl's held columns are dropped) -> u."""
        u = np.empty(self.nu)
        u[self.u_slot] = self.transmission.joint_command(self.transmission.servo_part(ctrl))
        return u

    def _servo_torque(self, x, u):
        """(demand, realised) joint torque for a command, through the servo law.

        The realised torque is the demand clipped to the DC envelope, which is
        what the actuation model integrates. Reporting both separates "the plan
        is inside the motor" from "the plan is asking for more than the motor
        has and the model is saturating on its behalf".
        """
        nj = self.nu
        x = np.asarray(x, float)
        q_j = x[self.state.nq - nj:self.state.nq]
        v_j = x[self.state.nq + self.state.nv - nj:]
        demand = self.actuation.demand(q_j, v_j, u)
        upper, lower, _, _ = self.actuation._envelope(v_j)
        return demand, np.clip(demand, lower, upper), upper, lower

    def _costs(self, knot, terminal):
        costs = self.b._costs(knot, terminal)
        # croc.OCP passes `weights ** 2` to ActivationModelWeightedQuad, which squares
        # nothing itself - so those coefficients are squared twice. Base orientation
        # lands at 500^2 = 250000 while joint posture lands at 0.01^2 = 0.0001, a ratio
        # of 2.5e9: a 0.15 rad stance error costs 0.0135 at the terminal node, so the
        # plan has no reason to return to the stance pose. Restated here as explicit
        # quadratic coefficients, 1/sigma^2 for a tolerance sigma, and scoped to this
        # class because the stand and walk pipelines share croc.OCP and are tuned
        # around its current behaviour.
        costs.removeCost("state")
        nj = self.nu
        # ATTITUDE IS TRACKED AT EVERY KNOT, flight included. ResidualModelState
        # takes state.diff(), the minimal geodesic on SO(3), so a reference a
        # full turn away is the identity and a reference half a turn away sits on
        # the log map's cut locus. That is a property of the (iterate, reference)
        # PAIR, not of the reference: it only bites when the current iterate is
        # ~pi from the reference, i.e. from a cold start. A version of this
        # method released the attitude in the push and flight phases and drove
        # the turn by base angular rate instead; measured on the same flip
        # problem from a contact-consistent warm start (motions/backflip_ocp.py),
        # tracking the attitude through the half turn solved to -360.0 deg with
        # every dynamics gap closed and a LOWER cost than the released variant.
        # The fix for the cold-start hop is the warm start, not the cost.
        # Tangent-space base rotation is (wx, wy, wz): roll, pitch, then yaw.
        sigma = np.concatenate([np.full(3, 1e3),
                                np.array([self.attitude_sigma, self.attitude_sigma, self.yaw_sigma]),
                                np.full(nj, .15),                          # joint posture
                                np.full(6, .30), np.full(nj, 2.)])         # base vel, joint vel
        costs.addCost("state", crocoddyl.CostModelResidual(
            self.state, crocoddyl.ActivationModelWeightedQuad(1. / sigma ** 2),
            crocoddyl.ResidualModelState(self.state, knot.posture, self.nu)),
            self.w.terminal_state if terminal else self.w.state)
        if not terminal:
            costs.removeCost("control")
            costs.addCost("control", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ResidualModelControl(self.state, self.u_ref)),
                self.w.control)
            if self.w.trunk_acceleration:
                # TRUNK SMOOTHNESS. Nothing priced the command's change between knots, and the
                # plans chattered: measured on the dog pace, every joint command reversed
                # direction on 47 % of ticks and trunk angular acceleration had 50-87 % of its
                # energy above 6 Hz (retargeted dog 10-15 %). Neither more iterations, a
                # continuity cost at the re-plan nor an executed-command low-pass removed it.
                # A quadratic price on the base's acceleration weighs high frequencies by w^4.
                rows = np.zeros(self.state.nv)
                rows[2:6] = 1.
                costs.addCost("trunk_acceleration", crocoddyl.CostModelResidual(
                    self.state, crocoddyl.ActivationModelWeightedQuad(rows),
                    crocoddyl.ResidualModelJointAcceleration(self.state, self.nu)),
                    self.w.trunk_acceleration)
            if self.torque_reserve_weight:
                costs.addCost("torque_reserve", crocoddyl.CostModelResidual(
                    self.state, TorqueReserveResidual(
                        self.state, self.actuation, self.torque_fraction)),
                    self.torque_reserve_weight)
        return costs

    def model(self, knot, terminal: bool = False, bound=None):
        costs = self._costs(knot, terminal)
        if knot.support:
            dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, self.b._contacts(knot.support, knot), costs, 0., True)
        else:
            dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation, costs)
        dam.u_lb, dam.u_ub = self.u_lb, self.u_ub
        return crocoddyl.IntegratedActionModelEuler(dam, 0. if terminal else self.dt)

    def build(self, x0, knots, terminal):
        running, labels = [], []
        for knot in knots:
            if knot.impulse:
                running.append(self.b.impulse_model(knot.support, knot.touchdown, knot))
                labels.append(f"{knot.phase}:impulse")
            running.append(self.model(knot))
            labels.append(knot.phase)
        self.labels = labels
        return crocoddyl.ShootingProblem(x0, running, self.model(terminal, terminal=True))

    def solve(self, x0, knots, terminal, max_iter: int = 400, xs_init=None, us_init=None,
              verbose: bool = False):
        """Solve and return a croc.Solution, the same package croc.OCP.solve returns.

        There is no DC outer loop here. croc.OCP re-solves against torque bounds
        re-evaluated at the previous pass's joint speeds because a torque box
        cannot express a speed-dependent envelope; in this control space the
        envelope is a function of v INSIDE the actuation model and is
        differentiated exactly, so one pass is the whole answer.

        `xs_init` / `us_init` are the previous horizon shifted forward. They are
        fitted to this problem rather than used raw: consecutive MPC problems
        differ in node count as impulse nodes enter and leave the horizon, and
        an unfitted guess is a gap on whichever node the count shifted at.
        """
        import time
        t0 = time.perf_counter()
        problem = self.build(x0, knots, terminal)
        solver = crocoddyl.SolverBoxFDDP(problem)
        solver.th_stop = 1e-9
        if verbose:
            solver.setCallbacks([crocoddyl.CallbackVerbose()])
        xs0 = (self.b._fit_states(xs_init, problem.T + 1, x0) if xs_init is not None
               else [np.asarray(x0, float)] * (problem.T + 1))
        # quasiStatic needs the actuation inverse; seeding with the standing
        # command is both simpler and closer to the answer for a motion that
        # starts from a stand. Impulse nodes carry nu=0, so a uniform seed is
        # the wrong shape for them.
        seed = [self.u_ref.copy() if mdl.nu == self.nu else np.zeros(mdl.nu)
                for mdl in problem.runningModels]
        us0 = (self.b._fit_controls(us_init, seed, self.labels) if us_init is not None
               else seed)
        converged = solver.solve(list(xs0), list(us0), max_iter, False)
        seconds = time.perf_counter() - t0

        # Utilisation against the motor, measured on the torque the actuation
        # actually applies. It is <= 1 by construction here, exactly as it is in
        # croc.OCP where the solver is handed the envelope as a control box --
        # the difference is only where the bound is enforced. The saturation
        # figure is what says whether the plan is leaning on that clip.
        worst, sat_ratio, sat_frac = 0.0, 0.0, 0.0
        reserve_violation = 0.
        reserve = TorqueReserveResidual(self.state, self.actuation, self.torque_fraction)
        n_sat = n_tot = 0
        for node, label in enumerate(self.labels):
            if label.endswith(":impulse"):
                continue
            demand, realised, upper, lower = self._servo_torque(
                solver.xs[node], solver.us[node])
            bound = np.where(realised >= 0.0, upper, -lower)
            bound = np.maximum(np.abs(bound), 1e-9)
            worst = max(worst, float(np.max(np.abs(realised) / bound)))
            sat_ratio = max(sat_ratio, float(np.max(np.abs(demand) / bound)))
            n_sat += int(np.count_nonzero(~np.isclose(demand, realised, atol=1e-9)))
            n_tot += demand.size
            hi, lo, _, _ = reserve._terms(solver.xs[node], solver.us[node])
            reserve_violation = max(reserve_violation, float(np.max(hi)), float(np.max(lo)))
        sat_frac = n_sat / max(n_tot, 1)
        return self.b._assemble(
            solver, problem, knots, converged, seconds, max_iter,
            utilisation=worst, labels=self.labels,
            extra={"control_space": "joint position command",
                   "torque_bound": "DC motor law inside the actuation, differentiated exactly",
                   "dc_passes": 1,
                   "torque_reserve_violation": reserve_violation,
                   "command_saturation_ratio": sat_ratio,
                   "command_saturation_fraction": sat_frac})


# --------------------------------------------------------------------- landing

class LandingOCP(ServoOCP):
    """ServoOCP with touchdown actually modelled.

    croc.OCP's impulse node carries `CostModelSum(state, 0)` - an EMPTY cost model. The
    planner therefore receives a free, perfectly inelastic velocity reset and pays
    nothing for how hard it lands, so it plans touchdown at whatever speed the ballistic
    arc produces (2.43 m/s vertical, 1.21 m/s horizontal for a 0.30 m apex) and treats
    the landing as solved. MuJoCo has to absorb that over real time through position
    servos and cannot: measured, the base collapsed to 0.110 m and rang for two seconds.

    Three things are added, all on or around the impulse node:

      * BEFORE impact, the feet are held to the landing geometry and their velocity is
        penalised - tangential hard (that is what slips and splays the stance) and
        vertical against a bounded approach speed.
      * AT impact, the impulses themselves are priced and kept inside the friction cone,
        so a violent or skidding touchdown costs something.
      * AFTER impact, the absorption phase drives base velocity to zero rather than only
        tracking a CoM position, so the energy has somewhere to go.
    """

    def __init__(self, base_ocp, stand_command, margin: float = .02, *,
                 touchdown_speed: float = 1.2, w_touch: float = 4e3,
                 w_impulse: float = 2e1, w_settle: float = 2e3,
                 w_momentum: float = 0., sigma_L: float = .10, momentum_ref=None,
                 momentum_phases=None, w_plane: float = 0., base_frame: int = 0,
                 w_unload: float = 0., pivot_on: str = "R"):
        super().__init__(base_ocp, stand_command, margin)
        self.w_momentum = float(w_momentum)
        self.sigma_L = float(sigma_L)
        # Six-vector [linear; angular] the launch phase should leave the ground with.
        # A jump wants angular zero; a backflip wants it large and backward, so this is
        # a reference rather than a null.
        self.momentum_ref = (np.zeros(6) if momentum_ref is None
                             else np.asarray(momentum_ref, float).reshape(6))
        # Where the momentum objective lives. A jump nulls it across the push; a flip
        # wants it AT the boundary - demanding the final momentum throughout the crouch
        # is counterproductive, since the push has to build up to it.
        self.momentum_phases = momentum_phases or ("crouch", "launch")
        self.w_plane = float(w_plane)
        self.w_unload = float(w_unload)
        self.pivot_on = pivot_on
        self.base_frame = int(base_frame)
        self.touchdown_speed = float(touchdown_speed)
        self.w_touch = float(w_touch)
        self.w_impulse = float(w_impulse)
        self.w_settle = float(w_settle)
        self.mu = base_ocp.mu

    def _costs(self, knot, terminal):
        costs = super()._costs(knot, terminal)
        feet = self.b.ad.feet
        if knot.phase == "touchdown":
            # Land with the feet already moving with the ground, not into it.
            for leg, target in (knot.swing or {}).items():
                costs.addCost(f"touch_v_{leg}", crocoddyl.CostModelResidual(
                    self.state,
                    # nr = 6: linear xyz then angular. Tangential foot speed is what
                    # skids and splays the stance, so it is weighted hardest; vertical is
                    # scaled against the allowed approach speed; angular barely matters.
                    crocoddyl.ActivationModelWeightedQuad(np.array([
                        1., 1., 1. / max(self.touchdown_speed, 1e-3) ** 2, .1, .1, .1])),
                    crocoddyl.ResidualModelFrameVelocity(
                        self.state, feet[leg].frame, pin.Motion.Zero(),
                        pin.LOCAL_WORLD_ALIGNED, self.nu)),
                    self.w_touch)
        if self.w_unload and knot.phase.startswith("unload"):
            k, n = knot.phase.split(":")[1].split("/")
            self._unload_front(costs, knot, (int(k) + 1) / int(n))
        if self.w_plane and (knot.phase.startswith("unload")
                             or knot.phase in ("launch", "takeoff", "flight", "touchdown")) \
                and not terminal:
            # Off-axis momentum alone does NOT hold the trunk in the sagittal plane, and
            # measured it did not: execution left the plane at frame 38, two frames before
            # takeoff, i.e. during the rear-only push where the support is a LINE with no
            # fore-aft area. Roll authority there comes from differential rear-foot load
            # and vanishes as either foot unloads, so the plane has to be held explicitly.
            #
            # FrameRotation's residual is log(Rref^T R) - rotation-vector components, not
            # Euler angles - so weighting x and z pins roll and yaw while leaving the y
            # component free for the flip to wind through.
            base = self.base_frame
            Rref = knot.posture[3:7]
            rot_ref = pin.Quaternion(Rref[3], Rref[0], Rref[1], Rref[2]).matrix()
            costs.addCost("base_plane", crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(
                    np.array([1. / .12 ** 2, 0., 1. / .12 ** 2])),
                crocoddyl.ResidualModelFrameRotation(self.state, base, rot_ref, self.nu)),
                self.w_plane)
            costs.addCost("base_offaxis_rate", crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(
                    np.array([0., 0., 0., 1. / .5 ** 2, 0., 1. / .5 ** 2])),
                crocoddyl.ResidualModelFrameVelocity(
                    self.state, base, pin.Motion.Zero(), pin.LOCAL_WORLD_ALIGNED, self.nu)),
                self.w_plane)
            # A flip is planar. Nothing else here forbids rolling or yawing out of the
            # sagittal plane, and the first plan that rotated did exactly that - roll
            # -2.96, yaw -1.18. Charge the off-axis angular momentum specifically, and
            # leave the pitch component free for the flip to use.
            off = np.array([0., 0., 0., 1. / .3 ** 2, 0., 1. / .3 ** 2])
            costs.addCost("Lg_planar", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ActivationModelWeightedQuad(off),
                crocoddyl.ResidualModelCentroidalMomentum(
                    self.state, np.zeros(6), self.nu)),
                self.w_momentum)
        if self.w_momentum and knot.phase in self.momentum_phases and not terminal:
            # Centroidal ANGULAR momentum about the CoM. For point contacts
            # dL_G = sum_i (p_i - c) x f_i, so a bigger propulsive force tips the trunk
            # unless its line of action is placed right - and nothing else here controls
            # that. Measured without this: the plan pitched -0.35 rad off a crouch
            # takeoff, drifted backwards, and landed 0.06 m below the stance. Base
            # angular velocity is not a substitute: the residual accounts for every
            # moving link, so swinging legs cannot hide trunk rotation from it.
            # Residual is six long, linear momentum first then angular.
            weights = np.concatenate([np.zeros(3), np.full(3, 1. / self.sigma_L ** 2)])
            costs.addCost("Lg", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ActivationModelWeightedQuad(weights),
                crocoddyl.ResidualModelCentroidalMomentum(
                    self.state, self.momentum_ref, self.nu)),
                self.w_momentum)
        if knot.phase in ("absorb", "recover") and not terminal:
            # Kill momentum, not just position error.
            nv = self.state.nv
            sigma = np.concatenate([np.full(nv, 1e3), np.full(3, .15), np.full(3, .4),
                                    np.full(nv - 6, 3.)])
            costs.addCost("settle", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ActivationModelWeightedQuad(1. / sigma ** 2),
                crocoddyl.ResidualModelState(self.state, knot.posture, self.nu)),
                self.w_settle)
        return costs

    def _unload_front(self, costs, knot, fraction):
        """Drive the FRONT contact normal forces to zero before they are released.

        The schedule goes 4 feet -> rear only -> flight, and the OCP simply drops the
        front constraints at that boundary. Nothing asks those feet to unload first, so
        the plan is free to lift them by fiat while the commanded joint motion is still
        pressing them into the ground. MuJoCo has no support set - a foot leaves only when
        the motion actually unweights it - so the executed robot stayed on four feet,
        turned 0.0 degrees against a plan that turned 348, and toppled sideways instead.

        `fraction` ramps 0 -> 1 across the load phase so the transfer is smooth rather
        than a step the servo cannot follow.
        """
        for leg in knot.support:
            if leg.startswith(self.pivot_on):
                continue
            costs.addCost(f"unload_{leg}", crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(np.array([.2, .2, 1.])),
                crocoddyl.ResidualModelContactForce(
                    self.state, self.b.ad.feet[leg].frame, pin.Force.Zero(), 3, self.nu)),
                self.w_unload * fraction ** 2)

    def impulse_model(self, legs):
        impulses = crocoddyl.ImpulseModelMultiple(self.state)
        for leg in legs:
            impulses.addImpulse(leg, crocoddyl.ImpulseModel3D(
                self.state, self.b.ad.feet[leg].frame, pin.LOCAL_WORLD_ALIGNED))
        costs = crocoddyl.CostModelSum(self.state, 0)
        for leg in legs:
            frame = self.b.ad.feet[leg].frame
            # Price the impulse. Tangential impulse is what skids the foot outward and
            # leaves the stance splayed, so it is weighted above the normal component.
            costs.addCost(f"impulse_{leg}", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ActivationModelWeightedQuad(np.array([4., 4., 1.])),
                crocoddyl.ResidualModelContactForce(
                    self.state, frame, pin.Force.Zero(), 3, 0)),
                self.w_impulse)
            cone = crocoddyl.FrictionCone(np.eye(3), self.mu, 4, False)
            costs.addCost(f"impulse_cone_{leg}", crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelQuadraticBarrier(
                    crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                crocoddyl.ResidualModelContactFrictionCone(self.state, frame, cone, 0)),
                self.w_impulse * 10.)
        return crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulses, costs, 0., 1e-12, True)

    def build(self, x0, knots, terminal):
        running, labels = [], []
        for knot in knots:
            if knot.impulse:
                running.append(self.impulse_model(knot.impulse))
                labels.append(f"{knot.phase}:impulse")
            running.append(self.model(knot))
            labels.append(knot.phase)
        self.labels = labels
        return crocoddyl.ShootingProblem(x0, running, self.model(terminal, terminal=True))

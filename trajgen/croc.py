"""Whole-body optimal control with Crocoddyl on the adapted Pinocchio model.

Formulation follows Crocoddyl's quadruped example
(crocoddyl/utils/quadruped.py, SimpleQuadrupedalGaitProblem, BSD-3-Clause,
LAAS-CNRS / University of Edinburgh), adapted rather than imported because
upstream marks it an academic example:

    StateMultibody + ActuationModelFloatingBase
    support phases  DifferentialActionModelContactFwdDynamics + ContactModel3D
    costs           CoM tracking, friction cones (quadratic barrier), state
                    regularisation, control regularisation, state bounds
    integration     IntegratedActionModelEuler, one knot per control interval
    solver          SolverBoxFDDP with torque box bounds

Differences from upstream, each for a measured reason:

  * contacts sit at the foot sphere CENTRES (adapter.feet), not *_foot_site:
    the site is a fixed point 25 mm below the centre in the calf frame, which
    is on the sphere surface at only one leg orientation.
  * torque bounds are min(MuJoCo forcerange, motor saturation) * margin, so the
    optimiser cannot plan torque the executed servo will clip.
  * joint limits come from the compiled MuJoCo plant (adapter already checks the
    URDF agrees).
  * contact forces are read back and checked for unilaterality and friction
    explicitly: the cone is a soft barrier, so convergence is not feasibility.

Only rigid support phases are implemented here. Body/limb contact phases for
inverted recovery are left to the getup search and are not faked with foot contacts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import crocoddyl
import numpy as np
import pinocchio as pin

from .adapter import LEGS, Adapter
from .config import RobotConfig
from .motor_schedule import control_end_velocities


@dataclass
class Knot:
    """Specification of one control interval."""

    support: tuple[str, ...]            # legs in rigid contact, subset of LEGS
    com: np.ndarray | None = None       # world CoM target
    posture: np.ndarray | None = None   # Pinocchio state reference (nx)
    phase: str = ""
    swing: dict | None = None           # leg -> world target for the foot sphere centre
    impulse: tuple[str, ...] = ()       # legs landing at the START of this knot
    touchdown: dict | None = None       # landing leg -> world target for the foot sphere centre
    roll: dict | None = None            # rolling stance leg -> world velocity its contact should roll at


@dataclass
class Weights:
    com: float = 1e6
    friction: float = 1e1
    tangential: float = 1e2
    state: float = 1e1
    control: float = 1e-1
    bounds: float = 1e3
    terminal_state: float = 1e3
    swing: float = 1e5
    joint_posture: float = 0.01     # weight on joint angles inside the state cost
    com_height: float = 1.0         # weight on the CoM z residual, relative to x/y
    touch_position: float = 1e6     # hold the landing foot to its latched foothold
    touch_velocity: float = 1e3     # price the speed the foot arrives with
    normal_force: float = 0.0      # independent soft fz >= 0 penalty, N^-2
    rolling_velocity: float = 0.0  # rolling stance: contact speed against knot.roll, (m/s)^-2
    trunk_acceleration: float = 0.0  # base heave/roll/pitch/yaw acceleration, (m/s^2, rad/s^2)^-2


@dataclass
class Solution:
    xs: np.ndarray            # [T+1, nx] Pinocchio states
    us: np.ndarray            # [T, nu] joint torques, Pinocchio joint order
    K: list                   # per node; impulse nodes have no control gains
    forces: np.ndarray        # [T, 4, 3] world contact force per leg (NaN if not in contact)
    phases: list[str]         # one label per node (impulse nodes end in ":impulse")
    diagnostics: dict = field(default_factory=dict)
    step_nodes: list = field(default_factory=list)   # nodes that advance time, in order
    impulses: np.ndarray | None = None  # [T, 4, 3], Ns; NaN outside impact nodes

    def control_steps(self):
        """(node index, phase) per executed control interval; impulses have no duration."""
        return [(n, self.phases[n]) for n in self.step_nodes]


TERMINAL_GAIN_HOLD = 0.2   # s of control steps at the end of a horizon that reuse the gain before them


def hold_terminal_gains(K: list, labels: list[str], dt: float, seconds: float = TERMINAL_GAIN_HOLD) -> list:
    """Riccati gains for execution: the last `seconds` of control steps take the gain from
    the step just before them.

    The terminal cost (Weights.terminal_state, 100x the running state cost) pins the final
    state, and the backward pass answers it with gains that grow over the last ~8 nodes to
    reach x_T in one or two steps (3.2x the steady gain on Go2, 3.3x on DHAV1, at the last
    running node). Executed through a 20 ms held servo command, that deadbeat correction
    kicks a robot that is meant to be standing still. Measured on the stand-squat perturbed
    starts (tests/test_bridge.py): base rotation in the hold 0.48 rad/s on Go2 (gate 0.30)
    and 0.27 on DHAV1; with the steady gain held to the end, 0.003 and 0.014. The robot does
    not stop at T, so its feedback should not either. The plan (xs, us) is untouched.
    """
    steps = [i for i, label in enumerate(labels) if not label.endswith(":impulse")]
    n = int(round(seconds / dt))
    if len(steps) <= n:
        return K
    tail = set(steps[-n:])
    source = K[steps[-n - 1]]
    return [source if i in tail else k for i, k in enumerate(K)]


def torque_bounds(ad: Adapter, cfg: RobotConfig, margin: float = 0.9) -> np.ndarray:
    """Per-joint bound in Pinocchio joint (v) order."""
    import mujoco

    mj = ad.mj
    tr = ad.transmission
    by_joint: dict[str, list[int]] = {}
    for a in range(len(tr.servo)):     # servo index (= actuator id without held ones)
        # The actuator's PRIMARY joint: a tendon actuator's trnid is a tendon id.
        jname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, int(tr.joints[a]))
        by_joint.setdefault(jname, []).append(a)
    bound = np.zeros(ad.pin.nv - 6)
    for k, name in enumerate(ad.joint_names):
        role = name.split("_")[1]
        act = by_joint.get(name, [])
        if len(act) != 1:
            raise ValueError(f"{name}: expected exactly one actuator, found {len(act)}")
        # Unit servos are clamped per actuator (read as before); a geared or
        # belt plant is clamped per JOINT, and its actuators carry no forcerange.
        force = float(mj.actuator_forcerange[tr.servo[act[0]]][1]) if tr.identity else float(tr.limit[act[0]])
        sat = cfg.motor[role][0]
        bound[ad.pin_vadr[k] - 6] = margin * min(force, sat)
    return bound


def dc_bounds(ad: Adapter, cfg: RobotConfig, dq: np.ndarray, margin: float = 0.95) -> np.ndarray:
    """Torque bound at joint speed `dq` from the DC motor law, in Pinocchio order.

    tau_max(v) = min(sat (1 - v / v_nl), forcerange). A flat box would let the
    optimiser plan torque the motor cannot deliver at speed, which the executed
    motion then fails on the torque-speed gate.
    """
    import mujoco

    mj = ad.mj
    tr = ad.transmission
    bound = np.zeros(ad.pin.nv - 6)
    by_joint = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, int(tr.joints[a])): a
                for a in range(len(tr.servo))}
    for k, name in enumerate(ad.joint_names):
        sat, v_nl = cfg.motor[name.split("_")[1]]
        a = by_joint[name]
        force = float(mj.actuator_forcerange[tr.servo[a]][1]) if tr.identity else float(tr.limit[a])
        v = abs(float(dq[ad.pin_vadr[k] - 6]))
        bound[ad.pin_vadr[k] - 6] = margin * max(1.0, min(force, sat * max(0.0, 1.0 - v / v_nl)))
    return bound


def static_torques(ad: Adapter, q: np.ndarray, support=LEGS):
    """Joint torques holding configuration q still with VERTICAL foot forces only.

    Static equilibrium h(q) = S^T tau + Jz^T fz. The six base rows fix fz
    (least squares; exact when the CoM lies inside the support polygon), and
    the joint rows then give tau. Returns (tau in Pinocchio joint order,
    fz per support leg, base-row residual).
    """
    pm = ad.pin
    pd = pm.createData()
    h = pin.rnea(pm, pd, q, np.zeros(pm.nv), np.zeros(pm.nv))
    pin.computeJointJacobians(pm, pd, q)
    pin.updateFramePlacements(pm, pd)
    Jz = np.array([pin.getFrameJacobian(pm, pd, ad.feet[leg].frame, pin.LOCAL_WORLD_ALIGNED)[2]
                   for leg in support])
    fz, *_ = np.linalg.lstsq(Jz[:, :6].T, h[:6], rcond=None)
    residual = float(np.max(np.abs(Jz[:, :6].T @ fz - h[:6])))
    return h[6:] - Jz[:, 6:].T @ fz, fz, residual


class OCP:
    def __init__(self, ad: Adapter, cfg: RobotConfig, dt: float | None = None,
                 weights: Weights | None = None, velocity_margin: float = 0.9,
                 inner_cone: bool = False):
        self.ad = ad
        self.cfg = cfg
        self.dt = cfg.control_dt if dt is None else dt
        self.w = weights or Weights()
        # The motor's no-load speed is a real limit: above it the joint can
        # produce no driving torque at all.
        # StateMultibody reads velocityLimit for its state bounds, so the plan
        # is bounded by the actuator envelope rather than by the URDF's blanks.
        for k, name in enumerate(ad.joint_names):
            ad.pin.velocityLimit[ad.pin_vadr[k]] = velocity_margin * cfg.motor[name.split("_")[1]][1]
        self.state = crocoddyl.StateMultibody(ad.pin)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.nu = self.actuation.nu
        self.tau_max = torque_bounds(ad, cfg)
        self.mu = cfg.mu * cfg.cone_margin
        self.inner_cone = bool(inner_cone)
        # ROLLING WHEEL STANCE (trajgen/rolling.py), off unless a caller turns it on (the
        # wheel-legged hybrid yaw). When on, a support foot is constrained across its wheel and
        # vertically only, along the knot's heading, and its rolling direction is free and
        # carries no force; touchdown impulses are projected the same way.
        self.rolling = False
        self.rolling_frames = None      # leg -> frame the rolling contact acts at (else the foot)
        self.rolling_yaw_rate = 0.0     # reference heading rate for the rolling constraint
        self.rolling_gap_kp = 0.0       # Baumgarte gain on the rolling contact's normal gap

    @staticmethod
    def _yaw(knot) -> float | None:
        if knot is None or knot.posture is None:
            return None
        qx, qy, qz, qw = knot.posture[3:7]
        return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

    # ------------------------------------------------------------- pieces

    def _contacts(self, support, knot=None):
        contacts = crocoddyl.ContactModelMultiple(self.state, self.nu)
        if self.rolling and knot is not None:
            from .rolling import ContactModelRolling, projection
            P = projection(self._yaw(knot))
            for leg in support:
                frame = (self.rolling_frames or {}).get(leg, self.ad.feet[leg].frame)
                contacts.addContact(leg, ContactModelRolling(
                    self.state, frame, P, self.nu, (self.rolling_gap_kp, 50.0),
                    yaw_rate=self.rolling_yaw_rate,
                    z_ref=0.0 if self.rolling_frames else self.ad.feet[leg].radius))
            return contacts
        for leg in support:
            contacts.addContact(leg, crocoddyl.ContactModel3D(
                self.state, self.ad.feet[leg].frame, np.zeros(3),
                pin.LOCAL_WORLD_ALIGNED, self.nu, np.array([0.0, 50.0])))
        return contacts

    def _costs(self, knot: Knot, terminal: bool):
        costs = crocoddyl.CostModelSum(self.state, self.nu)
        if knot.com is not None:
            # ANISOTROPIC, because the three axes of this residual want different
            # things. Fore-aft has to stay soft: it is the term the gait pushes
            # against to move, and stiffening it fights the stride. Height wants
            # to be held -- it is a ride-height reference, not a trajectory to
            # negotiate -- and with one scalar weight it could only be tightened
            # by tightening fore-aft with it. Measured on the isotropic cost,
            # base height swung 15.9 mm peak-to-peak at 0.11 m/s and 38.5 mm at
            # 0.48 m/s, against a 0.40 m nominal.
            costs.addCost("com", crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(
                    np.array([1.0, 1.0, self.w.com_height])),
                crocoddyl.ResidualModelCoMPosition(self.state, knot.com, self.nu)),
                self.w.com)
        for leg, target in (knot.swing or {}).items():
            costs.addCost(f"swing_{leg}", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ResidualModelFrameTranslation(
                    self.state, self.ad.feet[leg].frame, np.asarray(target, float), self.nu)),
                self.w.swing)
        if not terminal and self.rolling and self.w.rolling_velocity and knot.roll:
            # A FREE ROLLING DIRECTION IS A FREE DEGREE OF FREEDOM. Without a reference the
            # plan swept the stance legs fore-aft on their wheels to balance (cart-pole style),
            # rolling at +-20-30 rad/s with signs unrelated to the turn. Each stance contact is
            # asked to roll at the velocity the rigid turn gives it (gait.Scheduler.knots); the
            # weight is on horizontal velocity, whose axle part the contact already pins.
            for leg, vref in knot.roll.items():
                costs.addCost(f"roll_v_{leg}", crocoddyl.CostModelResidual(
                    self.state, crocoddyl.ActivationModelWeightedQuad(np.array([1., 1., 0., 0., 0., 0.])),
                    crocoddyl.ResidualModelFrameVelocity(
                        self.state, self.ad.feet[leg].frame,
                        pin.Motion(np.asarray(vref, float), np.zeros(3)),
                        pin.LOCAL_WORLD_ALIGNED, self.nu)), self.w.rolling_velocity)
        if not terminal and self.rolling:
            # Rolling stance: the force is (f_axle, f_z); only the axle component is
            # penalised (as [fx, fy] is below), and the cone bounds |f_axle| <= mu fz, fz >= 0.
            from .rolling import ResidualRollingCone, ResidualRollingForce
            for leg in knot.support:
                costs.addCost(f"tangential_{leg}", crocoddyl.CostModelResidual(
                    self.state, crocoddyl.ActivationModelWeightedQuad(np.array([1.0, 0.0])),
                    ResidualRollingForce(self.state, leg, self.nu)), self.w.tangential)
                costs.addCost(f"cone_{leg}", crocoddyl.CostModelResidual(
                    self.state, crocoddyl.ActivationModelQuadraticBarrier(
                        crocoddyl.ActivationBounds(np.full(3, -np.inf), np.zeros(3))),
                    ResidualRollingCone(self.state, leg, self.mu, self.nu)), self.w.friction)
        elif not terminal:
            # Penalise tangential contact force. Rigid 4-foot support leaves
            # internal forces undetermined, and the minimum-torque solution
            # pushes the feet apart by ~1.3 N each. MuJoCo's soft friction
            # creeps under sustained tangential load even at 2% of the cone:
            # measured 4 mm stance splay and 2.3 mm height loss over 10 s,
            # versus 0.5 mm with vertical-only static forces.
            for leg in knot.support:
                costs.addCost(f"tangential_{leg}", crocoddyl.CostModelResidual(
                    self.state, crocoddyl.ActivationModelWeightedQuad(np.array([1.0, 1.0, 0.0])),
                    crocoddyl.ResidualModelContactForce(
                        self.state, self.ad.feet[leg].frame, pin.Force.Zero(), 3, self.nu)),
                    self.w.tangential)
            for leg in knot.support:
                # LWA contact forces and horizontal ground: surface axes are
                # world axes. The last cone row ALREADY bounds fz >= 0.
                cone = crocoddyl.FrictionCone(np.eye(3), self.mu, 4, self.inner_cone)
                costs.addCost(f"cone_{leg}", crocoddyl.CostModelResidual(
                    self.state,
                    crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
                    crocoddyl.ResidualModelContactFrictionCone(
                        self.state, self.ad.feet[leg].frame, cone, self.nu)),
                    self.w.friction)
                if self.w.normal_force:
                    costs.addCost(f"normal_{leg}", crocoddyl.CostModelResidual(
                        self.state, crocoddyl.ActivationModelQuadraticBarrier(
                            crocoddyl.ActivationBounds(
                                np.array([-np.inf, -np.inf, 0.]), np.full(3, np.inf))),
                        crocoddyl.ResidualModelContactForce(
                            self.state, self.ad.feet[leg].frame, pin.Force.Zero(), 3, self.nu)),
                        self.w.normal_force)
        nj = self.ad.pin.nv - 6
        weights = np.array([0.0] * 3 + [500.0] * 3 + [self.w.joint_posture] * nj
                           + [10.0] * 6 + [1.0] * nj)
        posture = knot.posture
        costs.addCost("state", crocoddyl.CostModelResidual(
            self.state, crocoddyl.ActivationModelWeightedQuad(weights ** 2),
            crocoddyl.ResidualModelState(self.state, posture, self.nu)),
            self.w.terminal_state if terminal else self.w.state)
        if not terminal:
            costs.addCost("control", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ResidualModelControl(self.state, self.nu)), self.w.control)
        lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        costs.addCost("bounds", crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub)),
            crocoddyl.ResidualModelState(self.state, self.nu)), self.w.bounds)
        return costs

    def model(self, knot: Knot, terminal: bool = False, bound: np.ndarray | None = None):
        """One integrated action model. Empty support means flight."""
        if knot.posture is None:
            raise ValueError("every knot needs a posture reference")
        if not set(knot.support) <= set(LEGS):
            raise ValueError(f"unknown support legs in {knot.support}")
        costs = self._costs(knot, terminal)
        if knot.support:
            dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, self._contacts(knot.support, knot), costs, 0.0, True)
        else:
            dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(self.state, self.actuation, costs)
        bound = self.tau_max if bound is None else bound
        dam.u_lb, dam.u_ub = -bound, bound
        return crocoddyl.IntegratedActionModelEuler(dam, 0.0 if terminal else self.dt)

    def impulse_model(self, legs, touchdown=None, knot=None):
        """Touchdown: zero duration, so it adds no control interval to the clip.

        `legs` is the support set AFTER the landing, not just the legs that land:
        an impulse that constrains only the new foot lets the feet already
        standing take a free velocity jump through the reset.

        The cost model here used to be empty. An empty cost on an inelastic
        impulse means the planner is handed a free velocity reset and pays
        nothing for how hard it lands, so it lands at whatever speed the swing
        arc happens to produce and treats the touchdown as solved -- MuJoCo then
        has to absorb that through position servos over real time. The two costs
        below are evaluated on the PRE-impact state, which is the only place the
        approach can still be shaped; a velocity cost on the following stance
        node sees the post-reset velocity and misses it entirely.
        """
        impulses = crocoddyl.ImpulseModelMultiple(self.state)
        if self.rolling and knot is not None:
            from .rolling import ImpulseModelRolling, projection
            P = projection(self._yaw(knot))
            for leg in legs:
                frame = (self.rolling_frames or {}).get(leg, self.ad.feet[leg].frame)
                impulses.addImpulse(leg, ImpulseModelRolling(self.state, frame, P))
        else:
            for leg in legs:
                impulses.addImpulse(leg, crocoddyl.ImpulseModel3D(
                    self.state, self.ad.feet[leg].frame, pin.LOCAL_WORLD_ALIGNED))
        costs = crocoddyl.CostModelSum(self.state, 0)
        for leg, target in (touchdown or {}).items():
            frame = self.ad.feet[leg].frame
            costs.addCost(f"touch_p_{leg}", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ResidualModelFrameTranslation(
                    self.state, frame, np.asarray(target, float), 0)),
                self.w.touch_position)
            if self.rolling and knot is not None:
                # A rolling wheel may land rolling: price only its speed across the axle and
                # vertically, at the wheel bottom (Codex review item 6).
                from .rolling import ResidualProjectedVelocity
                P = projection(self._yaw(knot))
                costs.addCost(f"touch_v_{leg}", crocoddyl.CostModelResidual(
                    self.state, ResidualProjectedVelocity(
                        self.state, (self.rolling_frames or {}).get(leg, frame), P, 0)),
                    self.w.touch_velocity)
                continue
            # nr = 6: linear xyz then angular. Angular carries no weight -- the
            # foot is a sphere and spins freely about its contact point.
            costs.addCost(f"touch_v_{leg}", crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(
                    np.array([1., 1., 1., 0., 0., 0.])),
                crocoddyl.ResidualModelFrameVelocity(
                    self.state, frame, pin.Motion.Zero(),
                    pin.LOCAL_WORLD_ALIGNED, 0)),
                self.w.touch_velocity)
        return crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulses, costs)

    # -------------------------------------------------------------- solve

    def build(self, x0: np.ndarray, knots: list[Knot], terminal: Knot, bounds=None):
        running, labels = [], []
        for i, knot in enumerate(knots):
            if knot.impulse:
                running.append(self.impulse_model(knot.support, knot.touchdown, knot))
                labels.append(f"{knot.phase}:impulse")
            running.append(self.model(knot, bound=None if bounds is None else bounds[i]))
            labels.append(knot.phase)
        self.labels = labels
        return crocoddyl.ShootingProblem(x0, running, self.model(terminal, terminal=True))

    def _fit_states(self, xs, n: int, x0) -> list:
        """A warm start from the previous horizon, cut or padded to this one.

        Consecutive MPC problems differ in node count -- an impulse node appears
        the moment a touchdown enters the horizon and vanishes when it leaves --
        so a shifted guess never lines up exactly. Padding with the last state
        and truncating the tail keeps the part that matters (the near nodes,
        where the guess is good) aligned with the problem. Node 0 is always the
        MEASURED state: Crocoddyl takes the rollout from the problem's x0, and a
        guess that disagrees with it is a gap on the first node.
        """
        out = [np.asarray(x, float).copy() for x in xs][:n]
        while len(out) < n:
            out.append(out[-1].copy() if out else np.asarray(x0, float).copy())
        out[0] = np.asarray(x0, float).copy()
        return out

    def _fit_controls(self, us, quasi, labels) -> list:
        """Same for the controls, respecting that impulse nodes carry none."""
        src = [np.asarray(u, float) for u in us if np.size(u) == self.nu]
        out, j = [], 0
        for k, label in enumerate(labels):
            if label.endswith(":impulse"):
                out.append(np.zeros(0))
            elif j < len(src):
                out.append(src[j])
                j += 1
            else:
                out.append(np.asarray(quasi[k], float))
        return out

    def solve(self, x0, knots: list[Knot], terminal: Knot, max_iter: int = 200,
              xs_init=None, us_init=None, dc_passes: int = 3) -> Solution:
        """Solve, then re-solve with torque bounds re-evaluated at the planned
        joint speeds (the DC motor law is speed dependent, a box is not).

        `xs_init` / `us_init` are the previous horizon shifted forward. Without
        them each re-plan starts from a constant rollout at the measured state
        and has to rediscover the whole gait inside its iteration budget --
        which flat ground forgives and stairs do not (measured on the 20 cm
        staircase: 22-59% of solves reached the stopping criterion and the
        dynamics gaps stayed at 10-24, whatever the budget).
        """
        bounds, xs, us, solver, problem = None, xs_init, us_init, None, None
        used = None
        t0 = time.perf_counter()
        for _ in range(max(1, dc_passes)):
            used = bounds
            problem = self.build(x0, knots, terminal, bounds)
            solver = crocoddyl.SolverBoxFDDP(problem)
            solver.th_stop = 1e-9
            xs0 = (self._fit_states(xs, problem.T + 1, x0) if xs is not None
                   else [np.asarray(x0, float)] * (problem.T + 1))
            quasi = problem.quasiStatic(list(xs0[:-1]))
            us0 = self._fit_controls(us, quasi, self.labels) if us is not None else quasi
            converged = solver.solve(list(xs0), list(us0), max_iter, False)
            speeds = np.array(solver.xs)[:, self.ad.pin.nq + 6:]
            bounds = [dc_bounds(self.ad, self.cfg, speed)
                      for speed in control_end_velocities(speeds, self.labels)]
            xs, us = list(solver.xs), list(solver.us)
        seconds = time.perf_counter() - t0
        utilisation = float(
            np.max(np.abs(np.array([u if np.size(u) == self.nu else np.zeros(self.nu)
                                    for u in solver.us])) / self.tau_max) if used is None else
            max(np.max(np.abs(np.asarray(solver.us[node])) / used[i])
                for i, node in enumerate(n for n, label in enumerate(self.labels)
                                         if not label.endswith(":impulse"))))
        return self._assemble(solver, problem, knots, converged, seconds, max_iter,
                              utilisation=utilisation,
                              extra={"dc_passes": dc_passes,
                                     "torque_bound": "per node, DC motor law at the planned joint speed"})

    def _assemble(self, solver, problem, knots, converged, seconds, max_iter, *,
                  utilisation: float, extra: dict | None = None,
                  labels: list[str] | None = None) -> Solution:
        """Pack a solved problem into a Solution.

        Split out of solve() because ServoOCP solves the same problem in a
        different control space and needs the same package: everything here --
        contact forces, Riccati gains, the step-node map, the feasibility
        predicate -- reads the state and the contact schedule, not the control.
        The one control-space-specific number is the motor utilisation, which
        the caller computes and passes in.

        `labels` defaults to this OCP's own, but a wrapper that built the
        problem itself has to pass the labels it built it with -- they are what
        maps nodes to control intervals, and the wrapper's build set them.
        """
        labels = self.labels if labels is None else labels
        xs = np.array(solver.xs)
        us = np.array([u if np.size(u) == self.nu else np.zeros(self.nu) for u in solver.us])
        forces = np.full((problem.T, len(LEGS), 3), np.nan)
        impulses = np.full_like(forces, np.nan)
        supports = []
        for knot in knots:
            if knot.impulse:
                supports.append(())          # impulse node: no contact forces to read
            supports.append(knot.support)
        for k, (support, data) in enumerate(zip(supports, problem.runningDatas)):
            if labels[k].endswith(":impulse"):
                for leg in LEGS:
                    if leg in data.multibody.impulses.impulses:
                        impulses[k, LEGS.index(leg)] = data.multibody.impulses.impulses[leg].f.linear
            if not support or not hasattr(data, "differential"):
                continue
            if self.rolling:
                # Rolling contacts carry a 2-vector multiplier; their data's f is the world
                # force P^T lambda (trajgen/rolling.py), with nothing along the rolling direction.
                contacts = data.differential.multibody.contacts.contacts
                for leg in support:
                    forces[k, LEGS.index(leg)] = np.array(contacts[leg].f.linear)
                continue
            lam = np.array(data.differential.multibody.pinocchio.lambda_c).ravel()
            for i, leg in enumerate(sorted(support)):   # ContactModelMultiple is name-ordered
                forces[k, LEGS.index(leg)] = lam[3 * i: 3 * i + 3]

        K = hold_terminal_gains([np.asarray(k) for k in solver.K], labels, self.dt)
        active =np.array([[leg in support for leg in LEGS] for support in supports])
        active_forces = forces[active]
        fn = active_forces[:, 2]
        ft = np.linalg.norm(active_forces[:, :2], axis=1)
        pm = self.ad.pin
        q_joint = xs[:, 7:pm.nq]
        v_joint = xs[:, pm.nq + 6:]
        lower, upper = pm.lowerPositionLimit[7:], pm.upperPositionLimit[7:]
        v_limit = np.array([self.cfg.motor[pm.names[pm.getJointId(n)].split("_")[1]][1]
                            for n in pm.names[2:]])
        limit_violation = float(max(np.max(lower - q_joint), np.max(q_joint - upper), 0.0))
        speed_violation = float(max(np.max(np.abs(v_joint) - v_limit), 0.0))
        forces_finite = bool(np.all(np.isfinite(active_forces)))
        impact_rows = impulses[~np.isnan(impulses).all(axis=2)]
        impulse_min = float(impact_rows[:, 2].min()) if len(impact_rows) else 0.
        impulse_friction = (float(np.max(np.linalg.norm(impact_rows[:, :2], axis=1)
                                        / np.maximum(impact_rows[:, 2], 1e-9)))
                            if len(impact_rows) else 0.)
        diagnostics = {
            "solver": "BoxFDDP", "converged": bool(converged), "iterations": int(solver.iter),
            "max_iterations": max_iter, "cost": float(solver.cost), "stop": float(solver.stop),
            "dynamic_feasibility": float(solver.ffeas), "reg": float(solver.preg),
            "seconds": round(seconds, 3),
            "knots": problem.T, "dt": self.dt,
            "torque_utilisation_max": float(utilisation),
            "min_normal_force": float(fn.min()) if forces_finite and fn.size else float("nan"),
            "max_friction_ratio": float(np.max(ft / np.maximum(fn, 1e-9))) if forces_finite and fn.size else float("nan"),
            "mu_planned": self.mu,
            "min_normal_impulse": impulse_min,
            "max_impulse_friction_ratio": impulse_friction,
            "joint_limit_violation_rad": limit_violation,
            "joint_speed_violation": speed_violation,
            "finite": bool(np.all(np.isfinite(xs)) and np.all(np.isfinite(us))
                           and all(np.all(np.isfinite(k)) for k in K) and forces_finite
                           and np.all(np.isfinite(impact_rows))),
        }
        diagnostics["feasible"] = bool(
            diagnostics["finite"] and diagnostics["converged"]
            and diagnostics["dynamic_feasibility"] < 1e-6
            and diagnostics["torque_utilisation_max"] <= 1.0 + 1e-9
            and diagnostics["min_normal_force"] >= 0.0
            and diagnostics["max_friction_ratio"] <= self.cfg.mu
            and impulse_min >= 0. and impulse_friction <= self.cfg.mu
            and limit_violation <= 1e-6 and speed_violation <= 0.0)
        diagnostics.update(extra or {})
        step_nodes = [i for i, label in enumerate(labels) if not label.endswith(":impulse")]
        return Solution(xs=xs, us=us, K=K, forces=forces, phases=list(labels),
                        diagnostics=diagnostics, step_nodes=step_nodes, impulses=impulses)

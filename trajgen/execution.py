"""Snapshot-preserving, instrumented MuJoCo execution.

Execution and replay share this one stepping routine, so "replay agrees with
execution" is a statement about determinism, not about two implementations
happening to match:

    restore full mjSTATE_INTEGRATION into a fresh MjData, mj_forward
    per control step k:
        ctrl = controller(k, data)         (replay: the saved ctrl[k])
        substeps x { sample; mj_step }     contacts unioned per substep
        mj_forward; union once more        (the trainer's publish pass)

Sampling timing. After mj_step, data.contact / efc_force / actuator_force /
xpos / cdof all describe the PRE-step state (MuJoCo does not recompute
kinematics after integrating), while qpos/qvel are already post-step. Every
physics sample is therefore indexed to the pre-step state and uses the pre-step
qvel, including contact-point slip velocity. Nothing mixes the two.

Missing measurements are NaN, never zero: a foot with no contact has NaN slip
and NaN penetration, so no statistic can quietly count it as "no slip".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import plantmap
from .contract import FOOT_ORDER

STATE_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)
EXECUTION_VERSION = 3
SAMPLE_CHANNELS = (
    "time", "ctrl", "actuator_force", "joint_q", "joint_qvel", "joint_excess",
    "base_qpos", "base_qvel",
    "foot_normal", "foot_tangential", "foot_slip", "foot_penetration",
    "body_ground_contacts", "body_ground_force", "self_contacts", "warnings",
)
CONTACT_CHANNELS = ("sample", "step", "boundary", "geom", "pos", "frame", "force", "dist", "velocity")


def snapshot(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    state = np.zeros(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def restore(model: mujoco.MjModel, state: np.ndarray) -> mujoco.MjData:
    """A FRESH MjData holding exactly `state`.

    ACCEPTS EITHER CONVENTION. Two live in this codebase: `Plant` gets and sets
    with mjSTATE_FULLPHYSICS while everything here uses STATE_SPEC, and for this
    robot those are 38 and 194 values respectively. Three separate call sites
    have now handed a plant snapshot to an executor and died on the size check,
    each time one boundary further in - plan(), the MPC loop, then replay() - so
    the conversion belongs here rather than at every caller. The two sizes are
    unambiguous, so dispatching on length is safe; anything else is still an
    error.
    """
    state = np.asarray(state, float)
    want = mujoco.mj_stateSize(model, STATE_SPEC)
    plant_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
    plant_want = mujoco.mj_stateSize(model, plant_spec)
    if state.size == want:
        spec = STATE_SPEC
    elif state.size == plant_want:
        spec = plant_spec
    else:
        raise ValueError(f"snapshot has {state.size} values; this model takes "
                         f"{want} (STATE_SPEC) or {plant_want} (plant FULLPHYSICS)")
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state, spec)
    mujoco.mj_forward(model, data)
    return data


@dataclass
class Execution:
    initial_state: np.ndarray
    qpos: np.ndarray            # [T+1, nq]   frame k = after k control steps
    qvel: np.ndarray            # [T+1, nv]
    ctrl: np.ndarray            # [T+1, nu]   ctrl[0] = command in effect at t=0
    contacts: np.ndarray        # [T+1, 5]    interval union, FR FL RR RL other; row 0 unused
    samples: dict               # physics-rate channels, [S, ...]
    sample_step: np.ndarray     # [S] control step each sample belongs to
    control_dt: float
    physics_timestep: float
    substeps: int
    terminal: dict = field(default_factory=dict)   # the SAMPLE_CHANNELS, [1, ...], final state
    contact_table: dict = field(default_factory=dict)  # CONTACT_CHANNELS, one row per contact
    phases: list | None = None  # [T] label per control step
    info: list = field(default_factory=list)
    # mjSTATE_INTEGRATION after the final control boundary. An MPC loop must
    # hand the NEXT solve the state this execution actually reached;
    # re-simulating the slice to recover it runs a second, different rollout.
    final_state: np.ndarray | None = None

    @property
    def steps(self) -> int:
        return self.ctrl.shape[0] - 1


class _NoDc:
    """Explicit opt-out sentinel for Executor(cfg=...). See Executor.__init__."""
    def __repr__(self): return "NO_DC"


NO_DC = _NoDc()


class Executor:
    def __init__(self, model: mujoco.MjModel, substeps: int, cfg):
        """`cfg` enables the DC torque-speed envelope during execution.

        REQUIRED, and deliberately has no default. It used to default to None,
        which silently produced a flat-clamped servo -- and because `apply()`
        drives the curve by writing `actuator_forcerange`, a cfg-less run does
        not even get the RATED range: it inherits whatever the previous run left
        in the shared model. Four call sites had drifted into that path, and the
        resulting in-process replay disagreed with the authoritative execution by
        0.338 m and 1.62 rad on a turn while reporting it as a physics problem.
        Pass `NO_DC` to opt out explicitly.

        Without it the plant is a flat-clamped position servo and the executed
        motion is produced by a motor the robot does not have -- gates.py then
        scores that motion against the real curve, so the two disagree by
        construction. See trajgen/dcmotor.py.
        """
        self.model = model
        self.substeps = int(substeps)
        self.dc = None
        # COMMAND-CLAMP plants (config [contract].envelope = "command", wheel-legged): the DC law
        # clamps the requested command every substep, the command-clamp interface, and nothing
        # is written into the model. The recorded ctrl is the REQUEST.
        self.cmd = None
        if cfg is not NO_DC:
            if cfg is None:
                raise TypeError(
                    "Executor(cfg=...) is required; None used to mean 'flat servo' "
                    "and silently ran a different motor than the one the plan was "
                    "solved against. Pass the config, or execution.NO_DC to opt out.")
            if getattr(cfg, "envelope", "forcerange") == "command":
                from .dcmotor import CommandClamp
                self.cmd = CommandClamp(model, cfg.motor)
            else:
                from .dcmotor import DcClamp
                self.dc = DcClamp(model, cfg)
        m = model
        # The same per-leg foot spheres the adapter plans contacts on (plantmap).
        self.feet = plantmap.foot_geoms(m, FOOT_ORDER)
        self.foot_slot = {g: i for i, g in enumerate(self.feet)}
        self.world = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
        # TORQUE CHANNEL. For unit servos the actuator force IS the joint torque and
        # `actuator_force` records it as always. On a geared/belt plant the
        # actuator force is a MOTOR-side quantity -- 1/8 or 1/16 of the joint torque,
        # unclamped (the clamp acts per joint), and the belt's force lands on two
        # joints -- so the channel records what the limits and the motor curve are
        # stated in: the clamped actuator torque each primary joint receives,
        # qfrc_actuator at its dof, in actuator order.
        self.tr = plantmap.transmission(m)
        self.joint_torque = not self.tr.identity
        # Joint channels are over the SERVO actuators: all of them on unit-gear/geared plants; a
        # wheel-legged plant's held wheels are not recorded there (ctrl, which carries their
        # target, is full).
        self.servo_force = self.tr.held.size > 0
        self.qadr, self.dadr, self.lo, self.hi = [], [], [], []
        for a in range(len(self.tr.servo)):
            j = int(self.tr.joints[a])     # primary joint; a belt's trnid is a tendon id
            self.qadr.append(int(m.jnt_qposadr[j]))
            self.dadr.append(int(m.jnt_dofadr[j]))
            self.lo.append(m.jnt_range[j][0])
            self.hi.append(m.jnt_range[j][1])
        self.qadr, self.dadr = np.array(self.qadr), np.array(self.dadr)
        self.lo, self.hi = np.array(self.lo), np.array(self.hi)
        self._f6 = np.zeros(6)
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))

    # ------------------------------------------------------------ contacts

    def _union(self, data) -> np.ndarray:
        flags = np.zeros(5, bool)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.efc_address < 0:
                continue
            a, b = int(c.geom[0]), int(c.geom[1])
            aw, bw = a in self.world, b in self.world
            if aw == bw:
                continue
            other = b if aw else a
            slot = self.foot_slot.get(other)
            flags[4 if slot is None else slot] = True
        return flags

    def _sample(self, data, qpos, qvel, time, out, table, index, step, boundary=False):
        """One physics sample of the state that data's derived quantities describe.

        `qpos`, `qvel`, `time` must be that same state (pre-step values after
        mj_step, or the current values after mj_forward).
        """
        m = self.model
        normal = np.zeros(4)
        tangential = np.zeros(4)
        slip = np.full(4, np.nan)
        penetration = np.full(4, np.nan)
        body_contacts, body_force, self_contacts = 0, 0.0, 0
        for i in range(data.ncon):
            c = data.contact[i]
            if c.efc_address < 0:
                continue
            a, b = int(c.geom[0]), int(c.geom[1])
            aw, bw = a in self.world, b in self.world
            if aw and bw:
                continue
            mujoco.mj_contactForce(m, data, i, self._f6)
            # Relative velocity of the two material points at the contact:
            # geom b's body point minus geom a's (world geoms are static).
            velocity = np.zeros(3)
            for sign, geom, static in ((1.0, b, bw), (-1.0, a, aw)):
                if not static:
                    mujoco.mj_jac(m, data, self._jacp, self._jacr, c.pos, int(m.geom_bodyid[geom]))
                    velocity += sign * (self._jacp @ qvel)
            n = np.asarray(c.frame[:3])
            tangent_speed = float(np.linalg.norm(velocity - np.dot(velocity, n) * n))
            table["sample"].append(index)
            table["step"].append(step)
            table["boundary"].append(int(boundary))
            table["geom"].append((a, b))
            table["pos"].append(np.array(c.pos))
            table["frame"].append(np.array(c.frame))
            table["force"].append(self._f6.copy())
            table["dist"].append(float(c.dist))
            table["velocity"].append(velocity)
            if not aw and not bw:
                self_contacts += 1
                continue
            other = b if aw else a
            slot = self.foot_slot.get(other)
            if slot is None:
                body_contacts += 1
                body_force += abs(float(self._f6[0]))
                continue
            normal[slot] += abs(float(self._f6[0]))
            tangential[slot] += float(np.hypot(self._f6[1], self._f6[2]))
            depth = max(0.0, -float(c.dist))
            penetration[slot] = depth if np.isnan(penetration[slot]) else max(penetration[slot], depth)
            slip[slot] = tangent_speed if np.isnan(slip[slot]) else max(slip[slot], tangent_speed)
        q = np.asarray(qpos)
        out["time"].append(float(time))
        out["joint_q"].append(q[self.qadr])
        out["joint_excess"].append(float(max(np.max(self.lo - q[self.qadr], initial=0.0),
                                             np.max(q[self.qadr] - self.hi, initial=0.0))))
        out["base_qpos"].append(q[:7].copy())
        out["base_qvel"].append(np.asarray(qvel)[:6].copy())
        out["foot_normal"].append(normal)
        out["foot_tangential"].append(tangential)
        out["foot_slip"].append(slip)
        out["foot_penetration"].append(penetration)
        out["body_ground_contacts"].append(body_contacts)
        out["body_ground_force"].append(body_force)
        out["self_contacts"].append(self_contacts)
        out["actuator_force"].append(np.array(data.qfrc_actuator[self.dadr], float) if self.joint_torque
                                     else np.array(data.actuator_force[self.tr.servo], float)
                                     if self.servo_force else np.array(data.actuator_force, float))
        out["joint_qvel"].append(np.asarray(qvel)[self.dadr].copy())
        out["warnings"].append(int(sum(w.number for w in data.warning)))

    # ----------------------------------------------------------------- run

    def run(self, initial_state: np.ndarray, controller, steps: int, control_dt: float,
            phases: list | None = None) -> Execution:
        try:
            return self._run(initial_state, controller, steps, control_dt, phases)
        finally:
            # A per-JOINT envelope (geared/belt plant) writes jnt_actfrcrange, which MuJoCo and
            # every later reader treat as the rating; put the rating back. The
            # per-actuator path (unit-gear plant) never restored and still does not, so its
            # executions are unchanged.
            if self.dc is not None and self.dc.joint_clamp:
                self.dc.restore()

    def _run(self, initial_state: np.ndarray, controller, steps: int, control_dt: float,
             phases: list | None = None) -> Execution:
        m = self.model
        if not np.isclose(control_dt, m.opt.timestep * self.substeps, rtol=1e-6, atol=0):
            raise ValueError("control_dt must equal physics timestep x substeps")
        if phases is not None and len(phases) != steps:
            raise ValueError("phases must label every control step")
        data = restore(m, initial_state)
        nu = m.nu
        qpos = [np.array(data.qpos)]
        qvel = [np.array(data.qvel)]
        ctrl = [np.array(data.ctrl)]
        contacts = [np.zeros(5, bool)]
        samples = {k: [] for k in SAMPLE_CHANNELS}
        table = {k: [] for k in CONTACT_CHANNELS}
        sample_step, info = [], []
        index = 0
        for k in range(steps):
            command, extra = controller(k, data)
            command = np.asarray(command, float)
            if command.shape != (nu,) or not np.all(np.isfinite(command)):
                raise ValueError(f"controller returned an invalid command at step {k}")
            data.ctrl[:] = command
            union = np.zeros(5, bool)
            for _ in range(self.substeps):
                qpos_pre, qvel_pre, time_pre = np.array(data.qpos), np.array(data.qvel), float(data.time)
                samples["ctrl"].append(command.copy())
                if self.dc is not None:
                    self.dc.apply(data)
                elif self.cmd is not None:
                    self.cmd.apply(data, command)
                mujoco.mj_step(m, data)
                self._sample(data, qpos_pre, qvel_pre, time_pre, samples, table, index, k)
                union |= self._union(data)
                sample_step.append(k)
                index += 1
            # The envelope still holds the values written for the LAST SUBSTEP's
            # pre-step speed, but this forward evaluates the POST-step state.
            # Boundary and terminal samples are taken from it, so a stale
            # envelope mis-reports the force at every control boundary -- up to
            # 2.141 N.m measured on a 0.3 rad/s turn.
            if self.dc is not None:
                self.dc.apply(data)
            elif self.cmd is not None:
                self.cmd.apply(data, command)
            mujoco.mj_forward(m, data)
            union |= self._union(data)
            if k < steps - 1:   # the last boundary is the terminal sample below
                self._sample(data, np.array(data.qpos), np.array(data.qvel), float(data.time),
                             {c: [] for c in SAMPLE_CHANNELS}, table, index - 1, k, boundary=True)
            qpos.append(np.array(data.qpos))
            qvel.append(np.array(data.qvel))
            ctrl.append(command)
            contacts.append(union)
            info.append(extra or {})
        # Terminal boundary: the final state after the last control-boundary
        # mj_forward, whose derived quantities describe exactly this state.
        terminal = {k: [] for k in SAMPLE_CHANNELS}
        # A command-clamp plant's data.ctrl is the CLAMPED value; the channel holds requests.
        terminal["ctrl"].append(np.array(ctrl[-1]) if self.cmd is not None else np.array(data.ctrl))
        self._sample(data, np.array(data.qpos), np.array(data.qvel), float(data.time),
                     terminal, table, index, steps - 1, boundary=True)
        return Execution(
            final_state=snapshot(m, data),
            initial_state=np.asarray(initial_state, float).copy(),
            qpos=np.array(qpos), qvel=np.array(qvel), ctrl=np.array(ctrl),
            contacts=np.array(contacts), samples={k: np.array(v) for k, v in samples.items()},
            sample_step=np.array(sample_step), control_dt=float(control_dt),
            physics_timestep=float(m.opt.timestep), substeps=self.substeps,
            terminal={k: np.array(v) for k, v in terminal.items()},
            contact_table=_table_arrays(table), phases=list(phases) if phases is not None else None,
            info=info,
        )


def _table_arrays(table: dict) -> dict:
    n = len(table["sample"])
    shapes = {"sample": (0,), "step": (0,), "boundary": (0,), "geom": (0, 2), "pos": (0, 3),
              "frame": (0, 9), "force": (0, 6), "dist": (0,), "velocity": (0, 3)}
    out = {}
    for key in CONTACT_CHANNELS:
        out[key] = np.array(table[key]) if n else np.zeros(shapes[key])
    for key in ("sample", "step", "boundary", "geom"):
        out[key] = out[key].astype(np.int64)
    return out


def replay(model: mujoco.MjModel, initial_state: np.ndarray, ctrl: np.ndarray,
           substeps: int, control_dt: float, phases: list | None = None,
           *, cfg) -> Execution:
    """Forward replay of saved commands. ctrl[0] is the t=0 command (unused)."""
    saved = np.asarray(ctrl, float)
    return Executor(model, substeps, cfg).run(initial_state, lambda k, d: (saved[k + 1], None),
                                         saved.shape[0] - 1, control_dt, phases)


def _max_abs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return float("inf")
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.where(both_nan, 0.0, np.abs(a - b))
    return float(np.max(diff)) if diff.size else 0.0


def agreement(a: Execution, b: Execution) -> dict:
    """Replay agreement over every recorded channel, not just the base pose."""
    if a.qpos.shape != b.qpos.shape or a.ctrl.shape != b.ctrl.shape:
        return {"comparable": False}
    dq = []
    for qa, qb in zip(a.qpos, b.qpos):
        err = np.zeros(3)
        mujoco.mju_subQuat(err, qa[3:7], qb[3:7])
        dq.append(np.linalg.norm(err))
    samples = {k: _max_abs(a.samples.get(k, []), b.samples.get(k, [])) for k in SAMPLE_CHANNELS}
    terminal = {k: _max_abs(a.terminal.get(k, []), b.terminal.get(k, [])) for k in SAMPLE_CHANNELS}
    return {
        "comparable": True,
        "translation_max_m": float(np.max(np.abs(a.qpos[:, :3] - b.qpos[:, :3]))),
        "orientation_max_rad": float(np.max(dq)) if np.all(np.isfinite(dq)) else float("inf"),
        "joint_max_rad": _max_abs(a.qpos[:, 7:], b.qpos[:, 7:]),
        "velocity_max": _max_abs(a.qvel, b.qvel),
        "ctrl_max": _max_abs(a.ctrl, b.ctrl),
        "contact_mismatch_frames": int(np.sum(np.any(a.contacts != b.contacts, axis=1)))
        if a.contacts.shape == b.contacts.shape else a.contacts.shape[0],
        "sample_max": max(samples.values()) if samples else float("inf"),
        "terminal_max": max(terminal.values()) if terminal else float("inf"),
        "sample_channel_max": samples,
        "contact_table_max": max((_max_abs(a.contact_table.get(k, []), b.contact_table.get(k, []))
                                  for k in CONTACT_CHANNELS), default=float("inf")),
    }

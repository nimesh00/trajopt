"""Fail-closed MuJoCo acceptance gates.

Rules, each a fix for a false pass reproduced in an earlier validator:

  * integrity is a complete typed schema: exact shapes, finite values, value
    domains, stepping consistent with the model, timestamps with the right
    origin and spacing, the saved snapshot restoring exactly to frame 0,
    physics samples consistent with the frames and commands they belong to,
    and derived channels (joint excess) consistent with raw ones. Anything
    malformed FAILS; nothing below integrity runs on untrusted data
  * the terminal hold is a closed interval: frames T-H..T, every physics
    sample of steps T-H..T-1, and the terminal boundary sample
  * hold pose, height, tilt and speeds are checked at physics rate as well as
    at the frames, so an excursion between control frames is not invisible
  * support is SIMULTANEOUS loaded feet per sample, never contact-flag unions
  * a loaded foot with NaN slip or penetration fails; it is not zero
  * the DC torque-speed law is evaluated at the measured velocity, unclipped:
    beyond the corner speed the admissible interval is empty and any torque fails
  * body contact is allowed only in explicitly named phases; never in the hold
  * report-only diagnostics are labelled `info` and never contribute a PASS

Thresholds are the standing defaults: simulator acceptance criteria, not
hardware certification.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from . import plantmap
from .execution import CONTACT_CHANNELS, SAMPLE_CHANNELS, STATE_SPEC, Execution

GATES_VERSION = 2


@dataclass
class StandingSpec:
    hold_seconds: float = 1.0
    height_ref: float = float("nan")          # calibrated settled standing base height
    joints_ref: np.ndarray | None = None      # calibrated settled joints (actuator order)
    height_tol: float = 0.025
    tilt_max_deg: float = 10.0
    joint_rms_max: float = 0.12
    base_speed_max: float = 0.10
    angular_speed_max: float = 0.30
    joint_speed_rms_max: float = 0.30
    loaded_force: float = 5.0
    min_loaded_feet: int = 3
    all_four_fraction: float = 0.90
    slip_p95_max: float = 0.03
    slip_distance_max: float = 0.02
    penetration_max: float = 0.005
    penetration_p99_max: float = 0.002
    joint_excess_max: float = 0.01
    body_contact_phases: tuple = ()           # phases where non-foot ground contact is allowed
    min_duration: float = 1.0

    def as_dict(self):
        d = asdict(self)
        d["joints_ref"] = None if self.joints_ref is None else np.asarray(self.joints_ref).tolist()
        d["body_contact_phases"] = list(self.body_contact_phases)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown acceptance fields {sorted(unknown)}")
        if d.get("joints_ref") is not None:
            d["joints_ref"] = np.asarray(d["joints_ref"], float)
        d["body_contact_phases"] = tuple(d.get("body_contact_phases", ()))
        return cls(**d)


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str
    required: bool = True


@dataclass
class Report:
    gates: list[Gate] = field(default_factory=list)

    def add(self, name, passed, detail, required=True):
        self.gates.append(Gate(name, bool(passed), detail, required))

    @property
    def passed(self) -> bool:
        required = [g for g in self.gates if g.required]
        return bool(required) and all(g.passed for g in required)

    def as_dict(self):
        return {"version": GATES_VERSION, "passed": self.passed,
                "gates": [asdict(g) for g in self.gates]}

    def __str__(self):
        lines = [f"VALIDATION: {'PASS' if self.passed else 'FAIL'}"]
        for g in self.gates:
            tag = ("PASS" if g.passed else "FAIL") if g.required else "info"
            lines.append(f"  [{tag}] {g.name:<26s} {g.detail}")
        return "\n".join(lines)


def _tilt_deg(quat_wxyz):
    q = np.asarray(quat_wxyz, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    up = 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)
    return np.degrees(np.arccos(np.clip(up, -1.0, 1.0)))


def actuator_addresses(model):
    # Each actuator's PRIMARY joint (plantmap): a belt actuator's trnid is a tendon id.
    joints = [int(j) for j in plantmap.primary_joints(model)]
    return (np.array([model.jnt_qposadr[j] for j in joints]),
            np.array([model.jnt_dofadr[j] for j in joints]), joints)


def _integrity_problems(ex: Execution, model: mujoco.MjModel, spec: StandingSpec) -> list[str]:
    """Every structural, domain and consistency problem. Empty means trustworthy."""
    p: list[str] = []
    nq, nv, nu = model.nq, model.nv, model.nu
    qadr, dadr, joints = actuator_addresses(model)
    ns = len(joints)            # servo actuators: nu, less any held (a wheel-legged plant's wheels)

    # stepping
    if not isinstance(ex.substeps, (int, np.integer)) or ex.substeps < 1:
        return [f"substeps {ex.substeps!r} is not a positive integer"]
    if not np.isclose(ex.physics_timestep, model.opt.timestep, rtol=0, atol=1e-12):
        p.append(f"physics_timestep {ex.physics_timestep} != model {model.opt.timestep}")
    if not (np.isfinite(ex.control_dt) and np.isclose(ex.control_dt, model.opt.timestep * ex.substeps,
                                                      rtol=1e-6, atol=0)):
        p.append(f"control_dt {ex.control_dt} != timestep x substeps")
    if p:
        return p

    # frames
    T = ex.ctrl.shape[0] - 1 if np.ndim(ex.ctrl) == 2 else -1
    if T < 1:
        return ["no control steps"]
    S = T * ex.substeps
    shapes = {"qpos": (ex.qpos, (T + 1, nq)), "qvel": (ex.qvel, (T + 1, nv)),
              "ctrl": (ex.ctrl, (T + 1, nu)), "contacts": (ex.contacts, (T + 1, 5)),
              "sample_step": (ex.sample_step, (S,))}
    expected = {"time": (S,), "ctrl": (S, nu), "actuator_force": (S, ns), "joint_q": (S, ns),
                "joint_qvel": (S, ns), "joint_excess": (S,), "base_qpos": (S, 7), "base_qvel": (S, 6),
                "foot_normal": (S, 4), "foot_tangential": (S, 4), "foot_slip": (S, 4),
                "foot_penetration": (S, 4), "body_ground_contacts": (S,), "body_ground_force": (S,),
                "self_contacts": (S,), "warnings": (S,)}
    if set(expected) != set(SAMPLE_CHANNELS):
        p.append("validator and executor disagree on sample channels")
    for name, shape in expected.items():
        shapes[f"samples.{name}"] = (ex.samples.get(name), shape)
        shapes[f"terminal.{name}"] = (ex.terminal.get(name), (1,) + shape[1:])
    for name, (arr, shape) in shapes.items():
        if arr is None:
            p.append(f"missing {name}")
        elif np.shape(arr) != shape:
            p.append(f"{name} shape {np.shape(arr)} != {shape}")
    if p:
        return p

    def chan(name):
        return np.concatenate([ex.samples[name], ex.terminal[name]])

    for name in ("qpos", "qvel", "ctrl"):
        if not np.all(np.isfinite(getattr(ex, name))):
            p.append(f"{name} not finite")
    for name in ("time", "ctrl", "actuator_force", "joint_q", "joint_qvel", "joint_excess", "base_qpos",
                 "base_qvel", "foot_normal", "foot_tangential", "body_ground_contacts",
                 "body_ground_force", "self_contacts", "warnings"):
        if not np.all(np.isfinite(chan(name))):
            p.append(f"samples.{name} not finite")
    if p:
        return p
    if ex.contacts.dtype != bool or np.any(ex.contacts[0]):
        p.append("contact flags must be boolean with an empty frame-0 row")

    # domains
    for name in ("foot_normal", "foot_tangential", "body_ground_force", "joint_excess"):
        if np.any(chan(name) < 0):
            p.append(f"samples.{name} negative")
    for name in ("body_ground_contacts", "self_contacts", "warnings"):
        v = chan(name)
        if np.any(v < 0) or np.any(v != np.round(v)):
            p.append(f"samples.{name} not non-negative integers")
    if np.any(np.diff(chan("warnings")) < 0):
        p.append("warning counter decreased")
    for name in ("foot_slip", "foot_penetration"):
        v = chan(name)
        if np.any(v[np.isfinite(v)] < 0):
            p.append(f"samples.{name} negative")
        if np.any(~np.isfinite(v) & (chan("foot_normal") > 0)):
            p.append(f"samples.{name} missing for a foot carrying force")
    norms = np.linalg.norm(ex.qpos[:, 3:7], axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-6) or np.any(np.abs(np.linalg.norm(chan("base_qpos")[:, 3:7], axis=1) - 1) > 1e-6):
        p.append("base quaternion not unit")

    # timing
    t = chan("time")
    t0 = float(t[0])
    if not np.allclose(t, t0 + np.arange(S + 1) * model.opt.timestep, rtol=0, atol=1e-9):
        p.append("physics timestamps do not advance by exactly one timestep from the start")
    if not np.array_equal(ex.sample_step, np.repeat(np.arange(T), ex.substeps)):
        p.append("sample-to-step index is inconsistent")
    if ex.phases is not None and len(ex.phases) != T:
        p.append("phase labels do not cover every control step")
    if T * ex.control_dt < max(spec.min_duration, spec.hold_seconds) - 1e-9:
        p.append(f"duration {T * ex.control_dt:.3f} s shorter than required")

    # snapshot restores exactly to frame 0
    state = np.asarray(ex.initial_state, float)
    if state.size != mujoco.mj_stateSize(model, STATE_SPEC) or not np.all(np.isfinite(state)):
        p.append("initial snapshot has the wrong size or non-finite values")
    else:
        d = mujoco.MjData(model)
        mujoco.mj_setState(model, d, state, STATE_SPEC)
        if not (np.array_equal(d.qpos, ex.qpos[0]) and np.array_equal(d.qvel, ex.qvel[0])
                and np.array_equal(d.ctrl, ex.ctrl[0]) and d.time == t0):
            p.append("initial snapshot does not restore to frame 0 / first sample time")

    # samples consistent with frames and commands
    first = np.arange(T) * ex.substeps
    step_ctrl = np.repeat(ex.ctrl[1:], ex.substeps, axis=0)
    if not np.array_equal(ex.samples["ctrl"], step_ctrl) or not np.array_equal(ex.terminal["ctrl"][0], ex.ctrl[-1]):
        p.append("physics-rate commands differ from the command applied for their interval")
    if not (np.array_equal(ex.samples["joint_q"][first], ex.qpos[:-1, qadr])
            and np.array_equal(ex.samples["joint_qvel"][first], ex.qvel[:-1, dadr])
            and np.array_equal(ex.samples["base_qpos"][first], ex.qpos[:-1, :7])
            and np.array_equal(ex.samples["base_qvel"][first], ex.qvel[:-1, :6])
            and np.array_equal(ex.terminal["joint_q"][0], ex.qpos[-1, qadr])
            and np.array_equal(ex.terminal["base_qpos"][0], ex.qpos[-1, :7])
            and np.array_equal(ex.terminal["base_qvel"][0], ex.qvel[-1, :6])):
        p.append("physics samples disagree with the frame states they start from")
    # contact flags must match the contacts actually recorded in that interval
    table = ex.contact_table
    if set(table) != set(CONTACT_CHANNELS) or any(len(table[c]) != len(table["sample"]) for c in CONTACT_CHANNELS):
        p.append("contact table missing or ragged")
    else:
        feet = plantmap.foot_geoms(model, ("FL", "FR", "RL", "RR"))
        world = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0}
        union = np.zeros((T, 5), bool)
        steps_col = np.asarray(table["step"], int)
        if steps_col.size and (steps_col.min() < 0 or steps_col.max() >= T):
            p.append("contact table references steps outside the motion")
        else:
            for step, (ga, gb) in zip(steps_col, np.asarray(table["geom"], int)):
                aw, bw = int(ga) in world, int(gb) in world
                if aw == bw:
                    continue
                other = int(gb) if aw else int(ga)
                union[step, feet.index(other) if other in feet else 4] = True
            if not np.array_equal(union, ex.contacts[1:]):
                p.append("contact flags disagree with the recorded contacts")
            # per-foot forces and body/self counts must equal what the contacts say
            normal = np.zeros((S + 1, 4))
            tangential = np.zeros((S + 1, 4))
            body_n = np.zeros(S + 1, int)
            body_f = np.zeros(S + 1)
            self_n = np.zeros(S + 1, int)
            rows = zip(np.asarray(table["sample"], int), np.asarray(table["boundary"], int),
                       np.asarray(table["geom"], int), np.asarray(table["force"], float))
            for sample, boundary, (ga, gb), force in rows:
                if boundary and sample != S:
                    continue            # intermediate boundary passes are not samples
                aw, bw = int(ga) in world, int(gb) in world
                if aw and bw:
                    continue
                if not aw and not bw:
                    self_n[sample] += 1
                    continue
                other = int(gb) if aw else int(ga)
                if other in feet:
                    normal[sample, feet.index(other)] += abs(force[0])
                    tangential[sample, feet.index(other)] += np.hypot(force[1], force[2])
                else:
                    body_n[sample] += 1
                    body_f[sample] += abs(force[0])
            for name, recomputed in (("foot_normal", normal), ("foot_tangential", tangential),
                                     ("body_ground_contacts", body_n), ("body_ground_force", body_f),
                                     ("self_contacts", self_n)):
                if not np.allclose(chan(name), recomputed, rtol=0, atol=1e-9):
                    p.append(f"samples.{name} disagrees with the recorded contact forces")
    lo, hi = model.jnt_range[joints, 0], model.jnt_range[joints, 1]
    q = chan("joint_q")
    excess = np.maximum(np.max(lo - q, axis=1, initial=0.0), np.max(q - hi, axis=1, initial=0.0))
    if not np.allclose(excess, chan("joint_excess"), rtol=0, atol=1e-12):
        p.append("joint_excess disagrees with joint positions")
    return p


def _dc_exceedance(tau, dq, force_lim, sat, vnl):
    """Distance outside the DC motor region, in the SAME algebra execution enforces.

    The velocity clip is not cosmetic. Both trajgen/dcmotor.py and mjlab's
    dc_motor_clip evaluate the curve at a speed clipped to +/-vel_at_lim, so past
    that speed they hold the bound at -force_lim instead of letting it keep
    falling. Omitting the clip here made the gate a DIFFERENT specification from
    the thing it scores: it reported exceedance against a bound execution never
    applied, exactly in the regime -- a turn reaching 14.15 rad/s at the knee --
    where the clamp binds every step.
    """
    vel_at_lim = vnl * (1.0 + force_lim / sat)
    v = np.clip(dq, -vel_at_lim, vel_at_lim)
    hi = np.minimum(sat * (1.0 - v / vnl), force_lim)
    lo = np.maximum(sat * (-1.0 - v / vnl), -force_lim)
    lo = np.minimum(lo, hi)     # see dcmotor.apply: the endpoint rounds apart
    return np.maximum(np.maximum(tau - hi, lo - tau), 0.0), hi < lo


def validate_standing(ex: Execution, model: mujoco.MjModel, spec: StandingSpec,
                      motor: dict, action_scale: float) -> Report:
    report = Report()
    try:
        problems = _integrity_problems(ex, model, spec)
    except Exception as error:  # noqa: BLE001 - malformed input fails, never raises
        problems = [f"malformed execution: {type(error).__name__}: {error}"]
    if not problems:
        report.add("integrity", True, f"{ex.steps} control steps, {ex.steps * ex.substeps} physics "
                   "samples + terminal; schema, domains, timing, snapshot and consistency checked")
    else:
        report.add("integrity", False, "; ".join(problems[:6]) + (" ..." if len(problems) > 6 else ""))
        return report
    if not (np.isfinite(spec.height_ref) and spec.joints_ref is not None
            and np.shape(spec.joints_ref) == (len(plantmap.primary_joints(model)),)
            and np.all(np.isfinite(spec.joints_ref))):
        report.add("calibration", False, "standing reference height/joints not calibrated")
        return report
    if not (np.isfinite(action_scale) and action_scale > 0):
        report.add("configuration", False, f"action scale {action_scale} invalid")
        return report

    def chan(name):
        return np.concatenate([ex.samples[name], ex.terminal[name]])

    T, sub = ex.steps, ex.substeps
    H = int(round(spec.hold_seconds / ex.control_dt))
    qadr, dadr, joints = actuator_addresses(model)

    # ---------------------------------------------------------- whole motion
    tr = plantmap.transmission(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a)) for a in tr.servo]
    # THE RATING, not whatever the last DC step left in actuator_forcerange.
    # dcmotor.apply() drives the envelope by overwriting that field, so reading
    # it back here scores the motion against a clamped value from some earlier
    # rollout. jnt_actfrcrange carries the same numbers and is only ever read on
    # unit-servo plants; on a per-joint plant dcmotor writes IT (and restores it
    # after each rollout), so the rating comes from plantmap's cache, captured
    # before any write. `actuator_force` is the joint torque on such a plant.
    force_lim = plantmap.transmission(model).limit
    tau, dq = chan("actuator_force"), chan("joint_qvel")
    over = np.abs(tau) - force_lim
    report.add("torque", np.all(over <= 1e-6), f"peak |tau|/limit {np.max(np.abs(tau) / force_lim):.2f}")
    try:
        sat = np.array([motor[n.rsplit('_', 1)[-1]][0] for n in names], float)
        vnl = np.array([motor[n.rsplit('_', 1)[-1]][1] for n in names], float)
        motor_ok = np.all(np.isfinite(sat)) and np.all(sat > 0) and np.all(np.isfinite(vnl)) and np.all(vnl > 0)
    except (KeyError, IndexError, TypeError, ValueError):
        motor_ok = False
    if not motor_ok:
        report.add("torque-speed", False, "motor parameters missing or invalid")
    else:
        exceed, empty = _dc_exceedance(tau, dq, force_lim, sat, vnl)
        report.add("torque-speed", np.all(exceed <= 1e-6) and not np.any(empty),
                   f"max exceedance {exceed.max():.3f} N m, {int(np.sum(empty))} samples beyond "
                   f"corner speed (peak |dq| {np.abs(dq).max():.2f} rad/s)")

    lo_c, hi_c = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    ctrl_ok = bool(np.all((ex.ctrl >= lo_c - 1e-9) & (ex.ctrl <= hi_c + 1e-9)))
    # Deltas in SERVO-POSITION units (ctrl / gear), the units action_scale is stated
    # in; identical to raw ctrl deltas on a unit-gear plant.
    actions = np.diff(tr.positions(tr.servo_part(ex.ctrl)), axis=0) / action_scale
    report.add("commands", ctrl_ok and np.all(np.abs(actions) <= 1.0 + 1e-9),
               f"within ctrlrange: {ctrl_ok}; max |action| {np.abs(actions).max():.3f} "
               f"(command-delta encoding, scale {action_scale})")
    report.add("joint limits", chan("joint_excess").max() <= spec.joint_excess_max,
               f"max excursion {chan('joint_excess').max():.4f} rad (limit {spec.joint_excess_max})")
    warn = int(chan("warnings").max())
    report.add("no warnings", warn == 0, f"{warn} MuJoCo warnings")

    body = chan("body_ground_contacts")
    if spec.body_contact_phases:
        if ex.phases is None:
            report.add("body contact", False, "phase-scoped allowance needs phase labels")
            allowed = np.zeros(body.shape, bool)
        else:
            step_phase = np.array(list(ex.phases) + [None])
            sample_phase = np.concatenate([step_phase[ex.sample_step], [None]])
            allowed = np.isin(sample_phase, list(spec.body_contact_phases))
    else:
        allowed = np.zeros(body.shape, bool)
    hold_mask = np.concatenate([ex.sample_step >= T - H, [True]])
    allowed &= ~hold_mask
    n = int(np.sum((body > 0) & ~allowed))
    report.add("body contact", n == 0, f"{n} samples with disallowed non-foot ground contact "
               f"(allowed phases: {list(spec.body_contact_phases) or 'none'}; never in hold)")
    report.add("self contacts", True, f"{int(np.sum(chan('self_contacts') > 0))} samples", required=False)

    # ---------------------------------------------- hold: closed interval T-H..T
    frames = np.arange(T - H, T + 1)
    hs = np.nonzero(hold_mask)[0]                       # samples of steps T-H..T-1 + terminal
    base_q = np.concatenate([ex.qpos[frames, :7], chan("base_qpos")[hs]])
    base_v = np.concatenate([ex.qvel[frames, :6], chan("base_qvel")[hs]])
    joint_q = np.concatenate([ex.qpos[frames][:, qadr], chan("joint_q")[hs]])
    joint_v = np.concatenate([ex.qvel[frames][:, dadr], chan("joint_qvel")[hs]])

    height_err = np.abs(base_q[:, 2] - spec.height_ref)
    report.add("hold height", np.all(height_err <= spec.height_tol),
               f"max |z - {spec.height_ref:.4f}| = {height_err.max():.4f} m (tol {spec.height_tol}), "
               f"{len(frames)} frames + {len(hs)} samples")
    tilt = _tilt_deg(base_q[:, 3:7])
    report.add("hold tilt", np.all(tilt <= spec.tilt_max_deg), f"max {tilt.max():.2f} deg (limit {spec.tilt_max_deg})")
    rms = np.sqrt(np.mean((joint_q - np.asarray(spec.joints_ref)) ** 2, axis=1))
    report.add("hold pose", np.all(rms <= spec.joint_rms_max),
               f"max joint rms {rms.max():.4f} rad (limit {spec.joint_rms_max})")
    lin = np.linalg.norm(base_v[:, :3], axis=1)
    ang = np.linalg.norm(base_v[:, 3:6], axis=1)
    jspd = np.sqrt(np.mean(joint_v ** 2, axis=1))
    report.add("hold speed", np.all(lin <= spec.base_speed_max) and np.all(ang <= spec.angular_speed_max)
               and np.all(jspd <= spec.joint_speed_rms_max),
               f"max base {lin.max():.3f} m/s, ang {ang.max():.3f} rad/s, joint rms {jspd.max():.3f} rad/s")

    normal = chan("foot_normal")[hs]
    loaded = normal > spec.loaded_force
    count = loaded.sum(axis=1)
    all_four = float(np.mean(count == 4))
    report.add("hold support", count.min() >= spec.min_loaded_feet and all_four >= spec.all_four_fraction,
               f"min simultaneous loaded feet {count.min()}, all four {all_four:.1%} "
               f"(need >= {spec.min_loaded_feet}, {spec.all_four_fraction:.0%})")

    slip = chan("foot_slip")[hs]
    if np.any(loaded & ~np.isfinite(slip)):
        report.add("hold slip", False, "loaded foot with unmeasured slip")
    else:
        vals = slip[loaded]
        p95 = float(np.percentile(vals, 95)) if vals.size else float("nan")
        # integrate over the physics intervals only; the terminal boundary has no duration
        inner = loaded[:-1]
        dist = np.where(inner, slip[:-1], 0.0).sum(axis=0) * model.opt.timestep
        report.add("hold slip", vals.size > 0 and p95 <= spec.slip_p95_max and dist.max() <= spec.slip_distance_max,
                   f"loaded p95 {p95:.4f} m/s, worst foot {dist.max():.4f} m "
                   f"(limits {spec.slip_p95_max}, {spec.slip_distance_max})")

    pen = chan("foot_penetration")[hs]
    if np.any(loaded & ~np.isfinite(pen)):
        report.add("hold penetration", False, "loaded foot with unmeasured penetration")
    else:
        pvals = pen[loaded]
        finite = pen[np.isfinite(pen)]
        pmax = float(finite.max()) if finite.size else float("nan")
        p99 = float(np.percentile(pvals, 99)) if pvals.size else float("nan")
        report.add("hold penetration", pvals.size > 0 and pmax <= spec.penetration_max
                   and p99 <= spec.penetration_p99_max,
                   f"max {pmax * 1e3:.2f} mm, loaded p99 {p99 * 1e3:.2f} mm "
                   f"(limits {spec.penetration_max * 1e3:.0f}/{spec.penetration_p99_max * 1e3:.0f})")

    peak = chan("foot_normal").max()
    report.add("peak foot force", True,
               f"{peak:.0f} N ({peak / (np.sum(model.body_mass) * 9.81):.2f} x weight)", required=False)
    return report

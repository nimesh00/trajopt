"""Strict robot configuration (robots/*.toml).

Fail closed: every key is required, unknown keys are errors, and the contract
is checked against the compiled model rather than trusted.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

SCHEMA = {
    "model": {"mjb": str, "name": str},
    "contract": {"physics_timestep": float, "substeps": int, "kp": float, "kd": float,
                 "action_scale": float},
    "motor": {"hip": list, "thigh": list, "calf": list},
    "contact": {"mu": float, "cone_margin": float},
    "pose": {"hip": float, "thigh": float, "calf": float},
}
# OPTIONAL keys, still typed and still closed: an unknown key is an error either way.
# [model].name is required: the robot name in clip task ids.
#
#   [model].urdf | [model].mjcf   EXACTLY ONE: the Pinocchio description. A plant with no
#                                 URDF matching it has Pinocchio read the same MJCF the
#                                 trainer compiles (adapter.build dispatches on it).
#   [pose].rear_*                 rear-leg overrides of the per-role pose: a geared plant stands
#                                 front 0.721028/-1.524295, rear 0.809954/-1.522588
#                                 (thigh/calf, its 0.30 m `stand` keyframe); one value
#                                 per role cannot express that.
#   [gait]                        robot-sized walk scheduler values: leg_reach (hip-to-foot
#                                 extension, required to walk) and optional period / duty /
#                                 swing_height over the gait pattern's own. CLI flags still win.
#   [contract].integrator         the compiled model's integrator, "IMPLICITFAST" when
#                                 absent (the trainer's compiled plants). A wheel-legged
#                                 plant may keep its MJCF's "RK4".
#   [contract].envelope           how the DC law is enforced while stepping: "forcerange"
#                                 when absent (trajgen.dcmotor.DcClamp rewrites the force
#                                 range); "command" clamps the COMMAND every substep, the
#                                 command-clamp plant interface (dcmotor.CommandClamp).
#   [motor].wheel                 [stall, no-load] of a HELD velocity-servo wheel drive.
#   [build]                       how trajgen.build_plant compiles [model].mjb from a source
#                                 MJCF: `source` (the torque-motor MJCF), `feet` (its foot geom
#                                 name pattern, renamed {leg}_foot in the plant), optionally
#                                 `foot_solimp` / `foot_solref` / `foot_friction` (replace
#                                 the source's foot contact parameters), `effort`
#                                 (N m per role, replaces the source motors' ctrlrange as the
#                                 force limit) and `repo` + `commit` + `checkout` to fetch the
#                                 source with git.
OPTIONAL = {
    "model": {"urdf": str, "mjcf": str},
    "contract": {"integrator": str, "envelope": str},
    "motor": {"wheel": list},
    "pose": {"rear_hip": float, "rear_thigh": float, "rear_calf": float},
    "gait": {"leg_reach": float, "period": float, "duty": float, "swing_height": float},
    "build": {"source": str, "feet": str, "foot_solimp": list, "repo": str, "commit": str,
              "checkout": str, "effort": dict, "foot_solref": list,
              "foot_friction": list},
}
INTEGRATORS = {"IMPLICITFAST": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
               "RK4": mujoco.mjtIntegrator.mjINT_RK4}
ENVELOPES = ("forcerange", "command")


def _expand(value: str, base: Path) -> Path:
    def env(match):
        name, default = match.group(1), match.group(3)
        if name in os.environ:
            return os.environ[name]
        if default is None:
            raise KeyError(f"environment variable {name} is not set")
        return default

    value = re.sub(r"\$\{(\w+)(:-([^}]*))?\}", env, value)
    path = Path(os.path.expanduser(value))
    return path if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True)
class RobotConfig:
    path: Path
    sha256: str
    mjb: Path
    urdf: Path
    physics_timestep: float
    substeps: int
    kp: float
    kd: float
    action_scale: float
    motor: dict            # role -> (sat, v_nl)
    mu: float
    cone_margin: float
    pose: dict             # role -> default joint angle (rear_<role> overrides rear legs)
    raw: dict
    name: str
    description_format: str = "urdf"   # what `urdf` points at: "urdf" or "mjcf"
    gait: dict = None                  # optional walk defaults, {} when absent
    integrator: str = "IMPLICITFAST"
    envelope: str = "forcerange"

    @property
    def control_dt(self) -> float:
        return self.physics_timestep * self.substeps

    def pose_for(self, name: str) -> float:
        """Default angle for an actuator or joint NAME ("RL_calf", "FL_thigh_joint").

        Rear legs (R*) take [pose].rear_<role> when the config has it, so a robot whose
        front and rear stance differ is described exactly; otherwise every leg takes
        the per-role value, which is all a symmetric robot's config says."""
        leg, role = name.split("_")[0], name.split("_")[1]
        if leg.startswith("R") and f"rear_{role}" in self.pose:
            return self.pose[f"rear_{role}"]
        return self.pose[role]


def load(path: str | Path, check_files: bool = True) -> RobotConfig:
    """`check_files=False` skips the existence check of the plant and the description,
    for trajgen.build_plant, which reads the config to make them."""
    path = Path(path).resolve()
    text = path.read_bytes()
    raw = tomllib.loads(text.decode())
    unknown = set(raw) - set(SCHEMA) - set(OPTIONAL)
    if unknown:
        raise ValueError(f"{path}: unknown sections {sorted(unknown)}")
    for section in set(SCHEMA) | set(OPTIONAL):
        keys = SCHEMA.get(section, {})
        optional = OPTIONAL.get(section, {})
        if section not in raw:
            if keys:
                raise ValueError(f"{path}: missing section [{section}]")
            continue
        extra = set(raw[section]) - set(keys) - set(optional)
        missing = set(keys) - set(raw[section])
        if extra or missing:
            raise ValueError(f"{path}: [{section}] unknown {sorted(extra)} missing {sorted(missing)}")
        for key, kind in (keys | optional).items():
            if key not in raw[section]:
                continue
            value = raw[section][key]
            if kind is float and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if not isinstance(value, kind) or isinstance(value, bool):
                raise ValueError(f"{path}: [{section}].{key} must be {kind.__name__}")
            raw[section][key] = value
    for role, pair in raw["motor"].items():
        if len(pair) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                     and math.isfinite(v) and v > 0 for v in pair):
            raise ValueError(f"{path}: [motor].{role} must be finite [sat, v_nl] > 0")
    c, k = raw["contract"], raw["contact"]
    domains = [
        ("contract.physics_timestep", c["physics_timestep"], lambda v: 0 < v < 0.1),
        ("contract.substeps", c["substeps"], lambda v: 1 <= v <= 1000),
        ("contract.kp", c["kp"], lambda v: 0 < v < 1e5),
        ("contract.kd", c["kd"], lambda v: 0 <= v < 1e4),
        ("contract.action_scale", c["action_scale"], lambda v: 0 < v <= 10),
        ("contact.mu", k["mu"], lambda v: 0 < v <= 5),
        ("contact.cone_margin", k["cone_margin"], lambda v: 0 < v <= 1),
        ("pose.hip", raw["pose"]["hip"], lambda v: abs(v) < 3.2),
        ("pose.thigh", raw["pose"]["thigh"], lambda v: abs(v) < 3.2),
        ("pose.calf", raw["pose"]["calf"], lambda v: abs(v) < 3.2),
    ]
    domains += [(f"pose.{key}", raw["pose"][key], lambda v: abs(v) < 3.2)
                for key in OPTIONAL["pose"] if key in raw["pose"]]
    gait = raw.get("gait", {})
    gait_domains = {"leg_reach": lambda v: 0.05 < v < 2.0, "period": lambda v: 0.1 < v < 5.0,
                    "duty": lambda v: 0.0 < v < 1.0, "swing_height": lambda v: 0.0 < v < 0.5}
    domains += [(f"gait.{key}", value, gait_domains[key]) for key, value in gait.items()]
    for name, value, ok in domains:
        if not (math.isfinite(value) and ok(value)):
            raise ValueError(f"{path}: {name} = {value} is outside its valid range")
    c_opt = {key: raw["contract"][key] for key in OPTIONAL["contract"] if key in raw["contract"]}
    if c_opt.get("integrator", "IMPLICITFAST") not in INTEGRATORS:
        raise ValueError(f"{path}: [contract].integrator must be one of {sorted(INTEGRATORS)}")
    if c_opt.get("envelope", "forcerange") not in ENVELOPES:
        raise ValueError(f"{path}: [contract].envelope must be one of {ENVELOPES}")
    described = [key for key in ("urdf", "mjcf") if key in raw["model"]]
    if len(described) != 1:
        raise ValueError(f"{path}: [model] needs exactly one of urdf / mjcf, got {described}")
    # adapter.build dispatches on the suffix (.xml -> MJCF), so the key must agree with it.
    if (Path(raw["model"][described[0]]).suffix.lower() == ".xml") != (described[0] == "mjcf"):
        raise ValueError(f"{path}: [model].{described[0]} = {raw['model'][described[0]]!r}: "
                         "an MJCF must be given as mjcf (and end in .xml), a URDF as urdf")

    base = path.parent
    cfg = RobotConfig(
        path=path, sha256=hashlib.sha256(text).hexdigest(),
        mjb=_expand(raw["model"]["mjb"], base), urdf=_expand(raw["model"][described[0]], base),
        motor={k: (float(v[0]), float(v[1])) for k, v in raw["motor"].items()},
        mu=raw["contact"]["mu"], cone_margin=raw["contact"]["cone_margin"],
        pose={k: float(v) for k, v in raw["pose"].items()}, raw=raw,
        name=raw["model"]["name"], description_format=described[0],
        gait={k: float(v) for k, v in gait.items()},
        **{k: v for k, v in raw["contract"].items() if k not in OPTIONAL["contract"]},
        **c_opt,
    )
    for p in (cfg.mjb, cfg.urdf) if check_files else ():
        if not p.is_file():
            raise FileNotFoundError(f"{path}: {p} does not exist")
    return cfg


def assert_plant(cfg: RobotConfig, model: mujoco.MjModel) -> None:
    """The compiled model must be the plant this configuration describes.

    Two plant shapes are accepted, and both are checked against the SAME servo law
    (trajgen/plantmap.py): unit-gear joint servos, force-limited per actuator,
    and geared joint + belt-tendon servos with gainprm = kp/g^2,
    biasprm = [0, -kp/g^2, -kd/g^2] and the torque clamped per JOINT. In both, the
    joint-side law is tau = J^T (kp (ctrl/g - J q) - kd J qdot), so the config's kp/kd
    are the JOINT-side gains and are compared after undoing the g^2.
    """
    from . import plantmap

    problems = []
    if not np.isclose(model.opt.timestep, cfg.physics_timestep, rtol=0, atol=1e-9):
        problems.append(f"timestep {model.opt.timestep} != {cfg.physics_timestep}")
    if model.opt.integrator != INTEGRATORS[cfg.integrator]:
        problems.append(f"integrator is not {cfg.integrator}"
                        + (" (not the trainer's compiled plant)" if cfg.integrator == "IMPLICITFAST" else ""))
    try:
        tr = plantmap.transmission(model)
    except ValueError as error:
        raise ValueError(f"compiled model does not match the robot config:\n  {error}") from error
    for a in tr.held:
        # HELD actuators (velocity servos, plantmap.is_held): braked by the walk, locked in
        # the plan. Their drive law needs a [motor] entry; the command clamp reads it.
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a))
        role = name.rsplit("_", 1)[-1]
        if role not in cfg.motor:
            problems.append(f"{name}: held actuator has no [motor] entry for role '{role}'")
        if not (model.actuator_ctrllimited[a] and model.actuator_ctrlrange[a][0] <= 0
                <= model.actuator_ctrlrange[a][1]):
            problems.append(f"{name}: held actuator cannot be commanded to 0")
    if tr.held.size and cfg.envelope != "command":
        problems.append("held actuators need envelope = \"command\" (DcClamp is per servo joint)")
    for k, a in enumerate(tr.servo):      # k: servo index; a: actuator id (= k without held)
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        g2 = float(tr.gear[k]) ** 2
        kp = model.actuator_gainprm[a][0] * g2
        if model.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED or \
                model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE:
            problems.append(f"{name}: not a position servo")
        elif not (np.isclose(kp, cfg.kp) and np.isclose(-model.actuator_biasprm[a][1] * g2, cfg.kp)
                  and np.isclose(-model.actuator_biasprm[a][2] * g2, cfg.kd)):
            problems.append(f"{name}: joint-side gains kp={kp} kd={-model.actuator_biasprm[a][2] * g2} "
                            f"!= config {cfg.kp}/{cfg.kd} (gear {tr.gear[k]})")
        # The bridge inverts the servo law exactly, so every other term of MuJoCo's
        # general actuator must be absent.
        if model.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE:
            problems.append(f"{name}: has activation dynamics")
        if np.any(model.actuator_gainprm[a][1:] != 0) or model.actuator_biasprm[a][0] != 0 \
                or np.any(model.actuator_biasprm[a][3:] != 0):
            problems.append(f"{name}: servo has extra gain/bias terms")
        joint = int(tr.joints[k])
        if tr.joint_clamp:
            # Geared/belt plant: actuators are NOT force limited; MuJoCo clamps the summed
            # actuator torque on each joint to jnt_actfrcrange (12/12/24 N m).
            lo, hi = model.jnt_actfrcrange[joint]
            if not (lo < 0 < hi and np.isclose(-lo, hi)):
                problems.append(f"{name}: joint actfrcrange {model.jnt_actfrcrange[joint]} not symmetric")
        if model.actuator_forcelimited[a] or not tr.joint_clamp:
            if not model.actuator_forcelimited[a]:
                problems.append(f"{name}: no force limit")
            lo, hi = model.actuator_forcerange[a]
            if not (lo < 0 < hi and np.isclose(-lo, hi)):
                problems.append(f"{name}: forcerange {model.actuator_forcerange[a]} not symmetric")
        if not tr.identity and model.actuator_forcelimited[a]:
            # A per-actuator clamp on a coupled transmission bounds MOTOR force, which
            # the joint-space envelope (dcmotor, gates) does not model.
            problems.append(f"{name}: geared/tendon actuator is force limited per actuator")
        role = name.rsplit("_", 1)[-1]
        if role not in cfg.motor:
            problems.append(f"{name}: no [motor] entry for role '{role}'")
        if role not in cfg.pose:
            problems.append(f"{name}: no [pose] entry for role '{role}'")
        else:
            lo, hi = model.jnt_range[joint]
            if not lo <= cfg.pose_for(name) <= hi:
                problems.append(f"{name}: default pose {cfg.pose_for(name)} outside [{lo}, {hi}]")
    if problems:
        raise ValueError("compiled model does not match the robot config:\n  " + "\n  ".join(problems))

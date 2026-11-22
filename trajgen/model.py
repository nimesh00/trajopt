"""Load the trainer's compiled model and audit that it is the plant we think it is.

The raw MJCF is NOT the training plant. The trainer's model compiler rewrites it:
torque motors become force-limited position servos (gainprm[0]=kp,
biasprm[1]=-kp, biasprm[2]=-kd, forcerange=+-ctrlrange, ctrlrange=jnt_range),
the timestep becomes physics_timestep, the integrator becomes IMPLICITFAST, and
the solver becomes Newton with a pyramidal cone and 30 iterations.

Loading scene.xml instead would optimize against torque control at 0.0005 s.
Every trajectory produced that way would be unreachable in training, and
nothing downstream would notice. So this module refuses a model that has not
been through that transformation.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .contract import Contract

REQUIRED_MUJOCO = "3.11.0"


def load(path: str | Path) -> mujoco.MjModel:
    """Load a compiled .mjb exported by the training stack or trajgen.build_plant."""
    path = Path(path)
    if path.suffix != ".mjb":
        raise ValueError(
            f"{path} is not a compiled model. Compile one from a robot config with:\n"
            f"  python -m trajgen.build_plant robots/<robot>.toml\n"
            "or export it from the training stack.\n"
            "Loading the raw MJCF would target torque motors, not position servos."
        )
    if mujoco.__version__ != REQUIRED_MUJOCO:
        raise RuntimeError(
            f"MuJoCo {mujoco.__version__}; the trainer requires exactly {REQUIRED_MUJOCO}. "
            "Checkpoints and clips are rejected across versions."
        )
    return mujoco.MjModel.from_binary_path(str(path))


def stand_keyframe(model: mujoco.MjModel) -> int:
    """Index of the robot's standing keyframe.

    Most plants call it `stand`; the prepared Go1 plant calls its only keyframe
    `home`, and the trainer's own Go1 path just uses index 0. Requiring the literal
    name `stand` made every go1 plant fail at load, so fall back the same way the
    trainer does.
    """
    for name in ("stand", "home"):
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key >= 0:
            return key
    if model.nkey > 0:
        return 0
    raise ValueError("compiled model has no keyframe to use as the standing pose")


def read_contract(model: mujoco.MjModel, action_scale: float, substeps: int) -> Contract:
    """Derive the control contract from the compiled model itself."""
    names, lower, upper, qadr, dadr = [], [], [], [], []
    kp = kd = None
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if model.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED:
            raise ValueError(f"actuator {name} is not a fixed-gain servo; model is not compiled")
        if model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE:
            raise ValueError(f"actuator {name} is not an affine-bias servo; model is not compiled")
        gain = float(model.actuator_gainprm[a][0])
        bias_p = float(model.actuator_biasprm[a][1])
        bias_d = float(model.actuator_biasprm[a][2])
        if not np.isclose(gain, -bias_p):
            raise ValueError(f"actuator {name}: gainprm {gain} != -biasprm {bias_p}")
        if kp is None:
            kp, kd = gain, -bias_d
        elif not (np.isclose(gain, kp) and np.isclose(-bias_d, kd)):
            raise ValueError(f"actuator {name} gains differ from the rest of the robot")
        if not model.actuator_forcelimited[a]:
            raise ValueError(f"actuator {name} has no force limit; model is not compiled")
        joint = int(model.actuator_trnid[a][0])
        names.append(name)
        lower.append(float(model.jnt_range[joint][0]))
        upper.append(float(model.jnt_range[joint][1]))
        qadr.append(int(model.jnt_qposadr[joint]))
        dadr.append(int(model.jnt_dofadr[joint]))

    nominal = np.array([model.key_qpos[stand_keyframe(model)][i] for i in qadr], float)

    return Contract(
        control_dt=float(model.opt.timestep) * substeps,
        physics_timestep=float(model.opt.timestep),
        substeps=substeps,
        kp=float(kp),
        kd=float(kd),
        action_scale=float(action_scale),
        joint_lower=np.array(lower),
        joint_upper=np.array(upper),
        actuator_names=tuple(names),
        qpos_adr=np.array(qadr),
        dof_adr=np.array(dadr),
        nominal=nominal,
    )


def audit(model: mujoco.MjModel) -> dict:
    """Everything that must be recorded alongside a generated clip."""
    return {
        "mujoco_version": mujoco.__version__,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "timestep": float(model.opt.timestep),
        "integrator": int(model.opt.integrator),
        "solver": int(model.opt.solver),
        "cone": int(model.opt.cone),
        "iterations": int(model.opt.iterations),
        "impratio": float(model.opt.impratio),
        "gravity": [float(v) for v in model.opt.gravity],
        "total_mass": float(sum(model.body_mass)),
    }

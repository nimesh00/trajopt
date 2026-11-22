"""Standing pipeline fixture: supported squat and a one-second hold.

It exists to prove the pipeline end to end on the simplest problem with real
contact dynamics -- settled snapshot, Crocoddyl solve, servo bridge,
instrumented MuJoCo execution, fail-closed gates, bundle, fresh-process
replay -- which the other motions build on.

Start state. Commanding the stand keyframe directly sags to 0.351 m and keeps
creeping, because kp=40 needs position error to carry gravity. The fixture
commands the keyframe plus the static vertical-force torques through the
bridge, settles for `settle_seconds`, and calibrates the standing reference
(height, joints) from that settled state on this plant.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import pinocchio as pin

from .. import croc, execution, plantmap
from ..adapter import LEGS, Adapter
from ..bridge import ServoBridge
from ..config import RobotConfig
from ..gates import StandingSpec

NAME = "stand-squat"
STATUS = "standing pipeline fixture; belly-up recovery unsolved"


@dataclass
class Start:
    state: np.ndarray        # full mjSTATE_INTEGRATION
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    residual_speed: float
    info: dict


def default_qpos(model: mujoco.MjModel, ad: Adapter, cfg: RobotConfig, ground: float = 0.0):
    """The robot's default pose, standing on `ground`.

    The joint angles come from the robot config's [pose] section (per joint
    role, with rear-leg overrides where front and rear stand differently), and
    the base height is solved so the collision spheres just touch: a
    hand-written base height would either float the robot or bury its feet.
    """
    qpos = np.array(model.qpos0, float)
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    qpos[0:2] = 0.0
    tr = plantmap.transmission(model)
    joints = tr.joints                            # a belt's trnid is a tendon id
    for k, a in enumerate(tr.servo):              # held actuators' joints stay at qpos0
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a))
        qpos[model.jnt_qposadr[joints[k]]] = cfg.pose_for(name)
    qpos[2] = 0.0
    data = ad.pin.createData()
    pin.framesForwardKinematics(ad.pin, data, ad.q_from_mj(qpos))
    lowest = min(float(data.oMf[contact.frame].translation[2]) - contact.radius
                 for contact in ad.feet.values())
    qpos[2] = ground - lowest
    return qpos


def settled_stand(model: mujoco.MjModel, ad: Adapter, bridge: ServoBridge,
                  cfg: RobotConfig | None = None, min_seconds: float = 2.0,
                  max_seconds: float = 6.0, window: float = 0.5, speed_tol: float = 5e-3,
                  ground: float = 0.0) -> Start:
    """Settle until |qvel| <= speed_tol for a continuous `window`, at least
    `min_seconds`; refuse if that never happens within `max_seconds`."""
    if cfg is None:
        raise ValueError("settled_stand needs the robot config for the default pose")
    qkey = default_qpos(model, ad, cfg, ground)
    q = ad.q_from_mj(qkey)
    tau, fz, residual = croc.static_torques(ad, q)
    if residual > 1e-6 or np.any(fz <= 0):
        raise ValueError(f"keyframe is not statically supportable on four feet (fz {fz})")
    x = np.concatenate([q, np.zeros(ad.pin.nv)])
    ctrl, info = bridge.command(x, x, tau)
    if info["saturated"]:
        raise ValueError("static standing command saturates the command range")
    data = mujoco.MjData(model)
    data.qpos[:] = qkey
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    quiet_needed = int(round(window / dt))
    quiet, steps = 0, 0
    while True:
        mujoco.mj_step(model, data)
        steps += 1
        quiet = quiet + 1 if np.linalg.norm(data.qvel) <= speed_tol else 0
        if steps * dt >= min_seconds and quiet >= quiet_needed:
            break
        if steps * dt >= max_seconds:
            raise RuntimeError(f"standing start did not settle: |qvel| {np.linalg.norm(data.qvel):.2e} "
                               f"after {max_seconds} s (tol {speed_tol} for {window} s)")
        if not np.all(np.isfinite(data.qpos)) or any(w.number for w in data.warning):
            raise RuntimeError("standing start diverged or raised MuJoCo warnings while settling")
    mujoco.mj_forward(model, data)
    settle_seconds = steps * dt
    return Start(state=execution.snapshot(model, data), qpos=np.array(data.qpos),
                 qvel=np.array(data.qvel), ctrl=ctrl.copy(),
                 residual_speed=float(np.linalg.norm(data.qvel)),
                 info={"pose": dict(cfg.pose), "ground": ground,
                       "settle_seconds": settle_seconds,
                       "settle_criterion": {"speed_tol": speed_tol, "window": window,
                                            "min_seconds": min_seconds, "max_seconds": max_seconds},
                       "static_normal_forces": fz.tolist(), "static_torques": tau.tolist()})


def _smooth(a, b, n):
    s = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n + 1)[1:])
    return a + (b - a) * s


def schedule(ad: Adapter, cfg: RobotConfig, start: Start, depth: float = 0.06,
             hold_seconds: float = 1.2):
    """Phases as (label, seconds, CoM height offset profile)."""
    dt = cfg.control_dt
    x0 = ad.x_from_mj(start.qpos, start.qvel)
    posture = x0.copy()
    posture[ad.pin.nq:] = 0.0
    com0 = pin.centerOfMass(ad.pin, ad.pin.createData(), x0[:ad.pin.nq])
    n = lambda seconds: int(round(seconds / dt))  # noqa: E731
    plan = [
        ("settle", np.zeros(n(0.5))),
        ("lower", _smooth(0.0, -depth, n(1.0))),
        ("bottom", np.full(n(0.3), -depth)),
        ("rise", _smooth(-depth, 0.0, n(1.0))),
        ("hold", np.zeros(n(hold_seconds))),
    ]
    knots = [croc.Knot(support=LEGS, com=com0 + [0.0, 0.0, dz], posture=posture, phase=label)
             for label, profile in plan for dz in profile]
    terminal = croc.Knot(support=LEGS, com=com0, posture=posture, phase="hold")
    return x0, knots, terminal, {"depth": depth, "com0": com0.tolist(),
                                 "phases": [(label, len(p)) for label, p in plan]}


def acceptance(model: mujoco.MjModel, start: Start) -> StandingSpec:
    qadr = [model.jnt_qposadr[j] for j in plantmap.primary_joints(model)]
    return StandingSpec(hold_seconds=1.0, height_ref=float(start.qpos[2]),
                        joints_ref=start.qpos[qadr].copy(), body_contact_phases=())


def progression(ex, motion_info: dict, fraction: float = 0.8) -> dict:
    """The squat must actually happen: executed base descent at the bottom phase
    reaches `fraction` of the planned CoM depth, and the phases occur in order."""
    labels = ex.phases or []
    order = [label for i, label in enumerate(labels) if i == 0 or labels[i - 1] != label]
    expected = [label for label, _ in motion_info["phases"]]
    bottom = [k + 1 for k, label in enumerate(labels) if label == "bottom"]
    if order != expected or not bottom:
        return {"passed": False, "detail": f"phase order {order} != {expected}"}
    descent = float(ex.qpos[0, 2] - np.min(ex.qpos[bottom, 2]))
    need = fraction * motion_info["depth"]
    return {"passed": bool(descent >= need), "descent_m": descent,
            "detail": f"executed base descent {descent * 1e3:.1f} mm at bottom (need >= {need * 1e3:.1f} mm)"}


def hold_weights() -> croc.Weights:
    """Hold costs: pull the joints back to the nominal stance.

    The default joint weight (0.01) lets the robot hold whatever pose it landed
    in; measured after a jump, that is 0.15 rad rms from the standing pose and
    fails the pose gate at 0.12.
    """
    return croc.Weights(joint_posture=5.0)


def hold_knots(ad: Adapter, cfg: RobotConfig, model, qpos, qvel, reference: Start, seconds: float):
    """Supported hold from wherever the robot is now, back to standing height.

    Used after a landing: the feet end up somewhere other than the nominal
    stance, so commanding the nominal standing pose fights the geometry --
    measured, the robot crept to 15 degrees of tilt and lost 47 mm of height
    over two seconds. Optimising the hold from the MEASURED state keeps the
    contact forces consistent with where the feet actually are.
    """
    x0 = ad.x_from_mj(qpos, qvel)
    posture = ad.x_from_mj(reference.qpos, np.zeros(model.nv))
    posture[ad.pin.nq:] = 0.0
    data = ad.pin.createData()
    com = pin.centerOfMass(ad.pin, data, x0[:ad.pin.nq])
    target = pin.centerOfMass(ad.pin, data, ad.q_from_mj(reference.qpos))
    target[:2] = com[:2]                       # hold where it landed; x, y stay free
    steps = int(round(seconds / cfg.control_dt))
    knots = [croc.Knot(support=LEGS, com=target, posture=posture, phase="hold")
             for _ in range(steps)]
    return x0, knots, croc.Knot(support=LEGS, com=target, posture=posture, phase="hold")

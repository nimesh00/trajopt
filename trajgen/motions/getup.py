"""Belly-up getup: inverted start and contact discovery.

The robot starts ON ITS BACK and must end standing. This module provides the
two pieces that must exist before any optimal control problem can be posed:

  1. a deterministic, physically settled inverted start with a saved full
     snapshot, accepted only against explicit criteria (base-up . world-up
     <= -0.9, measured back contact, bounded penetration, no warnings, and a
     sustained quiet interval -- "we stepped for 2 s" is not settled);
  2. contact discovery by MuJoCo search over asymmetric command knots, whose
     finalists are re-run on the authoritative instrumented stepper so the
     contact sequence is read from measured geometry and load rather than
     assumed.

Rigid scheduled contacts cannot represent a torso rolling on its back, and
Crocoddyl 3.1 additionally cannot carry two contacts on one parent joint
(verified: 7.6e6 relative derivative error versus 1.5e-7 for contacts on
distinct joints), so the optimal-control stage must use at most one contact per
parent joint. Discovery therefore comes first, and what it finds decides which
segments a rigid-contact OCP can own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .. import execution
from ..config import RobotConfig

NAME = "getup-belly-up"
STATUS = "inverted start fixture; recovery unsolved"


@dataclass
class InvertedStart:
    state: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    upright: float
    contacts: tuple
    info: dict = field(default_factory=dict)


def upright_of(qpos) -> float:
    """World-z component of the torso z axis: +1 level, -1 fully inverted."""
    _, x, y, _ = np.asarray(qpos, float)[3:7]
    return float(1.0 - 2.0 * (x * x + y * y))


def ground_contacts(model: mujoco.MjModel, data: mujoco.MjData):
    """(robot geom names touching the world, deepest penetration in m)."""
    world = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0}
    names, deepest = set(), 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.efc_address < 0:
            continue
        a, b = int(c.geom[0]), int(c.geom[1])
        if (a in world) == (b in world):
            continue
        other = b if a in world else a
        names.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom{other}")
        deepest = max(deepest, -float(c.dist))
    return tuple(sorted(names)), deepest


def settled_inverted(model: mujoco.MjModel, cfg: RobotConfig, *, drop_height: float = 0.12,
                     keyframe: str = "stand", upright_max: float = -0.9, speed_tol: float = 5e-3,
                     window: float = 0.5, min_seconds: float = 1.0, max_seconds: float = 8.0,
                     penetration_max: float = 0.005) -> InvertedStart:
    """Settle the robot on its back, holding the keyframe joint command.

    The keyframe joints are held rather than going limp, matching how the
    training reset drives a fall; the command is what the recovery starts from.
    """
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key < 0:
        raise ValueError(f"no '{keyframe}' keyframe")
    qadr = np.array([model.jnt_qposadr[model.actuator_trnid[a][0]] for a in range(model.nu)])
    qpos = np.array(model.key_qpos[key], float)
    command = qpos[qadr].copy()
    qpos[2] = drop_height
    qpos[3:7] = [0.0, 1.0, 0.0, 0.0]          # pi about x: belly up
    qpos[0:2] = 0.0

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.ctrl[:] = command
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    needed, quiet, steps = int(round(window / dt)), 0, 0
    trace = []
    while True:
        mujoco.mj_step(model, data)
        steps += 1
        speed = float(np.linalg.norm(data.qvel))
        quiet = quiet + 1 if speed <= speed_tol else 0
        if steps % int(round(0.1 / dt)) == 0:
            trace.append({"t": round(steps * dt, 3), "z": float(data.qpos[2]),
                          "upright": upright_of(data.qpos), "speed": speed})
        if steps * dt >= min_seconds and quiet >= needed:
            break
        if steps * dt >= max_seconds:
            raise RuntimeError(f"inverted start did not settle: |qvel| {speed:.2e} after "
                               f"{max_seconds} s (tol {speed_tol} for {window} s)")
        if not np.all(np.isfinite(data.qpos)) or any(w.number for w in data.warning):
            raise RuntimeError("inverted start diverged or raised MuJoCo warnings while settling")
    mujoco.mj_forward(model, data)

    upright = upright_of(data.qpos)
    contacts, penetration = ground_contacts(model, data)
    problems = []
    if not upright <= upright_max:
        problems.append(f"upright {upright:.3f} > {upright_max} (not inverted)")
    if not contacts:
        problems.append("no measured ground contact")
    if any("foot" in name for name in contacts):
        problems.append(f"feet are on the ground: {contacts}")
    if penetration > penetration_max:
        problems.append(f"penetration {penetration * 1e3:.2f} mm > {penetration_max * 1e3:.0f} mm")
    if problems:
        raise RuntimeError("inverted start rejected: " + "; ".join(problems))

    return InvertedStart(
        state=execution.snapshot(model, data), qpos=np.array(data.qpos), qvel=np.array(data.qvel),
        ctrl=command, upright=upright, contacts=contacts,
        info={"keyframe": keyframe, "drop_height": drop_height, "settle_seconds": steps * dt,
              "settle_criterion": {"speed_tol": speed_tol, "window": window,
                                   "min_seconds": min_seconds, "max_seconds": max_seconds},
              "residual_speed": float(np.linalg.norm(data.qvel)), "upright": upright,
              "support_geoms": list(contacts), "penetration_m": penetration,
              "base_height": float(data.qpos[2]), "trace": trace})


# --------------------------------------------------------------- discovery

def commands_from_knots(knots: np.ndarray, start_command: np.ndarray, steps: int,
                        model: mujoco.MjModel, action_scale: float) -> np.ndarray:
    """Interpolate knots into per-step commands, then apply the trainer's own
    command law: u_k = clamp(u_{k-1} + scale * clip(a_k, -1, 1), joint range).

    Searching commands the consumer cannot express would waste the search.
    """
    knots = np.asarray(knots, float)
    joints = [int(model.actuator_trnid[a][0]) for a in range(model.nu)]
    lo = model.jnt_range[joints, 0]
    hi = model.jnt_range[joints, 1]
    grid = np.linspace(0.0, 1.0, steps)
    nodes = np.linspace(0.0, 1.0, len(knots))
    desired = np.array([np.interp(grid, nodes, knots[:, j]) for j in range(knots.shape[1])]).T
    out = np.empty((steps, model.nu))
    previous = np.asarray(start_command, float)
    for k in range(steps):
        step = np.clip(desired[k] - previous, -action_scale, action_scale)
        previous = np.clip(previous + step, lo, hi)
        out[k] = previous
    return out



def landed(model, execution_result) -> dict:
    """Is the robot standing on four feet with nothing else touching?"""
    terminal = execution_result.terminal
    normals = terminal["foot_normal"][0]
    body = int(terminal["body_ground_contacts"][0])
    return {"foot_normals": normals.tolist(), "body_contacts": body,
            "feet_loaded": int((normals > 5.0).sum()),
            "height": float(execution_result.qpos[-1, 2]),
            "upright": upright_of(execution_result.qpos[-1]),
            "speed": float(np.linalg.norm(execution_result.qvel[-1]))}


# ------------------------------------------------- contact schedule from data

def contact_schedule(model: mujoco.MjModel, ad, ex, min_steps: int = 5) -> list[dict]:
    """Phases read from the MEASURED contacts of an execution.

    One contact per PARENT JOINT: Crocoddyl 3.1 overwrites rather than
    accumulates `fext` for contacts sharing a parent joint (verified here:
    7.6e6 relative derivative error against numerical differentiation, versus
    1.5e-7 for contacts on distinct joints), so a link touching the ground at
    two geoms contributes one contact, placed at its most loaded geom.

    Short flickers are absorbed into the neighbouring phase (`min_steps`), since
    a two-step contact is not a phase worth scheduling.
    """
    by_geom = {info["mj_geom"]: name for name, info in ad.geoms.items()}
    world = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0}
    table = ex.contact_table
    steps = ex.steps
    per_step = []
    for k in range(steps):
        rows = np.nonzero((table["step"] == k) & (table["boundary"] == 0))[0]
        load: dict[int, tuple] = {}
        for r in rows:
            ga, gb = (int(v) for v in table["geom"][r])
            aw, bw = ga in world, gb in world
            if aw == bw:
                continue
            geom = gb if aw else ga
            joint = ad.pin.frames[ad.geoms[by_geom[geom]]["frame"]].parentJoint
            force = abs(float(table["force"][r][0]))
            if force > load.get(joint, (0.0, None))[0]:
                load[joint] = (force, geom)
        per_step.append({joint: geom for joint, (_, geom) in load.items()})

    phases, current, first = [], None, 0
    for k, contacts in enumerate(per_step + [None]):
        key = None if contacts is None else tuple(sorted(contacts))
        if current is None:
            current, first = key, k
            continue
        if key != current:
            if k - first >= min_steps or not phases:
                phases.append({"first": first, "last": k - 1, "joints": current,
                               "geoms": {j: per_step[first][j] for j in current}})
            else:                                  # absorb a flicker into the last phase
                phases[-1]["last"] = k - 1
            current, first = key, k
    for i, phase in enumerate(phases):
        names = sorted(by_geom[g] for g in phase["geoms"].values())
        phase["label"] = f"p{i}"
        phase["contacts"] = names
        phase["steps"] = phase["last"] - phase["first"] + 1
    return phases

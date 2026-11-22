"""MuJoCo command search for contact discovery.

Rigid scheduled contacts cannot be written down for a robot rolling on its
back until the roll and plant sequence is known, so the sequence is DISCOVERED
by searching commands in MuJoCo and then read off the measured contacts.

What this is and is not: a screening search. It uses MuJoCo's native batched
`rollout` (fast, no per-substep instrumentation) and scores candidates from
states alone. Finalists are re-run on the authoritative instrumented stepper,
which is the only result that may be reported. A sampling result is not a
Crocoddyl solution and is labelled with the backend that produced it.

Ranking is feasibility first, then progress: candidates that go non-finite or
raise simulator warnings are rejected outright rather than scored, and progress
is never required to improve monotonically.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout

from .motions.getup import commands_from_knots


@dataclass
class SearchConfig:
    knots: int = 10
    horizon: float = 3.0            # seconds of commanded motion
    hold: float = 1.0               # seconds holding the final command, scored
    population: int = 128
    elite_fraction: float = 0.1
    iterations: int = 25
    init_std: float = 0.35          # rad on joint command knots
    std_floor: float = 0.03
    seed: int = 0
    threads: int = 8
    # Cap on hip (abduction) command travel from the stance, radians. The objective says
    # nothing about posture along the trajectory, so an unconstrained search spends the
    # full +/-0.838 of hip range: the accepted getup clip averaged 0.285 and peaked at
    # 0.848, past the limit, and the policy imitating it inherited that sway. Rolling over
    # does not need that much abduction; None leaves the joint range alone.
    hip_limit: float | None = None
    # Robustness: evaluate every candidate under injected command noise, `replicas` times, and
    # score the MEAN. A search that scores one noise-free rollout selects knife-edge
    # trajectories - the accepted getup clip reached upright +0.999 replayed exactly and
    # -0.640 with a 5% systematic offset, which no cloned or learned policy can reproduce.
    command_noise: float = 0.0
    # Per-replica CONSTANT command offset. This is the perturbation that matters: the
    # accepted clip survives 0.15 rad of per-step noise 10 times out of 10, yet a 0.05 rad
    # systematic offset destroys it - and a cloned or learned policy's residual error is
    # exactly a systematic, state-dependent bias, not zero-mean noise.
    command_bias: float = 0.0
    replicas: int = 1

    def as_dict(self):
        return dict(self.__dict__)


@dataclass
class Result:
    knots: np.ndarray
    commands: np.ndarray
    score: float
    metrics: dict
    history: list = field(default_factory=list)
    config: dict = field(default_factory=dict)
    family: str = ""


def _metrics(states: np.ndarray, nq: int, nv: int, hold_seconds: float, dt: float,
             stance: np.ndarray | None = None, hip_qpos: list | None = None):
    """Per-candidate metrics from states alone: [batch, nstep, 1+nq+nv+na]."""
    qpos = states[:, :, 1:1 + nq]
    qvel = states[:, :, 1 + nq:1 + nq + nv]
    finite = np.all(np.isfinite(states), axis=(1, 2))
    quat = qpos[:, :, 3:7]
    upright = 1.0 - 2.0 * (quat[:, :, 1] ** 2 + quat[:, :, 2] ** 2)
    height = qpos[:, :, 2]
    tail = slice(-int(round(hold_seconds / dt)), None)
    # Terminal distance to the standing joint configuration. Without this the search
    # is free to end high, upright and on four feet in any posture at all, which is
    # what it did: the getup reference finished 0.358 rad from the stance and only
    # looked correct because a settled stance was concatenated onto the end.
    stance_rms = (np.sqrt(((qpos[:, tail, 7:7 + len(stance)] - stance) ** 2).mean(axis=2)).mean(axis=1)
                  if stance is not None else np.zeros(len(qpos)))
    # Mean hip abduction over the WHOLE trajectory, not just its end. A hard cap on hip
    # range is too blunt - capped at 0.3 rad the robot cannot roll at all - but left
    # unpriced the search spends the full range and the policy imitating the clip inherits
    # the sway. Pricing travel lets it buy abduction where the roll needs it.
    hip_travel = (np.abs(qpos[:, :, hip_qpos]).mean(axis=(1, 2))
                  if hip_qpos else np.zeros(len(qpos)))
    return {
        "hip_travel": hip_travel,
        "terminal_stance": stance_rms,
        "terminal_upright": upright[:, tail].mean(axis=1),
        "terminal_height": height[:, tail].mean(axis=1),
        "peak_upright": upright.max(axis=1),
        "peak_height": height.max(axis=1),
        "terminal_speed": np.linalg.norm(qvel[:, tail, :3], axis=2).mean(axis=1),
        "terminal_spin": np.linalg.norm(qvel[:, tail, 3:6], axis=2).mean(axis=1),
        "peak_speed": np.linalg.norm(qvel[:, :, :3], axis=2).max(axis=1),
        "finite": finite.astype(float),
    }


def _lowest_points(model: mujoco.MjModel, data: mujoco.MjData, geoms) -> np.ndarray:
    """Lowest world z of each geom, from its type, size and orientation."""
    out = np.empty(len(geoms))
    for i, g in enumerate(geoms):
        R = data.geom_xmat[g].reshape(3, 3)
        size = model.geom_size[g]
        kind = model.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            extent = size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            extent = float(np.abs(R[2, :]) @ size[:3])
        elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            axis = float(abs(R[2, 2]))                     # the geom's local z in world z
            extent = size[1] * axis + size[0] * np.sqrt(max(0.0, 1.0 - axis * axis))
        else:
            extent = float(np.max(size))                   # conservative fallback
        out[i] = data.geom_xpos[g][2] - extent
    return out


def terminal_features(model: mujoco.MjModel, states: np.ndarray, nq: int) -> dict:
    """Foot contact and body clearance of each candidate's FINAL state, by FK.

    `rollout` returns states only and this plant has no touch sensors, so a
    foot counts as down when its collision sphere's lowest point is within
    5 mm of the ground, and body clearance is the lowest point of every other
    collision geom (kneeling on the calves is what this has to notice). One
    mj_kinematics per candidate; finalists are measured for real on the
    instrumented stepper.
    """
    data = mujoco.MjData(model)
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{f}_foot_collision")
            for f in ("FL", "FR", "RL", "RR")]
    others = [g for g in range(model.ngeom)
              if model.geom_bodyid[g] != 0 and g not in feet
              and (model.geom_contype[g] or model.geom_conaffinity[g])]
    foot_z = np.zeros((states.shape[0], len(feet)))
    clearance = np.zeros(states.shape[0])
    for i in range(states.shape[0]):
        data.qpos[:] = states[i, -1, 1:1 + nq]
        mujoco.mj_kinematics(model, data)
        foot_z[i] = _lowest_points(model, data, feet)
        clearance[i] = _lowest_points(model, data, others).min()
    return {"foot_clearance": foot_z, "feet_down": (foot_z < 0.005).sum(axis=1).astype(float),
            "body_clearance": clearance}


def score(m, target_height: float, hip_cost: float = 0.0) -> np.ndarray:
    """One objective: end upright, on all four feet, clear of the ground, high and still.

    Support and height credit is GATED on being upright. Ungated, the search
    found a robot lying on its side with two feet touching, which scores well
    on "feet down" and is not a recovery. Peak uprightness is also rewarded, so
    a candidate that rolls and falls back ranks above one that never moved;
    progress is never required to improve monotonically in time.
    """
    upright = np.clip(m["terminal_upright"], -1.0, 1.0)
    gate = np.clip((upright - 0.5) / 0.4, 0.0, 1.0)
    return (3.0 * upright
            + 1.0 * np.clip(m["peak_upright"], -1.0, 1.0)
            + gate * (1.0 * m["feet_down"]
                      + 2.0 * np.clip(m["body_clearance"] / 0.03, -1.0, 1.0)
                      + 3.0 * np.clip(m["terminal_height"] / target_height, 0.0, 1.0)
                      # Ending high on four feet says nothing about the posture. Zero
                      # when no stance is supplied, so callers that only want a
                      # recovery are unaffected.
                      - 3.0 * np.clip(m["terminal_stance"] / 0.5, 0.0, 1.0))
            - hip_cost * np.clip(m["hip_travel"] / 0.4, 0.0, 1.0)
            - 1.0 * np.clip(m["terminal_speed"], 0.0, 2.0)
            - 0.5 * np.clip(m["terminal_spin"], 0.0, 5.0))


def _evaluate(model, datas, full, warmstart, knots, start_command, steps, cfg, target_height,
              hold_steps=0, stance=None, hip_qpos=None, hip_cost=0.0, noise=0.0, replicas=1,
              seed=0, joints=None, bias=0.0):
    commands = np.array([commands_from_knots(k, start_command, steps, model, cfg.action_scale)
                         for k in knots])
    if hold_steps:
        # Hold the final command and score THAT: a candidate that looks upright
        # at the end of its commanded motion and then topples is not a recovery.
        commands = np.concatenate([commands, np.repeat(commands[:, -1:], hold_steps, axis=1)], axis=1)
    # Robustness: each candidate is rolled out `replicas` times with independent command
    # noise and scored on the mean, so a trajectory that only survives exact replay loses to
    # one that tolerates the error a learned policy will always have.
    reps = max(1, int(replicas))
    batch = np.repeat(commands, reps, axis=0)
    if noise > 0 or bias > 0:
        rng = np.random.default_rng(seed)
        if noise > 0:
            batch = batch + rng.normal(0, noise, batch.shape)
        if bias > 0:
            # one constant offset per replica, held for the whole trajectory
            offs = rng.normal(0, bias, (batch.shape[0], 1, batch.shape[2]))
            batch = batch + offs
        lo_b, hi_b = model.jnt_range[joints, 0], model.jnt_range[joints, 1]
        batch = np.clip(batch, lo_b, hi_b)
    control = np.repeat(batch, cfg.substeps, axis=1)
    states, _ = mj_rollout.rollout(model, datas, np.tile(full, (len(batch), 1)), control,
                                   nstep=batch.shape[1] * cfg.substeps,
                                   initial_warmstart=np.tile(warmstart, (len(batch), 1)))
    metrics = _metrics(states, model.nq, model.nv, 0.5, model.opt.timestep, stance, hip_qpos)
    finite = metrics["finite"] > 0
    metrics.update(terminal_features(model, np.where(finite[:, None, None], states, 0.0), model.nq))
    raw = np.where(finite, score(metrics, target_height, hip_cost), -np.inf)
    if reps > 1:
        raw = raw.reshape(len(commands), reps).mean(axis=1)
        metrics = {k: (v.reshape(len(commands), reps).mean(axis=1)
                       if getattr(v, "ndim", 0) == 1 and len(v) == len(commands) * reps else v)
                   for k, v in metrics.items()}
    return commands, raw, metrics


def _initial_state(model, state):
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    full = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(model, data, full, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    return full, np.array(data.qacc_warmstart)


def cem(model: mujoco.MjModel, state: np.ndarray, start_command: np.ndarray, cfg,
        search: SearchConfig, target_height: float, *,
        mean_knots: np.ndarray | None = None, family: str = "",
        stance: np.ndarray | None = None, hip_cost: float = 0.0) -> Result:
    """Cross-entropy search over absolute joint-command knots.

    With no `mean_knots`, the first population is drawn UNIFORMLY over the joint
    ranges and the elite statistics seed the search: from a settled inverted
    pose, a mean seeded on "ramp towards standing" explores no roll at all,
    while roughly 0.4% of uniform samples do get past upright 0.
    """
    rng = np.random.default_rng(search.seed)
    steps = int(round(search.horizon / cfg.control_dt))
    hold_steps = int(round(search.hold / cfg.control_dt))
    joints = [int(model.actuator_trnid[a][0]) for a in range(model.nu)]
    lo, hi = model.jnt_range[joints, 0].copy(), model.jnt_range[joints, 1].copy()
    if search.hip_limit is not None:
        for a, j in enumerate(joints):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if "_hip_" in name:
                lo[a] = max(lo[a], -search.hip_limit)
                hi[a] = min(hi[a], search.hip_limit)
    hip_qpos = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
                if "_hip_" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "")]
    full, warmstart = _initial_state(model, state)
    datas = [mujoco.MjData(model) for _ in range(search.threads)]
    elites = max(4, int(search.population * search.elite_fraction))
    shape = (search.knots, model.nu)

    if mean_knots is None:
        knots = rng.uniform(lo, hi, size=(search.population,) + shape)
    else:
        mean = np.array(mean_knots, float)
        knots = np.clip(mean + search.init_std * rng.normal(size=(search.population,) + shape), lo, hi)
        knots[0] = mean
    best, history = None, []
    for iteration in range(search.iterations):
        commands, scores, metrics = _evaluate(model, datas, full, warmstart, knots, start_command,
                                              steps, cfg, target_height, hold_steps, stance,
                                              hip_qpos, hip_cost, search.command_noise,
                                              search.replicas, search.seed + iteration, joints,
                                              search.command_bias)
        order = np.argsort(-scores)
        top = order[:elites]
        if np.isfinite(scores[order[0]]) and (best is None or scores[order[0]] > best.score):
            i = order[0]
            best = Result(knots=knots[i].copy(), commands=commands[i].copy(), score=float(scores[i]),
                          metrics={k: (float(v[i]) if np.ndim(v[i]) == 0 else np.asarray(v[i]).tolist())
                                   for k, v in metrics.items()},
                          config=search.as_dict(), family=family)
        history.append({"iteration": iteration, "best": float(scores[order[0]]),
                        "elite_mean": float(scores[top].mean()),
                        "rejected": int(np.sum(~np.isfinite(scores))),
                        "terminal_upright": float(metrics["terminal_upright"][order[0]]),
                        "peak_upright": float(metrics["peak_upright"][order[0]]),
                        "feet_down": float(metrics["feet_down"][order[0]]),
                        "body_clearance": float(metrics["body_clearance"][order[0]]),
                        "terminal_height": float(metrics["terminal_height"][order[0]])})
        mean = knots[top].mean(axis=0)
        std = np.maximum(knots[top].std(axis=0), search.std_floor)
        knots = np.clip(mean + std * rng.normal(size=(search.population,) + shape), lo, hi)
        knots[0] = best.knots                              # keep the incumbent
    if best is None:
        raise RuntimeError("every candidate was rejected as non-finite")
    best.history = history
    return best


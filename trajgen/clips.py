"""Write reference clips in trajgen's clip format, which the trainer reads.

The trainer reads these exactly, so every field name,
array length and frame convention has to match; a clip that merely looks right
is worse than one that fails loudly, because the mismatch surfaces as bad
training rather than an error.

Frame convention: one initial frame carrying the start state with a
null action, then one POST-action frame per control step.

    frame 0    t=0,  qpos=x0.qpos, qvel=x0.qvel, ctrl=u_previous, action=null
    frame k+1  u_k = clamp(u_{k-1} + scale*clip(a_k), joint_limits)
               simulate `substeps` physics steps with u_k
               t=(k+1)*control_dt, qpos/qvel = result, ctrl=u_k, action=a_k
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .contract import CONTACT_COLUMNS, Contract

CLIP_SCHEMA = 1
CLIP_KIND = "reference-clip"


def fingerprint(model_bytes: bytes) -> str:
    """FNV-1a 64-bit, matching the trainer's clip `fingerprint`.

    Lets a clip be checked against the plant it was generated on instead of
    trusting a filename.
    """
    h = 14695981039346656037
    for b in model_bytes:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return format(h, "x")


def _finite(name: str, values: np.ndarray) -> list:
    array = np.asarray(values, float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values; refusing to write the clip")
    return [float(v) for v in array.ravel()]


class Frame:
    """One recorded control step."""

    __slots__ = ("t", "qpos", "qvel", "ctrl", "action", "feet", "reward")

    def __init__(self, t, qpos, qvel, ctrl, action=None, feet=None, reward=None):
        self.t = float(t)
        self.qpos = np.asarray(qpos, float)
        self.qvel = np.asarray(qvel, float)
        self.ctrl = np.asarray(ctrl, float)
        self.action = None if action is None else np.asarray(action, float)
        # Contacts must be the UNION across the control interval, matching
        # mujoco_batch.cpp; sampling only the final substep misses short impacts,
        # which for getup is exactly where the informative contacts happen.
        self.feet = None if feet is None else [int(bool(v)) for v in feet]
        self.reward = None if reward is None else float(reward)

    def to_json(self, nq: int, nv: int, nu: int) -> dict:
        if self.qpos.size != nq or self.qvel.size != nv or self.ctrl.size != nu:
            raise ValueError(
                f"frame shape mismatch: qpos {self.qpos.size}/{nq}, "
                f"qvel {self.qvel.size}/{nv}, ctrl {self.ctrl.size}/{nu}"
            )
        return {
            "t": self.t,
            "qpos": _finite("qpos", self.qpos),
            "qvel": _finite("qvel", self.qvel),
            "ctrl": _finite("ctrl", self.ctrl),
            "action": None if self.action is None else _finite("action", self.action),
            "feet": [] if self.feet is None else self.feet,
            "reward": self.reward,
        }


class Episode:
    """One generated trajectory."""

    def __init__(self, index: int, frames: Sequence[Frame], ending: str, success: bool):
        self.index = int(index)
        self.frames = list(frames)
        self.ending = str(ending)
        self.success = bool(success)

    def to_json(self, nq: int, nv: int, nu: int) -> dict:
        heights = [float(f.qpos[2]) for f in self.frames]
        rewards = [f.reward for f in self.frames if f.reward is not None]
        return {
            "index": self.index,
            "frames": [f.to_json(nq, nv, nu) for f in self.frames],
            "steps": len(self.frames) - 1,
            "ending": self.ending,
            "success": self.success,
            "return": float(sum(rewards)) if rewards else 0.0,
            "peak_height": max(heights),
            "final_height": heights[-1],
        }


def write_clip(
    path: str | Path,
    *,
    episodes: Iterable[Episode],
    contract: Contract,
    task: str,
    task_schema: str,
    model_bytes: bytes,
    nq: int,
    nv: int,
    nu: int,
    target_height: float,
    base_height: float,
    balance_version: int = 0,
    balance_height: float = 0.0,
    provenance: dict | None = None,
) -> Path:
    """Write a clip, plus a sidecar recording how it was generated.

    Generator provenance stays in the sidecar rather than the clip: the clip's
    `iteration` field means "PPO iteration" to every existing reader, so faking
    one there would misrepresent an optimized trajectory as a trained policy.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    episodes = list(episodes)
    if not episodes:
        raise ValueError("refusing to write a clip with no episodes")

    successes = sum(1 for e in episodes if e.success)
    frame_count = sum(len(e.frames) for e in episodes)
    lengths = [len(e.frames) - 1 for e in episodes]
    peaks = [max(float(f.qpos[2]) for f in e.frames) for e in episodes]
    returns = [sum(f.reward for f in e.frames if f.reward is not None) for e in episodes]

    document = {
        "schema": CLIP_SCHEMA,
        "kind": CLIP_KIND,
        "task": task,
        "task_schema": task_schema,
        "model_id": fingerprint(model_bytes),
        # Not a PPO checkpoint: generated references carry iteration 0 and
        # describe themselves in the sidecar instead.
        "iteration": 0,
        "seed": 0,
        "deterministic": True,
        "control_dt": contract.control_dt,
        "physics_timestep": contract.physics_timestep,
        "substeps": contract.substeps,
        "nq": int(nq),
        "nv": int(nv),
        "nu": int(nu),
        "parameters": {
            "kp": contract.kp,
            "kd": contract.kd,
            "action_scale": contract.action_scale,
            "balance_version": int(balance_version),
            "balance_height": float(balance_height),
            "base_height": float(base_height),
        },
        "actuator_names": list(contract.actuator_names),
        "joint_lower": [float(v) for v in contract.joint_lower],
        "joint_upper": [float(v) for v in contract.joint_upper],
        "nominal": [float(v) for v in contract.nominal],
        "foot_columns": list(CONTACT_COLUMNS),
        "episodes": [e.to_json(nq, nv, nu) for e in episodes],
        "summary": {
            "episodes": len(episodes),
            "successes": successes,
            "success_rate": successes / len(episodes),
            "frames": frame_count,
            "mean_length": float(np.mean(lengths)),
            "mean_peak_height": float(np.mean(peaks)),
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "target_height": float(target_height),
        },
    }

    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(document) + "\n")
    temporary.rename(path)

    sidecar = path.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(provenance or {}, indent=2) + "\n")
    return path

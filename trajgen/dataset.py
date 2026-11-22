"""Turn accepted trajopt bundles into a reference dataset.

A bundle is the unit of ACCEPTANCE: it carries the plant it was solved against,
every gate result, and the replay agreement that says the clip reproduces. It is
not a convenient unit of CONSUMPTION -- a training run wants one array per clip
and one place to read what each clip was commanded to do.

This writes that view WITHOUT re-deriving anything. Every number here is copied
from a bundle that already passed; nothing is recomputed, so the dataset cannot
disagree with the acceptance that admitted it. Clips that were rejected are
refused rather than exported, and clips solved against different plants are
refused as a set, because a reference is only meaningful on the geometry and the
actuator model it was solved for.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from . import bundle

SCHEMA = 1


def _achieved(mi: dict, travel: dict) -> np.ndarray:
    """Measured (vx, vy, wz) of a clip, resolved onto the world axes.

    `progression` reports speed along the COMMANDED heading, so a sidestep's
    speed is its lateral speed. Projecting it back onto the heading recovers the
    world components without recomputing anything from the trajectory.
    """
    speed = float(travel.get("speed_mps") or 0.0)
    heading = travel.get("heading") or [1.0, 0.0]
    return np.array([speed * float(heading[0]), speed * float(heading[1]),
                     float(travel.get("yaw_rate_rps") or 0.0)], float)


def export(sources: list[Path], out: Path, name: str = "") -> dict:
    """Write `sources` (accepted bundle dirs) to `out` as an indexed dataset."""
    out = Path(out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    entries, models, contracts = [], set(), set()
    for src in sources:
        man, clip, arr = bundle.load(src)
        # validation.passed is the verdict. `status` is a free-text note about how
        # the clip was solved ("fixed contact schedule") and says nothing about
        # whether it was accepted.
        validation = man.get("validation", {}) or {}
        if not validation.get("passed"):
            failed = [g["name"] for g in validation.get("gates", []) if not g.get("passed")]
            raise ValueError(f"{src}: bundle did not pass validation"
                             + (f" (failed: {', '.join(failed)})" if failed else ""))
        models.add(man["model"]["mjb_sha256"])
        step = man["stepping"]
        contracts.add((step["control_dt"], step["physics_timestep"], step["substeps"]))
        mi = man.get("motion_info", {}) or {}
        travel = mi.get("travel", {}) or {}
        tag = Path(src).name
        qpos = np.asarray(arr["qpos"]); ctrl = np.asarray(arr["ctrl"])
        qvel = np.asarray(arr["qvel"])
        frames = int(len(qpos))
        dt = float(step["control_dt"])
        # Phase label per frame, so a consumer can cut the walk out of the clip
        # without knowing how the motion was scheduled.
        phase = np.full(frames, "", dtype=object)
        spans = man.get("phases", {}) or {}
        for label, (lo, hi) in spans.items():
            phase[int(lo):int(hi) + 1] = label
        # The spans index CONTROL INTERVALS and the clip carries one more frame
        # than intervals (the terminal state). Leaving it blank would hand the
        # consumer an unlabelled frame it has to guess about; it belongs to the
        # phase that produced it.
        if spans and phase[-1] == "":
            phase[-1] = max(spans.items(), key=lambda kv: int(kv[1][1]))[0]
        np.savez_compressed(
            out / "clips" / f"{tag}.npz",
            time=np.arange(frames, dtype=float) * dt,
            qpos=qpos, qvel=qvel, ctrl=ctrl,
            phase=phase.astype("U8"),
            command=np.array([mi.get("velocity", 0.0) or 0.0,
                              mi.get("velocity_y", 0.0) or 0.0,
                              mi.get("yaw_rate", 0.0) or 0.0], float),
            achieved=_achieved(mi, travel))
        entries.append({
            "name": tag,
            "file": f"clips/{tag}.npz",
            "command": {"lin_vel_x": mi.get("velocity", 0.0) or 0.0,
                        "lin_vel_y": mi.get("velocity_y", 0.0) or 0.0,
                        "ang_vel_z": mi.get("yaw_rate", 0.0) or 0.0},
            "achieved_xyz": [float(v) for v in _achieved(mi, travel)],
            "achieved": {"speed_mps": travel.get("speed_mps"),
                         "forward_m": travel.get("forward_m"),
                         "turned_rad": travel.get("turned_rad"),
                         "yaw_rate_rps": travel.get("yaw_rate_rps"),
                         "heading": travel.get("heading")},
            "gait": {"pattern": mi.get("pattern"), "period": mi.get("period"),
                     "duty": mi.get("duty"), "swing_height": mi.get("swing_height")},
            "frames": frames, "seconds": round(frames * dt, 4),
            "phases": man.get("phases", {}),
            "source_bundle": str(Path(src).resolve()),
        })
    if len(models) != 1:
        raise ValueError(f"clips span {len(models)} different plants; a reference is only "
                         f"meaningful on the plant it was solved against: {sorted(models)}")
    if len(contracts) != 1:
        raise ValueError(f"clips span {len(contracts)} different stepping contracts")
    dt, physics, substeps = next(iter(contracts))
    man0, _, _ = bundle.load(sources[0])
    # JOINT NAMES, EXPLICITLY. qpos[:, 7:] is in the trajopt plant's joint order.
    # A consumer whose model orders joints differently and slices by position
    # gets a silently wrong reference -- legs swapped, no error raised -- so the
    # order is written down and the consumer maps by name.
    import mujoco
    _m = mujoco.MjModel.from_binary_path(man0["model"]["mjb"])
    joint_names = [n for n in
                   (mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(_m.njnt))
                   if n and _m.jnt_type[mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                   != mujoco.mjtJoint.mjJNT_FREE]
    index = {
        "schema": SCHEMA,
        "name": name or out.name,
        "kind": "trajopt-reference-dataset",
        "robot": man0.get("robot_config", {}),
        "model": man0.get("model", {}),
        "contract": {"control_dt": dt, "physics_timestep": physics, "substeps": substeps,
                     "action_encoding": man0.get("action_encoding", {})},
        "arrays": {
            "time": "[T] seconds from the clip start",
            "qpos": "[T, nq] MuJoCo qpos; base is free-joint xyz + wxyz quaternion, "
                    "then the 12 joints in `joint_names` order -- map by NAME, not position",
            "qvel": "[T, nv] MuJoCo qvel",
            "ctrl": "[T, nu] joint POSITION command in MuJoCo actuator order",
            "phase": "[T] one of walk|stop|hold",
            "command": "[3] the command GIVEN TO THE SOLVER (lin_vel_x, lin_vel_y, ang_vel_z)",
            "achieved": "[3] the velocity the clip actually MOVES AT, same order. "
                        "These differ -- the solver is commanded above the speed it "
                        "reaches -- and `achieved` is the one to condition a policy on, "
                        "because it is what the robot in the clip is doing.",
        },
        "joint_names": joint_names,
        "clips": sorted(entries, key=lambda e: e["name"]),
    }
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index

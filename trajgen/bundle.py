"""Atomic, hash-bound artifact bundles.

    <bundle>/
      manifest.json   provenance, diagnostics, validation, file hashes
      clip.json       schema-1 reference-clip of the EXECUTED MuJoCo rollout
      arrays.npz      full initial snapshot, ctrl, executed states, physics-rate
                      samples, planned states/torques/gains/forces

The bundle is assembled in a sibling temporary directory and renamed into place,
so a crash can leave a stale temporary but never a half-written bundle. A bundle
whose files do not match the manifest hashes is refused on load: missing or
mismatched provenance means unverified.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from . import plantmap
from .clips import CLIP_KIND, CLIP_SCHEMA, fingerprint
from .contract import CONTACT_COLUMNS
from .execution import CONTACT_CHANNELS, EXECUTION_VERSION, SAMPLE_CHANNELS, STATE_SPEC, Execution

BUNDLE_VERSION = 2
BUNDLE_KIND = "trajopt-bundle"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort, recorded as null
        return None


def versions() -> dict:
    import crocoddyl
    import pinocchio

    return {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
            "mujoco": mujoco.__version__, "pinocchio": pinocchio.__version__,
            "crocoddyl": crocoddyl.__version__, "numpy": np.__version__}


def clip_document(ex: Execution, model: mujoco.MjModel, model_bytes: bytes, *, task: str,
                  task_schema: str, kp: float, kd: float, action_scale: float,
                  success: bool, ending: str) -> dict:
    """Schema-1 reference-clip of an execution. Frame 0: t=0, action=null."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)]
    tr = plantmap.transmission(model)
    joints = [int(j) for j in tr.joints]          # primary joints: a belt's trnid is a tendon
    qadr = [int(model.jnt_qposadr[j]) for j in joints]
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    nominal = [float(model.key_qpos[max(key, 0)][i]) for i in qadr]
    # The action is the command delta in SERVO-POSITION units (ctrl / gear), the units
    # action_scale is stated in; on a unit-gear plant that is the raw ctrl delta.
    position = tr.positions(tr.servo_part(ex.ctrl))
    frames = []
    for k in range(ex.steps + 1):
        action = None if k == 0 else ((position[k] - position[k - 1]) / action_scale).tolist()
        frames.append({"t": k * ex.control_dt, "qpos": ex.qpos[k].tolist(), "qvel": ex.qvel[k].tolist(),
                       "ctrl": ex.ctrl[k].tolist(), "action": action,
                       "feet": [] if k == 0 else [int(v) for v in ex.contacts[k]], "reward": None})
    heights = ex.qpos[:, 2]
    return {
        "schema": CLIP_SCHEMA, "kind": CLIP_KIND, "task": task, "task_schema": task_schema,
        "model_id": fingerprint(model_bytes), "iteration": 0, "seed": 0, "deterministic": True,
        "control_dt": ex.control_dt, "physics_timestep": ex.physics_timestep, "substeps": ex.substeps,
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "parameters": {"kp": kp, "kd": kd, "action_scale": action_scale, "balance_version": 0,
                       "balance_height": 0.0, "base_height": float(heights[0])},
        "actuator_names": names,
        "joint_lower": [float(model.jnt_range[j][0]) for j in joints],
        "joint_upper": [float(model.jnt_range[j][1]) for j in joints],
        "nominal": nominal, "foot_columns": list(CONTACT_COLUMNS),
        "episodes": [{"index": 0, "frames": frames, "steps": ex.steps, "ending": ending,
                      "success": bool(success), "return": 0.0,
                      "peak_height": float(heights.max()), "final_height": float(heights[-1])}],
        "summary": {"episodes": 1, "successes": int(bool(success)), "success_rate": float(bool(success)),
                    "frames": ex.steps + 1, "mean_length": float(ex.steps),
                    "mean_peak_height": float(heights.max()), "mean_return": 0.0,
                    "target_height": float(heights[0])},
    }


def execution_arrays(ex: Execution) -> dict:
    out = {"initial_state": ex.initial_state, "qpos": ex.qpos, "qvel": ex.qvel, "ctrl": ex.ctrl,
           "contacts": ex.contacts, "sample_step": ex.sample_step}
    out.update({f"sample_{k}": ex.samples[k] for k in SAMPLE_CHANNELS})
    out.update({f"terminal_{k}": ex.terminal[k] for k in SAMPLE_CHANNELS})
    out.update({f"contact_{k}": ex.contact_table[k] for k in CONTACT_CHANNELS})
    return out


def phases_from_ranges(ranges: dict, steps: int) -> list[str]:
    """Expand {label: [first, last]} into one label per step; must tile 0..steps-1 exactly."""
    labels: list = [None] * steps
    for label, (first, last) in ranges.items():
        if not (isinstance(first, int) and isinstance(last, int) and 0 <= first <= last < steps):
            raise ValueError(f"phase {label} range {first}..{last} outside 0..{steps - 1}")
        for k in range(first, last + 1):
            if labels[k] is not None:
                raise ValueError(f"phases overlap at step {k}")
            labels[k] = label.split("@")[0]
    if any(v is None for v in labels):
        raise ValueError("phases do not cover every control step")
    return labels


def execution_from_arrays(arrays, manifest: dict) -> Execution:
    stepping = manifest["stepping"]
    steps = arrays["ctrl"].shape[0] - 1
    return Execution(
        initial_state=arrays["initial_state"], qpos=arrays["qpos"], qvel=arrays["qvel"],
        ctrl=arrays["ctrl"], contacts=arrays["contacts"],
        samples={k: arrays[f"sample_{k}"] for k in SAMPLE_CHANNELS},
        sample_step=arrays["sample_step"], control_dt=stepping["control_dt"],
        physics_timestep=stepping["physics_timestep"], substeps=stepping["substeps"],
        terminal={k: arrays[f"terminal_{k}"] for k in SAMPLE_CHANNELS},
        contact_table={k: arrays[f"contact_{k}"] for k in CONTACT_CHANNELS},
        phases=phases_from_ranges(manifest["phases"], steps))


REQUIRED_MANIFEST = {
    "motion": str, "status": str, "created": str, "robot_config": dict, "model": dict, "adapter": dict,
    "stepping": dict, "action_encoding": dict, "controller": dict, "planner": dict, "phases": dict,
    "motion_info": dict, "acceptance": dict, "tracking": dict, "replay_agreement": dict,
    "validation": dict, "versions": dict, "source": dict, "state_spec": dict, "files": dict,
    "execution_version": int,
}


def verify(manifest: dict, clip: dict, arrays: dict, model: mujoco.MjModel, model_bytes: bytes) -> list[str]:
    """Semantic verification beyond hashes. Returns every problem found."""
    p: list[str] = []
    for key, kind in REQUIRED_MANIFEST.items():
        if not isinstance(manifest.get(key), kind):
            p.append(f"manifest.{key} missing or not {kind.__name__}")
    if p:
        return p
    if manifest["execution_version"] != EXECUTION_VERSION:
        p.append(f"execution version {manifest['execution_version']} unsupported")
    for key in ("mjb", "mjb_sha256", "model_id", "urdf_sha256"):
        if not manifest["model"].get(key):
            p.append(f"manifest.model.{key} missing")
    if manifest["model"].get("mjb_sha256") != hashlib.sha256(model_bytes).hexdigest():
        p.append("model bytes do not match manifest.model.mjb_sha256")
    if fingerprint(model_bytes) != manifest["model"].get("model_id"):
        p.append("model fingerprint does not match manifest.model.model_id")
    for key in ("path", "sha256"):
        if not manifest["robot_config"].get(key):
            p.append(f"manifest.robot_config.{key} missing")
    enc = manifest["action_encoding"]
    scale = enc.get("scale")
    if enc.get("mode") != "command-delta" or enc.get("version") != 1 or enc.get("bound") != 1.0 \
            or not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
        p.append(f"unsupported action encoding {enc}")
        scale = None
    acceptance = manifest["acceptance"]
    if not isinstance(acceptance.get("spec"), dict) or not isinstance(acceptance.get("motor"), dict):
        p.append("manifest.acceptance must embed the resolved spec and motor law")
    st = manifest["stepping"]
    if not (isinstance(st.get("substeps"), int) and st["substeps"] >= 1
            and np.isclose(st.get("physics_timestep", np.nan), model.opt.timestep, rtol=0, atol=1e-12)
            and np.isclose(st.get("control_dt", np.nan), model.opt.timestep * st["substeps"], rtol=1e-6)):
        p.append(f"stepping {st} inconsistent with the model")
        return p
    ss = manifest["state_spec"]
    if ss.get("value") != STATE_SPEC or ss.get("size") != mujoco.mj_stateSize(model, STATE_SPEC):
        p.append(f"state spec {ss} does not match this MuJoCo build")

    for key in ["initial_state", "qpos", "qvel", "ctrl", "contacts", "sample_step"] + \
            [f"sample_{k}" for k in SAMPLE_CHANNELS] + [f"terminal_{k}" for k in SAMPLE_CHANNELS] + \
            [f"contact_{k}" for k in CONTACT_CHANNELS]:
        if key not in arrays:
            p.append(f"arrays.npz missing {key}")
    if p:
        return p
    T = arrays["ctrl"].shape[0] - 1
    try:
        phases_from_ranges(manifest["phases"], T)
    except (ValueError, TypeError) as error:
        p.append(f"phases: {error}")

    # ---- clip semantics and clip <-> arrays correspondence
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)]
    joints = [int(j) for j in plantmap.primary_joints(model)]
    checks = [
        (clip.get("schema") == CLIP_SCHEMA, "clip.schema"), (clip.get("kind") == CLIP_KIND, "clip.kind"),
        ((clip.get("nq"), clip.get("nv"), clip.get("nu")) == (model.nq, model.nv, model.nu), "clip dims"),
        (clip.get("model_id") == manifest["model"].get("model_id"), "clip.model_id"),
        (clip.get("actuator_names") == names, "clip.actuator_names"),
        (clip.get("substeps") == st["substeps"], "clip.substeps"),
        (np.isclose(clip.get("control_dt", np.nan), st["control_dt"], rtol=1e-9), "clip.control_dt"),
        (clip.get("joint_lower") == [float(model.jnt_range[j][0]) for j in joints], "clip.joint_lower"),
        (clip.get("joint_upper") == [float(model.jnt_range[j][1]) for j in joints], "clip.joint_upper"),
        (isinstance(clip.get("episodes"), list) and len(clip["episodes"]) == 1, "clip.episodes"),
    ]
    p += [f"{what} invalid" for ok, what in checks if not ok]
    if p:
        return p
    ep = clip["episodes"][0]
    frames = ep.get("frames", [])
    if len(frames) != T + 1 or ep.get("steps") != T:
        return p + [f"clip has {len(frames)} frames / steps {ep.get('steps')}, arrays have {T} steps"]
    if ep.get("success") is not bool(manifest["validation"].get("passed")):
        p.append("clip success disagrees with the recorded validation verdict")
    try:
        qpos = np.array([f["qpos"] for f in frames], float)
        qvel = np.array([f["qvel"] for f in frames], float)
        ctrl = np.array([f["ctrl"] for f in frames], float)
        times = np.array([f["t"] for f in frames], float)
        feet = np.array([f["feet"] for f in frames[1:]], int)
    except (KeyError, TypeError, ValueError) as error:
        return p + [f"clip frames malformed: {error}"]
    if not (np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel)) and np.all(np.isfinite(ctrl))):
        p.append("clip frames not finite")
    if not np.allclose(times, np.arange(T + 1) * st["control_dt"], rtol=0, atol=1e-9):
        p.append("clip timestamps are not k * control_dt")
    if not (np.array_equal(qpos, arrays["qpos"]) and np.array_equal(qvel, arrays["qvel"])
            and np.array_equal(ctrl, arrays["ctrl"])):
        p.append("clip states/commands differ from arrays.npz")
    if frames[0].get("action") is not None or frames[0].get("feet"):
        p.append("clip frame 0 must have a null action and no contact flags")
    if feet.shape != (T, 5) or not np.array_equal(feet.astype(bool), arrays["contacts"][1:]):
        p.append("clip contact flags differ from arrays.npz")
    if scale is not None:
        try:
            actions = np.array([f["action"] for f in frames[1:]], float)
            tr = plantmap.transmission(model)
            expect = np.diff(tr.positions(tr.servo_part(ctrl)), axis=0) / scale
            if actions.shape != expect.shape or not np.all(np.isfinite(actions)) \
                    or not np.allclose(actions, expect, rtol=0, atol=1e-9) or np.any(np.abs(actions) > 1 + 1e-9):
                p.append("clip actions do not reconstruct from ctrl within the encoding bound")
        except (TypeError, ValueError):
            p.append("clip actions malformed")
    return p


def source_provenance(repo: Path) -> dict:
    files = sorted((repo / "trajgen").rglob("*.py")) + sorted((repo / "robots").glob("*.toml"))
    digests = {str(f.relative_to(repo)): _sha(f) for f in files}
    dirty = None
    try:
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "."],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except Exception:  # noqa: BLE001
        pass
    tree = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in digests.items()).encode()).hexdigest()
    return {"git_commit": _git_commit(repo), "git_dirty": dirty, "tree_sha256": tree, "files": digests}


def write(path: str | Path, *, manifest: dict, clip: dict, arrays: dict, repo: Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; bundles are immutable, choose another name")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        (tmp / "clip.json").write_text(json.dumps(clip) + "\n")
        np.savez_compressed(tmp / "arrays.npz", **arrays)
        manifest = dict(manifest)
        manifest.update({
            "kind": BUNDLE_KIND, "bundle_version": BUNDLE_VERSION, "versions": versions(),
            "source": source_provenance(repo), "execution_version": EXECUTION_VERSION,
            "state_spec": {"name": "mjSTATE_INTEGRATION", "value": STATE_SPEC,
                           "size": int(arrays["initial_state"].size)},
            "files": {name: _sha(tmp / name) for name in ("clip.json", "arrays.npz")},
        })
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json) + "\n")
        os.rename(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return path


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def load(path: str | Path):
    """(manifest, clip, arrays). Refuses any hash or version mismatch."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("kind") != BUNDLE_KIND or manifest.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(f"{path}: not a version-{BUNDLE_VERSION} {BUNDLE_KIND}")
    for name, digest in manifest.get("files", {}).items():
        actual = _sha(path / name)
        if actual != digest:
            raise ValueError(f"{path}/{name}: sha256 {actual[:12]} does not match manifest {digest[:12]}")
    if set(manifest.get("files", {})) != {"clip.json", "arrays.npz"}:
        raise ValueError(f"{path}: manifest does not bind every bundle file")
    clip = json.loads((path / "clip.json").read_text())
    arrays = dict(np.load(path / "arrays.npz"))
    return manifest, clip, arrays

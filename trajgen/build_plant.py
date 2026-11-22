"""Compile a robot's plant from its torque-motor MJCF: the `.mjb` a robot config names.

    .venv/bin/python -m trajgen.build_plant robots/dhav1.toml

trajgen plans and validates against a COMPILED plant (trajgen/model.py says why a raw
MJCF is refused). A training stack usually exports that plant itself; this module is
the same transformation for a robot that only has an MJCF, driven by the config's
[build] section:

  source      the MJCF, read through MjSpec and never edited on disk. When it is missing
              and the config names `repo` + `commit` + `checkout`, the repository is
              cloned there at that commit first.
  servos      every actuator becomes a joint position servo, gainprm kp, biasprm
              [0, -kp, -kd] (the config's [contract]), force limited at the source motor's
              ctrlrange, or at `effort` (N m per joint role) when the motor's usable torque
              is below the rating the MJCF declares; the limit is also written to the joint's
              actfrcrange: the un-mutated limit trajgen.dcmotor.DcClamp reads. ctrlrange
              becomes the joint range and is not enforced.
  stepping    timestep and integrator from [contract]; solver, cone and contact stay as
              the MJCF has them.
  feet        the geoms `feet` names ("{leg}" -> "FL", ...) are renamed {leg}_foot, the
              name trajgen.plantmap looks up; `foot_solimp` / `foot_solref` / `foot_friction`,
              when given, replace their solimp / solref / friction (the only contact
              changes, and recorded in the provenance).
  `stand`     key: the config's default pose placed on the ground and settled on THIS
              plant under the command that holds it against gravity
              (trajgen.plant.Plant.settled_stance). key_ctrl is that command.

A provenance file next to the plant records the source commit and hashes, and the build
ends by loading the result the way every tool does (config.assert_plant, adapter.build).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from . import config as cm
from .plant import LEGS, Plant


def fetch(cfg: cm.RobotConfig) -> Path:
    """The source MJCF. With repo + commit + checkout, the checkout is cloned when missing and
    moved to the pinned commit when it is at another one (a re-pinned config must not build
    from the old checkout); without them, `source` must already exist."""
    build = cfg.raw["build"]
    source = cm._expand(build["source"], cfg.path.parent)
    if not all(k in build for k in ("repo", "commit", "checkout")):
        if source.is_file():
            return source
        raise FileNotFoundError(f"{source} does not exist and [build] names no repo / commit / checkout")
    checkout = cm._expand(build["checkout"], cfg.path.parent)
    commit = build["commit"]
    git = lambda *args, **kw: subprocess.run(["git", "-C", str(checkout), *args], **kw)  # noqa: E731
    if checkout.exists() and git("rev-parse", "HEAD", capture_output=True, text=True).stdout.strip() == commit \
            and source.is_file():
        return source
    sparse = source.relative_to(checkout).parts[0]
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--sparse", build["repo"],
                        str(checkout)], check=True)
    if git("cat-file", "-e", f"{commit}^{{commit}}", capture_output=True).returncode != 0:
        git("fetch", "--quiet", "origin", commit, check=True)
    git("sparse-checkout", "set", sparse, check=True)
    git("checkout", "--quiet", commit, check=True)
    if not source.is_file():
        raise FileNotFoundError(f"{source} is not in {build['repo']} at {build['commit']}")
    return source


def _spec(cfg: cm.RobotConfig, source: Path) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(source))
    spec.option.timestep = cfg.physics_timestep
    spec.option.integrator = cm.INTEGRATORS[cfg.integrator]
    pattern = cfg.raw["build"].get("feet", "{leg}_foot")
    geoms = {g.name: g for g in spec.geoms}
    for leg in LEGS:
        name = pattern.format(leg=leg)
        if name not in geoms:
            raise ValueError(f"{source}: no foot geom {name!r}")
        geoms[name].name = f"{leg}_foot"
        if "foot_solimp" in cfg.raw["build"]:
            geoms[name].solimp = [float(v) for v in cfg.raw["build"]["foot_solimp"]]
        if "foot_solref" in cfg.raw["build"]:
            geoms[name].solref = [float(v) for v in cfg.raw["build"]["foot_solref"]]
        if "foot_friction" in cfg.raw["build"]:
            geoms[name].friction = [float(v) for v in cfg.raw["build"]["foot_friction"]]
    joints = {j.name: j for j in spec.joints}
    effort = cfg.raw["build"].get("effort", {})
    unknown = set(effort) - set(cfg.motor)
    if unknown:
        raise ValueError(f"[build].effort: unknown roles {sorted(unknown)}")
    for a in spec.actuators:
        role = a.name.rsplit("_", 1)[-1]
        if role not in cfg.motor:
            raise ValueError(f"{a.name}: no [motor] entry for role {role!r}")
        if a.trntype != mujoco.mjtTrn.mjTRN_JOINT or a.target not in joints:
            raise ValueError(f"{a.name}: only joint transmissions are supported")
        if a.gaintype != mujoco.mjtGain.mjGAIN_FIXED or a.biastype != mujoco.mjtBias.mjBIAS_NONE:
            raise ValueError(f"{a.name}: the source actuator is not a torque motor")
        rating = float(a.ctrlrange[1])
        if not (rating > 0 and np.isclose(-a.ctrlrange[0], rating)):
            raise ValueError(f"{a.name}: torque ctrlrange {list(a.ctrlrange)} is not symmetric")
        rating = float(effort.get(role, rating))
        j = joints[a.target]
        a.gainprm = [cfg.kp] + [0.0] * 9
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm = [0.0, -cfg.kp, -cfg.kd] + [0.0] * 7
        a.gear = [1.0] + [0.0] * 5
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-rating, rating]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        a.ctrlrange = list(j.range)
        j.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.actfrcrange = [-rating, rating]
    for key in list(spec.keys):
        spec.delete(key)
    return spec


def _placed_pose(cfg: cm.RobotConfig, model: mujoco.MjModel) -> np.ndarray:
    """qpos of the default pose, upright, with the lowest foot sphere on z = 0."""
    d = mujoco.MjData(model)
    d.qpos[:] = model.qpos0
    d.qpos[3:7] = [1, 0, 0, 0]
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        d.qpos[model.jnt_qposadr[j]] = cfg.pose_for(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a))
    mujoco.mj_kinematics(model, d)
    feet = [model.geom(f"{leg}_foot").id for leg in LEGS]
    d.qpos[2] -= min(d.geom_xpos[g, 2] - model.geom_size[g, 0] for g in feet)
    return d.qpos.copy()


def build(cfg: cm.RobotConfig, settle_seconds: float = 2.5) -> tuple[mujoco.MjModel, dict]:
    source = fetch(cfg)
    spec = _spec(cfg, source)
    m = spec.compile()
    key = spec.add_key()
    key.name = "stand"
    key.qpos = _placed_pose(cfg, m).tolist()
    m = spec.compile()
    plant = Plant(m, cfg.substeps, cfg)
    state, ctrl, height = plant.settled_stance(settle_seconds)
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    key.qpos = d.qpos.tolist()
    key.qvel = d.qvel.tolist()
    key.ctrl = np.asarray(ctrl).tolist()
    m = spec.compile()
    return m, {"source": str(source), "settle_seconds": settle_seconds, "settled_base_z": height,
               "settled_qvel_norm": float(np.linalg.norm(d.qvel))}


def provenance(cfg: cm.RobotConfig, model_path: Path, info: dict) -> dict:
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    m = mujoco.MjModel.from_binary_path(str(model_path))
    source = Path(info["source"])
    build = cfg.raw["build"]
    return {
        "robot": cfg.name, "config": str(cfg.path), "config_sha256": cfg.sha256,
        "mjb": str(model_path), "mjb_sha256": sha(model_path), "mujoco": mujoco.__version__,
        "source": str(source), "source_sha256": sha(source),
        "repo": build.get("repo"), "commit": build.get("commit"), "foot_solimp": build.get("foot_solimp"),
        "foot_solref": build.get("foot_solref"), "foot_friction": build.get("foot_friction"),
        "effort": build.get("effort"),
        "nq": m.nq, "nv": m.nv, "nu": m.nu, "mass_kg": float(m.body_mass.sum()),
        "timestep": float(m.opt.timestep), "substeps": cfg.substeps, "kp": cfg.kp, "kd": cfg.kd,
        "force_limits": {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a): float(m.actuator_forcerange[a, 1])
                         for a in range(m.nu)},
        "stand_key": {k: info[k] for k in ("settle_seconds", "settled_base_z", "settled_qvel_norm")},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="robot config with a [build] section, e.g. robots/dhav1.toml")
    a = ap.parse_args(argv)
    cfg = cm.load(a.config, check_files=False)
    if "build" not in cfg.raw:
        raise SystemExit(f"{a.config}: no [build] section; this plant is exported by its training stack")
    m, info = build(cfg)
    cfg.mjb.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveModel(m, str(cfg.mjb), None)
    prov = provenance(cfg, cfg.mjb, info)
    # Load the result the way every tool does: the contract, then the Pinocchio pair.
    from . import adapter
    cfg = cm.load(a.config)
    cm.assert_plant(cfg, mujoco.MjModel.from_binary_path(str(cfg.mjb)))
    ad = adapter.build(cfg.mjb, cfg.urdf)
    prov["adapter_differences"] = list(getattr(ad, "differences", []))
    cfg.mjb.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=1))
    print(json.dumps(prov, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

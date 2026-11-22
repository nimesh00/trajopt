"""Adapter tests: MuJoCo plant vs Pinocchio model, numerically.

Matching total mass is not evidence the models agree, so every comparison here
is pose-by-pose at neutral, standing, crouched, inverted and randomised poses
with rotated, moving bases:

  state round trips              <= 1e-9
  velocity convention            numerical integration agreement
  body FK, foot sphere centres   <= 1e-5 m / rad
  COM                            <= 1e-5 m
  Jacobians                      relative <= 1e-4
  mass matrix, bias forces       relative <= 1e-4   (armature transferred)
  free-space forward dynamics    relative <= 1e-4   (damper, spring, frictionloss, limits, contact off)
  structural mismatches          refused (renamed / swapped joints, changed axis, link
                                 length or mass in the Pinocchio description)

    python tests/test_adapter.py [--robot robots/dhav1.toml]

The robot config names the compiled plant and its description (URDF or MJCF); the
leg and joint names follow trajgen's convention ({leg}_{role}_joint, legs FL/FR/RL/RR).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import pathlib
import sys
import tempfile

import mujoco
import numpy as np
import pinocchio as pin

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trajgen import adapter as adapter_mod  # noqa: E402
from trajgen import config as config_mod  # noqa: E402

MJB = URDF = None          # the robot config's plant and description, set in main()
TOL_STATE, TOL_POS, TOL_REL = 1e-9, 1e-5, 1e-4
RESULTS: dict[str, dict] = {}


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) /
                 max(np.linalg.norm(b), 1e-9))


def record(name, value, tol):
    ok = bool(np.isfinite(value) and value <= tol)
    prev = RESULTS.get(name)
    if prev is None or value > prev["max"]:
        RESULTS[name] = {"max": float(value), "tol": tol, "ok": ok if prev is None else ok and prev["ok"]}
    elif not ok:
        RESULTS[name]["ok"] = False


def poses(ad, rng):
    m = ad.mj
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    stand = m.key_qpos[key].copy()
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    jq = ad.mj_qadr

    neutral = m.qpos0.copy(); neutral[2] = 0.6
    crouch = stand.copy()
    roles = [n.split("_")[1] for n in ad.joint_names]
    bend = np.array([{"thigh": 0.5, "calf": -0.8}.get(r, 0.0) for r in roles])
    crouch[jq] = np.clip(stand[jq] + bend, lo, hi); crouch[2] = 0.5 * stand[2]
    inverted = stand.copy()
    inverted[3:7] = [0.0, 1.0, 0.0, 0.0]; inverted[2] = 0.15   # pi about x, belly up
    out = [("neutral", neutral, np.zeros(m.nv)),
           ("stand", stand, np.zeros(m.nv)),
           ("crouch", crouch, np.zeros(m.nv)),
           ("inverted", inverted, np.zeros(m.nv))]
    for i in range(12):
        qpos = stand.copy()
        qpos[0:3] = rng.normal(0, 0.5, 3) + [0, 0, 1.0]
        quat = rng.normal(size=4); qpos[3:7] = quat / np.linalg.norm(quat)
        qpos[jq] = rng.uniform(lo, hi)
        qvel = rng.normal(0, 1.5, m.nv)
        out.append((f"random{i}", qpos, qvel))
    # The same poses with a moving base, so velocity conventions are exercised.
    out += [(n + "+moving", q, rng.normal(0, 1.5, m.nv)) for n, q, _ in out[:4]]
    return out


def check_pose(ad, name, qpos, qvel):
    m, pm = ad.mj, ad.pin
    d = mujoco.MjData(m)
    pd = pm.createData()
    d.qpos[:], d.qvel[:] = qpos, qvel
    mujoco.mj_forward(m, d)

    # --- state round trips (quaternion sign equivalence tolerated)
    q = ad.q_from_mj(qpos)
    v = ad.v_from_mj(qpos, qvel)
    qpos_back = ad.qpos_from_pin(q)
    if np.dot(qpos_back[3:7], qpos[3:7]) < 0:
        qpos_back[3:7] *= -1
    record("roundtrip_qpos", np.max(np.abs(qpos_back - qpos)), TOL_STATE)
    record("roundtrip_qvel", np.max(np.abs(ad.qvel_from_pin(q, v) - qvel)), TOL_STATE)
    q_neg = q.copy(); q_neg[3:7] *= -1
    record("quat_sign_equivalence", np.max(np.abs(pin.difference(pm, q, q_neg))), TOL_STATE)

    # --- velocity convention by numerical integration (first order in dt)
    dt = 1e-6
    qpos_next = qpos.copy()
    mujoco.mj_integratePos(m, qpos_next, qvel, dt)
    q_next = pin.integrate(pm, q, v * dt)
    delta = pin.difference(pm, ad.q_from_mj(qpos_next), q_next) / dt
    record("velocity_convention", np.max(np.abs(delta)) / max(np.max(np.abs(v)), 1.0), 1e-5)

    # --- FK for every body, foot sphere centres, COM
    pin.forwardKinematics(pm, pd, q, v)
    pin.updateFramePlacements(pm, pd)
    err_p = err_r = 0.0
    for b, fid in ad.body_frames.items():
        oMf = pd.oMf[fid]
        err_p = max(err_p, np.max(np.abs(oMf.translation - d.xpos[b])))
        R = d.xmat[b].reshape(3, 3)
        err_r = max(err_r, np.linalg.norm(pin.log3(R.T @ oMf.rotation)))
    record("fk_body_position", err_p, TOL_POS)
    record("fk_body_rotation", err_r, TOL_POS)
    record("foot_sphere_centre", max(np.max(np.abs(pd.oMf[c.frame].translation - d.geom_xpos[c.mj_geom]))
                                     for c in ad.feet.values()), TOL_POS)
    com = pin.centerOfMass(pm, pd, q)
    record("com", np.max(np.abs(com - d.subtree_com[1])), TOL_POS)

    # --- Jacobians (world-aligned, body origins) in MuJoCo velocity coordinates
    pin.computeJointJacobians(pm, pd, q)
    T = ad.transform(qpos)
    worst = 0.0
    for b, fid in ad.body_frames.items():
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jacp, jacr, b)
        J = pin.getFrameJacobian(pm, pd, fid, pin.LOCAL_WORLD_ALIGNED) @ T
        worst = max(worst, rel(J, np.vstack([jacp, jacr])))
    record("jacobian", worst, TOL_REL)

    # --- mass matrix and bias forces
    M_mj = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M_mj)
    M_p, h_p = ad.dynamics_mj(qpos, qvel, pd)
    record("mass_matrix", rel(M_p, M_mj), TOL_REL)
    record("bias_forces", rel(h_p, d.qfrc_bias), TOL_REL)
    record("damper_force", np.max(np.abs(ad.damping_mj(qvel) - d.qfrc_damper)), TOL_STATE)
    record("collision_geom_centres", max(np.max(np.abs(pd.oMf[g["frame"]].translation - d.geom_xpos[g["mj_geom"]]))
                                         for g in ad.geoms.values()), TOL_POS)

    # --- sphere surface point on horizontal and tilted planes: the point must
    # lie on the sphere and its outward normal must oppose the plane normal.
    for normal in ([0, 0, 1], [np.sin(0.3), 0, np.cos(0.3)], [0.2, -0.3, 0.93]):
        n = np.asarray(normal, float) / np.linalg.norm(normal)
        for c in ad.feet.values():
            p = ad.sphere_contact_point(pd, c, n)
            centre = d.geom_xpos[c.mj_geom]
            record("sphere_point_on_surface", abs(np.linalg.norm(p - centre) - c.radius), TOL_STATE)
            record("sphere_point_normal", np.linalg.norm((centre - p) / c.radius - n), TOL_STATE)


def check_free_dynamics(ad, rng, trials=10):
    """qacc under random applied torques with everything non-rigid-body disabled."""
    m = ad.mj
    flags = m.opt.disableflags
    m.opt.disableflags = flags | int(mujoco.mjtDisableBit.mjDSBL_CONTACT) \
        | int(mujoco.mjtDisableBit.mjDSBL_DAMPER) | int(mujoco.mjtDisableBit.mjDSBL_SPRING) | int(mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS) \
        | int(mujoco.mjtDisableBit.mjDSBL_LIMIT) | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    try:
        pd = ad.pin.createData()
        for _, qpos, _ in poses(ad, rng)[4:4 + trials]:
            d = mujoco.MjData(m)
            d.qpos[:] = qpos
            d.qvel[:] = rng.normal(0, 1.5, m.nv)
            d.qfrc_applied[6:] = rng.normal(0, 10.0, m.nv - 6)
            mujoco.mj_forward(m, d)
            T = ad.transform(d.qpos)
            q, v = ad.q_from_mj(d.qpos), ad.v_from_mj(d.qpos, d.qvel)
            tau_pin = np.linalg.solve(T.T, d.qfrc_applied)
            a_pin = pin.aba(ad.pin, pd, q, v, tau_pin)
            Tdot_v = np.zeros(m.nv); Tdot_v[0:3] = -np.cross(v[3:6], v[0:3])
            qacc = np.linalg.solve(T, a_pin - Tdot_v)
            record("free_forward_dynamics", rel(qacc, d.qacc), TOL_REL)
    finally:
        m.opt.disableflags = flags


def _element(text, tag, name):
    """Span of the <tag name="name" ...> element (to its closing tag, or self-closing)."""
    m = re.search(rf'<{tag}\s+name="{re.escape(name)}"[^>]*?(/>|>.*?</{tag}>)', text, re.S)
    return m.span() if m else None


def _parent_body(text, joint):
    """Span of the MJCF <body ...> opening tag that holds `joint`."""
    at = text.find(f'name="{joint}"')
    open_at = text.rfind("<body", 0, at)
    return (open_at, text.index(">", open_at) + 1) if at >= 0 and open_at >= 0 else None


def _scale_attr(text, span, attr, index, factor):
    """Scale one number of an attribute inside text[span]."""
    a, b = span
    part = text[a:b]
    m = re.search(rf'{attr}="([^"]+)"', part)
    if not m:
        return text
    values = m.group(1).split()
    values[index] = repr(float(values[index]) * factor)
    part = part[:m.start(1)] + " ".join(values) + part[m.end(1):]
    return text[:a] + part + text[b:]


def _swap(text, a, b):
    return text.replace(f'"{a}"', '"TMP_SWAP"').replace(f'"{b}"', f'"{a}"').replace('"TMP_SWAP"', f'"{b}"')


def check_refusals():
    text = pathlib.Path(URDF).read_text()
    mjcf = pathlib.Path(URDF).suffix.lower() == ".xml"
    cases = {
        "renamed joint": text.replace('name="FL_calf_joint"', 'name="FL_knee_joint"', 1),
        # names/axes/types all still match; only kinematics can catch this
        "swapped FL/RL thigh names": _swap(text, "FL_thigh_joint", "RL_thigh_joint"),
    }
    if mjcf:
        hip = _element(text, "joint", "FL_hip_joint")
        if hip:
            a, b = hip
            joint = re.sub(r'\s+axis="[^"]*"', "", text[a:b])
            joint = joint.replace('name="FL_hip_joint"', 'name="FL_hip_joint" axis="0 1 0"', 1)
            cases["axis changed"] = text[:a] + joint + text[b:]
        calf = _parent_body(text, "FL_calf_joint")
        cases["shortened calf"] = _scale_attr(text, calf, "pos", 2, 0.96) if calf else None
        thigh = _parent_body(text, "FL_thigh_joint")
        if thigh:
            inertial = re.compile(r'<inertial[^>]*mass="[^"]+"').search(text, thigh[1])
            cases["heavier thigh"] = (_scale_attr(text, inertial.span(), "mass", 0, 1.05)
                                      if inertial else None)
    else:
        hip = _element(text, "joint", "FL_hip_joint")
        if hip:
            a, b = hip
            cases["axis changed"] = text[:a] + re.sub(r'axis xyz="[^"]*"', 'axis xyz="0 1 0"',
                                                      text[a:b], count=1) + text[b:]
        calf = _element(text, "joint", "FL_calf_joint")
        if calf:
            origin = re.compile(r'<origin[^>]*xyz="[^"]+"').search(text, calf[0], calf[1])
            cases["shortened calf"] = _scale_attr(text, origin.span(), "xyz", 2, 0.96) if origin else None
        link = re.search(r'<joint\s+name="FL_thigh_joint".*?<child\s+link="([^"]+)"', text, re.S)
        thigh = _element(text, "link", link.group(1)) if link else None
        if thigh:
            mass = re.compile(r'<mass\s+value="[^"]+"').search(text, thigh[0], thigh[1])
            cases["heavier thigh"] = _scale_attr(text, mass.span(), "value", 0, 1.05) if mass else None
    for label, body in cases.items():
        if body is None or body == text:
            RESULTS[f"refuses {label}"] = {"ok": False, "note": "mutation not applied"}
            continue
        # Next to the original, so relative mesh / include paths still resolve.
        with tempfile.NamedTemporaryFile("w", suffix=pathlib.Path(URDF).suffix, delete=False,
                                         dir=pathlib.Path(URDF).parent) as f:
            f.write(body)
        try:
            adapter_mod.build(MJB, f.name)
            RESULTS[f"refuses {label}"] = {"ok": False, "note": "accepted a mismatched description"}
        except ValueError as error:
            lines = str(error).splitlines()
            RESULTS[f"refuses {label}"] = {"ok": True, "note": (lines[1] if len(lines) > 1 else lines[0]).strip()}
        finally:
            os.unlink(f.name)


def main():
    global MJB, URDF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default=str(ROOT / "robots/dhav1.toml"))
    cfg = config_mod.load(ap.parse_args().robot)
    MJB, URDF = str(cfg.mjb), str(cfg.urdf)
    ad = adapter_mod.build(MJB, URDF)
    rng = np.random.default_rng(7)
    for name, qpos, qvel in poses(ad, rng):
        check_pose(ad, name, qpos, qvel)
    check_free_dynamics(ad, rng)

    # Armature matters: without it the mass matrix must disagree.
    bare = adapter_mod.build(MJB, URDF, transfer_armature=False)
    stand = poses(bare, rng)[1]
    d = mujoco.MjData(bare.mj); d.qpos[:] = stand[1]; mujoco.mj_forward(bare.mj, d)
    M_mj = np.zeros((bare.mj.nv, bare.mj.nv)); mujoco.mj_fullM(bare.mj, d, M_mj)
    err = rel(bare.dynamics_mj(stand[1], stand[2])[0], M_mj)
    if np.allclose(bare.pin.armature, ad.pin.armature):
        # An MJCF carries <joint armature>, which Pinocchio's parser reads itself: nothing
        # is transferred, so there is nothing to switch off. A URDF has no armature field.
        RESULTS["armature is load-bearing"] = {"ok": err <= TOL_REL, "note": "read from the MJCF itself",
                                              "rel_err": err}
    else:
        RESULTS["armature is load-bearing"] = {"ok": err > 10 * TOL_REL, "no_armature_rel_err": err}

    check_refusals()

    failed = 0
    for name, r in RESULTS.items():
        ok = r["ok"]
        failed += not ok
        detail = (f"max {r['max']:.3e}  tol {r['tol']:.0e}" if "max" in r
                  else json.dumps({k: v for k, v in r.items() if k != "ok"}))
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<28s} {detail}")
    print("\nrecorded modelling differences:")
    for line in ad.differences:
        print("  -", line)
    out = ROOT / "out/tests" / f"adapter_{cfg.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": RESULTS, "manifest": ad.manifest()}, indent=2) + "\n")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} checks passed   (manifest: {out.relative_to(ROOT)})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

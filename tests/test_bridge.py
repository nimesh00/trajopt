"""Servo bridge tests.

  static        the command produced for a static pose makes MuJoCo's servos
                deliver exactly the planned torque
  mapping       commands are built by joint NAME: reversing the adapter's
                internal order must not change them, while scrambling the
                torque map must (this plant's two joint orders coincide, so
                identity order alone would prove nothing)
  interval      the 10-substep response is measured, not assumed from the
                instantaneous torque identity
  perturbed     tracking from perturbed starts, open loop and closed loop,
                reported separately

    python tests/test_bridge.py [--robot robots/dhav1.toml] [bundle]

The bundle is a stand-squat solved on the same robot (default
out/bundles/<robot name>-stand-squat).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trajgen import adapter as adapter_mod  # noqa: E402
from trajgen import bridge as bridge_mod  # noqa: E402
from trajgen import bundle, config, croc, execution, gates  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))


def servo_force(model, qpos, ctrl):
    """JOINT torque the servos deliver, per actuator's primary joint. For unit-gear joint
    servos that is actuator_force; geared or tendon servos act through J^T."""
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    d.ctrl[:] = ctrl
    mujoco.mj_forward(model, d)
    _, dadr, _ = gates.actuator_addresses(model)
    return np.array(d.qfrc_actuator[dadr])


def static_case(cfg, model, ad, label):
    """Static gravity compensation: planned torque in, same torque out."""
    br = bridge_mod.ServoBridge(ad, cfg.kp, cfg.kd)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    qkey = np.array(model.key_qpos[key], float)
    q = ad.q_from_mj(qkey)
    tau, _, _ = croc.static_torques(ad, q)
    x = np.concatenate([q, np.zeros(ad.pin.nv)])
    ctrl, _ = br.command(x, x, tau)
    produced = servo_force(model, qkey, ctrl)
    expected = np.array([tau[br.pin_v[a] - 6] for a in range(model.nu)])
    err = float(np.max(np.abs(produced - expected)))
    check(f"static torque reproduced ({label})", err < 1e-9, f"max error {err:.2e} N m")
    return br, tau, q


def permuted_mapping(ad):
    """A copy of the adapter whose Pinocchio joint order is reversed.

    Reordering the URDF file does NOT reorder Pinocchio's joints (it builds
    them by kinematic-tree traversal), and this plant's two orders happen to
    coincide, so identity order is all a real model offers here. Reversing the
    map exercises the name-based indexing the bridge relies on.
    """
    import copy

    other = copy.copy(ad)
    order = np.arange(len(ad.joint_names))[::-1]
    other.joint_names = tuple(np.array(ad.joint_names)[order])
    other.mj_qadr, other.mj_dadr = ad.mj_qadr[order], ad.mj_dadr[order]
    other.pin_qadr, other.pin_vadr = ad.pin_qadr[order], ad.pin_vadr[order]
    return other


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default=str(ROOT / "robots/dhav1.toml"))
    ap.add_argument("bundle", nargs="?")
    args = ap.parse_args()
    cfg = config.load(args.robot)
    path = pathlib.Path(args.bundle) if args.bundle else ROOT / "out/bundles" / f"{cfg.name}-stand-squat"
    model = mujoco.MjModel.from_binary_path(str(cfg.mjb))
    config.assert_plant(cfg, model)
    ad = adapter_mod.build(cfg.mjb, cfg.urdf)
    br, tau, q_stand = static_case(cfg, model, ad, "nominal order")

    # ---- non-identity joint mapping (indexing must follow names, not positions)
    p_ad = permuted_mapping(ad)
    p_br = bridge_mod.ServoBridge(p_ad, cfg.kp, cfg.kd)
    same_maps = np.array_equal(p_br.pin_v, br.pin_v)
    check("permuted adapter gives the same actuator map", same_maps,
          "bridge resolves actuator -> joint by name, so reversing the adapter's "
          "internal order must not change the command it builds")
    x = np.concatenate([q_stand, np.zeros(ad.pin.nv)])
    ctrl_a, _ = br.command(x, x, tau)
    ctrl_b, _ = p_br.command(x, x, tau)
    check("permuted adapter produces the same command",
          float(np.max(np.abs(ctrl_a - ctrl_b))) < 1e-15,
          f"max difference {float(np.max(np.abs(ctrl_a - ctrl_b))):.2e}")
    wrong = bridge_mod.ServoBridge(ad, cfg.kp, cfg.kd)
    wrong.pin_v = wrong.pin_v[::-1].copy()
    ctrl_wrong, _ = wrong.command(x, x, tau)
    check("a scrambled torque map changes the command",
          float(np.max(np.abs(ctrl_a - ctrl_wrong))) > 1e-3,
          "the static test would not detect mapping errors otherwise")

    # ---- whole-interval response and perturbed starts, from the accepted bundle
    manifest, clip, arrays = bundle.load(path)
    acceptance = manifest["acceptance"]
    spec = gates.StandingSpec.from_dict(acceptance["spec"])
    motor = {k: tuple(v) for k, v in acceptance["motor"].items()}
    scale = manifest["action_encoding"]["scale"]
    stepping = manifest["stepping"]
    xs, us, K = arrays["planned_xs"], arrays["planned_us"], arrays["planned_K"]
    phases = bundle.phases_from_ranges(manifest["phases"], arrays["ctrl"].shape[0] - 1)
    steps = us.shape[0]

    def run(state, feedback):
        b = bridge_mod.ServoBridge(ad, cfg.kp, cfg.kd, bridge_mod.BridgeConfig(feedback=feedback))

        def controller(k, data):
            x_meas = ad.x_from_mj(data.qpos, data.qvel) if feedback != "none" else None
            return b.command(xs[k], xs[k + 1], us[k], K[k], x_meas)

        return execution.Executor(model, stepping["substeps"], cfg).run(
            state, controller, steps, stepping["control_dt"], phases=phases)

    planned_qpos = arrays["planned_qpos"]
    nominal = run(arrays["initial_state"], "none")
    interval_err = float(np.max(np.abs(planned_qpos[:, 7:] - nominal.qpos[:, 7:])))
    check("open-loop interval response tracks the plan", interval_err < 0.05,
          f"max joint error over the 10-substep response {interval_err:.4f} rad "
          f"(instantaneous torque equality would predict 0)")

    rng = np.random.default_rng(3)
    qadr, dadr, _ = gates.actuator_addresses(model)
    results = {"none": [], "riccati": []}
    trials = 8
    for _ in range(trials):
        d = mujoco.MjData(model)
        mujoco.mj_setState(model, d, arrays["initial_state"], execution.STATE_SPEC)
        d.qpos[qadr] += rng.uniform(-0.03, 0.03, model.nu)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, axis, np.radians(rng.uniform(-3.0, 3.0)))
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, d.qpos[3:7], dq)
        d.qpos[3:7] = quat
        mujoco.mj_forward(model, d)
        state = execution.snapshot(model, d)
        for mode in results:
            ex = run(state, mode)
            report = gates.validate_standing(ex, model, spec, motor, scale)
            results[mode].append((report.passed,
                                  float(np.max(np.abs(planned_qpos[:, 7:] - ex.qpos[:, 7:]))),
                                  float(abs(ex.qpos[-1, 2] - spec.height_ref)),
                                  [g.name for g in report.gates if g.required and not g.passed]))
    for mode, rows in results.items():
        passed = sum(r[0] for r in rows)
        joint = max(r[1] for r in rows)
        height = max(r[2] for r in rows)
        # Reported per mode; only the closed loop is gated.
        failures = sorted({g for r in rows for g in r[3]})
        check(f"perturbed starts ({mode})", passed >= trials - 1 if mode == "riccati" else True,
              f"{passed}/{trials} pass every gate, worst joint error {joint:.4f} rad, "
              f"worst final height error {height * 1e3:.1f} mm"
              + (f", failed gates: {failures}" if failures else "")
              + ("" if mode == "riccati" else "  [reported, not gated]"))

    failed = 0
    for name, ok, detail in RESULTS:
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<46s} {detail}")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

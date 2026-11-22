"""Pipeline rejection tests.

Every false pass reproduced in an earlier validator has a test here, run against
a known-good execution from an accepted bundle. The known-good execution must
pass, or the rejections prove nothing.

  validator   missing/short/zero-width channels, NaN or out-of-domain values,
              wrong stepping, broken timestamps, snapshot that does not restore,
              physics samples inconsistent with their frames or commands,
              hold-boundary excursions, between-frame spikes, overspeed torque
  bundle      tampered files, wrong schema, unreconstructable actions, missing
              provenance, clip disagreeing with arrays, unsupported versions
  replay      joint / velocity / command / contact / sample disagreement

    python tests/test_pipeline.py [bundle]      (default out/bundles/dhav1-stand-squat)
"""
from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trajgen import bundle, execution, gates, pipeline, plantmap  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))


def failing(report):
    return [g.name for g in report.gates if g.required and not g.passed]


def main():
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out/bundles/dhav1-stand-squat"
    manifest, clip, arrays = bundle.load(path)
    mjb = pathlib.Path(manifest["model"]["mjb"])
    model_bytes = mjb.read_bytes()
    model = mujoco.MjModel.from_binary_path(str(mjb))
    acceptance = manifest["acceptance"]
    spec = gates.StandingSpec.from_dict(acceptance["spec"])
    motor = {k: tuple(v) for k, v in acceptance["motor"].items()}
    scale = manifest["action_encoding"]["scale"]
    good = bundle.execution_from_arrays(arrays, manifest)

    def run(ex, sp=spec):
        return gates.validate_standing(ex, model, sp, motor, scale)

    base = run(good)
    check("known-good execution passes", base.passed, ", ".join(failing(base)))
    check("known-good bundle verifies", not bundle.verify(manifest, clip, arrays, model, model_bytes))

    T, sub = good.steps, good.substeps
    hold_first_frame = T - int(round(spec.hold_seconds / good.control_dt))
    hold = np.nonzero(good.sample_step >= T - 50)[0]

    def mutated(name, mutate, expect, sp=spec):
        ex = copy.deepcopy(good)
        mutate(ex)
        try:
            rep = run(ex, sp)
            fails = failing(rep)
            ok = (not rep.passed) and (expect is None or expect in fails)
            check(f"rejects {name}", ok, f"failed: {fails}")
        except Exception as error:   # a crash is not a clean rejection
            check(f"rejects {name}", False, f"raised {type(error).__name__}: {error}")

    # ---- structure and domains
    mutated("missing instrumentation", lambda e: e.samples.pop("foot_normal"), "integrity")
    mutated("truncated samples", lambda e: e.samples.__setitem__("foot_slip", e.samples["foot_slip"][:-10]),
            "integrity")
    mutated("zero-width channel", lambda e: e.samples.__setitem__("joint_q", np.zeros((T * sub, 0))), "integrity")
    mutated("missing terminal sample", lambda e: e.terminal.pop("foot_normal"), "integrity")
    mutated("NaN torque", lambda e: e.samples["actuator_force"].__setitem__((500, 3), np.nan), "integrity")
    mutated("NaN state", lambda e: e.qpos.__setitem__((120, 2), np.nan), "integrity")
    mutated("NaN body contact count", lambda e: e.samples.__setitem__(
        "body_ground_contacts", np.full(T * sub, np.nan)), "integrity")
    mutated("NaN snapshot", lambda e: e.__setattr__("initial_state", np.full_like(e.initial_state, np.nan)),
            "integrity")
    mutated("snapshot of a different state", lambda e: e.__setattr__(
        "initial_state", e.initial_state + 0.01), "integrity")
    mutated("empty contact flags", lambda e: e.__setattr__("contacts", np.zeros((T + 1, 5), bool)),
            "integrity")
    mutated("invented contact flags", lambda e: e.contacts.__setitem__((slice(1, None), 4), True), "integrity")
    mutated("emptied contact table", lambda e: e.contact_table.__setitem__(
        "sample", e.contact_table["sample"][:0]), "integrity")
    mutated("negative slip", lambda e: e.samples["foot_slip"].__setitem__((hold[3], 0), -1.0), "integrity")
    mutated("fractional contact count", lambda e: e.samples.__setitem__(
        "body_ground_contacts", e.samples["body_ground_contacts"].astype(float) + 0.5), "integrity")

    def decreasing_warnings(e):
        e.samples["warnings"][1000:] = 1
        e.terminal["warnings"][:] = 1
        e.samples["warnings"][1500] = 0
    mutated("decreasing warning counter", decreasing_warnings, "integrity")

    # ---- stepping and timing
    mutated("inflated control_dt", lambda e: e.__setattr__("control_dt", 1.0), "integrity")
    mutated("wrong substeps", lambda e: e.__setattr__("substeps", 5), "integrity")
    mutated("equal timestamps", lambda e: e.samples["time"].__setitem__(slice(None), 0.0), "integrity")
    mutated("shifted timestamp origin", lambda e: e.samples["time"].__setitem__(slice(None),
            e.samples["time"] + 1.0), "integrity")

    def shorten(e):
        keep = 10
        e.qpos, e.qvel, e.ctrl = e.qpos[:keep + 1], e.qvel[:keep + 1], e.ctrl[:keep + 1]
        e.contacts = e.contacts[:keep + 1]
        n = keep * e.substeps
        e.samples = {k: v[:n] for k, v in e.samples.items()}
        e.sample_step = e.sample_step[:n]
        e.phases = e.phases[:keep]
    mutated("hold shorter than required", shorten, "integrity")

    # ---- consistency between frames, samples and commands
    mutated("physics command differs from frame command",
            lambda e: e.samples["ctrl"].__setitem__((700, 2), 1000.0), "integrity")
    mutated("physics joint position inconsistent",
            lambda e: e.samples["joint_q"].__setitem__((700, 2), 100.0), "integrity")
    mutated("joint excess inconsistent with positions",
            lambda e: e.samples["joint_excess"].__setitem__(700, 0.5), "integrity")

    # ---- physical gates
    mutated("zero normal forces", lambda e: (e.samples["foot_normal"].__setitem__(slice(None), 0.0),
                                             e.terminal["foot_normal"].__setitem__(slice(None), 0.0)),
            "integrity")

    def table_rows(e, keep):
        e.contact_table.update({k: v[keep] for k, v in e.contact_table.items()})

    def unsupported_hold(e):   # a genuinely unloaded hold: contacts, forces and flags all gone
        first_step = T - 60
        table_rows(e, e.contact_table["step"] < first_step)
        e.contacts[first_step + 1:] = False
        for ch in ("foot_normal", "foot_tangential"):
            e.samples[ch][e.sample_step >= first_step] = 0.0
            e.terminal[ch][:] = 0.0
        for ch in ("foot_slip", "foot_penetration"):
            e.samples[ch][e.sample_step >= first_step] = np.nan
            e.terminal[ch][:] = np.nan
    mutated("unsupported hold", unsupported_hold, "hold support")

    def terminal_unsupported(e):   # the final boundary state has no support
        S = T * sub
        keep = ~((e.contact_table["sample"] == S) & (e.contact_table["boundary"] == 1))
        table_rows(e, keep)
        for ch in ("foot_normal", "foot_tangential"):
            e.terminal[ch][:] = 0.0
        for ch in ("foot_slip", "foot_penetration"):
            e.terminal[ch][:] = np.nan
    mutated("terminal boundary unsupported", terminal_unsupported, "hold support")

    def body_contact(e):      # torso touching the floor, recorded consistently
        sample, step = 700, 70
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "torso_collision")
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        e.contact_table["sample"] = np.append(e.contact_table["sample"], sample)
        e.contact_table["step"] = np.append(e.contact_table["step"], step)
        e.contact_table["boundary"] = np.append(e.contact_table["boundary"], 0)
        e.contact_table["geom"] = np.vstack([e.contact_table["geom"], [floor, torso]])
        e.contact_table["pos"] = np.vstack([e.contact_table["pos"], [0.0, 0.0, 0.0]])
        e.contact_table["frame"] = np.vstack([e.contact_table["frame"], np.eye(3).ravel()])
        e.contact_table["force"] = np.vstack([e.contact_table["force"], [12.0, 0, 0, 0, 0, 0]])
        e.contact_table["dist"] = np.append(e.contact_table["dist"], -0.001)
        e.contact_table["velocity"] = np.vstack([e.contact_table["velocity"], np.zeros(3)])
        e.samples["body_ground_contacts"][sample] = 1
        e.samples["body_ground_force"][sample] = 12.0
        e.contacts[step + 1, 4] = True
    mutated("body contact", body_contact, "body contact")

    mutated("foot force not backed by a contact",
            lambda e: e.samples["foot_normal"].__setitem__((hold[5], 1), 200.0), "integrity")
    # 0.9 in SERVO-POSITION units (ctrl / gear), what action_scale bounds: 0.9 ctrl on
    # a unit-gear plant, 7.2 ctrl on an 8:1 geared thigh (0.9 ctrl there is a reachable 0.11 rad).
    jump = 0.9 * float(plantmap.transmission(model).gear[4])

    def command_jump(e):      # the command and its physics samples move together
        e.ctrl[100, 4] += jump
        e.samples["ctrl"][99 * sub:100 * sub, 4] += jump
    mutated("unreachable command jump", command_jump, "commands")
    mutated("torque over limit", lambda e: e.samples["actuator_force"].__setitem__((50, 2), 85.0), "torque")
    mutated("torque-speed violation", lambda e: (e.samples["actuator_force"].__setitem__((65, 0), 33.0),
                                                 e.samples["joint_qvel"].__setitem__((65, 0), 15.0)),
            "torque-speed")
    # Past the corner speed the only admissible torque is full braking, -force limit; half
    # of it is inside the force limit (the torque gate passes) and outside the DC law.
    brake = -0.5 * float(model.actuator_forcerange[0, 1])
    mutated("overspeed beyond corner speed",
            lambda e: (e.samples["actuator_force"].__setitem__((65, 0), brake),
                       e.samples["joint_qvel"].__setitem__((65, 0), 100.0)), "torque-speed")
    mutated("MuJoCo warning", lambda e: (e.samples["warnings"].__setitem__(slice(1500, None), 1),
                                         e.terminal["warnings"].__setitem__(slice(None), 1)), "no warnings")

    # ---- the hold interval: boundary frame and between-frame excursions
    mutated("hold opening frame height", lambda e: (
        e.qpos.__setitem__((hold_first_frame, 2), 0.1),
        e.samples["base_qpos"].__setitem__((hold_first_frame * sub, 2), 0.1)), "hold height")
    mutated("between-frame joint speed spike",
            lambda e: e.samples["joint_qvel"].__setitem__(hold[5], 1.0), "hold speed")
    mutated("between-frame tilt excursion", lambda e: e.samples["base_qpos"].__setitem__(
        hold[5], [0, 0, 0.4, np.sqrt(0.5), np.sqrt(0.5), 0, 0]), "hold tilt")

    uncal = gates.StandingSpec.from_dict({**acceptance["spec"], "joints_ref": None})
    mutated("uncalibrated spec", lambda e: None, "calibration", sp=uncal)

    # ---- bundle semantics (hashes recomputed, so only meaning can catch these)
    def bundle_case(name, mutate_clip=None, mutate_manifest=None, expect_load_failure=False):
        with tempfile.TemporaryDirectory() as tmp:
            copied = pathlib.Path(tmp) / "b"
            shutil.copytree(path, copied)
            m = json.loads((copied / "manifest.json").read_text())
            c = json.loads((copied / "clip.json").read_text())
            if mutate_clip:
                mutate_clip(c)
                (copied / "clip.json").write_text(json.dumps(c) + "\n")
                m["files"]["clip.json"] = hashlib.sha256((copied / "clip.json").read_bytes()).hexdigest()
            if mutate_manifest:
                mutate_manifest(m)
            (copied / "manifest.json").write_text(json.dumps(m) + "\n")
            try:
                m2, c2, a2 = bundle.load(copied)
            except ValueError as error:
                check(f"rejects {name}", expect_load_failure, str(error)[:70])
                return
            problems = bundle.verify(m2, c2, a2, model, model_bytes)
            check(f"rejects {name}", bool(problems), "; ".join(problems)[:90])

    bundle_case("tampered clip.json", mutate_clip=lambda c: c["episodes"][0].__setitem__("success", False),
                mutate_manifest=lambda m: m["files"].__setitem__(
                    "clip.json", manifest["files"]["clip.json"]), expect_load_failure=True)
    bundle_case("unknown clip schema", mutate_clip=lambda c: c.__setitem__("schema", 999))
    bundle_case("NaN action", mutate_clip=lambda c: c["episodes"][0]["frames"][5].__setitem__(
        "action", [float("nan")] * model.nu))
    bundle_case("actions not reconstructable", mutate_clip=lambda c: c["episodes"][0]["frames"][5].__setitem__(
        "action", [0.5] * model.nu))
    bundle_case("duplicate frame timestamps", mutate_clip=lambda c: c["episodes"][0]["frames"][5].__setitem__(
        "t", c["episodes"][0]["frames"][4]["t"]))
    bundle_case("clip state differing from arrays",
                mutate_clip=lambda c: c["episodes"][0]["frames"][7]["qpos"].__setitem__(2, 0.9))
    bundle_case("success disagreeing with the verdict",
                mutate_clip=lambda c: c["episodes"][0].__setitem__("success", not c["episodes"][0]["success"]))
    bundle_case("missing provenance", mutate_manifest=lambda m: m.pop("source"))
    bundle_case("missing embedded acceptance config", mutate_manifest=lambda m: m["acceptance"].pop("motor"))
    bundle_case("unsupported execution version", mutate_manifest=lambda m: m.__setitem__("execution_version", 99))
    bundle_case("phases not covering the motion",
                mutate_manifest=lambda m: m["phases"].__setitem__("hold", [m["phases"]["hold"][0], 10]))
    bundle_case("infinite action scale",
                mutate_manifest=lambda m: m["action_encoding"].__setitem__("scale", float("inf")))

    # ---- replay agreement must see more than the base pose
    stepping = manifest["stepping"]
    # Replay under the ratings the bundle was solved with. Passing no cfg used to
    # be allowed and silently ran a flat-clamped servo, so "replay is exact" was
    # asserted against a different motor than the recording.
    replayed = execution.replay(model, arrays["initial_state"], arrays["ctrl"], stepping["substeps"],
                                stepping["control_dt"], phases=good.phases,
                                cfg=pipeline._BundleMotor(
                                    {k: tuple(v) for k, v in manifest["acceptance"]["motor"].items()}))
    exact = gates.Report()
    pipeline.replay_gate(exact, execution.agreement(good, replayed), "replay")
    check("replay of the saved commands is exact", exact.passed, exact.gates[0].detail)

    for label, mutate in (
        ("joint mismatch", lambda e: e.qpos.__setitem__((50, 9), e.qpos[50, 9] + 0.5)),
        ("velocity mismatch", lambda e: e.qvel.__setitem__((50, 7), e.qvel[50, 7] + 100.0)),
        ("contact mismatch", lambda e: e.contacts.__setitem__(slice(1, None), ~e.contacts[1:])),
        ("command mismatch", lambda e: e.ctrl.__setitem__((50, 0), e.ctrl[50, 0] + 1e-6)),
        ("sample force mismatch", lambda e: e.samples["foot_normal"].__setitem__((500, 0), 500.0)),
    ):
        ex = copy.deepcopy(good)
        mutate(ex)
        rep = gates.Report()
        pipeline.replay_gate(rep, execution.agreement(ex, replayed), "replay")
        check(f"replay rejects {label}", not rep.passed, rep.gates[0].detail[:80])

    failed = 0
    for name, ok, detail in RESULTS:
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<44s} {detail}")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

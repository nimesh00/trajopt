"""Optimisation pipeline: solve -> execute -> validate -> bundle, and replay.

    python -m trajgen.pipeline solve stand-squat --out out/bundles/stand-squat
    python -m trajgen.pipeline replay out/bundles/stand-squat --video out/stand.mp4 --show

`solve` exports a bundle whether or not validation passes, marked accordingly
(`validation.passed`, clip `success`), so failures can be inspected; it exits
nonzero on failure. `replay` is the independent check: in a fresh process it
verifies hashes, restores the saved full snapshot into a fresh MjData, applies
the saved commands on the same stepping schedule, requires agreement with the
recorded execution (< 1 mm, < 1 mrad), re-runs every gate on the REPLAYED
execution, and renders the replayed states.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from . import adapter as adapter_mod
from . import bundle, config, croc, execution, gait, gates
from .bridge import BridgeConfig, ServoBridge

ROOT = Path(__file__).resolve().parents[1]
REPLAY_TOL = {"translation_max_m": 1e-3, "orientation_max_rad": 1e-3, "joint_max_rad": 1e-3,
              "velocity_max": 1e-2, "ctrl_max": 1e-12, "contact_mismatch_frames": 0}


def replay_gate(report, agree: dict, name: str) -> None:
    """Agreement on EVERY channel: base pose, joints, velocities, commands, contacts, samples."""
    if not agree.get("comparable"):
        report.add(name, False, "executions are not comparable (different lengths)")
        return
    over = [f"{k} {agree[k]:.2e} > {tol:g}" for k, tol in REPLAY_TOL.items() if not agree[k] <= tol]
    channels = agree["sample_channel_max"]
    kinematic = {"base_qpos", "joint_q", "joint_excess", "foot_penetration"}
    for key, value in channels.items():
        tol = 1e-3 if key in kinematic else (1.0 if "force" in key or "tangential" in key or "normal" in key
                                              else 1e-2 if key != "time" and key != "warnings" and "contacts" not in key
                                              else 0.0)
        if not value <= tol:
            over.append(f"sample {key} {value:.2e} > {tol:g}")
    if not agree["terminal_max"] <= 1.0:
        over.append(f"terminal sample {agree['terminal_max']:.2e}")
    report.add(name, not over, "; ".join(over) if over else
               f"translation {agree['translation_max_m']:.1e} m, orientation {agree['orientation_max_rad']:.1e} rad, "
               f"joints {agree['joint_max_rad']:.1e}, velocity {agree['velocity_max']:.1e}, "
               f"ctrl {agree['ctrl_max']:.1e}, samples {agree['sample_max']:.1e}, "
               f"contacts {agree['contact_mismatch_frames']} mismatched frames")


def _phase_ranges(phases: list[str]) -> dict:
    ranges: dict[str, list[int]] = {}
    for k, label in enumerate(phases):
        if label in ranges and ranges[label][1] != k - 1:
            label = f"{label}@{k}"
        ranges.setdefault(label, [k, k])[1] = k
    return ranges


def solve_stand(cfg: config.RobotConfig, out: Path, feedback: str = "riccati") -> int:
    from .motions import stand

    model = mujoco.MjModel.from_binary_path(str(cfg.mjb))
    config.assert_plant(cfg, model)
    ad = adapter_mod.build(cfg.mjb, cfg.urdf)
    bridge_cfg = BridgeConfig(feedback=feedback)
    br = ServoBridge(ad, cfg.kp, cfg.kd, bridge_cfg)

    start = stand.settled_stand(model, ad, ServoBridge(ad, cfg.kp, cfg.kd), cfg)
    print(f"start: settled z {start.qpos[2]:.4f} m, residual |qvel| {start.residual_speed:.2e}")
    x0, knots, terminal, motion_info = stand.schedule(ad, cfg, start)

    ocp = croc.OCP(ad, cfg)
    sol = ocp.solve(x0, knots, terminal, max_iter=200)
    diag = sol.diagnostics
    print(f"solve: converged={diag['converged']} iters={diag['iterations']} "
          f"stop={diag['stop']:.1e} feasible={diag['feasible']} ({diag['seconds']} s)")

    steps = sol.control_steps()

    def controller(k, data):
        node = steps[k][0]
        x_meas = ad.x_from_mj(data.qpos, data.qvel) if feedback != "none" else None
        return br.command(sol.xs[node], sol.xs[node + 1], sol.us[node], sol.K[node], x_meas)

    ex = execution.Executor(model, cfg.substeps, cfg).run(start.state, controller, len(steps), cfg.control_dt,
                                                     phases=[p for _, p in steps])
    planned_qpos = np.array([ad.qpos_from_pin(sol.xs[n][:ad.pin.nq]) for n, _ in steps]
                            + [ad.qpos_from_pin(sol.xs[-1][:ad.pin.nq])])
    tracking = {
        "base_z_max_m": float(np.max(np.abs(planned_qpos[:, 2] - ex.qpos[:, 2]))),
        "base_xy_max_m": float(np.max(np.abs(planned_qpos[:, :2] - ex.qpos[:, :2]))),
        "joint_max_rad": float(np.max(np.abs(planned_qpos[:, 7:] - ex.qpos[:, 7:]))),
        "command_saturations": int(sum(i["saturated"] for i in ex.info)),
        "feedback_max_nm": float(max(i["feedback_max"] for i in ex.info)),
    }
    print(f"execute: tracking z {tracking['base_z_max_m'] * 1e3:.2f} mm, "
          f"joints {tracking['joint_max_rad']:.4f} rad, saturations {tracking['command_saturations']}")

    spec = stand.acceptance(model, start)
    report = gates.validate_standing(ex, model, spec, cfg.motor, cfg.action_scale)
    # Solver feasibility is a required gate too: a converged but infeasible plan is not accepted.
    report.add("planner feasibility", diag["feasible"],
               f"converged {diag['converged']}, dynamics defect {diag['dynamic_feasibility']:.1e}, "
               f"torque use {diag['torque_utilisation_max']:.2f}, min fz {diag['min_normal_force']:.1f} N, "
               f"friction ratio {diag['max_friction_ratio']:.3f}")
    progress = stand.progression(ex, motion_info)
    report.add("motion progression", progress["passed"], progress["detail"])
    rp = execution.replay(model, start.state, ex.ctrl, cfg.substeps, cfg.control_dt,
                          phases=[p for _, p in steps], cfg=cfg)
    agree = execution.agreement(ex, rp)
    replay_gate(report, agree, "in-process replay")
    print(report)

    model_bytes = cfg.mjb.read_bytes()
    clip = bundle.clip_document(ex, model, model_bytes, task=f"{cfg.name}-{stand.NAME}",
                                task_schema="trajopt-stand-squat-v1", kp=cfg.kp, kd=cfg.kd,
                                action_scale=cfg.action_scale, success=report.passed, ending="hold")
    arrays = bundle.execution_arrays(ex)
    arrays.update({"planned_xs": sol.xs, "planned_us": sol.us, "planned_K": sol.K,
                   "planned_forces": sol.forces, "planned_qpos": planned_qpos,
                   "spec_joints_ref": np.asarray(spec.joints_ref)})
    manifest = {
        "motion": stand.NAME, "status": stand.STATUS,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "robot_config": {"path": str(cfg.path), "sha256": cfg.sha256},
        "model": {"mjb": str(cfg.mjb), "mjb_sha256": adapter_mod.sha256(cfg.mjb),
                  "model_id": clip["model_id"], "urdf": str(cfg.urdf),
                  "urdf_sha256": adapter_mod.sha256(cfg.urdf)},
        "adapter": ad.manifest(),
        "stepping": {"physics_timestep": float(model.opt.timestep), "substeps": cfg.substeps,
                     "control_dt": cfg.control_dt, "boundary_mj_forward": True,
                     "contact_union": "per substep plus boundary pass",
                     "samples": "pre-step state after each mj_step"},
        "action_encoding": {"mode": "command-delta", "version": 1, "scale": cfg.action_scale,
                            "bound": 1.0, "authoritative": "ctrl (standalone forward replay)",
                            "rl_integration": "none claimed"},
        "controller": {"bridge": bridge_cfg.as_dict(), "start": start.info,
                       "start_residual_speed": start.residual_speed},
        "planner": {**diag, "backend": "crocoddyl", "weights": ocp.w.__dict__},
        "phases": _phase_ranges([p for _, p in steps]),
        "motion_info": motion_info,
        "acceptance": {"validator": "standing", "gates_version": gates.GATES_VERSION,
                       "spec": spec.as_dict(), "motor": {k: list(v) for k, v in cfg.motor.items()},
                       "progression": progress},
        "tracking": tracking,
        "replay_agreement": agree,
        "validation": report.as_dict(),
    }
    if not report.passed:
        out = out.with_name(out.name + "-REJECTED")   # failures never occupy the requested name
    path = bundle.write(out, manifest=manifest, clip=clip, arrays=arrays, repo=ROOT)
    print(f"\nbundle: {path}  ({'ACCEPTED' if report.passed else 'REJECTED'})")
    print(f"replay: {sys.executable} -m trajgen.pipeline replay {path} --show")
    return 0 if report.passed else 1


def solve_getup(cfg: config.RobotConfig, out: Path, seeds=(1, 2, 3), feedback: str = "riccati") -> int:
    """Belly-up getup: MuJoCo search for the recovery, then validate the whole run.

    The rolling part cannot be posed as a rigid-contact problem (the contacts
    change, and Crocoddyl 3.1 cannot even carry two contacts on one parent
    joint), so the search owns it and every phase records the backend that
    produced it. The final hold is handed to Crocoddyl when, and only when, the
    robot has actually landed on four feet with nothing else touching.
    """
    from . import search as search_mod
    from .motions import getup, stand

    model = mujoco.MjModel.from_binary_path(str(cfg.mjb))
    config.assert_plant(cfg, model)
    ad = adapter_mod.build(cfg.mjb, cfg.urdf)
    reference = stand.settled_stand(model, ad, ServoBridge(ad, cfg.kp, cfg.kd), cfg)
    start = getup.settled_inverted(model, cfg)
    print(f"start: inverted, upright {start.upright:+.3f}, on {', '.join(start.contacts)}, "
          f"settled in {start.info['settle_seconds']:.2f} s")

    target = float(reference.qpos[2])
    best = None
    for round_name, mean in (("explore", None), ("refine", "best")):
        for seed in seeds:
            sc = search_mod.SearchConfig(seed=seed, init_std=0.35 if mean is None else 0.25)
            warm = None if mean is None or best is None else best.knots
            if mean is not None and warm is None:
                continue
            result = search_mod.cem(model, start.state, start.ctrl, cfg, sc, target,
                                    mean_knots=warm, family=f"{round_name}{seed}")
            m = result.metrics
            print(f"  search {result.family}: score {result.score:.2f}, upright "
                  f"{m['terminal_upright']:+.3f}, height {m['terminal_height']:.3f} m, "
                  f"{m['feet_down']:.0f} feet down, clearance {m['body_clearance'] * 1e3:.0f} mm")
            if best is None or result.score > best.score:
                best = result
    hold_steps = int(round(best.config["hold"] / cfg.control_dt))
    phases = ["recover"] * (len(best.commands) - hold_steps) + ["hold"] * hold_steps
    backends = {"recover": "mujoco-search", "hold": "mujoco-search"}

    executor = execution.Executor(model, cfg.substeps, cfg)
    ex = executor.run(start.state, lambda k, d: (best.commands[k], None), len(best.commands),
                      cfg.control_dt, phases=phases)
    landing = getup.landed(model, ex)
    print(f"recovery: height {landing['height']:.3f} m, upright {landing['upright']:+.3f}, "
          f"{landing['feet_loaded']}/4 feet loaded, {landing['body_contacts']} body contacts, "
          f"|qvel| {landing['speed']:.3f}")

    spec = stand.acceptance(model, reference)
    spec.body_contact_phases = ("recover",)
    report = gates.validate_standing(ex, model, spec, cfg.motor, cfg.action_scale)
    report.add("landed on four feet", landing["feet_loaded"] == 4 and landing["body_contacts"] == 0,
               f"{landing['feet_loaded']}/4 feet loaded, {landing['body_contacts']} non-foot contacts")
    replayed = execution.replay(model, start.state, ex.ctrl, cfg.substeps, cfg.control_dt, phases=phases, cfg=cfg)
    replay_gate(report, execution.agreement(ex, replayed), "in-process replay")
    print(report)

    model_bytes = cfg.mjb.read_bytes()
    clip = bundle.clip_document(ex, model, model_bytes, task=f"{cfg.name}-{getup.NAME}",
                                task_schema="trajopt-getup-belly-up-v1", kp=cfg.kp, kd=cfg.kd,
                                action_scale=cfg.action_scale, success=report.passed, ending="hold")
    arrays = bundle.execution_arrays(ex)
    arrays.update({"search_knots": best.knots, "search_commands": best.commands,
                   "spec_joints_ref": np.asarray(spec.joints_ref)})
    manifest = {
        "motion": getup.NAME,
        "status": ("belly-up getup" if report.passed else getup.STATUS),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "robot_config": {"path": str(cfg.path), "sha256": cfg.sha256},
        "model": {"mjb": str(cfg.mjb), "mjb_sha256": adapter_mod.sha256(cfg.mjb),
                  "model_id": clip["model_id"], "urdf": str(cfg.urdf),
                  "urdf_sha256": adapter_mod.sha256(cfg.urdf)},
        "adapter": ad.manifest(),
        "stepping": {"physics_timestep": float(model.opt.timestep), "substeps": cfg.substeps,
                     "control_dt": cfg.control_dt, "boundary_mj_forward": True,
                     "contact_union": "per substep plus boundary pass",
                     "samples": "pre-step state after each mj_step"},
        "action_encoding": {"mode": "command-delta", "version": 1, "scale": cfg.action_scale,
                            "bound": 1.0, "authoritative": "ctrl (standalone forward replay)",
                            "rl_integration": "none claimed"},
        "controller": {"bridge": None, "start": start.info, "start_residual_speed":
                       start.info["residual_speed"]},
        "planner": {"backend": "mujoco-search", "backends_by_phase": backends,
                    "search": best.config, "family": best.family, "score": best.score,
                    "metrics": best.metrics, "history": best.history,
                    "converged": None, "iterations": best.config["iterations"], "feasible": None},
        "phases": _phase_ranges(phases),
        "motion_info": {"landing": landing, "reference_height": target,
                        "inverted_start": {k: v for k, v in start.info.items() if k != "trace"}},
        "acceptance": {"validator": "standing", "gates_version": gates.GATES_VERSION,
                       "spec": spec.as_dict(), "motor": {k: list(v) for k, v in cfg.motor.items()},
                       "progression": landing},
        "tracking": {"backend": "mujoco-search: commands are the searched sequence, "
                                "so there is no planned-versus-executed state error to report"},
        "replay_agreement": execution.agreement(ex, replayed),
        "validation": report.as_dict(),
    }
    if not report.passed:
        out = out.with_name(out.name + "-REJECTED")
    path = bundle.write(out, manifest=manifest, clip=clip, arrays=arrays, repo=ROOT)
    print(f"\nbundle: {path}  ({'ACCEPTED' if report.passed else 'REJECTED'})")
    print(f"replay: {sys.executable} -m trajgen.pipeline replay {path} --show")
    return 0 if report.passed else 1


def _lateral(args) -> dict:
    """The lateral foot-path flags that were given (walk.walk `lateral`)."""
    from .motions.walk import LATERAL
    return {k: getattr(args, k) for k in LATERAL if getattr(args, k, None) is not None}


def solve_walk(cfg: config.RobotConfig, out: Path, *, pattern: str = "trot",
               velocity: float = 0.25, velocity_y: float = 0.0, yaw_rate: float = 0.0,
               seconds: float = 6.0, feedback: str = "riccati",
               start_x: float | None = None, period: float | None = None,
               swing_height: float | None = None, max_lead: float | None = None,
               duty: float | None = None, accel_seconds: float | None = None,
               horizon: float | None = None, max_iter: int | None = None,
               rolling: bool = False, rolling_weight: float | None = None,
               dog_style: str | None = None, dog_gait: str | None = None,
               dog_parts: str | None = None, dog_trunk_scale: float = 1.0,
               stop_seconds: float | None = None, dog_duty_ramp: float | None = None,
               dog_bounds_weight: float | None = None, dog_support_sway: float = 0.0,
               trunk_acceleration_weight: float | None = None,
               swing_flat: float | None = None, lateral: dict | None = None) -> int:
    """Walk forward with a fixed gait schedule, then stop and hold.

    The gait is phase-clocked (trajgen/gait.py) and re-planned from the measured
    state every 0.1 s, so the plan can start anywhere in the cycle. Every
    command still comes from a Crocoddyl solve through the servo bridge, and the
    whole thing is one continuous rollout from one snapshot.
    """
    from .motions import stand, walk

    model = mujoco.MjModel.from_binary_path(str(cfg.mjb))
    config.assert_plant(cfg, model)
    ad = adapter_mod.build(cfg.mjb, cfg.urdf)
    bridge_cfg = BridgeConfig(feedback=feedback, rate_limit=cfg.action_scale)
    br = ServoBridge(ad, cfg.kp, cfg.kd, bridge_cfg)
    start = stand.settled_stand(model, ad, ServoBridge(ad, cfg.kp, cfg.kd), cfg)
    if start_x is not None:
        # TRANSLATE THE SETTLED STAND ALONG x. Flat ground is invariant under a
        # horizontal translation, so moving a settled state is exact there --
        # and it is what gives a walk room to run. The compiled stair plant is
        # flat from x = -12.00 to +0.75, i.e. 12 m BEHIND the spawn, so a walk
        # started back there has metres of level ground ahead of it instead of
        # the 0.75 m in front of the keyframe. Refused off the flat region,
        # because the translation is only exact while the ground is level.
        terrain_chk = gait.Terrain(model)
        span = [terrain_chk.height(start_x + d, 0.0) for d in (0.0, 0.25, 0.5)]
        if max(span) - min(span) > 1e-6 or abs(span[0]) > 1e-6:
            raise SystemExit(f"[ERROR] --start_x {start_x}: ground there is not level "
                             f"(heights {span}); the translation would be invalid")
        data0 = mujoco.MjData(model)
        mujoco.mj_setState(model, data0, start.state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data0)
        data0.qpos[0] += float(start_x)
        mujoco.mj_forward(model, data0)
        start = dataclasses.replace(
            start, qpos=np.array(data0.qpos),
            state=execution.snapshot(model, data0))
        print(f"start: translated to x = {start.qpos[0]:.3f} m on level ground")
    br.start_from(start.ctrl)
    print(f"start: settled z {start.qpos[2]:.4f} m, residual |qvel| {start.residual_speed:.2e}")
    dog = None
    if dog_style is not None:
        from . import dogstyle
        parts = tuple(p for p in (dog_parts or ",".join(dogstyle.PARTS)).split(",") if p)
        unknown = set(parts) - set(dogstyle.PARTS)
        if unknown:
            raise SystemExit(f"[ERROR] --dog-parts: unknown {sorted(unknown)} (known {dogstyle.PARTS})")
        h0, turn_radius = dogstyle.leg_root_scale(model, start.qpos)
        dog = dogstyle.DogGaitModel(dog_style, h0, turn_radius).targets(
            velocity, velocity_y, yaw_rate, family=dog_gait, parts=parts, trunk_scale=dog_trunk_scale,
            duty_ramp=dog_duty_ramp, support_sway=dog_support_sway)
        print(f"dog: scaled to leg-root height {h0:.4f} m, hip radius {turn_radius:.3f} m")
        pattern = f"dog-{dog.family}"
        print(f"dog: {dog.family} period {dog.period:.3f} s, duty "
              + "/".join(f"{dog.duties[k]:.2f}" for k in ("FL", "FR", "RL", "RR")) + ", offsets "
              + "/".join(f"{dog.offsets[k]:.3f}" for k in ("FR", "RL", "RR")) + f", parts {','.join(parts)}"
              + "".join(f"\n  [{lab}]" for lab in dog.labels))

    commands, phases, info = walk.walk(model, ad, cfg, start, br, pattern=pattern,
                                       velocity=velocity, velocity_y=velocity_y,
                                       yaw_rate=yaw_rate, seconds=seconds,
                                       period=period, swing_height=swing_height,
                                       max_lead=max_lead, duty=duty,
                                       accel_seconds=accel_seconds,
                                       **({} if horizon is None else {"horizon_seconds": horizon}),
                                       **({} if max_iter is None else {"max_iter": max_iter}),
                                       **({"rolling": True} if rolling else {}),
                                       **({} if rolling_weight is None else {"rolling_weight": rolling_weight}),
                                       **({} if dog is None else {"dog": dog}),
                                       **({} if not swing_flat else {"swing_flat": swing_flat}),
                                       **({} if not lateral else {"lateral": lateral}),
                                       **({} if dog_bounds_weight is None and trunk_acceleration_weight is None else {
                                           "costs": dataclasses.replace(walk.weights(), **{
                                               k: v for k, v in (("bounds", dog_bounds_weight),
                                                                 ("trunk_acceleration", trunk_acceleration_weight))
                                               if v is not None})}),
                                       **({} if stop_seconds is None else {"stop_seconds": stop_seconds}))
    print(f"solve: {info['solves']} re-plans in {info['solve_seconds']} s, "
          f"converged={info['converged']}, defect {info['max_dynamic_feasibility']:.1e}")

    ex = execution.replay(model, start.state, np.vstack([start.ctrl, commands]), cfg.substeps,
                          cfg.control_dt, phases=phases, cfg=cfg)
    # Score the motion that was ASKED FOR. A sidestep makes no x progress and a
    # turn in place makes none at all; measuring either against a forward
    # distance rejects them for the wrong reason.
    heading = (velocity, velocity_y) if (velocity or velocity_y) else (1.0, 0.0)
    # A turn's progress is its rotation, also when it walks while turning: a curved walk
    # returns towards its start, so its straight-line displacement says nothing.
    yaw_min = 0.0
    if yaw_rate:
        yaw_min = 0.5 * abs(yaw_rate) * max(seconds, 1e-9)
    progress = walk.progression(ex, heading=heading, yaw_min=yaw_min)
    print(f"execute: {progress['detail']}")

    spec = walk.acceptance(model, start)
    report = gates.validate_standing(ex, model, spec, cfg.motor, cfg.action_scale)
    report.add("travelled", progress["passed"], progress["detail"])
    # Feasibility of a receding horizon is per solve: every plan the robot
    # actually executed must be dynamically consistent. Hitting the iteration
    # cap on some solves is normal and is reported, not gated -- what matters is
    # that no executed plan had a dynamics defect.
    feasible = info["max_dynamic_feasibility"] < 1e-6
    report.add("planner feasibility", feasible,
               f"{info['solves']} solves, max dynamics defect {info['max_dynamic_feasibility']:.1e}, "
               f"{info['converged_fraction']:.0%} reached the stopping criterion "
               f"(max {info['max_iterations_used']} iterations)")
    replayed = execution.replay(model, start.state, ex.ctrl, cfg.substeps, cfg.control_dt,
                                phases=phases, cfg=cfg)
    replay_gate(report, execution.agreement(ex, replayed), "in-process replay")
    print(report)

    model_bytes = cfg.mjb.read_bytes()
    clip = bundle.clip_document(ex, model, model_bytes, task=f"{cfg.name}-{walk.NAME}",
                                task_schema="trajopt-walk-v1", kp=cfg.kp, kd=cfg.kd,
                                action_scale=cfg.action_scale, success=report.passed, ending="hold")
    arrays = bundle.execution_arrays(ex)
    arrays["spec_joints_ref"] = np.asarray(spec.joints_ref)
    manifest = {
        "motion": walk.NAME, "status": walk.STATUS,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "robot_config": {"path": str(cfg.path), "sha256": cfg.sha256},
        "model": {"mjb": str(cfg.mjb), "mjb_sha256": adapter_mod.sha256(cfg.mjb),
                  "model_id": clip["model_id"], "urdf": str(cfg.urdf),
                  "urdf_sha256": adapter_mod.sha256(cfg.urdf)},
        "adapter": ad.manifest(),
        "stepping": {"physics_timestep": float(model.opt.timestep), "substeps": cfg.substeps,
                     "control_dt": cfg.control_dt, "boundary_mj_forward": True,
                     "contact_union": "per substep plus boundary pass",
                     "samples": "pre-step state after each mj_step",
                     # recorded only off the default, so forcerange-plant manifests are unchanged
                     **({"envelope": cfg.envelope} if cfg.envelope != "forcerange" else {})},
        "action_encoding": {"mode": "command-delta", "version": 1, "scale": cfg.action_scale,
                            "bound": 1.0, "authoritative": "ctrl (standalone forward replay)",
                            "rl_integration": "none claimed"},
        "controller": {"bridge": bridge_cfg.as_dict(), "start": start.info,
                       "start_residual_speed": start.residual_speed},
        "planner": {**info, "backend": "crocoddyl", "feasible": feasible},
        "phases": _phase_ranges(phases),
        # The WHOLE command, not just its x component: a sidestep and a turn are
        # the same motion type with a different command, and a clip that does not
        # record what it was asked for cannot be used as a reference for it.
        "motion_info": {"travel": progress, "pattern": pattern, "velocity": velocity,
                        "velocity_y": velocity_y, "yaw_rate": yaw_rate,
                        "period": info.get("period"), "duty": info.get("duty"),
                        "swing_height": info.get("swing_height"),
                        "accel_seconds": info.get("accel_seconds"),
                        **({"rolling": True} if rolling else {}),
                        **({"swing_flat": swing_flat} if swing_flat else {}),
                        **({"lateral": dict(lateral)} if lateral else {}),
                        **({} if dog is None else {"dog_style": {"model": str(dog_style), **info["dog"]}})},
        "acceptance": {"validator": "standing", "gates_version": gates.GATES_VERSION,
                       "spec": spec.as_dict(), "motor": {k: list(v) for k, v in cfg.motor.items()},
                       "progression": progress},
        "tracking": {"note": "receding horizon: each command comes from a solve at the "
                             "measured state, so there is no single planned trajectory to "
                             "difference against"},
        "replay_agreement": execution.agreement(ex, replayed),
        "validation": report.as_dict(),
    }
    if not report.passed:
        out = out.with_name(out.name + "-REJECTED")
    path = bundle.write(out, manifest=manifest, clip=clip, arrays=arrays, repo=ROOT)
    print(f"\nbundle: {path}  ({'ACCEPTED' if report.passed else 'REJECTED'})")
    print(f"replay: {sys.executable} -m trajgen.pipeline replay {path} --show")
    return 0 if report.passed else 1


def solve_stairs(cfg: config.RobotConfig, out: Path, *, risers: int = 4,
                pattern: str = "pairs", velocity: float | None = None, seconds: float | None = None,
                feedback: str = "riccati") -> int:
    """Climb a measured staircase, stop on a tread and hold.

    Same loop as the walk -- phase clock, Crocoddyl solve from the measured
    state, servo bridge, one continuous rollout -- with the scheduler reading
    ground height from the plant's static geoms and snapping footholds onto
    treads. The acceptance is stair-specific: risers actually climbed, stance
    placement inside the treads, and a closing hold scored against the plane
    through the feet rather than world zero.
    """
    from .motions import stairs, stand

    model = mujoco.MjModel.from_binary_path(str(cfg.mjb))
    config.assert_plant(cfg, model)
    ad = adapter_mod.build(cfg.mjb, cfg.urdf)
    terrain = gait.Terrain(model)
    profile = stairs.measure(model, terrain)
    print(f"terrain: {profile.count} risers, first edge x = {profile.edges[0]:.3f} m, "
          f"rise {profile.rise * 1e3:.0f} mm, tread {profile.tread * 1e3:.0f} mm, "
          f"{profile.slope_deg:.1f} deg")
    bridge_cfg = BridgeConfig(feedback=feedback, rate_limit=cfg.action_scale)
    br = ServoBridge(ad, cfg.kp, cfg.kd, bridge_cfg)
    start = stand.settled_stand(model, ad, ServoBridge(ad, cfg.kp, cfg.kd), cfg)
    br.start_from(start.ctrl)
    print(f"start: settled z {start.qpos[2]:.4f} m, residual |qvel| {start.residual_speed:.2e}")

    extra = {} if seconds is None else {"seconds": seconds}
    commands, phases, info = stairs.climb(model, ad, cfg, start, br, profile=profile,
                                          terrain=terrain, pattern=pattern, risers=risers,
                                          velocity=velocity, **extra)
    print(f"solve: {info['solves']} re-plans in {info['solve_seconds']} s, "
          f"converged={info['converged']}, defect {info['max_dynamic_feasibility']:.1e}")

    ex = execution.replay(model, start.state, np.vstack([start.ctrl, commands]), cfg.substeps,
                          cfg.control_dt, phases=phases, cfg=cfg)
    progress = stairs.progression(ex, profile, ad)
    placement = stairs.treads(ex, model, profile)
    print(f"execute: {progress['detail']}")
    print(f"placement: {placement['detail']}")

    spec = stairs.acceptance(model, start, ex, ad, profile)
    report = gates.validate_standing(ex, model, spec, cfg.motor, cfg.action_scale)
    report.add("ascent", progress["passed"], progress["detail"])
    report.add("stance inside treads", placement["passed"], placement["detail"])
    feasible = info["max_dynamic_feasibility"] < 1e-6
    report.add("planner feasibility", feasible,
               f"{info['solves']} solves, max dynamics defect {info['max_dynamic_feasibility']:.1e}, "
               f"{info['converged_fraction']:.0%} reached the stopping criterion "
               f"(max {info['max_iterations_used']} iterations)")
    replayed = execution.replay(model, start.state, ex.ctrl, cfg.substeps, cfg.control_dt,
                                phases=phases, cfg=cfg)
    replay_gate(report, execution.agreement(ex, replayed), "in-process replay")
    print(report)

    model_bytes = cfg.mjb.read_bytes()
    clip = bundle.clip_document(ex, model, model_bytes, task=f"{cfg.name}-{stairs.NAME}",
                                task_schema="trajopt-stairs-v1", kp=cfg.kp, kd=cfg.kd,
                                action_scale=cfg.action_scale, success=report.passed, ending="hold")
    arrays = bundle.execution_arrays(ex)
    arrays["spec_joints_ref"] = np.asarray(spec.joints_ref)
    arrays["stair_edges_x"] = np.asarray(profile.edges)
    arrays["stair_tread_heights"] = np.asarray(profile.heights)
    manifest = {
        "motion": stairs.NAME, "status": stairs.STATUS,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "robot_config": {"path": str(cfg.path), "sha256": cfg.sha256},
        "model": {"mjb": str(cfg.mjb), "mjb_sha256": adapter_mod.sha256(cfg.mjb),
                  "model_id": clip["model_id"], "urdf": str(cfg.urdf),
                  "urdf_sha256": adapter_mod.sha256(cfg.urdf)},
        "adapter": ad.manifest(),
        "terrain": {**profile.as_dict(), "mu": cfg.mu, "cone_margin": cfg.cone_margin,
                    "hash": adapter_mod.sha256(cfg.mjb)},
        "stepping": {"physics_timestep": float(model.opt.timestep), "substeps": cfg.substeps,
                     "control_dt": cfg.control_dt, "boundary_mj_forward": True,
                     "contact_union": "per substep plus boundary pass",
                     "samples": "pre-step state after each mj_step"},
        "action_encoding": {"mode": "command-delta", "version": 1, "scale": cfg.action_scale,
                            "bound": 1.0, "authoritative": "ctrl (standalone forward replay)",
                            "rl_integration": "none claimed"},
        "controller": {"bridge": bridge_cfg.as_dict(), "start": start.info,
                       "start_residual_speed": start.residual_speed},
        "planner": {**info, "backend": "crocoddyl", "feasible": feasible},
        "phases": _phase_ranges(phases),
        "motion_info": {"ascent": progress, "placement": placement, "pattern": pattern,
                        "velocity": info["velocity"]},
        "acceptance": {"validator": "standing (support-plane referenced)",
                       "gates_version": gates.GATES_VERSION, "spec": spec.as_dict(),
                       "motor": {k: list(v) for k, v in cfg.motor.items()},
                       "progression": progress, "placement": placement},
        "tracking": {"note": "receding horizon: each command comes from a solve at the "
                             "measured state, so there is no single planned trajectory to "
                             "difference against"},
        "replay_agreement": execution.agreement(ex, replayed),
        "validation": report.as_dict(),
    }
    if not report.passed:
        out = out.with_name(out.name + "-REJECTED")
    path = bundle.write(out, manifest=manifest, clip=clip, arrays=arrays, repo=ROOT)
    print(f"\nbundle: {path}  ({'ACCEPTED' if report.passed else 'REJECTED'})")
    print(f"replay: {sys.executable} -m trajgen.pipeline replay {path} --show")
    return 0 if report.passed else 1


class _BundleMotor:
    """The attributes the executor's envelope reads, sourced from a bundle's own ratings:
    `motor`, and `envelope` ("forcerange" unless the bundle's stepping records otherwise)."""
    __slots__ = ("motor", "envelope")

    def __init__(self, motor, envelope="forcerange"):
        self.motor = motor
        self.envelope = envelope


def replay(path: Path, video: Path | None, show: bool, slowmo: float) -> int:
    try:
        manifest, clip, arrays = bundle.load(path)
        mjb = Path(manifest["model"]["mjb"])
        model_bytes = mjb.read_bytes()
        model = mujoco.MjModel.from_binary_path(str(mjb))
        problems = bundle.verify(manifest, clip, arrays, model, model_bytes)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"REFUSED: {path}: {type(error).__name__}: {error}")
        return 2
    if problems:
        print(f"REFUSED: {path} failed semantic verification:\n  " + "\n  ".join(problems))
        return 2

    stepping = manifest["stepping"]
    acceptance = manifest["acceptance"]
    spec = gates.StandingSpec.from_dict(acceptance["spec"])
    motor = {k: tuple(v) for k, v in acceptance["motor"].items()}
    scale = manifest["action_encoding"]["scale"]
    recorded = bundle.execution_from_arrays(arrays, manifest)
    # The bundle carries the motor ratings it was solved under; replaying without
    # them scores a stored clip against a motor the robot does not have.
    replayed = execution.replay(model, arrays["initial_state"], arrays["ctrl"], stepping["substeps"],
                                stepping["control_dt"], phases=recorded.phases,
                                cfg=_BundleMotor(motor, stepping.get("envelope", "forcerange")))

    print(f"bundle {path}: {manifest['motion']} -- {manifest['status']}")
    recorded_report = gates.validate_standing(recorded, model, spec, motor, scale)
    report = gates.validate_standing(replayed, model, spec, motor, scale)
    report.add("recorded execution valid", recorded_report.passed,
               "all gates pass on the stored execution" if recorded_report.passed else
               "failed: " + ", ".join(g.name for g in recorded_report.gates if g.required and not g.passed))
    replay_gate(report, execution.agreement(recorded, replayed), "replay agreement")
    planner = manifest["planner"]
    if planner.get("backend") == "crocoddyl":
        report.add("planner feasibility (recorded)", planner.get("feasible") is True,
                   f"converged {planner.get('converged')}, iterations {planner.get('iterations')}")
    else:
        report.add("planner backend", True,
                   f"{planner.get('backend')} per phase {planner.get('backends_by_phase')}; "
                   "a sampled command sequence carries no solver feasibility certificate",
                   required=False)
    progress = acceptance.get("progression", {})
    if "passed" in progress:
        report.add("motion progression (recorded)", progress["passed"] is True,
                   progress.get("detail", "missing"))
    else:
        report.add("landed on four feet (recorded)",
                   progress.get("feet_loaded") == 4 and progress.get("body_contacts") == 0,
                   f"{progress.get('feet_loaded')}/4 feet loaded, "
                   f"{progress.get('body_contacts')} non-foot contacts")
    cfg_path = Path(manifest["robot_config"]["path"])
    current = adapter_mod.sha256(cfg_path) if cfg_path.is_file() else None
    report.add("robot config unchanged", True,
               "same as at solve time" if current == manifest["robot_config"]["sha256"] else
               "robot config changed or missing since solve; the embedded acceptance config was used",
               required=False)
    print(report)

    if video or show:
        from .video import render_document

        video = video or path / "replay.mp4"
        doc = bundle.clip_document(replayed, model, model_bytes, task=clip["task"],
                                   task_schema=clip["task_schema"], kp=clip["parameters"]["kp"],
                                   kd=clip["parameters"]["kd"], action_scale=scale,
                                   success=report.passed, ending=clip["episodes"][0]["ending"])
        phases = {k: tuple(v) for k, v in manifest["phases"].items()}
        render_document(doc, mjb, video, provenance={"phases": phases}, slowmo=slowmo)
        print(f"video (rendered from this fresh replay): {video}")
        if show:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.run([opener, str(video)], check=False)
    return 0 if report.passed else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("solve", help="optimise, execute in MuJoCo, validate, write a bundle")
    s.add_argument("motion", choices=["stand-squat", "walk", "getup-belly-up", "stairs"])
    s.add_argument("--robot", default=str(ROOT / "robots/dhav1.toml"))
    s.add_argument("--out", required=True, help="bundle directory (must not exist)")
    s.add_argument("--feedback", choices=["riccati", "none"], default="riccati")
    s.add_argument("--gait", choices=["trot", "crawl", "pairs", "walk", "pace"], default=None,
                   help="contact pattern; walk defaults to trot, stairs to pairs")
    s.add_argument("--velocity", type=float, default=None,
                   help="m/s; walk defaults to 0.25, stairs to one tread per gait cycle")
    s.add_argument("--seconds", type=float, default=None,
                   help="cruise duration; walk defaults to 6.0, stairs to the climb it needs")
    s.add_argument("--risers", type=int, default=4, help="stairs only, steps to climb")
    s.add_argument("--period", type=float, default=None,
                   help="walk only: gait period in seconds. The RL side schedules it "
                        "as f = clamp(1.64 * v^0.4, 1.0, 3.0) (Alexander & Jayes stride "
                        "law at h = 0.40 m); pass 1/f to plan on the same clock.")
    s.add_argument("--horizon", type=float, default=None,
                   help="walk only: MPC horizon in seconds (default 0.6). It must "
                        "span at least one full gait cycle or the optimiser cannot "
                        "see a complete stride -- at the 0.66-0.70 s periods the "
                        "high-speed band needs, the default is SHORTER than the "
                        "cycle it is planning.")
    s.add_argument("--max_iter", type=int, default=None,
                   help="walk only: BoxFDDP iteration cap per re-plan (default 60).")
    s.add_argument("--velocity_y", type=float, default=0.0,
                   help="Commanded LATERAL velocity, m/s (positive is the robot's left).")
    s.add_argument("--yaw_rate", type=float, default=0.0,
                   help="Commanded turn rate about world z, rad/s (positive is counter-clockwise).")
    s.add_argument("--accel_seconds", type=float, default=None,
                   help="walk only: ramp the commanded velocity from 0 to target over "
                        "this many seconds at the start, mirroring the ramp into the "
                        "stop. Without it the full speed is step-commanded to a "
                        "standing robot and the CoM target runs away from the feet.")
    s.add_argument("--duty", type=float, default=None,
                   help="walk only: stance fraction of the gait cycle (trot default "
                        "0.6). Lowering it lengthens the swing window, so the foot "
                        "covers the same stride more slowly -- the lever that keeps "
                        "joint speeds inside the DC motor envelope at high stride.")
    s.add_argument("--max_lead", type=float, default=None,
                   help="walk only: how far (m) the CoM target may lead the stance "
                        "centroid. The default 0.08 exists to stop the body leaning "
                        "into a riser, but on flat ground it CAPS THE STRIDE: the "
                        "body cannot advance further per cycle than it may lead, so "
                        "the achieved speed falls short of the command.")
    s.add_argument("--swing_height", type=float, default=None,
                   help="walk only: swing apex in metres (the RL side uses 0.10).")
    s.add_argument("--start_x", type=float, default=None,
                   help="walk only: translate the settled stand to this x before "
                        "walking. The stair plant is flat from -12.00 to +0.75, so a "
                        "negative value buys metres of level run-up.")
    s.add_argument("--rolling", action="store_true",
                   help="walk only, wheel feet: stance wheels roll along their heading "
                        "(trajgen/rolling.py) and are driven at the plan's rolling speed; "
                        "stepping takes only the motion across the wheels")
    s.add_argument("--dog-style", dest="dog_style", default=None, metavar="MODEL",
                   help="walk only: plan to the dog gait model MODEL (data/dog/dog-gait-model.json, "
                        "tools/dog/dog_gait_model.py): its contact schedule, swing paths, footholds and trunk waveform "
                        "(trajgen/dogstyle.py); the gait is chosen by speed unless --dog-gait")
    s.add_argument("--dog-gait", dest="dog_gait", default=None,
                   help="walk --dog-style only: the model's gait family (walk, pace, trot, gallop-R, gallop-L)")
    s.add_argument("--dog-parts", dest="dog_parts", default=None,
                   help="walk --dog-style only: comma list of timing,swing,foothold,trunk,vfilter (default all)")
    s.add_argument("--dog-duty-ramp", dest="dog_duty_ramp", type=float, default=None, metavar="D0",
                   help="walk --dog-style only: stance duty at standstill, blended to the dog's duties at "
                        "cruise with the commanded speed fraction (a gallop from a stand needs it)")
    s.add_argument("--dog-bounds-weight", dest="dog_bounds_weight", type=float, default=None,
                   help="walk --dog-style only: weight on the joint-limit barrier (walk default 1e3, at which a "
                        "dog turn's outer knee overshot its straight-leg limit by 0.03 rad)")
    s.add_argument("--trunk_acceleration_weight", type=float, default=None,
                   help="walk only: price on the planned base heave/roll/pitch/yaw acceleration "
                        "(default 0: off)")
    s.add_argument("--dog-support-sway", dest="dog_support_sway", type=float, default=0.0, metavar="G",
                   help="walk --dog-style only: CoM sway toward the support side, G x the first stride "
                        "harmonic of the support centre's lateral position, with footholds at the stand's "
                        "track (default 0: off)")
    s.add_argument("--dog-trunk-scale", dest="dog_trunk_scale", type=float, default=1.0,
                   help="walk --dog-style only: scale on the dog trunk waveform amplitude")
    s.add_argument("--swing_flat", type=float, default=None, metavar="F",
                   help="walk only (running): hold the swing foot at its peak horizontal speed over this "
                        "fraction of the swing instead of the cycloid's single peak at twice the mean "
                        "(0 = cycloid). Lowers the thigh speed the swing asks for at a run.")
    for flag, text in (("front_inward", "front footholds this much closer to the midline"),
                       ("rear_outward", "rear footholds this much further out"),
                       ("front_bow", "front swing bows in by this much between liftoff and touchdown"),
                       ("rear_bow", "rear swing bows out by this much between liftoff and touchdown")):
        s.add_argument(f"--{flag}", type=float, default=None, metavar="M",
                       help=f"walk only (running lateral foot paths, gait.Scheduler): {text}; metres at the "
                            "cruise command, scaled down with the commanded speed below it")
    for end in ("front", "rear"):
        s.add_argument(f"--{end}_bow_peak", type=float, default=None, metavar="S",
                       help=f"walk --{end}_bow only: fraction of the {end} swing's horizontal advance at which "
                            "the bow peaks (default 0.5)")
    s.add_argument("--stop_seconds", type=float, default=None,
                   help="walk only: length of the ramp from cruise to a standstill before the "
                        "closing hold (default 2.0). A slow gait (a 1 s dog walk) re-places each "
                        "foot only twice in 2 s and can leave the closing stance off the stand pose.")
    s.add_argument("--rolling_weight", type=float, default=None,
                   help="walk --rolling only: weight on each stance wheel rolling at the commanded "
                        "rigid motion's contact speed (default 1e5)")
    d = sub.add_parser("dataset", help="index accepted bundles as a reference dataset")
    d.add_argument("bundles", nargs="+", help="accepted bundle directories")
    d.add_argument("--out", required=True, help="dataset directory to write")
    d.add_argument("--name", default="", help="dataset name (defaults to the directory name)")

    r = sub.add_parser("replay", help="fresh-process forward replay + gates (+ video)")
    r.add_argument("bundle")
    r.add_argument("--video", type=Path)
    r.add_argument("--show", action="store_true", help="render and open the replay video")
    r.add_argument("--slowmo", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.command == "dataset":
        from . import dataset as dataset_mod
        index = dataset_mod.export([Path(b) for b in args.bundles], Path(args.out), args.name)
        print(f"dataset: {len(index['clips'])} clips -> {args.out}")
        for c in index["clips"]:
            cmd = c["command"]
            print(f"  {c['name']:14} cmd=({cmd['lin_vel_x']:+.2f}, {cmd['lin_vel_y']:+.2f}, "
                  f"{cmd['ang_vel_z']:+.2f})  {c['frames']} frames")
        return 0
    if args.command == "solve":
        cfg = config.load(args.robot)
        if args.motion == "walk":
            return solve_walk(cfg, Path(args.out),
                              pattern=args.gait or "trot",
                              velocity=0.25 if args.velocity is None else args.velocity,
                              seconds=6.0 if args.seconds is None else args.seconds,
                              feedback=args.feedback, start_x=args.start_x,
                              period=args.period, swing_height=args.swing_height,
                              max_lead=args.max_lead, duty=args.duty,
                              velocity_y=args.velocity_y, yaw_rate=args.yaw_rate,
                              accel_seconds=args.accel_seconds,
                              horizon=args.horizon, max_iter=args.max_iter,
                              rolling=args.rolling, rolling_weight=args.rolling_weight,
                              **({} if args.stop_seconds is None else {"stop_seconds": args.stop_seconds}),
                              **({} if args.swing_flat is None else {"swing_flat": args.swing_flat}),
                              **({} if not _lateral(args) else {"lateral": _lateral(args)}),
                              **({} if args.trunk_acceleration_weight is None else {
                                  "trunk_acceleration_weight": args.trunk_acceleration_weight}),
                              **({} if args.dog_style is None else dict(
                                  dog_style=args.dog_style, dog_gait=args.dog_gait,
                                  dog_parts=args.dog_parts, dog_trunk_scale=args.dog_trunk_scale,
                                  dog_duty_ramp=args.dog_duty_ramp, dog_bounds_weight=args.dog_bounds_weight,
                                  dog_support_sway=args.dog_support_sway)))
        if args.motion == "stairs":
            return solve_stairs(cfg, Path(args.out), risers=args.risers,
                                pattern=args.gait or "pairs",
                                velocity=args.velocity, seconds=args.seconds,
                                feedback=args.feedback)
        if args.motion == "getup-belly-up":
            return solve_getup(cfg, Path(args.out), feedback=args.feedback)
        return solve_stand(cfg, Path(args.out), args.feedback)
    return replay(Path(args.bundle), args.video, args.show, args.slowmo)


if __name__ == "__main__":
    raise SystemExit(main())

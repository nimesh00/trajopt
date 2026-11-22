"""Render a generated clip to video.

Ideas borrowed from the MATLAB reference project (`cube_animate.m` in
NaCl-1374/Trajectory-Optimization-for-Legged-Robots): contact-versus-swing foot
distinction, foot force arrows, a base velocity vector, a trajectory trace, a
text panel carrying time and pose, a configurable view angle, and an animated
file as the deliverable. Nothing is copied -- that project is MATLAB and carries
no licence -- only the choice of what is worth drawing.

MuJoCo draws contact points and contact force arrows natively, so those are
scene flags here rather than hand-built primitives, which also means they show
true solver forces instead of a separately computed estimate.

Actuator load is not stored in a clip, but the servo law is, so torque is
recovered from the clip's own fields:  tau = kp * (u - q) - kd * qdot.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from .clips import fingerprint

# Foot order matches the trainer's contact columns.
FOOT_ORDER = ("FL", "FR", "RL", "RR")


def load_clip(path: str | Path) -> dict:
    clip = json.loads(Path(path).read_text())
    if clip.get("kind") != "reference-clip":
        raise ValueError(f"{path} is not a reference clip")
    return clip


def check_model(model_path: str | Path, clip: dict) -> None:
    """Refuse to render a clip against a different plant than it was made on."""
    actual = fingerprint(Path(model_path).read_bytes())
    if actual != clip.get("model_id"):
        raise ValueError(
            f"model mismatch: {model_path} fingerprints {actual}, "
            f"clip was generated on {clip.get('model_id')}. Rendering would show "
            "a different robot than the one the trajectory was solved for."
        )


def _torque(clip: dict, frame: dict, qpos_adr: np.ndarray, dof_adr: np.ndarray,
            transmission=None) -> np.ndarray:
    """Recover joint torque from the position-servo law (unclamped demand).

    On a geared/belt plant the law is the coupled joint-space one,
    Kp (q* - q) - Kd qd with q* decoded from ctrl (trajgen/plantmap.py)."""
    kp = clip["parameters"]["kp"]
    kd = clip["parameters"]["kd"]
    q = np.asarray(frame["qpos"], float)[qpos_adr]
    qd = np.asarray(frame["qvel"], float)[dof_adr]
    if transmission is not None and not transmission.identity:
        q_cmd = transmission.joint_command(frame["ctrl"])
        return transmission.stiffness(kp) @ (q_cmd - q) - transmission.stiffness(kd) @ qd
    return kp * (np.asarray(frame["ctrl"], float) - q) - kd * qd


def _phase_of(step: int, phases: dict | None) -> str:
    if not phases:
        return ""
    for label, (lo, hi) in phases.items():
        if lo <= step - 1 <= hi:
            return label
    return ""


def _overlay(frame_rgb: np.ndarray, lines: list[str], progress: float) -> np.ndarray:
    """Draw the telemetry panel and a progress bar. Returns the frame unchanged
    if Pillow is unavailable, so video export never hard-depends on it."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame_rgb

    image = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:  # older Pillow: no size argument
        font = ImageFont.load_default()

    pad, line_h = 8, 18
    box_h = pad * 2 + line_h * len(lines)
    box_w = max(
        (int(draw.textlength(text, font=font)) for text in lines), default=120
    ) + pad * 2
    draw.rectangle([(0, 0), (box_w, box_h)], fill=(0, 0, 0, 140))
    for i, text in enumerate(lines):
        draw.text((pad, pad + i * line_h), text, fill=(255, 255, 255, 235), font=font)

    width, height = image.size
    bar_h = 5
    draw.rectangle([(0, height - bar_h), (width, height)], fill=(0, 0, 0, 140))
    draw.rectangle(
        [(0, height - bar_h), (int(width * progress), height)], fill=(90, 220, 170, 235)
    )
    return np.asarray(image)


def render_clip(
    clip_path: str | Path,
    model_path: str | Path,
    output: str | Path,
    *,
    provenance: dict | None = None,
    width: int = 960,
    height: int = 720,
    fps: float | None = None,
    slowmo: float = 1.0,
    azimuth: float = 135.0,
    elevation: float = -15.0,
    distance: float = 1.8,
    track: bool = True,
    auto_cam: bool = False,
    contacts: bool = True,
    trace: bool = True,
    overlay: bool = True,
    trace_length: int = 120,
) -> Path:
    """Render every control step of a clip file to an mp4 (or .gif) file."""
    return render_document(load_clip(clip_path), model_path, output, provenance=provenance,
                           width=width, height=height, fps=fps, slowmo=slowmo, azimuth=azimuth,
                           elevation=elevation, distance=distance, track=track, auto_cam=auto_cam,
                           contacts=contacts, trace=trace, overlay=overlay, trace_length=trace_length)


def render_document(
    clip: dict,
    model_path: str | Path,
    output: str | Path,
    *,
    provenance: dict | None = None,
    width: int = 960,
    height: int = 720,
    fps: float | None = None,
    slowmo: float = 1.0,
    azimuth: float = 135.0,
    elevation: float = -15.0,
    distance: float = 1.8,
    track: bool = True,
    auto_cam: bool = False,
    contacts: bool = True,
    trace: bool = True,
    overlay: bool = True,
    trace_length: int = 120,
) -> Path:
    """Render an in-memory clip document (e.g. a fresh forward replay)."""
    if clip.get("kind") != "reference-clip":
        raise ValueError("not a reference clip")
    check_model(model_path, clip)

    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)

    # libx264 needs even dimensions.
    width, height = (width // 2) * 2, (height // 2) * 2

    # The offscreen framebuffer defaults to 640x480 and the Renderer refuses
    # anything larger. The usual fix is <visual><global offwidth=...> in the
    # XML, but we render from a compiled MJB, so raise it on the model instead.
    # Clamping the resolution down would silently produce a smaller video than
    # was asked for.
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)

    from . import plantmap

    transmission = plantmap.transmission(model)
    joint_of = [int(j) for j in transmission.joints]   # a belt's trnid is a tendon id
    qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_of])
    dof_adr = np.array([model.jnt_dofadr[j] for j in joint_of])
    limit = (np.array([float(model.actuator_forcerange[a][1]) for a in range(model.nu)])
             if transmission.identity else transmission.limit)

    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    if contacts:
        # The reference project drew foot forces as quivers by hand; MuJoCo
        # renders the solver's own contact forces, which is strictly better.
        option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if body < 0:
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
    if track and body >= 0:
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = body

    # Motion-following chase camera (trajgen/camera.py): frame
    # the limbs while the robot is down, pull back as it comes up.
    chase = None
    if auto_cam and body >= 0:
        from .camera import MotionChaseCam

        chase = MotionChaseCam(model, body)

    episode = clip["episodes"][0]
    frames = episode["frames"]
    control_dt = clip["control_dt"]
    if fps is None:
        fps = max(1.0, (1.0 / control_dt) / max(slowmo, 1e-6))

    phases = (provenance or {}).get("phases")
    renderer = mujoco.Renderer(model, height=height, width=width)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio

    writer_args = {"fps": fps}
    if path.suffix.lower() != ".gif":
        writer_args.update(codec="libx264", macro_block_size=None, quality=8)
    writer = imageio.get_writer(str(path), **writer_args)

    history: list[np.ndarray] = []
    try:
        for step, frame in enumerate(frames):
            data.qpos[:] = np.asarray(frame["qpos"], float)
            data.qvel[:] = np.asarray(frame["qvel"], float)
            data.ctrl[:] = np.asarray(frame["ctrl"], float)
            mujoco.mj_forward(model, data)

            active = chase.update(data, step * control_dt) if chase else camera
            renderer.update_scene(data, camera=active, scene_option=option)
            scene = renderer.scene

            base_pos = np.array(data.qpos[:3], float)
            history.append(base_pos.copy())
            if trace and hasattr(mujoco, "mjv_connector"):
                # Trajectory trace, as the reference project plotted with plot3.
                recent = history[-trace_length:]
                for i in range(1, len(recent)):
                    if scene.ngeom >= scene.maxgeom:
                        break
                    geom = scene.geoms[scene.ngeom]
                    fade = i / max(1, len(recent) - 1)
                    mujoco.mjv_initGeom(
                        geom, int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                        np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.array([0.35, 0.85, 0.70, 0.25 + 0.55 * fade], np.float32),
                    )
                    mujoco.mjv_connector(
                        geom, int(mujoco.mjtGeom.mjGEOM_CAPSULE), 0.004,
                        recent[i - 1], recent[i],
                    )
                    scene.ngeom += 1

            velocity = np.array(data.qvel[:3], float)
            if trace and np.linalg.norm(velocity) > 1e-3 and scene.ngeom < scene.maxgeom:
                if hasattr(mujoco, "mjv_connector"):
                    geom = scene.geoms[scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom, int(mujoco.mjtGeom.mjGEOM_ARROW),
                        np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.array([1.0, 0.75, 0.2, 0.9], np.float32),
                    )
                    mujoco.mjv_connector(
                        geom, int(mujoco.mjtGeom.mjGEOM_ARROW), 0.008,
                        base_pos, base_pos + 0.25 * velocity,
                    )
                    scene.ngeom += 1

            pixels = renderer.render()

            if overlay:
                quat = np.asarray(frame["qpos"], float)[3:7]
                upright = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
                tilt = np.degrees(np.arccos(np.clip(upright, -1.0, 1.0)))
                feet = frame.get("feet") or []
                labels = " ".join(
                    f"{n}{'+' if feet and i < len(feet) and feet[i] else '-'}"
                    for i, n in enumerate(FOOT_ORDER)
                )
                body_contact = bool(feet and len(feet) > 4 and feet[4])
                tau = _torque(clip, frame, qpos_adr, dof_adr, transmission)
                phase = _phase_of(step, phases)
                lines = [
                    f"{clip['task']}   t = {step * control_dt:5.2f} s   "
                    f"step {step}/{len(frames) - 1}" + (f"   [{phase}]" if phase else ""),
                    f"height {base_pos[2]:.3f} m    tilt {tilt:5.1f} deg",
                    f"feet {labels}" + ("   BODY CONTACT" if body_contact else ""),
                    f"peak |tau| {np.max(np.abs(tau)):5.1f} N-m  of {limit.max():.0f}",
                ]
                if slowmo != 1.0:
                    lines.append(f"{slowmo:g}x slow motion")
                pixels = _overlay(pixels, lines, (step + 1) / len(frames))

            writer.append_data(pixels)
    finally:
        writer.close()
        renderer.close()
    return path

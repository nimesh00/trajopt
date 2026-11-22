"""Vertical jump with a stable landing.

The contact schedule is FIXED and known in advance, so no search is involved:

    stance -> crouch -> launch -> flight (no contact) -> touchdown impulse
           -> absorb -> hold

Everything else is the standing pipeline: settled start, Crocoddyl solve,
servo bridge, instrumented MuJoCo execution, fail-closed gates, bundle, replay.

Acceptance adds two motion-specific measurements to the standing hold: a real
flight phase (no foot carrying load for a continuous interval) and a CoM rise
above the standing reference. Both are measured from the executed rollout at
physics rate, never from the plan.
"""
from __future__ import annotations

import mujoco
import numpy as np
import pinocchio as pin

from .. import croc
from ..adapter import LEGS, Adapter
from ..config import RobotConfig
from ..gates import StandingSpec
from .stand import settled_stand  # noqa: F401  (re-exported: the jump starts from a settled stand)

NAME = "jump"
STATUS = "fixed contact schedule"
FLIGHT_SECONDS_MIN = 0.10
COM_RISE_MIN = 0.15


def schedule(ad: Adapter, cfg: RobotConfig, start, *, crouch: float = 0.08, drive: float = 0.22,
             seconds=(0.3, 0.3, 0.2, 0.30, 0.4), hold_seconds: float = 1.2):
    """(x0, knots, terminal, info) for stance/crouch/launch/flight/absorb.

    The final hold is NOT part of the optimal control problem. Touchdown never
    happens at exactly the planned instant, so a time-indexed reference is out
    of phase precisely when the robot needs to settle; the hold is executed as
    the static standing command instead -- the same command the stand fixture
    settles on, whose behaviour is already validated.
    """
    dt = cfg.control_dt
    x0 = ad.x_from_mj(start.qpos, start.qvel)
    posture = x0.copy()
    posture[ad.pin.nq:] = 0.0
    data = ad.pin.createData()
    com0 = pin.centerOfMass(ad.pin, data, x0[:ad.pin.nq])
    pin.framesForwardKinematics(ad.pin, data, x0[:ad.pin.nq])
    feet0 = {leg: np.array(data.oMf[ad.feet[leg].frame].translation) for leg in LEGS}

    n = lambda s: max(1, int(round(s / dt)))  # noqa: E731
    stance_n, crouch_n, launch_n, flight_n, absorb_n = (n(s) for s in seconds)
    hold_n = n(hold_seconds)
    smooth = lambda a, b, k: a + (b - a) * (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, k + 1)[1:]))  # noqa: E731

    knots: list[croc.Knot] = []
    add = lambda phase, dz, **kw: knots.extend(  # noqa: E731
        croc.Knot(support=kw.pop("support", LEGS), com=com0 + [0, 0, z], posture=posture,
                  phase=phase, **kw) for z in np.atleast_1d(dz))
    add("stance", np.zeros(stance_n))
    add("crouch", smooth(0.0, -crouch, crouch_n))
    add("launch", smooth(-crouch, drive, launch_n))
    # Flight: no contact and NO CoM cost. In flight the CoM is ballistic --
    # commanding it fights the physics, and with a CoM target here the solver
    # traded it against foot placement and jumped 159 mm FORWARD instead of up.
    # What is worth asking for in flight is where the feet land.
    for _ in range(flight_n):
        knots.append(croc.Knot(support=(), com=None, posture=posture, phase="flight", swing=feet0))
    knots.append(croc.Knot(support=LEGS, com=com0 + [0, 0, -crouch], posture=posture,
                           phase="absorb", impulse=LEGS))
    add("absorb", smooth(-crouch, 0.0, absorb_n - 1))
    terminal = croc.Knot(support=LEGS, com=com0, posture=posture, phase="absorb")
    info = {"crouch_m": crouch, "drive_m": drive, "seconds": list(seconds),
            "hold_seconds": hold_seconds, "hold_steps": hold_n, "com0": com0.tolist(),
            "phases": [("stance", stance_n), ("crouch", crouch_n), ("launch", launch_n),
                       ("flight", flight_n), ("absorb", absorb_n), ("hold", hold_n)]}
    return x0, knots, terminal, info


def hold_commands(last_command, stand_command, steps: int, blend: int, rate: float) -> np.ndarray:
    """Commands for the post-landing hold: ease onto the standing command.

    Snapping to the standing command at the command-rate limit kicks the legs
    and the knee rebounds past the calf motor's no-load speed (measured: one
    knee 35 % past v_nl, 34 ms outside the torque-speed
    envelope). Easing over `blend` steps removes that kick.
    """
    last = np.asarray(last_command, float)
    target = np.asarray(stand_command, float)
    blend = max(1, min(blend, steps))
    weights = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, blend + 1)[1:])
    ramp = last + (target - last) * weights[:, None]
    commands = np.vstack([ramp, np.tile(target, (steps - blend, 1))])
    out, previous = [], last
    for command in commands:            # still respect the command-rate limit
        previous = np.clip(command, previous - rate, previous + rate)
        out.append(previous.copy())
    return np.array(out)


def acceptance(model: mujoco.MjModel, start) -> StandingSpec:
    qadr = [model.jnt_qposadr[model.actuator_trnid[a][0]] for a in range(model.nu)]
    return StandingSpec(hold_seconds=1.0, height_ref=float(start.qpos[2]),
                        joints_ref=start.qpos[qadr].copy(), body_contact_phases=())


def progression(model: mujoco.MjModel, ex, start, loaded_force: float = 5.0) -> dict:
    """Measured flight time and CoM rise, from the executed rollout."""
    normals = np.concatenate([ex.samples["foot_normal"], ex.terminal["foot_normal"]])
    airborne = np.all(normals <= loaded_force, axis=1)
    longest = best = 0
    for value in airborne:
        best = best + 1 if value else 0
        longest = max(longest, best)
    flight = longest * ex.physics_timestep

    data = mujoco.MjData(model)
    com_z = []
    for q in ex.qpos:
        data.qpos[:] = q
        mujoco.mj_comPos(model, data) if False else mujoco.mj_forward(model, data)
        com_z.append(float(data.subtree_com[0][2]))
    com_z = np.array(com_z)
    rise = float(com_z.max() - com_z[0])
    passed = bool(flight >= FLIGHT_SECONDS_MIN and rise >= COM_RISE_MIN)
    return {"passed": passed, "flight_seconds": flight, "com_rise_m": rise,
            "com_start_m": float(com_z[0]), "com_peak_m": float(com_z.max()),
            "peak_base_height_m": float(ex.qpos[:, 2].max()),
            "detail": f"flight {flight * 1e3:.0f} ms (need >= {FLIGHT_SECONDS_MIN * 1e3:.0f}), "
                      f"CoM rise {rise * 1e3:.0f} mm (need >= {COM_RISE_MIN * 1e3:.0f})"}

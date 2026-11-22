"""Forward walking with a fixed contact schedule.

The schedule is fixed and known -- a trot or a crawl -- so there is no search.
What walking does need is a plan that can start at ANY point in the gait, so
the robot re-plans from where it actually is rather than from where the last
plan said it would be. `trajgen/gait.py` provides that phase clock; this module
is the control loop around it:

    measure state -> build a horizon from the current phase -> solve (BoxFDDP)
    -> execute the first slice through the servo bridge -> advance the clock

Each re-plan is solved from the MEASURED state, and the executed commands
concatenate into one continuous rollout from one snapshot, so the result
replays and validates like any other motion in this pipeline.

What this replaces: Crocoddyl's quadruped helper builds a problem that always
starts at the beginning of a cycle, so re-planning mid-cycle restarts the gait
(measured: stable but -0.026 m of travel in 6 s) while re-planning once per
cycle lets error grow for a whole cycle (measured: sank 0.40 -> 0.25 m and fell
by the fifth cycle).
"""
from __future__ import annotations

import dataclasses
import time

import mujoco
import numpy as np

from .. import croc, execution, plantmap, servo
from ..adapter import Adapter
from ..config import RobotConfig
from ..gait import PATTERNS, Scheduler
from ..gates import StandingSpec

LEGS_ORDER = ("FL", "FR", "RL", "RR")    # croc.Solution.forces leg order (croc.LEGS)

NAME = "walk"
STATUS = "fixed contact schedule"
DISTANCE_MIN = 1.0
LATERAL = ("front_inward", "rear_outward", "front_bow", "front_bow_peak", "rear_bow", "rear_bow_peak")
SECONDS_MIN = 5.0


def weights() -> croc.Weights:
    """Walking costs.

    The tangential-force penalty that keeps a STANDING robot from creeping is
    wrong here: shear force is what propels a walk, so it is only lightly
    penalised. The swing cost has to win against posture regularisation, or the
    feet never reach their footholds.

    touch_position is the lever on landing quality and it is bounded ABOVE by
    the motor envelope, not by the landing itself: holding the foot harder to
    its foothold through the impulse buys penetration and slip, but the torque
    it takes to do so runs into the DC torque-speed curve. Measured at 0.46 m/s,
    exceedance against that curve is 0.000 N m at 2e5, 0.074 at 3e5, 0.879 at
    5e5 and 1.149 at 1e6 -- so 2e5 is the most the envelope allows, and it is
    what the gates pass at. Against the free-impulse baseline it takes loaded
    penetration p99 from 8.57 to 6.26 mm, slip p95 from 0.0260 to 0.0148 and
    touchdown approach speed p95 from 0.433 to 0.280 m/s.

    control is an EFFORT regulariser whose units changed with the control space.
    In torque it penalised |tau|, order 20 N m, so 1e-1 was a real cost; against
    a position command it penalises (u - u_stand), order 0.5 rad, whose square is
    ~1600x smaller, and the same 1e-1 left large command excursions almost free.
    Measured at 0.46 m/s, raising it recovers the landing: peak 1.17 xBW at 1e-1,
    1.09 at 3e2, 0.73 at 3e3, and it breaks up again by 1e4 (slip 0.065, contact
    onsets 118). 3e3 is the floor of that basin.

    com_height holds the RIDE HEIGHT. The CoM residual's three axes want opposite
    things -- fore-aft has to stay soft because it is what the gait pushes
    against, while height is a reference to be held -- and with one scalar weight
    the only way to tighten height was to tighten the stride with it. Measured at
    0.48 m/s, base height swung 38.5 mm peak-to-peak; at 3 it swings 21.6 mm.
    Past that the trade turns: 10 gives 18.7 mm but takes slip p95 from .0136 to
    .0555 and contact onsets from 53 to 79, and 30 breaks the torque-speed gate.
    A taller swing arc pays for most of that -- at 3 with a 0.035 m arc, slip
    (.0130) and onsets (48) are slightly BETTER than the untightened baseline,
    not worse, so the two are tuned together rather than in sequence.

    touch_velocity is nearly inert by comparison -- 1e3 through 2e5 differ by
    less than the run-to-run spread -- because pinning WHERE the foot arrives
    already bounds how fast it gets there. It is kept as a separate term because
    on terrain the position target is the speculative half.
    """
    return croc.Weights(com=1e6, swing=1e6, tangential=1.0, state=1e1,
                        bounds=1e3, terminal_state=1e3, joint_posture=0.01,
                        touch_position=2e5, touch_velocity=1e3, control=3e3,
                        com_height=3.0)


def walk(model: mujoco.MjModel, ad: Adapter, cfg: RobotConfig, start, bridge, *,
         pattern: str = "trot", velocity: float = 0.25, velocity_y: float = 0.0,
         yaw_rate: float = 0.0, yaw_sigma: float = 0.01, seconds: float = 8.0,
         stop_seconds: float = 2.0, hold_seconds: float = 2.0, horizon_seconds: float = 0.6,
         replan_seconds: float = 0.1, max_iter: int = 60, terrain=None,
         snap_footholds: bool = False, period: float | None = None,
         swing_height: float | None = None, clearance_scale: float = 1.0,
         max_lead: float | None = None, costs: croc.Weights | None = None,
         scheduler_overrides: dict | None = None, duty: float | None = None,
         accel_seconds: float | None = None, accel_limit: float = 0.6,
         rolling: bool = False, rolling_weight: float = 1e5, rolling_gap_kp: float = 100.0,
         dog=None, swing_flat: float = 0.0, lateral: dict | None = None):
    """Walk forward, then decelerate to a standstill. Returns (commands, phases, info).

    The stop matters: a robot halted mid-gait has its feet wherever the gait
    left them, which is not a stance it can hold (measured: 0.25 rad of joint
    error against the standing pose, and still moving at 0.33-0.57 rad/s).
    Ramping the commanded velocity to zero lets the foothold planner walk the
    feet back under the hips, and the closing hold is the same loop with an
    all-feet-down pattern -- so the whole motion, walk and stand, comes out of
    one scheduler and one solver.

    `rolling` (wheel-legged hybrid, wheel feet only): stance wheels ROLL. The planner constrains
    each stance wheel across its axle and vertically only (trajgen/rolling.py), footholds
    keep only the lead across the wheel, and each stance wheel's velocity target is the
    plan's rolling speed of its contact point over the wheel radius (plus the feedforward that
    makes it roll without traction); swing wheels are pre-spun to their next stance's rolling
    speed, and every wheel in the closing hold is braked (target 0). Off, nothing here changes.

    `dog` (trajgen/dogstyle.py DogTargets): plan to the dog gait model's targets - its per-leg
    contact schedule replaces `pattern`/`period`/`duty`, and the scheduler follows its swing paths,
    mid-stance footholds and trunk waveform. None (the default) changes nothing.

    `swing_flat` (running): plateau the swing foot's horizontal speed over this fraction of the
    swing (gait.Scheduler.swing_flat); 0 keeps the cycloid.

    `lateral` (running): lateral foot paths, {front_inward, rear_outward, front_bow, rear_bow} in
    metres at the cruise command and {front_bow_peak, rear_bow_peak} as fractions of the swing
    (gait.Scheduler); None or empty keeps the stand's track.
    """
    dt = cfg.control_dt
    # RAMP AT A FIXED ACCELERATION, not over a fixed time. Stepping the command
    # to full speed on the first control interval asks the servos for a torque
    # the motor does not have at that joint speed: measured at 0.60 m/s, the
    # executed motion broke the DC torque-speed envelope by 9.4 N m on the front
    # calf, on 56 samples out of 60000, ALL of them inside the first 0.4 s.
    # A fixed ramp TIME gets the other end wrong -- 1.0 s of ramp at a 0.12 m/s
    # command costs enough distance to fail the travelled gate -- so the ramp is
    # velocity / accel_limit and every command gets the same acceleration.
    if accel_seconds is None:
        # Ramp on the fastest thing commanded. A pure turn or a pure sidestep
        # has velocity 0 and still has to be ramped, or the first control
        # interval steps the whole command onto a standing robot.
        peak = max(abs(velocity), abs(velocity_y), abs(yaw_rate) * 0.4)
        accel_seconds = peak / max(accel_limit, 1e-6)
    horizon = max(2, int(round(horizon_seconds / dt)))
    slice_steps = max(1, int(round(replan_seconds / dt)))
    walk_steps = int(round(seconds / dt))
    stop_steps = int(round(stop_seconds / dt))
    hold_steps = int(round(hold_seconds / dt))
    total_steps = walk_steps + stop_steps + hold_steps

    gait_pattern = PATTERNS[pattern] if dog is None else None
    # ROBOT-SIZED VALUES. The leg reach comes from the robot config ([gait].leg_reach);
    # PATTERNS carry generic trot / walk timings that [gait] may restate. Explicit
    # arguments still win.
    robot_gait = cfg.gait or {}
    if "leg_reach" not in robot_gait:
        raise ValueError(f"{cfg.path}: walking needs [gait].leg_reach (hip-to-foot extension, m)")
    period = robot_gait.get("period") if period is None else period
    swing_height = robot_gait.get("swing_height") if swing_height is None else swing_height
    duty = robot_gait.get("duty") if duty is None else duty
    if dog is None and (period is not None or swing_height is not None or duty is not None):
        # DUTY is the lever at speed. It is the fraction of the cycle each leg
        # spends in stance, so 1 - duty is the swing window: LOWERING it gives
        # the foot MORE time to cover the same stride, which is what keeps the
        # joint speeds inside the DC motor envelope as the stride grows. The
        # 0.6 default is a walking trot; at 1 m/s this robot is at Froude 0.5,
        # the walk-trot transition, where a real quadruped is already reducing
        # duty toward a brief flight phase.
        gait_pattern = dataclasses.replace(
            gait_pattern, period=period or gait_pattern.period,
            swing_height=swing_height or gait_pattern.swing_height,
            duty=duty if duty is not None else gait_pattern.duty)
    if dog is not None:
        if period is not None or duty is not None:
            raise ValueError("walk: a dog gait model sets the period and duties; drop period/duty")
        gait_pattern = dog.pattern(swing_height)
    scheduler = Scheduler.from_stand(ad, gait_pattern, start.qpos,
                                     velocity=(velocity, velocity_y), terrain=terrain,
                                     snap_footholds=snap_footholds)
    if dog is not None:
        from ..dogstyle import scheduler_from
        scheduler = scheduler_from(scheduler, dog, (velocity, velocity_y, yaw_rate))
    scheduler.yaw_rate = yaw_rate
    scheduler.leg_reach = robot_gait["leg_reach"]
    scheduler.clearance *= clearance_scale
    for key, value in (scheduler_overrides or {}).items():
        if not hasattr(scheduler, key):
            raise ValueError(f"no scheduler field {key!r}")
        setattr(scheduler, key, value)
    if max_lead is not None:
        scheduler.max_lead = max_lead
    scheduler.swing_flat = float(swing_flat)
    if lateral:
        for key, value in lateral.items():
            if key not in LATERAL:
                raise ValueError(f"walk: unknown lateral parameter {key!r}")
            setattr(scheduler, key, float(value))
        scheduler.lateral_speed = float(np.hypot(velocity, velocity_y))
    if rolling:
        import pinocchio as pin

        from ..rolling import add_wheel_bottom_frames, rolling_direction
        scheduler.rolling = True
        # The rolling contact acts at a calf-fixed point at the bottom of each wheel (before
        # the OCP is built: its StateMultibody copies the model).
        bottom = add_wheel_bottom_frames(ad, start.qpos)
        wheel_act, radius = {}, {}
        for leg, contact in ad.feet.items():
            body = int(model.geom_bodyid[contact.mj_geom])
            joint = [j for j in range(model.njnt) if model.jnt_bodyid[j] == body][0]
            held = [int(a) for a in ad.transmission.held if model.actuator_trnid[a][0] == joint]
            if len(held) != 1:
                raise ValueError(f"rolling: {leg}'s foot is not a held wheel")
            wheel_act[leg], radius[leg] = held[0], contact.radius
        pin_data = ad.pin.createData()
        # Hinge-side loads the drive must supply for the wheel to roll WITHOUT traction, which
        # is what the plan assumes (its rolling direction carries no force): spin inertia
        # (wheel Iyy + armature), joint damping, frictionloss and rolling friction (mu_roll x
        # the planned normal force). A velocity servo makes no torque at zero speed error, so
        # these go in as feedforward, u = theta' + tau / kv (Codex review item 3).
        wheel_dof, wheel_inertia, kv, roll_mu = {}, {}, {}, {}
        for leg, a in wheel_act.items():
            j = int(model.actuator_trnid[a][0])
            dof = int(model.jnt_dofadr[j])
            wheel_dof[leg] = dof
            wheel_inertia[leg] = float(model.dof_armature[dof] + model.body_inertia[model.jnt_bodyid[j], 1])
            kv[leg] = float(model.actuator_gainprm[a][0])
            roll_mu[leg] = float(model.geom_friction[ad.feet[leg].mj_geom][2])

        def rolling_speed(leg, x_node, yaw):
            """Wheel hinge speed (rad/s) that rolls the wheel bottom without slip: the
            calf-fixed bottom point's rolling-direction speed over r (it includes the calf's
            own rotation about the axle)."""
            q_, v_ = x_node[:ad.pin.nq], x_node[ad.pin.nq:]
            pin.forwardKinematics(ad.pin, pin_data, q_, v_)
            pin.updateFramePlacements(ad.pin, pin_data)
            vel = pin.getFrameVelocity(ad.pin, pin_data, bottom[leg], pin.LOCAL_WORLD_ALIGNED)
            return float(rolling_direction(yaw) @ vel.linear) / radius[leg]

        def wheel_command(leg, x0, x1, yaw, fz):
            """x1 None on a wheel's LAST stance step: the next node's foot is already lifting,
            and averaging with it (or differencing it for spin-up) put a +4 -> -9 rad/s jump on
            the still-loaded wheel once per cycle, 11.5 of 14 N m."""
            w0 = rolling_speed(leg, x0, yaw)
            w1 = w0 if x1 is None else rolling_speed(leg, x1, yaw)
            w = 0.5 * (w0 + w1)
            dof = wheel_dof[leg]
            tau = (wheel_inertia[leg] * (w1 - w0) / dt + model.dof_damping[dof] * w
                   + (model.dof_frictionloss[dof] + roll_mu[leg] * max(fz, 0.0)) * np.tanh(w / 0.1))
            return w + tau / kv[leg]
    posture = ad.x_from_mj(start.qpos, np.zeros(model.nv))
    posture[ad.pin.nq:] = 0.0
    # PLAN IN THE VARIABLE THE PLANT ACCEPTS. The plant's actuators are position
    # servos, so a torque plan has to be inverted per step -- q_cmd = q_ref +
    # tau/kp -- and substituting that inversion back into the servo law gives
    # tau_applied = tau_plan + kp (q_ref - q) + kd (v_ref - v): the robot runs
    # the optimised controller PLUS a joint PD term that was never in the model
    # it was optimised against. ServoOCP puts the servo law inside the dynamics
    # and optimises the command directly, so there is nothing to invert and
    # nothing unmodelled. It also restates croc.OCP's state cost, whose weights
    # are squared twice there.
    # A COMMANDED TURN NEEDS THE TRACKING WEIGHT, NOT THE HOLDING ONE. With no
    # yaw command the heading has one correct value and is held as tightly as
    # roll and pitch. With one, the reference MOVES, and the same stiffness asks
    # for the whole heading change on the first knot: measured at 0.6 rad/s it
    # put 0.36 rad of error on the horizon and the solver answered with 37 rad/s
    # of joint speed and fell over.
    ocp = servo.ServoOCP(croc.OCP(ad, cfg, weights=costs if costs is not None else weights()),
                         start.ctrl,
                         yaw_sigma=(yaw_sigma if yaw_rate else None))
    if rolling:
        ocp.b.w.rolling_velocity = rolling_weight
        ocp.b.rolling_frames = bottom
        ocp.b.rolling_gap_kp = rolling_gap_kp

    state, commands, phases, diagnostics = start.state, [], [], []
    data = mujoco.MjData(model)
    executed = 0
    guess_xs = guess_us = None      # previous horizon, shifted: see ocp.solve
    t0 = time.perf_counter()
    while executed < total_steps:
        # Cruise, ramp to a standstill, then hold with all four feet down.
        if executed < walk_steps:
            # RAMP UP, mirroring the ramp down into the stop. The loop used to
            # step-command the full velocity to a standing robot on the first
            # control interval; the CoM target then runs away from the feet and
            # the solve diverges. Measured: a step command at 1.2 m/s left the
            # robot at -0.04 m/s with the dynamics defect over 1e2, while the
            # same gait reached 0.64 m/s when commanded 1.0. Acceleration is a
            # physical requirement, not a tuning knob.
            label = "walk"
            if accel_seconds > 0.0:
                ramp = min(1.0, (executed * dt) / accel_seconds)
            else:
                ramp = 1.0
            scheduler.velocity = np.array([velocity * ramp, velocity_y * ramp])
            scheduler.yaw_rate = yaw_rate * ramp
        elif executed < walk_steps + stop_steps:
            label = "stop"
            fraction = 1.0 - (executed - walk_steps) / max(stop_steps, 1)
            scheduler.velocity = np.array([velocity * fraction, velocity_y * fraction])
            scheduler.yaw_rate = yaw_rate * fraction
        else:
            if scheduler.pattern is not PATTERNS["stand"]:
                scheduler.pattern = PATTERNS["stand"]
                scheduler.phase = 0.0
            label, scheduler.velocity = "hold", np.zeros(2)
            scheduler.yaw_rate = 0.0
        if rolling:
            ocp.b.rolling = label != "hold"     # the closing hold stands on braked wheels
            ocp.b.rolling_yaw_rate = float(scheduler.yaw_rate)
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        x = ad.x_from_mj(data.qpos, data.qvel)
        # A TURNING WALK'S VELOCITY IS IN THE BODY'S HEADING FRAME. The scheduler takes a
        # world-frame velocity, so a fixed (vx, vy) under a yaw command kept translating
        # along the start direction while the body spun: measured, 0.5 m/s with 1.0 rad/s
        # cruised at -0.02 m/s along its own heading, and 0.3 m/s with 0.3 rad/s moved
        # sideways (vy -0.27). Rotating the command by the measured heading at every re-plan
        # makes it a curved walk. Straight walks keep their world-frame command unchanged.
        if yaw_rate:
            qx, qy, qz, qw = x[3:7]
            heading = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            c, s_ = np.cos(heading), np.sin(heading)
            vb = np.array(scheduler.velocity, float)
            scheduler.velocity = np.array([c * vb[0] - s_ * vb[1], s_ * vb[0] + c * vb[1]])
        scheduler.measured_velocity = np.array(data.qvel[:2], float)  # world frame
        scheduler.measured_yaw_rate = float(data.qvel[5])  # body z, level enough to be world z
        knots = scheduler.knots(x[:ad.pin.nq], posture, horizon, dt)
        # A TURN'S TERMINAL KNOT FACES THE TURNED HEADING. The terminal posture used to be
        # the starting stance for every walk, so under a yaw command each horizon ended
        # pulling the heading back to its start. Once the robot had turned ~pi that pull
        # crossed the SO(3) log map's cut locus and reversed, and every in-place turn fell
        # at +-pi (measured: +-0.5 rad/s fell at 6.6-7.0 s, |yaw| 3.0-3.9 rad). The last
        # knot's posture already carries the commanded heading - also through the stop and
        # hold, where the ramped rate is zero but the robot still faces where it turned to.
        # Straight walks keep the stance, so their plans are unchanged.
        # The same may hold for ATTITUDE on stairs (the knots pitch to the slope, the
        # terminal stays level), but on 2026-10-03 the stairs motion stopped before riser
        # 0 with either terminal, so that change is untested and not made.
        terminal = croc.Knot(support=tuple(knots[-1].support), com=knots[-1].com,
                             posture=knots[-1].posture if yaw_rate else posture,
                             phase=knots[-1].phase)
        # One DC pass: this is a receding horizon, and the executed motion is
        # scored against the motor law by the gates, not by the planner alone.
        was_warm = guess_xs is not None
        solution = ocp.solve(x, knots, terminal, max_iter=max_iter,
                             xs_init=guess_xs, us_init=guess_us)
        steps = solution.control_steps()
        take = min(slice_steps, len(steps), total_steps - executed)

        def controller(k, d, steps=steps, solution=solution, knots=knots, rolls=ocp.b.rolling):
            # u IS the joint position command. The only things left to do are
            # the Riccati correction and the projection onto what the consumer
            # can express -- both in command units, no inversion anywhere.
            node = steps[k][0]
            u = np.asarray(solution.us[node], float)
            if bridge.config.feedback == "riccati":
                dx = ocp.state.diff(np.asarray(solution.xs[node], float),
                                    ad.x_from_mj(d.qpos, d.qvel))
                # The gain is radians of COMMAND per unit state error, so the
                # clamp is in radians too -- but it has to mean the same
                # AUTHORITY the torque-space clamp meant, not the same number.
                # feedback_limit is N m; a command offset buys kp N m per rad,
                # so the equivalent bound is feedback_limit / kp. Clamping at
                # action_scale (0.5 rad) instead is 20 N m of correction against
                # a 5 N m intent, and the executed motion failed torque-speed by
                # 24.8 N m on exactly that.
                limit = bridge.config.feedback_limit / cfg.kp
                u = u + np.clip(-np.asarray(solution.K[node]) @ dx, -limit, limit)
            if not rolls:
                return bridge.project(ocp.mj_from_u(u))
            ctrl, info = bridge.project(ocp.mj_from_u(u))
            knot = knots[k]
            yaw = ocp.b._yaw(knot)
            for leg in knot.support:
                fz = solution.forces[node, LEGS_ORDER.index(leg), 2] if solution.forces is not None else 0.0
                lifting = k + 1 >= len(knots) or leg not in knots[k + 1].support
                target = wheel_command(leg, np.asarray(solution.xs[node], float),
                                       None if lifting else np.asarray(solution.xs[node + 1], float),
                                       yaw, float(fz) if np.isfinite(fz) else 0.0)
                lo, hi = model.actuator_ctrlrange[wheel_act[leg]]
                ctrl[wheel_act[leg]] = float(np.clip(target, lo, hi))
            # SWING WHEELS ARE PRE-SPUN to the rolling speed their next stance asks for (its
            # first knot's rigid-motion reference over r). Braked in the air, a wheel landed at
            # 0 rad/s and had to be spun up within one step: measured, the drive hit 13-14 of
            # its 14 N m at touchdown, a traction impulse the plan (no rolling force) lacks.
            for leg in set(wheel_act) - set(knot.support):
                nxt = next((kn for kn in knots[k:] if leg in kn.support and kn.roll), None)
                if nxt is not None:
                    ref = nxt.roll[leg]
                    lo, hi = model.actuator_ctrlrange[wheel_act[leg]]
                    speed = float(rolling_direction(ocp.b._yaw(nxt)) @ ref) / radius[leg]
                    ctrl[wheel_act[leg]] = float(np.clip(speed, lo, hi))
            return ctrl, info

        ex = execution.Executor(model, cfg.substeps, cfg).run(state, controller, take, dt)
        commands.append(ex.ctrl[1:])
        phases += [label] * take
        # REUSE the state the executor actually reached. This used to call
        # _advance(), which re-simulated the same slice with bare mj_step and no
        # DC envelope, so the next solve was anchored to a state the executed
        # motion never passed through -- measured 0.002735 in qpos and 0.043927
        # in qvel over one five-command slice of a 0.3 rad/s turn.
        state = ex.final_state
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        scheduler.advance(take, dt, ad.q_from_mj(data.qpos))
        # Shift the horizon by the slice that was executed. The node the loop
        # restarts from is the one after the last EXECUTED control interval, so
        # the guess and the clock advance together; node 0 is overwritten with
        # the measured state, which is what the next solve is anchored to.
        # Always carry the previous horizon forward. An MPC that cold-starts
        # whenever the last solve fell short spends its budget rediscovering the
        # same plan and never builds the consistency between consecutive solves
        # that makes the loop track; the answer to a poor guess is a problem the
        # solver can close, not a thrown-away guess.
        drop = (solution.step_nodes[take] if take < len(solution.step_nodes)
                else len(solution.xs) - 1)
        guess_xs = [np.asarray(row) for row in solution.xs[drop:]]
        guess_us = [u for u, label in zip(solution.us[drop:], solution.phases[drop:])
                    if not label.endswith(":impulse")]
        diagnostics.append({"step": executed, "phase": round(scheduler.phase, 3),
                            "converged": bool(solution.diagnostics["converged"]),
                            "iterations": solution.diagnostics["iterations"],
                            "stop": solution.diagnostics["stop"],
                            "dynamic_feasibility": solution.diagnostics["dynamic_feasibility"],
                            "cost": solution.diagnostics["cost"],
                            "warm_started": was_warm,
                            "x": float(ex.qpos[-1, 0]), "z": float(ex.qpos[-1, 2])})
        executed += take

    info = {"pattern": pattern, "period": gait_pattern.period, "duty": gait_pattern.duty,
            **({"rolling": True, "rolling_weight": rolling_weight} if rolling else {}),
            **({"swing_flat": swing_flat} if swing_flat else {}),
            **({"lateral": dict(lateral)} if lateral else {}),
            "swing_height": gait_pattern.swing_height, "velocity": velocity,
            "velocity_y": velocity_y, "yaw_rate": yaw_rate,
            "accel_seconds": accel_seconds, "seconds": seconds,
            "stop_seconds": stop_seconds, "hold_seconds": hold_seconds,
            "accel_seconds": accel_seconds,
            "horizon_seconds": horizon_seconds, "replan_seconds": replan_seconds,
            "solver": "BoxFDDP", "scheduler": "phase-clocked (trajgen/gait.py)",
            "clearance": scheduler.clearance, "max_lead": scheduler.max_lead,
            "attitude_source": scheduler.attitude_source, "attitude_tau": scheduler.attitude_tau,
            "slope_span_x": scheduler.slope_span_x,
            "snap_footholds": snap_footholds, "terrain": terrain is not None,
            "solves": len(diagnostics), "solve_seconds": round(time.perf_counter() - t0, 1),
            "converged": all(d["converged"] for d in diagnostics),
            "converged_fraction": sum(d["converged"] for d in diagnostics) / len(diagnostics),
            "max_dynamic_feasibility": max(d["dynamic_feasibility"] for d in diagnostics),
            "max_iterations_used": max(d["iterations"] for d in diagnostics),
            "per_solve": diagnostics}
    if dog is not None:
        info["dog"] = dog.as_dict()
    return np.vstack(commands), phases, info


def acceptance(model: mujoco.MjModel, start) -> StandingSpec:
    """Standing gates for the closing hold; the walk itself is scored by travel."""
    qadr = [model.jnt_qposadr[j] for j in plantmap.primary_joints(model)]
    return StandingSpec(hold_seconds=1.0, height_ref=float(start.qpos[2]),
                        joints_ref=start.qpos[qadr].copy(), body_contact_phases=())


def progression(ex, duration_min: float = SECONDS_MIN, distance_min: float = DISTANCE_MIN,
                heading=(1.0, 0.0), yaw_min: float = 0.0) -> dict:
    """Measured progress of the executed rollout, ALONG WHAT WAS COMMANDED.

    This used to measure displacement in +x and call everything else lateral
    error. That is right for a forward walk and wrong for the other two things
    the scheduler can be asked for: a sidestep makes no x progress and a turn in
    place makes none at all, so both were rejected by a gate that was measuring
    the wrong axis rather than by anything about the motion.

    `heading` is the commanded direction in the plane (a unit vector; the
    default is the forward walk, unchanged). `yaw_min`, when set, replaces the
    distance test with a rotation test, which is the only meaningful progress
    measure for a turn.
    """
    h = np.asarray(heading, float)
    h = h / max(float(np.linalg.norm(h)), 1e-9)
    delta = ex.qpos[:, :2] - ex.qpos[0, :2]
    travel = float(delta[-1] @ h)
    lateral = float(np.max(np.abs(delta @ np.array([-h[1], h[0]]))))
    height = ex.qpos[:, 2]
    seconds = ex.steps * ex.control_dt
    yaw = _yaw_series(ex.qpos)
    turned = float(yaw[-1] - yaw[0])
    if yaw_min > 0.0:
        passed = bool(abs(turned) >= yaw_min and seconds >= duration_min)
        need = f"need >= {yaw_min} rad over >= {duration_min} s"
        detail = (f"turned {turned:+.3f} rad in {seconds:.2f} s "
                  f"({turned / max(seconds, 1e-9):+.3f} rad/s, drift {abs(travel) * 1e3:.0f} mm, "
                  f"height {height.min():.3f}-{height.max():.3f} m); {need}")
    else:
        passed = bool(travel >= distance_min and seconds >= duration_min)
        need = f"need >= {distance_min} m over >= {duration_min} s"
        detail = (f"travelled {travel:.3f} m forward in {seconds:.2f} s "
                  f"({travel / max(seconds, 1e-9):.3f} m/s, lateral {lateral * 1e3:.0f} mm, "
                  f"height {height.min():.3f}-{height.max():.3f} m); {need}")
    return {"passed": passed, "forward_m": travel, "lateral_max_m": lateral, "seconds": seconds,
            "speed_mps": travel / max(seconds, 1e-9), "height_min_m": float(height.min()),
            "height_max_m": float(height.max()), "turned_rad": turned,
            "yaw_rate_rps": turned / max(seconds, 1e-9),
            "heading": [float(h[0]), float(h[1])], "detail": detail}


def _yaw_series(qpos) -> np.ndarray:
    """Unwrapped base yaw per frame, from the free joint's wxyz quaternion."""
    w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    return np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

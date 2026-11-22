"""The DC torque-speed law, enforced DURING simulation rather than checked after.

trajopt's plants are MuJoCo position servos with a flat force clamp. The robot's
motors are not flat: they hold their rating to a corner speed and droop to zero
at no-load. Scoring the executed motion against that curve afterwards (gates.py)
catches a plan that violated it, but the plan was still SOLVED against a motor
that does not exist, and a search free to spend torque it will not have spends it.

MuJoCo has no speed-dependent actuator clamp, but `actuator_forcerange` is read
every step, so writing the curve's value at the current joint speed into it makes
MuJoCo's own clamp the DC law exactly.

This is mjlab's `dc_motor_clip` (mjlab/actuator/dc_actuator.py), same algebra and
same branches, so a clip solved here is solved against the actuator the policy
will be trained and deployed with. Above no-load speed the upper bound goes
NEGATIVE: the motor is forced to brake. ServoActuation shares this law.

PER-JOINT PLANTS (geared, belt-coupled). Their actuators are not force limited at all: MuJoCo clamps
the summed actuator torque on each JOINT to jnt_actfrcrange, so writing
actuator_forcerange there would be inert and the curve would silently not apply.
The envelope is written into jnt_actfrcrange instead, evaluated at the joint speed,
with the rating read from plantmap's cache (captured before any write) and the
field restored after each rollout. A joint-space curve is exact for unit servos and
only a stand-in for a belt-coupled motor; without a measured curve, its
[motor] entries make the envelope flat (corner 900 rad/s), so the stand-in only ever
writes the rating back.
"""
from __future__ import annotations

import numpy as np


def envelope(velocity, saturation, vnl, effort):
    """Signed DC bounds and velocity derivatives, including forced braking.

    Clipping both affine bounds also collapses the interval at extreme speed,
    without roundoff reversing the interval at the collapse point.
    """
    velocity, saturation, vnl, effort = np.broadcast_arrays(
        velocity, saturation, vnl, effort)
    top = saturation * (1. - velocity / vnl)
    bottom = saturation * (-1. - velocity / vnl)
    slope = -saturation / vnl
    return (np.clip(top, -effort, effort), np.clip(bottom, -effort, effort),
            np.where((top > -effort) & (top < effort), slope, 0.),
            np.where((bottom > -effort) & (bottom < effort), slope, 0.))


class DcClamp:
    """Per-joint (saturation, no-load speed, force limit) in MuJoCo actuator order."""

    def __init__(self, model, cfg) -> None:
        import mujoco

        from .plantmap import transmission
        self.m = model
        tr = transmission(model)
        # Per-JOINT clamping when the actuators are not force limited (geared/belt plants);
        # unit-gear plants keep the per-actuator path below, unchanged.
        self.joint_clamp = bool(not tr.identity or not model.actuator_forcelimited.all())
        roles = []
        for a in range(model.nu):
            j = int(tr.joints[a])          # the PRIMARY joint, not a tendon id
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            roles.append(name.split("_")[1])
        self.sat = np.array([cfg.motor[r][0] for r in roles], float)
        self.vnl = np.array([cfg.motor[r][1] for r in roles], float)
        # THE FORCE LIMIT COMES FROM A FIELD THIS CLASS DOES NOT WRITE.
        # apply() drives the curve by overwriting actuator_forcerange, so reading
        # the limit back out of it captures whatever the LAST STEP of some
        # earlier rollout happened to leave there -- a clamped value, not the
        # rating. Two replays of one command sequence then run different motors:
        # measured on a turn, in-process replay disagreed by 0.338 m of
        # translation and 1.62 rad of orientation. The walk clips never showed it
        # because they peak at 74% of the envelope and the clamp does not bind;
        # a turn takes the knee past its no-load speed, and it
        # binds every step. jnt_actfrcrange carries the same numbers and is only
        # ever read.
        joint_ids = tr.joints
        if self.joint_clamp:
            # The per-joint path WRITES jnt_actfrcrange, so its rating must come from
            # somewhere it does not write: plantmap's per-model cache, captured the
            # first time the model was mapped.
            self.flim = np.array(tr.limit, float)
        else:
            self.flim = model.jnt_actfrcrange[joint_ids, 1].copy()
        if not np.all(self.flim > 0):
            raise ValueError("plant has no joint actuatorfrcrange; the DC envelope "
                             "has no un-mutated source for its force limit")
        self.joints = np.asarray(joint_ids, int)
        # qvel address per actuated joint, to read the speed the curve is evaluated at.
        self.vadr = np.array([int(model.jnt_dofadr[int(j)]) for j in joint_ids], int)
        self.base = np.column_stack([-self.flim, self.flim])

    def apply(self, data) -> None:
        """Write the envelope at the CURRENT joint speeds into the plant's clamp."""
        vel = data.qvel[self.vadr]
        upper, lower, _, _ = envelope(vel, self.sat, self.vnl, self.flim)
        if self.joint_clamp:
            self.m.jnt_actfrcrange[self.joints, 1] = upper
            self.m.jnt_actfrcrange[self.joints, 0] = lower
            return
        self.m.actuator_forcerange[:, 1] = upper
        self.m.actuator_forcerange[:, 0] = lower

    def restore(self) -> None:
        if self.joint_clamp:
            self.m.jnt_actfrcrange[self.joints] = self.base
            return
        self.m.actuator_forcerange[:] = self.base


class CommandClamp:
    """The DC envelope as a clamp on the COMMAND, every substep: the command-clamp plant interface.

    The trainer's plant step leaves the model's force range alone and instead clamps each
    requested command to the interval whose servo torque lies inside the DC curve at the
    current joint speed, with the curve FLOORED at zero (no forced braking past no-load):

        up = clip(sat (1 - qd/v_nl), 0, effort)     dn = clip(sat (-1 - qd/v_nl), -effort, 0)
        position servo  tau = kp (u - q) - kd qd    ->  u in [q + (dn + kd qd)/kp, q + (up + kd qd)/kp]
        velocity servo  tau = kv (u - qd)           ->  u in [qd + dn/kv, qd + up/kv]

    Unlike DcClamp it writes nothing into the model, so a rollout cannot leave a clamped
    force range behind for the next one. The request is what a policy (or a reference's
    .commands) holds; d.ctrl is what the substep applied.

    Every actuator must be a unit-gear joint servo with an affine bias: a position servo
    (biasprm = [0, -kp, -kd]) or a velocity servo (biasprm = [0, 0, -kv]). `motor` maps the
    joint ROLE (the second token of the joint name: hip, thigh, calf, wheel) to
    [saturation, no-load speed]; `effort` is the joint's jnt_actfrcrange, the motor's
    plateau, which nothing writes. Used by a wheel-legged plant; unit-gear and geared
    plants keep DcClamp.
    """

    def __init__(self, model, motor) -> None:
        import mujoco

        self.m = model
        nu = model.nu
        self.qadr = np.zeros(nu, int)
        self.vadr = np.zeros(nu, int)
        self.sat, self.vnl, self.effort = np.zeros(nu), np.zeros(nu), np.zeros(nu)
        self.kp, self.kd, self.kv = np.zeros(nu), np.zeros(nu), np.zeros(nu)
        self.position = np.zeros(nu, bool)
        for a in range(nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT or \
                    np.any(model.actuator_gear[a] != [1, 0, 0, 0, 0, 0]):
                raise ValueError(f"{name}: CommandClamp needs a unit-gear joint transmission")
            if model.actuator_gaintype[a] != mujoco.mjtGain.mjGAIN_FIXED or \
                    model.actuator_biastype[a] != mujoco.mjtBias.mjBIAS_AFFINE or \
                    model.actuator_dyntype[a] != mujoco.mjtDyn.mjDYN_NONE:
                raise ValueError(f"{name}: not an affine servo")
            g, b = model.actuator_gainprm[a], model.actuator_biasprm[a]
            j = int(model.actuator_trnid[a][0])
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.qadr[a], self.vadr[a] = model.jnt_qposadr[j], model.jnt_dofadr[j]
            if b[1] != 0:                               # position servo
                if not (np.isclose(-b[1], g[0]) and b[0] == 0):
                    raise ValueError(f"{name}: position servo bias {b[:3]} != [0, -kp, -kd]")
                self.position[a] = True
                self.kp[a], self.kd[a] = g[0], -b[2]
            else:                                       # velocity servo
                if not (b[0] == 0 and np.isclose(-b[2], g[0]) and g[0] > 0):
                    raise ValueError(f"{name}: velocity servo bias {b[:3]} != [0, 0, -kv]")
                self.kv[a] = g[0]
            role = jname.split("_")[1]
            if role not in motor:
                raise ValueError(f"{name}: no motor entry for role {role!r}")
            self.sat[a], self.vnl[a] = float(motor[role][0]), float(motor[role][1])
            if not model.jnt_actfrclimited[j] or not model.jnt_actfrcrange[j][1] > 0:
                raise ValueError(f"{jname}: no joint actuatorfrcrange to read the plateau from")
            self.effort[a] = float(model.jnt_actfrcrange[j][1])
        self.pos = np.nonzero(self.position)[0]
        self.vel = np.nonzero(~self.position)[0]

    def torque_bounds(self, data):
        """(dn, up) torque the motor can deliver at the current joint speeds."""
        qd = data.qvel[self.vadr]
        up = np.minimum(np.maximum(self.sat * (1. - qd / self.vnl), 0.), self.effort)
        dn = np.maximum(np.minimum(self.sat * (-1. - qd / self.vnl), 0.), -self.effort)
        return dn, up

    def bounds(self, data):
        """(lo, hi) command interval per actuator at the current state."""
        q, qd = data.qpos[self.qadr], data.qvel[self.vadr]
        dn, up = self.torque_bounds(data)
        lo, hi = np.empty(self.m.nu), np.empty(self.m.nu)
        p, v = self.pos, self.vel
        # The trainer's command-ceiling expression, term for term.
        lo[p] = q[p] + (dn[p] + self.kd[p] * qd[p]) / self.kp[p]
        hi[p] = q[p] + (up[p] + self.kd[p] * qd[p]) / self.kp[p]
        lo[v] = qd[v] + dn[v] / self.kv[v]
        hi[v] = qd[v] + up[v] / self.kv[v]
        return lo, hi

    def apply(self, data, request) -> None:
        """d.ctrl = the request clamped to the envelope at the current state."""
        lo, hi = self.bounds(data)
        data.ctrl[:] = np.minimum(np.maximum(np.asarray(request, float), lo), hi)

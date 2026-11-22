"""Direct access to the compiled plant, for motions that are planned ON it.

Most of trajgen plans against Pinocchio and executes on MuJoCo. That split is fine for
motions whose plan the plant can follow, and wrong for the ones it cannot: measured, a
Pinocchio plan and this plant disagree by 0.108 rad per control interval at jump command
magnitudes, which over a hundred-step flip turns a 350 degree plan into a 1 degree replay.

Motions in that second class are searched and solved on the plant itself, which needs the
index tables MuJoCo does not hand out directly - which geoms are feet, which qpos slots the
actuators drive, which actuators share a sagittal channel - plus a settled stance to start
from. `Plant` is those tables, derived once from the model.

Everything here is pure MuJoCo. No Pinocchio, no Crocoddyl; `settle.py` adds those on top.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import plantmap

# Task/actuator leg order, matching the trainer's contact columns.
LEGS = ("FL", "FR", "RL", "RR")

# Sagittal command channels. A backflip is a planar motion, so searching four mirrored
# channels instead of twelve joints makes a lateral tumble unrepresentable in the
# parameters rather than merely penalised - every earlier attempt failed by leaving the
# sagittal plane.
FRONT_THIGH, FRONT_CALF, REAR_THIGH, REAR_CALF = range(4)

# Lateral command channels, for a BARREL ROLL - rotation about the long axis. The same
# argument, turned ninety degrees: a roll is planar in the frontal plane, so the
# channels mirror FRONT/REAR (both left legs move together, both right legs move
# together) and a pitching tumble is unrepresentable rather than penalised.
#
# THE HIPS ARE CHANNELS HERE, NOT PINNED. `_channel_of` holds them at stance because a
# flip is sagittal and hip abduction can only take it out of plane. A roll is the
# opposite case: abduction is the only joint that can put the ground reaction OUTSIDE
# the track, which is where the roll moment comes from. Pinning them leaves the motion
# with a 0.148 m half-track to push against and nothing to aim it with.
LEFT_HIP, LEFT_THIGH, LEFT_CALF, RIGHT_HIP, RIGHT_THIGH, RIGHT_CALF = range(6)


class Plant:
    """A compiled plant plus the index tables motion search needs."""

    def __init__(self, model: mujoco.MjModel, substeps: int = 10, cfg=None):
        """`cfg` enables the DC torque-speed envelope. Without it the plant is a
        flat-clamped position servo, which is a motor the robot does not have:
        the search then spends torque at speed that the real actuator cannot
        deliver, and the gate rejects the result afterwards. See trajgen/dcmotor.py."""
        self.m = model
        self.substeps = int(substeps)
        self.dc = None
        if cfg is not None:
            from .dcmotor import DcClamp
            self.dc = DcClamp(model, cfg)
        self.dt = model.opt.timestep * self.substeps
        # The same foot spheres and primary joints the adapter and executor use
        # (plantmap): {leg}_foot_collision on a unit-gear plant, {leg}_foot on a belt-coupled one, whose
        # belt actuators' trnid is a tendon id rather than a joint.
        self.foot_geom = plantmap.foot_geoms(model, LEGS)
        self.foot_radius = [model.geom_size[g][0] for g in self.foot_geom]
        self.footset = set(self.foot_geom)
        self.floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.transmission = plantmap.transmission(model)
        joints = self.transmission.joints
        self.joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                            for j in joints]
        self.qadr = np.array([model.jnt_qposadr[j] for j in joints])
        self.lo = model.jnt_range[joints, 0]
        self.hi = model.jnt_range[joints, 1]
        self.channel = np.array([_channel_of(n) for n in self.joint_names])
        self.lateral = np.array([_lateral_channel_of(n) for n in self.joint_names])

    # ------------------------------------------------------------------ stance

    def settled_stance(self, seconds: float = 2.5):
        """(state, command, height) of a stand that has been allowed to settle.

        The command is the gravity-compensating one: the standing posture plus the bias
        torque the feet do not carry, divided by the servo gain. Settling afterwards means
        a clip that opens here opens from rest rather than from a pose that is still
        sinking.
        """
        d = mujoco.MjData(self.m)
        key = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        mujoco.mj_resetDataKeyframe(self.m, d, key)
        mujoco.mj_forward(self.m, d)
        d.qpos[2] -= min(d.geom_xpos[g, 2] - self.foot_radius[j]
                         for j, g in enumerate(self.foot_geom))
        mujoco.mj_forward(self.m, d)
        jac = []
        for g in self.foot_geom:
            jp = np.zeros((3, self.m.nv))
            mujoco.mj_jacGeom(self.m, d, jp, None, g)
            jac.append(jp[2])
        jac = np.array(jac).T
        fz = np.linalg.lstsq(jac[:6], d.qfrc_bias[:6], rcond=None)[0]
        dofs = self.m.jnt_dofadr[self.transmission.joints]
        if not self.transmission.identity:
            raise NotImplementedError("settled_stance inverts a unit-gear servo; a geared/belt "
                                      "plant needs plantmap's coupled inversion (see bridge.py)")
        cmd = d.qpos[self.qadr] + (d.qfrc_bias - jac @ fz)[dofs] / self.m.actuator_gainprm[:, 0]
        for _ in range(int(seconds / self.m.opt.timestep)):
            d.ctrl[:] = cmd
            self.step(d)
        state = np.zeros(mujoco.mj_stateSize(self.m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(self.m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return state, cmd, float(d.qpos[2])

    def step(self, d) -> None:
        """ONE physics step, through the actuator this plant is configured with.

        Every rollout goes through here. Calling mujoco.mj_step directly is how
        the DC envelope came to be applied in `data_at` but not in the search's
        own `trace`/`to_touchdown`/`settleable`, so the gate scored a
        flat-clamped robot and wrote a clip for a different one: the accepted
        flip finished at z 0.399 under the numbers it was judged by and at
        z 0.264, pitched 67 deg, under the motor it is actually for.
        """
        if self.dc is not None:
            self.dc.apply(d)
        mujoco.mj_step(self.m, d)

    # ----------------------------------------------------------------- rollout

    def data_at(self, start, cmds, upto=None):
        """Replay `cmds[:upto]` from `start` and return the MjData there."""
        d = mujoco.MjData(self.m)
        mujoco.mj_setState(self.m, d, start, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        # Part of MuJoCo's state for an incompletely converged solve. Zeroing it here and
        # everywhere else makes planning and replay agree deterministically rather than
        # depending on what ran before.
        d.qacc_warmstart[:] = 0.
        mujoco.mj_forward(self.m, d)
        for c in cmds[: len(cmds) if upto is None else upto]:
            d.ctrl[:] = c
            for _ in range(self.substeps):
                self.step(d)
        return d

    def foot_heights(self, d):
        return [d.geom_xpos[g, 2] - self.foot_radius[j]
                for j, g in enumerate(self.foot_geom)]

    def feet_down(self, d, tol: float = .005):
        return sum(1 for h in self.foot_heights(d) if h < tol)

    def non_foot_contacts(self, d):
        """Contacts between the floor and anything that is not a foot, i.e. kneeling."""
        n = 0
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            for g, other in ((g1, g2), (g2, g1)):
                if other == self.floor and g not in self.footset:
                    n += 1
        return n


def _channel_of(name: str) -> int:
    if "_hip" in name:
        return -1                     # hips are held at stance; a flip is planar
    front = name[0] == "F"
    if "_thigh" in name:
        return FRONT_THIGH if front else REAR_THIGH
    return FRONT_CALF if front else REAR_CALF


def _lateral_channel_of(name: str) -> int:
    """Side x joint role. FL and RL share a channel, as do FR and RR."""
    left = name[1] == "L"
    if "_hip" in name:
        return LEFT_HIP if left else RIGHT_HIP
    if "_thigh" in name:
        return LEFT_THIGH if left else RIGHT_THIGH
    return LEFT_CALF if left else RIGHT_CALF


def lateral_angle(d, body: int = 1):
    """Unwrappable ROLL about the world x axis, and out-of-axis error.

    The mirror of `sagittal_angle`. Roll is read as atan2(R[2,1], R[1,1]) - the heading
    of the body's y axis in the world y-z plane - rather than from Euler angles,
    because a barrel roll passes straight through the orientation where roll/pitch/yaw
    coordinates degenerate, and "inverted = 180 deg" has to be a value on a continuous
    curve rather than a wrap. Out-of-axis error is arccos(R[0,0]), which stays zero
    through a pure roll including upside down, and rises if the motion starts to pitch
    or yaw out of the frontal plane.
    """
    w, x, y, z = d.xquat[body]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return np.arctan2(R[2, 1], R[1, 1]), np.arccos(np.clip(R[0, 0], -1., 1.))


def sagittal_angle(d, body: int = 1):
    """Unwrappable pitch about the world y axis, and out-of-plane error.

    Pitch is read as atan2(-R[2,0], R[0,0]) rather than from Euler angles, because a flip
    passes through the orientation where roll/yaw coordinates degenerate and "upright =
    -1" is correct rather than a failure. Out-of-plane error is arccos(R[1,1]), which is
    zero throughout a pure pitch rotation including upside down.
    """
    w, x, y, z = d.xquat[body]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return np.arctan2(-R[2, 0], R[0, 0]), np.arccos(np.clip(R[1, 1], -1., 1.))

"""What the compiled plant's actuators and feet ARE, read from the MjModel alone.

Every module that touches the plant used to answer two questions by convention:

  * "which joint does actuator a drive?"  ->  actuator_trnid[a][0]
  * "which geom is a foot?"               ->  the geom named {leg}_foot_collision

Both are true of a unit-gear point-foot plant and false of a geared, belt-coupled one. Its calf motor
does not drive the calf joint: it drives a fixed TENDON {leg}_belt = 0.5 thigh + 1 calf
(the belt runs over the thigh pulley), and its hip and thigh motors sit behind an 8:1
reduction, so actuator_trnid[a][0] of a calf actuator is a TENDON id that indexes
joint 0 -- the free joint -- if read as a joint. Its feet are spheres named {leg}_foot.
Read through the unit-gear convention, the calf's qpos address, torque limit, DC role and
foot set all come out wrong without anything failing loudly.

This module is the one place those answers come from, for the Pinocchio adapter and
for the pure-MuJoCo modules (execution, gates, dcmotor, bundle) alike, so the planner,
the executor and the acceptance gates cannot disagree on them.

The servo law, for a general MuJoCo position servo with gear g on a transmission whose
length is g * (J_a . q) (J_a = 1 on its joint for a JOINT transmission, the tendon's
coefficients for a fixed tendon), with gainprm = kp/g^2 and biasprm = [0, -kp/g^2,
-kd/g^2], is

    actuator force  F_a = (kp/g)(ctrl_a/g - J_a q) - (kd/g) J_a qdot
    joint torque    tau = sum_a g J_a^T F_a = J^T ( kp (m* - J q) - kd J qdot ),
                    m* = ctrl / g   (the motor target in joint-side radians)

then clamped per JOINT by jnt_actfrcrange. Writing m* = J q* for a joint-space command
q* (J is square and invertible: one actuator per joint), the servo is

    tau = clamp( Kp (q* - q) - Kd qdot ),   Kp = kp J^T J,   Kd = kd J^T J

-- a joint PD with COUPLED stiffness. The belt-coupled plant's per-leg J is [[1,0,0],[0,1,0],[0,.5,1]],
so its thigh sees kp (1 + 0.25) and a thigh-calf cross term of kp/2. A unit-gear plant is J = I,
g = 1: the scalar law every module was written against, kept as an exact fast path
(`identity`) so no unit-gear number moves.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass

import mujoco
import numpy as np

from .contract import FOOT_ORDER

# Tried in order. A unit-gear plant names its foot spheres {leg}_foot_collision; a belt-coupled
# plant names them {leg}_foot (its training task keeps exactly these four for collision). A
# wheel-legged plant's feet are
# its WHEELS, {leg}_wheel_collision cylinders (see _wheel_foot).
FOOT_GEOM_NAMES = ("{leg}_foot_collision", "{leg}_foot")
WHEEL_GEOM_NAME = "{leg}_wheel_collision"


def foot_geom(model: mujoco.MjModel, leg: str) -> int:
    """Geom id of `leg`'s foot: a collidable SPHERE, or a held WHEEL, by name, or an error."""
    for pattern in FOOT_GEOM_NAMES:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, pattern.format(leg=leg))
        if gid < 0:
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_SPHERE:
            raise ValueError(f"foot geom {pattern.format(leg=leg)} is not a sphere")
        if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
            raise ValueError(f"foot geom {pattern.format(leg=leg)} does not collide")
        return int(gid)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, WHEEL_GEOM_NAME.format(leg=leg))
    if gid >= 0:
        return _wheel_foot(model, gid)
    raise ValueError(f"no foot sphere for {leg}: tried "
                     f"{[p.format(leg=leg) for p in FOOT_GEOM_NAMES + (WHEEL_GEOM_NAME,)]}")


def _wheel_foot(model: mujoco.MjModel, gid: int) -> int:
    """A wheel stands in for a foot sphere only when it is exactly that while braked.

    A sphere foot's contact is planned at its CENTRE with its radius (adapter.ContactPoint):
    the nonpenetration condition is centre_z >= r and the contact point is centre - r z,
    whatever the orientation. A wheel satisfies the same construction iff it is a collidable
    CYLINDER centred on its own body origin, that body hangs from exactly one hinge, the
    cylinder axis is that hinge's axis (so the wheel spins on its axle), and the hinge is
    driven by a HELD actuator (braked, locked in the plan): then the axle centre is a fixed
    point of the calf, and with the wheel upright its lowest point is axle - r z. Its width
    is not modelled (the contact is a point under the axle), nor is rolling: a held wheel's
    brake has no position term, so the executed wheel can roll where the plan's cannot.
    """
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CYLINDER:
        raise ValueError(f"wheel geom {name} is not a cylinder")
    if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
        raise ValueError(f"wheel geom {name} does not collide")
    body = int(model.geom_bodyid[gid])
    if not np.allclose(model.geom_pos[gid], 0.0, atol=1e-12):
        raise ValueError(f"wheel geom {name} is not centred on its body (the axle)")
    joints = [j for j in range(model.njnt) if model.jnt_bodyid[j] == body]
    if len(joints) != 1 or model.jnt_type[joints[0]] != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError(f"wheel geom {name}: its body must hang from exactly one hinge")
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, model.geom_quat[gid])
    axis = R.reshape(3, 3)[:, 2]                     # cylinder axis in the body frame
    if abs(abs(float(axis @ model.jnt_axis[joints[0]])) - 1.0) > 1e-9:
        raise ValueError(f"wheel geom {name}: cylinder axis is not its hinge axis")
    if joints[0] not in held_joints(model):
        raise ValueError(f"wheel geom {name}: its hinge is not driven by a held actuator")
    return int(gid)


def is_held(model: mujoco.MjModel, a: int) -> bool:
    """A HELD actuator: an affine VELOCITY servo (gain kv > 0, bias [0, 0, -kv]) on a unit
    joint transmission. Its command is a speed, not a pose, so it is not part of the planned
    joint space: the walk holds it at a constant target (0 = braked) and locks its joint in
    the planning model. Unit-gear and geared plants have none; a wheel-legged plant's four wheels are held."""
    b, g = model.actuator_biasprm[a], model.actuator_gainprm[a]
    return bool(model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
                and model.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
                and model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
                and g[0] > 0 and b[0] == 0 and b[1] == 0 and np.isclose(-b[2], g[0]))


def held_actuators(model: mujoco.MjModel) -> np.ndarray:
    return np.array([a for a in range(model.nu) if is_held(model, a)], int)


def held_joints(model: mujoco.MjModel) -> list[int]:
    return [int(model.actuator_trnid[a][0]) for a in held_actuators(model)]


def foot_geoms(model: mujoco.MjModel, legs=FOOT_ORDER) -> list[int]:
    return [foot_geom(model, leg) for leg in legs]


@dataclass(frozen=True)
class Transmission:
    """Per-actuator transmission, in MuJoCo ACTUATOR order throughout.

    `joints[a]` is actuator a's PRIMARY joint: its own joint for a joint transmission,
    and for a tendon the one joint of its path that no joint actuator claims (the
    belt's calf). Joint-indexed quantities everywhere else in trajgen are in this
    order -- joint k is the primary joint of actuator k -- which for a unit-gear plant is exactly
    actuator_trnid[:, 0], so nothing about its indexing changes.
    """

    joints: np.ndarray        # [ns] primary MuJoCo joint id per SERVO actuator
    J: np.ndarray             # [ns, ns] motor coordinate a = J[a] . q[joints]
    gear: np.ndarray          # [ns]
    limit: np.ndarray         # [ns] joint torque limit of each primary joint, N m
    joint_clamp: bool         # torque clamped per joint (jnt_actfrcrange), not per actuator
    identity: bool            # unit-gear joint servos, one per joint: the scalar law
    # The position servos among the model's nu actuators, and the HELD ones (is_held). With
    # nothing held (unit-gear, geared) servo = arange(nu) and every [ns] array above is the old
    # [nu] one; with held actuators (a wheel-legged plant's wheels) everything above is over servos only,
    # and a full MuJoCo ctrl vector is assembled with `full`.
    servo: np.ndarray = None  # [ns] actuator ids
    held: np.ndarray = None   # [nh] actuator ids
    nu: int = 0               # model.nu

    def addresses(self, model: mujoco.MjModel):
        """(qpos address, dof address) of each actuator's primary joint."""
        return (np.array([model.jnt_qposadr[j] for j in self.joints], int),
                np.array([model.jnt_dofadr[j] for j in self.joints], int))

    def ctrl(self, q_cmd) -> np.ndarray:
        """Joint-space command q* (actuator order) -> MuJoCo ctrl = g * (J q*)."""
        q_cmd = np.asarray(q_cmd, float)
        if self.identity:
            return q_cmd.copy()
        return self.gear * (self.J @ q_cmd)

    def joint_command(self, ctrl) -> np.ndarray:
        """MuJoCo ctrl -> the joint-space command q* it encodes."""
        ctrl = np.asarray(ctrl, float)
        if self.identity:
            return ctrl.copy()
        return np.linalg.solve(self.J, ctrl / self.gear)

    def positions(self, ctrl) -> np.ndarray:
        """ctrl in SERVO-POSITION units, ctrl / gear: the motor target in joint-side
        radians. The command-delta encoding and the slew limit are stated in these
        units, so `action_scale` means the same angle on an 8:1 hip as on a unit-gear 1:1,
        instead of an eighth of it. Identity plants return ctrl unchanged."""
        ctrl = np.asarray(ctrl, float)
        return ctrl if self.identity else ctrl / self.gear

    def stiffness(self, k: float) -> np.ndarray:
        """Joint-space servo gain matrix k J^T J (actuator order)."""
        return float(k) * (self.J.T @ self.J)

    def full(self, servo_ctrl) -> np.ndarray:
        """Servo commands [ns] -> a MuJoCo ctrl [nu], held actuators at target 0 (braked).
        Without held actuators the input is returned as it is."""
        if not self.held.size:
            return servo_ctrl
        servo_ctrl = np.asarray(servo_ctrl, float)
        out = np.zeros(servo_ctrl.shape[:-1] + (self.nu,))
        out[..., self.servo] = servo_ctrl
        return out

    def servo_part(self, ctrl):
        """MuJoCo ctrl [..., nu] -> its servo columns [..., ns]; unchanged without held ones."""
        if not self.held.size:
            return ctrl
        return np.asarray(ctrl)[..., self.servo]


_CACHE: "weakref.WeakKeyDictionary[mujoco.MjModel, Transmission]" = weakref.WeakKeyDictionary()


def transmission(model: mujoco.MjModel) -> Transmission:
    """Derive (and cache per model object) the transmission, or refuse the plant.

    Refused: anything but hinge-joint or fixed joint-tendon transmissions, gear with
    more than a scalar component, a joint driven by no actuator or claimed twice, and
    a singular J. Cached because it is read per control step by the executor's
    callers; it depends only on fields nothing in trajgen writes.
    """
    cached = _CACHE.get(model)
    if cached is not None:
        return cached
    held = held_actuators(model)
    servo = np.array([a for a in range(model.nu) if a not in set(held.tolist())], int)
    nu = len(servo)
    rows: list[dict[int, float]] = []
    joint_trn: list[int | None] = []
    for a in servo:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        if np.any(model.actuator_gear[a][1:] != 0):
            raise ValueError(f"actuator {name}: gear has non-scalar components")
        if not model.actuator_gear[a][0] > 0:
            raise ValueError(f"actuator {name}: gear {model.actuator_gear[a][0]} is not positive")
        kind, target = model.actuator_trntype[a], int(model.actuator_trnid[a][0])
        if kind == mujoco.mjtTrn.mjTRN_JOINT:
            if model.jnt_type[target] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"actuator {name}: drives a non-hinge joint")
            rows.append({target: 1.0})
            joint_trn.append(target)
        elif kind == mujoco.mjtTrn.mjTRN_TENDON:
            adr, num = model.tendon_adr[target], model.tendon_num[target]
            path = {}
            for w in range(adr, adr + num):
                if model.wrap_type[w] != mujoco.mjtWrap.mjWRAP_JOINT:
                    raise ValueError(f"actuator {name}: tendon is not a fixed joint tendon")
                path[int(model.wrap_objid[w])] = float(model.wrap_prm[w])
            rows.append(path)
            joint_trn.append(None)
        else:
            raise ValueError(f"actuator {name}: unsupported transmission type {kind}")
    claimed = [j for j in joint_trn if j is not None]
    if len(set(claimed)) != len(claimed):
        raise ValueError("two joint actuators drive the same joint")
    primary = []
    for a in range(nu):              # a indexes the SERVO actuators (= actuators w/o held)
        if joint_trn[a] is not None:
            primary.append(joint_trn[a])
            continue
        free = [j for j in rows[a] if j not in claimed]
        if len(free) != 1:
            raise ValueError(f"actuator {a}: tendon has {len(free)} joints no joint actuator "
                             "drives; cannot assign it a primary joint")
        primary.append(free[0])
        claimed.append(free[0])
    index = {j: k for k, j in enumerate(primary)}
    J = np.zeros((nu, nu))
    for a, path in enumerate(rows):
        for j, coef in path.items():
            if j not in index:
                raise ValueError(f"actuator {a}: its path moves joint {j}, which no actuator drives")
            J[a, index[j]] = coef
    if not np.isfinite(np.linalg.cond(J)) or np.linalg.cond(J) > 1e8:
        raise ValueError("transmission matrix is singular")
    gear = np.array(model.actuator_gear[servo, 0], float)
    joints = np.array(primary, int)
    identity = bool(np.array_equal(J, np.eye(nu)) and np.all(gear == 1.0)
                    and all(j is not None for j in joint_trn))
    joint_clamp = bool(np.all(model.jnt_actfrclimited[joints]))
    if not joint_clamp and not np.all(model.actuator_forcelimited[servo]):
        raise ValueError("plant has neither per-joint nor per-actuator torque limits")
    # THE RATING, captured on first use and cached with the model. jnt_actfrcrange is
    # never written by dcmotor's per-actuator path (unit-gear), and its per-joint path
    # (geared/belt) reads its rating from HERE, never back out of the field it writes --
    # whereas actuator_forcerange in a saved model can hold a DC-clamped value: the
    # point-foot flat-servo mjb carries 0.855 / 1.717 / 0.212 there against 34.6/34.6/69.2.
    limit = np.array(model.jnt_actfrcrange[joints, 1], float) if joint_clamp else \
        np.array(model.actuator_forcerange[servo, 1], float)
    tr = Transmission(joints=joints, J=J, gear=gear, limit=limit, joint_clamp=joint_clamp,
                      identity=identity, servo=servo, held=held, nu=int(model.nu))
    _CACHE[model] = tr
    return tr


def primary_joints(model: mujoco.MjModel) -> np.ndarray:
    return transmission(model).joints

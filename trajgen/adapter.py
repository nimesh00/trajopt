"""MuJoCo execution plant <-> Pinocchio planning model.

Planning runs in Pinocchio/Crocoddyl on the URDF; execution runs in MuJoCo on
the compiled MJB. Both describe the same robot with different conventions, and
every one of these differences has silently broken a comparison before:

  quaternion      MuJoCo qpos[3:7] = (w, x, y, z)   Pinocchio q[3:7] = (x, y, z, w)
  base linear v   MuJoCo: world frame               Pinocchio: base frame
  base angular v  both: base frame
  joint order     matched by NAME, never by array position
  armature        MuJoCo 0.013841 per actuated DOF; the URDF has none
  damping         MuJoCo 0.1 N m s/rad; Pinocchio/Crocoddyl have no damping term
  frictionloss    MuJoCo 0.2 N m; a constraint, not representable in ABA
  foot contact    MuJoCo: 25 mm sphere; URDF *_foot_site is not its surface point

Coordinates: with v_pin = T(q) v_mj and T = diag(R^T, I_3, I_nj),
power consistency gives tau_mj = T^T tau_pin, so

    M_mj = T^T M_pin T
    h_mj = T^T (h_pin + M_pin * Tdot v_mj),   Tdot v_mj = (-omega x v_base, 0, 0)

The extra term is the classical-vs-spatial acceleration of the base origin.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

ADAPTER_VERSION = 1
LEGS = ("FL", "FR", "RL", "RR")

_PIN_AXIS = {"RX": (1.0, 0.0, 0.0), "RY": (0.0, 1.0, 0.0), "RZ": (0.0, 0.0, 1.0)}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], float)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], float)


def _base_rotation_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat_wxyz, float) / np.linalg.norm(quat_wxyz))
    return R.reshape(3, 3)


@dataclass
class ContactPoint:
    """A collision sphere attached to a Pinocchio frame.

    `frame` sits at the sphere CENTRE, so for flat ground the contact point is
    centre - radius * world_z, and the nonpenetration condition is centre_z >=
    radius. A fixed body point on the sphere surface is only correct at one
    orientation, which is why *_foot_site cannot be used as the contact point.
    """

    name: str
    frame: int
    radius: float
    mj_geom: int


@dataclass
class Adapter:
    mj: mujoco.MjModel
    pin: pin.Model
    joint_names: tuple[str, ...]      # actuated joints, in MuJoCo order
    mj_qadr: np.ndarray
    mj_dadr: np.ndarray
    pin_qadr: np.ndarray
    pin_vadr: np.ndarray
    body_frames: dict                 # MuJoCo body id -> Pinocchio BODY frame id
    feet: dict                        # leg -> ContactPoint
    differences: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    geoms: dict = field(default_factory=dict)  # geom name -> {frame, mj_geom, type, size}
    transmission: object = None       # plantmap.Transmission of `mj`

    # ----------------------------------------------------------- state maps

    def q_from_mj(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, float)
        quat = qpos[3:7]
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or abs(norm - 1.0) > 1e-6:
            raise ValueError(f"base quaternion norm {norm} is not unit")
        q = pin.neutral(self.pin)
        q[0:3] = qpos[0:3]
        q[3:7] = quat_wxyz_to_xyzw(quat / norm)
        q[self.pin_qadr] = qpos[self.mj_qadr]
        return q

    def qpos_from_pin(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, float)
        norm = np.linalg.norm(q[3:7])
        if not np.isfinite(norm) or abs(norm - 1.0) > 1e-6:
            raise ValueError(f"base quaternion norm {norm} is not unit")
        qpos = np.zeros(self.mj.nq)
        qpos[0:3] = q[0:3]
        qpos[3:7] = quat_xyzw_to_wxyz(q[3:7] / norm)
        qpos[self.mj_qadr] = q[self.pin_qadr]
        return qpos

    def transform(self, qpos: np.ndarray) -> np.ndarray:
        """T with v_pin = T v_mj (square, nv x nv; pin.nv x mj.nv when joints are locked)."""
        if self.pin.nv != self.mj.nv:
            # LOCKED joints (held wheels): T selects the planned dofs, so T^T M_pin T is the
            # MuJoCo mass matrix with zero rows/columns at the locked dofs.
            T = np.zeros((self.pin.nv, self.mj.nv))
            T[0:3, 0:3] = _base_rotation_wxyz(np.asarray(qpos)[3:7]).T
            T[3:6, 3:6] = np.eye(3)
            T[self.pin_vadr, self.mj_dadr] = 1.0
            return T
        T = np.eye(self.mj.nv)
        T[0:3, 0:3] = _base_rotation_wxyz(np.asarray(qpos)[3:7]).T
        # Joint velocities permute by name; with T acting on MuJoCo dof order.
        P = np.zeros((self.mj.nv - 6, self.mj.nv - 6))
        P[self.pin_vadr - 6, self.mj_dadr - 6] = 1.0
        T[6:, 6:] = P
        return T

    def v_from_mj(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        qvel = np.asarray(qvel, float)
        v = np.zeros(self.pin.nv)
        v[0:3] = _base_rotation_wxyz(np.asarray(qpos)[3:7]).T @ qvel[0:3]
        v[3:6] = qvel[3:6]
        v[self.pin_vadr] = qvel[self.mj_dadr]
        return v

    def qvel_from_pin(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, float)
        R = pin.Quaternion(np.asarray(q, float)[3:7] / np.linalg.norm(q[3:7])).matrix()
        qvel = np.zeros(self.mj.nv)
        qvel[0:3] = R @ v[0:3]
        qvel[3:6] = v[3:6]
        qvel[self.mj_dadr] = v[self.pin_vadr]
        return qvel

    def x_from_mj(self, qpos, qvel) -> np.ndarray:
        return np.concatenate([self.q_from_mj(qpos), self.v_from_mj(qpos, qvel)])

    def mj_from_x(self, x):
        q, v = x[: self.pin.nq], x[self.pin.nq:]
        return self.qpos_from_pin(q), self.qvel_from_pin(q, v)

    def tau_pin_to_mj_joints(self, tau_joint_pin: np.ndarray) -> np.ndarray:
        """Joint torques in Pinocchio joint order -> MuJoCo actuated-joint order."""
        out = np.zeros(len(self.joint_names))
        out[:] = np.asarray(tau_joint_pin)[self.pin_vadr - 6]
        return out

    def joints_mj_to_pin(self, values_mj_order: np.ndarray) -> np.ndarray:
        out = np.zeros(len(self.joint_names))
        out[self.pin_vadr - 6] = np.asarray(values_mj_order)
        return out

    # ------------------------------------------------ dynamics in MJ coords

    def dynamics_mj(self, qpos, qvel, data: pin.Data | None = None):
        """Pinocchio mass matrix and bias forces expressed in MuJoCo coordinates.

        Comparable to mj_fullM(qM) and qfrc_bias (which excludes passive
        damping/frictionloss -- those are reported as modelling differences).
        """
        data = data or self.pin.createData()
        q = self.q_from_mj(qpos)
        v = self.v_from_mj(qpos, qvel)
        M = pin.crba(self.pin, data, q)
        M = np.triu(M) + np.triu(M, 1).T
        h = pin.nonLinearEffects(self.pin, data, q, v)
        T = self.transform(qpos)
        Tdot_v = np.zeros(self.pin.nv)
        Tdot_v[0:3] = -np.cross(v[3:6], v[0:3])
        return T.T @ M @ T, T.T @ (h + M @ Tdot_v)

    def damping_mj(self, qvel) -> np.ndarray:
        """MuJoCo's joint damper force (qfrc_damper), which Crocoddyl lacks."""
        return -self.mj.dof_damping * np.asarray(qvel, float)

    def tau_mj_from_pin(self, qpos, tau_pin) -> np.ndarray:
        """Generalised force: tau_mj = T^T tau_pin."""
        return self.transform(qpos).T @ np.asarray(tau_pin, float)

    def sphere_contact_point(self, data: pin.Data, contact: "ContactPoint",
                             normal=(0.0, 0.0, 1.0)) -> np.ndarray:
        """Surface point of a sphere against a plane with unit `normal`.

        Geometric point c - r n; its material velocity is not the centre's
        velocity when the sphere rolls. `data` must hold updated frame placements.
        """
        n = np.asarray(normal, float)
        n = n / np.linalg.norm(n)
        return data.oMf[contact.frame].translation - contact.radius * n

    # ------------------------------------------------------------- manifest

    def manifest(self) -> dict:
        m = self.mj
        return {
            "adapter_version": ADAPTER_VERSION,
            **self.provenance,
            "versions": {"mujoco": mujoco.__version__, "pinocchio": pin.__version__},
            "conventions": {
                "mj_quat": "wxyz", "pin_quat": "xyzw",
                "mj_base_linear_velocity": "world", "pin_base_linear_velocity": "local",
                "base_angular_velocity": "local (both)",
                "foot_contact": "sphere centre frame + radius, flat-ground normal",
            },
            "joint_names": list(self.joint_names),
            "mj_qadr": self.mj_qadr.tolist(), "mj_dadr": self.mj_dadr.tolist(),
            "pin_qadr": self.pin_qadr.tolist(), "pin_vadr": self.pin_vadr.tolist(),
            "actuators": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
                          for a in range(m.nu)],
            "timestep": float(m.opt.timestep),
            "gravity": m.opt.gravity.tolist(),
            "armature_transferred": self.pin.armature.tolist(),
            "feet": {k: {"frame": self.pin.frames[c.frame].name, "radius": c.radius,
                         "mj_geom": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.mj_geom)}
                     for k, c in self.feet.items()},
            "total_mass": {"mujoco": float(np.sum(m.body_mass)),
                           "pinocchio": float(pin.computeTotalMass(self.pin))},
            "collision_geoms": {k: {kk: vv for kk, vv in v.items() if kk != "frame"}
                                | {"frame": self.pin.frames[v["frame"]].name}
                                for k, v in self.geoms.items()},
            "differences": self.differences,
        }


# ---------------------------------------------------------------- building

def _pinocchio_model(description: str | Path) -> tuple[pin.Model, list[str]]:
    """Pinocchio model from a URDF, or from an MJCF when the robot has no URDF.

    A geared, belt-coupled plant may have no URDF matching its MJCF, so Pinocchio reads
    the very MJCF the trainer compiles. Its parser turns the MJCF <freejoint> into the root
    free flyer but keeps the body's `pos` as that joint's placement (0.445 m up on one such plant),
    whereas MuJoCo's free-joint qpos IS the body's world pose. Left alone, every
    Pinocchio placement sits 0.445 m above MuJoCo's and the self-check refuses the
    pair; the placement is reset to identity so q[0:7] means what qpos[0:7] means.
    """
    notes = []
    if Path(description).suffix.lower() == ".xml":
        model = pin.buildModelFromMJCF(str(description))
        if model.njoints < 2 or model.joints[1].shortname() != "JointModelFreeFlyer":
            raise ValueError(f"{description}: MJCF root is not a free joint")
        offset = model.jointPlacements[1]
        if not offset.isIdentity():
            notes.append(f"MJCF free-joint placement {np.round(offset.translation, 6).tolist()} "
                         "reset to identity (MuJoCo free qpos is the absolute body pose)")
            model.jointPlacements[1] = pin.SE3.Identity()
        # Crocoddyl's ActuationModelFloatingBase finds the floating base BY NAME
        # ("root_joint", buildModelFromUrdf's name for it); under the MJCF's own name
        # ("floating_base") it takes the universe's nv of 1 and makes nu = nv - 1 = 17.
        if model.names[1] != "root_joint":
            notes.append(f"MJCF free joint {model.names[1]!r} renamed 'root_joint' (Crocoddyl "
                         "locates the floating base by that name)")
            model.names[1] = "root_joint"
        return model, notes
    return pin.buildModelFromUrdf(str(description), pin.JointModelFreeFlyer()), notes


def build(mjb: str | Path, urdf: str | Path, transfer_armature: bool = True) -> Adapter:
    """Build and audit the adapter. Refuses structural mismatches outright.

    `urdf` is the Pinocchio description: a URDF, or an MJCF (.xml) for a robot
    without one (a belt-coupled plant). Either way the audit below is the same."""
    from . import plantmap

    mj = mujoco.MjModel.from_binary_path(str(mjb))
    model, notes = _pinocchio_model(urdf)
    problems, differences = [], list(notes)
    # HELD actuators' joints (a wheel-legged plant's braked wheels, plantmap.is_held) are not planned:
    # they are LOCKED in the Pinocchio model at angle 0, which merges each wheel's inertia
    # into its calf. MuJoCo's mass matrix restricted to the remaining dofs is exactly the
    # locked model's (the locked dof's own armature drops out with its row and column), and
    # the self-check below compares over those dofs. Robots without held actuators lock
    # nothing and build exactly as before.
    locked = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j) for j in plantmap.held_joints(mj)]
    if locked:
        missing = [n for n in locked if not model.existJointName(n)]
        if missing:
            raise ValueError(f"adapter refused the model pair: held joints {missing} absent from "
                             "the Pinocchio description")
        # buildReducedModel looks frames up by name and refuses duplicates; the MJCF parser
        # names a <site> after itself, and a wheel-legged plant's `base` site shares the `base` body's
        # name. Such non-body frames are renamed <name>_site; no adapter lookup uses them.
        bodies = {f.name for f in model.frames if f.type == pin.FrameType.BODY}
        for f in model.frames:
            if f.type != pin.FrameType.BODY and f.name in bodies:
                differences.append(f"Pinocchio frame {f.name!r} ({f.type}) renamed {f.name + '_site'!r}")
                f.name = f.name + "_site"
        model = pin.buildReducedModel(model, [model.getJointId(n) for n in locked], pin.neutral(model))
        differences.append(f"held joints {locked} locked at 0 in the planning model "
                           "(braked: velocity target 0); their rolling is not planned")

    # Exactly one free joint, at the root, in each.
    free = [j for j in range(mj.njnt) if mj.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if free != [0] or mj.jnt_qposadr[0] != 0 or mj.jnt_dofadr[0] != 0:
        problems.append(f"MuJoCo must have one root free joint at address 0, got {free}")
    if model.joints[1].shortname() != "JointModelFreeFlyer":
        problems.append("Pinocchio joint 1 is not the free flyer")

    # Hinges by name, with type and axis checked.
    names, mj_q, mj_d, pin_q, pin_v = [], [], [], [], []
    for j in range(mj.njnt):
        if mj.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name in locked:
            continue
        if mj.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            problems.append(f"{name}: unsupported MuJoCo joint type {mj.jnt_type[j]}")
            continue
        if not model.existJointName(name):
            problems.append(f"{name}: absent from URDF")
            continue
        pj = model.joints[model.getJointId(name)]
        kind = pj.shortname().replace("JointModel", "")
        if kind not in _PIN_AXIS or pj.nq != 1:
            problems.append(f"{name}: Pinocchio joint {kind} is not an axis-aligned revolute")
            continue
        if not np.allclose(mj.jnt_axis[j], _PIN_AXIS[kind], atol=1e-9):
            problems.append(f"{name}: axis MuJoCo {mj.jnt_axis[j]} vs Pinocchio {kind}")
        names.append(name)
        mj_q.append(int(mj.jnt_qposadr[j]))
        mj_d.append(int(mj.jnt_dofadr[j]))
        pin_q.append(int(pj.idx_q))
        pin_v.append(int(pj.idx_v))

        lo_mj, hi_mj = mj.jnt_range[j]
        lo_p, hi_p = model.lowerPositionLimit[pj.idx_q], model.upperPositionLimit[pj.idx_q]
        if not (np.isclose(lo_mj, lo_p, atol=1e-4) and np.isclose(hi_mj, hi_p, atol=1e-4)):
            differences.append(f"{name}: range MuJoCo [{lo_mj:.4f},{hi_mj:.4f}] "
                               f"vs URDF [{lo_p:.4f},{hi_p:.4f}]; MuJoCo range transferred")
        if not mj.jnt_limited[j]:
            problems.append(f"{name}: MuJoCo joint is unlimited")
        model.lowerPositionLimit[pj.idx_q] = lo_mj
        model.upperPositionLimit[pj.idx_q] = hi_mj
    extra = {model.names[i] for i in range(2, model.njoints)} - set(names)
    if extra:
        problems.append(f"URDF joints absent from MuJoCo: {sorted(extra)}")
    if len(set(names)) != len(names):
        problems.append("duplicate joint names")

    # Actuators: one per mapped hinge, through a joint transmission (unit gear on a
    # point-foot plant) or a fixed joint tendon (a belt-driven calf). The servo law each implies
    # is derived in plantmap and used by the bridge, the servo OCP and the gates.
    transmission = None
    try:
        transmission = plantmap.transmission(mj)
    except ValueError as error:
        problems.append(f"actuators: {error}")
    else:
        for k, a in enumerate(transmission.servo):
            aname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a))
            jname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, int(transmission.joints[k]))
            if jname not in names:
                problems.append(f"actuator {aname}: drives unmapped joint {jname}")
        if not transmission.identity:
            differences.append(
                "geared/tendon transmission: joint-side servo tau = J^T (kp (ctrl/g - J q) - "
                f"kd J qdot), gears {sorted(set(transmission.gear.tolist()))}, "
                f"J rows {sorted({tuple(float(v) for v in r[r != 0]) for r in transmission.J})}; "
                "torque clamped per " + ("joint" if transmission.joint_clamp else "actuator"))

    # Bodies by name.
    body_frames = {}
    for b in range(1, mj.nbody):
        bname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, b)
        if not model.existBodyName(bname):
            problems.append(f"body {bname}: no Pinocchio BODY frame")
            continue
        body_frames[b] = model.getBodyId(bname)

    if problems:
        raise ValueError("adapter refused the model pair:\n  " + "\n  ".join(problems))

    mj_q, mj_d = np.array(mj_q), np.array(mj_d)
    pin_q, pin_v = np.array(pin_q), np.array(pin_v)

    # Gravity.
    if not np.allclose(mj.opt.gravity, model.gravity.linear, atol=1e-9):
        differences.append(f"gravity MuJoCo {mj.opt.gravity} vs Pinocchio {model.gravity.linear}; "
                           "set to MuJoCo")
        model.gravity.linear = mj.opt.gravity.copy()

    # Passive joint terms.
    arm = mj.dof_armature[mj_d]
    if transfer_armature:
        model.armature[pin_v] = arm
        model.armature[:6] = mj.dof_armature[:6]
    else:
        differences.append(f"armature not transferred (MuJoCo {arm.max():.6g})")
    damping, fricloss = mj.dof_damping[mj_d], mj.dof_frictionloss[mj_d]
    if np.any(damping):
        differences.append(f"joint damping {np.unique(damping).tolist()} N m s/rad in MuJoCo "
                           "is not modelled by Pinocchio ABA/Crocoddyl")
    if np.any(fricloss):
        differences.append(f"joint frictionloss {np.unique(fricloss).tolist()} N m in MuJoCo "
                           "is not modelled by Pinocchio ABA/Crocoddyl")
    if np.any(mj.dof_damping[:6]) or np.any(mj.dof_frictionloss[:6]):
        differences.append("free-joint damping/frictionloss nonzero in MuJoCo")

    # Foot collision spheres -> Pinocchio frames at the sphere centre.
    feet = {}
    for leg in LEGS:
        gid = plantmap.foot_geom(mj, leg)     # the same lookup execution and gates use
        body = int(mj.geom_bodyid[gid])
        parent = model.frames[body_frames[body]]
        offset = pin.SE3(np.eye(3), mj.geom_pos[gid].copy())
        if not np.allclose(mj.geom_quat[gid], [1, 0, 0, 0]):
            offset.rotation = _base_rotation_wxyz(mj.geom_quat[gid])
        fname = f"{leg}_foot_contact"
        fid = model.addFrame(pin.Frame(fname, parent.parentJoint, body_frames[body],
                                       parent.placement * offset, pin.FrameType.OP_FRAME))
        feet[leg] = ContactPoint(fname, fid, float(mj.geom_size[gid][0]), gid)
        site = f"{leg}_foot_site"
        if model.existFrame(site):
            gap = model.frames[model.getFrameId(site)].placement.translation - \
                (parent.placement * offset).translation
            differences.append(f"{site} is {np.round(gap, 5).tolist()} m from the sphere centre "
                               f"in the calf frame (a fixed surface point); use {fname}")

    # Every robot collision geom gets a frame at its centre, for body/limb
    # contacts (getup needs trunk, thigh and calf support, not just feet).
    geoms = {}
    for g in range(mj.ngeom):
        body = int(mj.geom_bodyid[g])
        if body == 0 or not (mj.geom_contype[g] or mj.geom_conaffinity[g]):
            continue
        gname = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g) or \
            f"{mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, body)}_geom{g}"
        parent = model.frames[body_frames[body]]
        offset = pin.SE3(_base_rotation_wxyz(mj.geom_quat[g]), mj.geom_pos[g].copy())
        fid = model.addFrame(pin.Frame(f"geom:{gname}", parent.parentJoint, body_frames[body],
                                       parent.placement * offset, pin.FrameType.OP_FRAME))
        geoms[gname] = {"frame": fid, "mj_geom": g, "type": int(mj.geom_type[g]),
                        "size": mj.geom_size[g].tolist()}

    ad = Adapter(mj=mj, pin=model, joint_names=tuple(names), mj_qadr=mj_q, mj_dadr=mj_d,
                 pin_qadr=pin_q, pin_vadr=pin_v, body_frames=body_frames, feet=feet,
                 differences=differences,
                 provenance={"mjb": str(mjb), "mjb_sha256": sha256(mjb),
                             "urdf": str(urdf), "urdf_sha256": sha256(urdf)})
    ad.geoms = geoms
    ad.transmission = transmission
    _self_check(ad, mass_tol=1e-4 if transfer_armature else np.inf)
    return ad


def _self_check(ad: Adapter, poses: int = 4, tol: float = 1e-6,
                mass_tol: float = 1e-4) -> None:
    """Refuse a mapping whose kinematics disagree.

    Names and axes can all match while the mapping is still wrong (two legs'
    joint names swapped, a joint zero offset, a changed link length), so
    compare body and geom placements at seeded random configurations.
    """
    rng = np.random.default_rng(12345)
    mj, pm = ad.mj, ad.pin
    d, pd = mujoco.MjData(mj), pm.createData()
    worst = worst_M = 0.0
    for _ in range(poses):
        d.qpos[:] = mj.qpos0
        d.qpos[0:3] = rng.normal(size=3)
        quat = rng.normal(size=4)
        d.qpos[3:7] = quat / np.linalg.norm(quat)
        if len(ad.mj_qadr) == mj.njnt - 1:
            lo, hi = mj.jnt_range[1:, 0], mj.jnt_range[1:, 1]
        else:   # locked joints stay at qpos0 (0), the angle they are locked at
            mapped = [j for j in range(1, mj.njnt) if mj.jnt_qposadr[j] in set(ad.mj_qadr.tolist())]
            lo, hi = mj.jnt_range[mapped, 0], mj.jnt_range[mapped, 1]
        d.qpos[ad.mj_qadr] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(mj, d)
        pin.framesForwardKinematics(pm, pd, ad.q_from_mj(d.qpos))
        for b, fid in ad.body_frames.items():
            worst = max(worst, np.max(np.abs(pd.oMf[fid].translation - d.xpos[b])),
                        np.max(np.abs(pd.oMf[fid].rotation - d.xmat[b].reshape(3, 3))))
        for info in ad.geoms.values():
            worst = max(worst, np.max(np.abs(pd.oMf[info["frame"]].translation
                                             - d.geom_xpos[info["mj_geom"]])))
        # Inertial parameters: the mass matrix exercises every body's mass,
        # COM and inertia tensor (plus armature) at this configuration.
        mujoco.mj_forward(mj, d)
        M_mj = np.zeros((mj.nv, mj.nv))
        mujoco.mj_fullM(mj, d, M_mj)
        M_pin = ad.dynamics_mj(d.qpos, np.zeros(mj.nv), pd)[0]
        if pm.nv != mj.nv:      # compare over the planned dofs only
            sel = np.concatenate([np.arange(6), ad.mj_dadr])
            M_pin, M_mj = M_pin[np.ix_(sel, sel)], M_mj[np.ix_(sel, sel)]
        worst_M = max(worst_M, np.linalg.norm(M_pin - M_mj) / np.linalg.norm(M_mj))
    if not worst_M <= mass_tol:
        raise ValueError(f"adapter refused the model pair: mass matrices disagree by relative "
                         f"{worst_M:.3g} (> {mass_tol:g}); inertial parameters or armature differ")
    if not worst <= tol:
        raise ValueError(f"adapter refused the model pair: kinematics disagree by {worst:.3g} "
                         f"(> {tol:g}) at random poses; mapping or link geometry differs")
    ad.differences.append(f"build-time self-check: kinematics max error {worst:.2e}, "
                          f"mass matrix relative error {worst_M:.2e}")

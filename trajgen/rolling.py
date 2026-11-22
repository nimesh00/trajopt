"""A ROLLING stance contact for wheel feet: the axle may move along the wheel's heading, not across it.

A braked wheel planned as a point foot (adapter: contact at the axle centre, radius 0.086 m) is a
3D contact: the axle point is pinned. A wheel that may ROLL pins only two directions of that
point - across the wheel (the axle direction a) and vertically (z) - and leaves the rolling
direction t = z x a free, with no force along it (a free wheel carries no longitudinal force).

How it is built, and why this way. Crocoddyl 3.1's ContactModel2D constrains the frame's LOCAL x
and z; a frame rigid to the calf pitches with the leg, so its local z mixes vertical and rolling
motion as the knee moves (0.1 rad of leg pitch makes 10% of the rolling speed look like
vertical motion). ContactModel1D takes a rotation but is one row, and two contacts on one parent
joint corrupt fext in ContactModelMultiple (trajopt memory: 7.6e6 relative derivative error) -
one contact per joint. So this is one Python ContactModelAbstract that wraps a native
LOCAL_WORLD_ALIGNED ContactModel3D and keeps two of its rows through a CONSTANT projection

    P = [a^T; z^T]  (2 x 3),   a = (-sin psi, cos psi, 0),

psi the wheel's heading at that node (the knot posture's yaw; wheels point along the body x).
Then Jc = P Jc3, a0 = P a0_3 and da0/dx = P da0_3/dx exactly (P does not depend on x), and the
2-vector force lambda maps back as f3 = P^T lambda, so the native model's force update, fext
and LOCAL_WORLD_ALIGNED torque derivative (dtau_dq) are reused unchanged. With P = I it IS the
3D contact (tests/test_rolling.py: equal to rounding), and with two rows its derivatives match
finite differences.

The heading is held over a node rather than followed: in a node the body turns by wz dt
(0.02 rad at 1 rad/s), which tilts the constraint by that much - the same first-order
approximation every knot reference makes.
"""
from __future__ import annotations

import crocoddyl
import numpy as np
import pinocchio as pin


def projection(yaw: float) -> np.ndarray:
    """[axle; vertical] rows for a wheel heading `yaw` (world)."""
    return np.array([[-np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])


def rolling_direction(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw), np.sin(yaw), 0.0])


class ContactModelRolling(crocoddyl.ContactModelAbstract):
    """Rows [axle; vertical] of the frame point's acceleration, with two terms the constant-P
    projection alone lacks (Codex review 2026-10-15, items 2):

      axle row      a.(c'' + kd c') - psi' (t.c')     a = (-sin psi, cos psi, 0) turns at psi',
                                                       so d/dt (a.c') = a.c'' - psi' t.c'
      vertical row  z.(c'' + kd c') + kp (c_z - z_ref) Baumgarte on the normal GAP only; no
                                                       lateral spring (that would pin the
                                                       touchdown point, a holonomic constraint
                                                       the wheel does not have)

    psi' is the knot's reference yaw rate (constant over the node, as P is)."""

    def __init__(self, state, frame: int, P: np.ndarray, nu: int, gains=(0.0, 50.0),
                 yaw_rate: float = 0.0, z_ref: float = 0.0):
        P = np.asarray(P, float)
        crocoddyl.ContactModelAbstract.__init__(self, state, pin.LOCAL_WORLD_ALIGNED, P.shape[0], nu)
        self.id = frame
        self.P = P
        self.kp_gap = float(gains[0])
        self.kd = float(gains[1])
        self.yaw_rate = float(yaw_rate)
        self.z_ref = float(z_ref)
        self.t = np.array([P[0, 1], -P[0, 0], 0.0])        # rolling direction, a x z
        # The native model carries the velocity gain; position terms are added here.
        self.inner = crocoddyl.ContactModel3D(state, frame, np.zeros(3), pin.LOCAL_WORLD_ALIGNED, nu,
                                              np.array([0.0, self.kd]))

    def _extra(self):
        return bool(self.yaw_rate or self.kp_gap) and self.P.shape[0] == 2

    def createData(self, pinocchio_data):
        return ContactDataRolling(self, pinocchio_data)

    def calc(self, data, x):
        self.inner.calc(data.inner, x)
        data.Jc[:, :] = self.P @ data.inner.Jc
        a0 = self.P @ data.inner.a0
        if self._extra():
            model, pd = self.state.pinocchio, data.pinocchio
            v = pin.getFrameVelocity(model, pd, self.id, pin.LOCAL_WORLD_ALIGNED).linear
            a0[0] -= self.yaw_rate * float(self.t @ v)
            a0[1] += self.kp_gap * (float(pd.oMf[self.id].translation[2]) - self.z_ref)
        data.a0[:] = a0

    def calcDiff(self, data, x):
        # The native LOCAL_WORLD_ALIGNED derivative rebuilds its world drift from the FULL
        # frame acceleration at the current joint acceleration (contact-3d.hxx), so projecting
        # it is exact for the free row too (checked against finite differences of the drift).
        self.inner.calcDiff(data.inner, x)
        da0 = self.P @ data.inner.da0_dx
        if self._extra():
            # world velocity of the point, v_w = R v_l: d/dq = R (dv_l/dq - [v_l]x J_ang),
            # d/dv = R J_lin; world position: d/dq = R J_lin (the LWA linear Jacobian)
            model, pd = self.state.pinocchio, data.pinocchio
            nv = model.nv
            v_dq, v_dv = pin.getFrameVelocityDerivatives(model, pd, self.id, pin.LOCAL)
            v_l = pin.getFrameVelocity(model, pd, self.id, pin.LOCAL).linear
            J = pin.getFrameJacobian(model, pd, self.id, pin.LOCAL)
            R = pd.oMf[self.id].rotation
            dvw_dq = R @ (v_dq[:3] - pin.skew(v_l) @ J[3:])
            dvw_dv = R @ v_dv[:3]
            da0[0, :nv] -= self.yaw_rate * (self.t @ dvw_dq)
            da0[0, nv:] -= self.yaw_rate * (self.t @ dvw_dv)
            da0[1, :nv] += self.kp_gap * (R @ J[:3])[2]
        data.da0_dx[:, :] = da0

    def updateForce(self, data, force):
        data.lam = np.array(force, float)
        self.inner.updateForce(data.inner, self.P.T @ data.lam)
        data.f = data.inner.f
        data.fext = data.inner.fext
        # The force-rotation torque derivative is set HERE, as the native model does: the
        # contact dynamics sums it (ContactModelMultiple::updateRneaDiff) BEFORE it calls the
        # contacts' calcDiff, so copying it in calcDiff is one evaluation late.
        data.dtau_dq[:, :] = data.inner.dtau_dq


class ContactDataRolling(crocoddyl.ContactDataAbstract):
    def __init__(self, model, pinocchio_data):
        crocoddyl.ContactDataAbstract.__init__(self, model, pinocchio_data)
        self.inner = model.inner.createData(pinocchio_data)
        # ContactModelMultiple routes fext and dtau_dq by THIS frame's parent joint.
        self.frame = model.id
        self.lam = np.zeros(model.nc)
        self.jMf = self.inner.jMf
        self.fXj = self.inner.fXj


class ResidualRollingCone(crocoddyl.ResidualModelAbstract):
    """Friction and unilaterality of a rolling contact's 2-vector force lambda = (f_axle, f_z):

        r = [f_a - mu f_z,  -f_a - mu f_z,  -f_z]   (each <= 0, under a quadratic barrier)

    the lateral friction limit and fz >= 0. There is no rolling-direction force to bound."""

    def __init__(self, state, contact_name: str, mu: float, nu: int):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 3, nu, True, True, True)
        self.name = contact_name
        self.A = np.array([[1.0, -mu], [-1.0, -mu], [0.0, -1.0]])

    def createData(self, collector):
        data = ResidualRollingData(self, collector)
        return data

    def calc(self, data, x, u=None):
        data.r[:] = self.A @ data.contact.lam

    def calcDiff(self, data, x, u=None):
        data.Rx[:, :] = self.A @ data.contact.df_dx
        data.Ru[:, :] = self.A @ data.contact.df_du


class ResidualRollingForce(crocoddyl.ResidualModelAbstract):
    """The rolling contact's force 2-vector (f_axle, f_z) itself, for the tangential (lateral)
    penalty, the counterpart of croc.OCP's ResidualModelContactForce on the [fx, fy] rows."""

    def __init__(self, state, contact_name: str, nu: int):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 2, nu, True, True, True)
        self.name = contact_name

    def createData(self, collector):
        return ResidualRollingData(self, collector)

    def calc(self, data, x, u=None):
        data.r[:] = data.contact.lam

    def calcDiff(self, data, x, u=None):
        data.Rx[:, :] = data.contact.df_dx
        data.Ru[:, :] = data.contact.df_du


class ResidualRollingData(crocoddyl.ResidualDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.ResidualDataAbstract.__init__(self, model, collector)
        self.contact = collector.contacts.contacts[model.name]



class ImpulseModelRolling(crocoddyl.ImpulseModelAbstract):
    """The touchdown counterpart: the impulse removes the landing (and standing) wheel's
    velocity across its axle and vertically, and leaves its rolling velocity alone. The same
    wrapper as ContactModelRolling around a native LOCAL_WORLD_ALIGNED ImpulseModel3D."""

    def __init__(self, state, frame: int, P: np.ndarray):
        P = np.asarray(P, float)
        crocoddyl.ImpulseModelAbstract.__init__(self, state, pin.LOCAL_WORLD_ALIGNED, P.shape[0])
        self.id = frame
        self.P = P
        self.inner = crocoddyl.ImpulseModel3D(state, frame, pin.LOCAL_WORLD_ALIGNED)

    def createData(self, pinocchio_data):
        return ImpulseDataRolling(self, pinocchio_data)

    def calc(self, data, x):
        self.inner.calc(data.inner, x)
        data.Jc[:, :] = self.P @ data.inner.Jc

    def calcDiff(self, data, x):
        self.inner.calcDiff(data.inner, x)
        data.dv0_dq[:, :] = self.P @ data.inner.dv0_dq

    def updateForce(self, data, force):
        data.lam = np.array(force, float)
        self.inner.updateForce(data.inner, self.P.T @ data.lam)
        data.f = data.inner.f
        data.fext = data.inner.fext
        data.dtau_dq[:, :] = data.inner.dtau_dq


class ImpulseDataRolling(crocoddyl.ImpulseDataAbstract):
    def __init__(self, model, pinocchio_data):
        crocoddyl.ImpulseDataAbstract.__init__(self, model, pinocchio_data)
        self.inner = model.inner.createData(pinocchio_data)
        self.frame = model.id
        self.lam = np.zeros(model.nc)


class ResidualProjectedVelocity(crocoddyl.ResidualModelAbstract):
    """P (world linear velocity of a frame): the landing wheel's speed across its axle and
    vertically, leaving the rolling direction free (the touchdown cost of a rolling foot)."""

    def __init__(self, state, frame: int, P: np.ndarray, nu: int):
        crocoddyl.ResidualModelAbstract.__init__(self, state, P.shape[0], nu, True, True, False)
        self.P = np.asarray(P, float)
        self.inner = crocoddyl.ResidualModelFrameVelocity(state, frame, pin.Motion.Zero(),
                                                          pin.LOCAL_WORLD_ALIGNED, nu)

    def createData(self, collector):
        return _ProjectedData(self, collector)

    def calc(self, data, x, u=None):
        self.inner.calc(data.inner, x)
        data.r[:] = self.P @ data.inner.r[:3]

    def calcDiff(self, data, x, u=None):
        self.inner.calcDiff(data.inner, x)
        data.Rx[:, :] = self.P @ data.inner.Rx[:3]


class _ProjectedData(crocoddyl.ResidualDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.ResidualDataAbstract.__init__(self, model, collector)
        self.inner = model.inner.createData(collector)


def add_wheel_bottom_frames(ad, qpos) -> dict:
    """Per wheel, an OP_FRAME on the CALF at the wheel's lowest point at the stance `qpos`
    (axle centre - r z, upright wheel). Returns leg -> frame id.

    Why there (Codex review items 3, 5): fixed to the calf, the point's velocity is the
    axle's plus the calf's rotation about the axle, v_p = v_c + w x (-r z). So t.v_p / r is
    exactly the wheel hinge speed relative to the calf that rolls without slip
    (t.v_c / r - a.w_calf), the lateral row sees the wheel's camber rate (r w_t) and a
    lateral force acts AT THE GROUND (its moment about the axle included). The price: the
    point rides on the calf, so a calf pitch delta from the stance moves it r (1 - cos delta)
    off the true lowest point (1.7 mm at 0.2 rad), and a wheel tilt e from abduction or roll
    misplaces the support height by up to r (1 - cos e) + h sin e (3.2 mm at 10 deg).
    """
    data = ad.pin.createData()
    q = ad.q_from_mj(qpos)
    pin.framesForwardKinematics(ad.pin, data, q)
    frames = {}
    for leg, c in ad.feet.items():
        parent = ad.pin.frames[c.frame]
        joint = parent.parentJoint
        point = data.oMf[c.frame].translation - np.array([0.0, 0.0, c.radius])
        local = data.oMi[joint].actInv(point)
        name = f"{leg}_wheel_bottom"
        if ad.pin.existFrame(name):
            frames[leg] = ad.pin.getFrameId(name)
            continue
        frames[leg] = ad.pin.addFrame(pin.Frame(name, joint, parent.parentFrame,
                                                pin.SE3(np.eye(3), local), pin.FrameType.OP_FRAME))
    return frames

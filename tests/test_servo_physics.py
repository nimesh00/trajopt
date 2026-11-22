"""Small algebra/derivative regressions; no optimisation or trajectory rollout.

Remote: .venv/bin/python -m unittest discover -s tests -p test_servo_physics.py
"""
import types
import unittest

import crocoddyl
import numpy as np
import pinocchio as pin

from trajgen.dcmotor import envelope
from trajgen.servo import ServoActuation, TorqueReserveResidual


class ServoPhysicsTest(unittest.TestCase):
    def setUp(self):
        model = pin.Model()
        root = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root")
        model.addJoint(root, pin.JointModelRY(), pin.SE3.Identity(), "motor1")
        model.addJoint(root, pin.JointModelRY(), pin.SE3.Identity(), "motor2")
        self.state = crocoddyl.StateMultibody(model)
        self.motor = ServoActuation(self.state, 40., 2., [20., 20.], [30., 30.], [10., 10.], 2)
        self.x = np.r_[pin.neutral(model), np.zeros(model.nv)]

    def test_envelope_matches_plant_law_including_overspeed(self):
        v = np.linspace(-40., 40., 1001)
        up, lo, _, _ = envelope(v, 30., 10., 20.)
        limited_v = np.clip(v, -10. * (1. + 20./30.), 10. * (1. + 20./30.))
        np.testing.assert_allclose(up, np.minimum(30. * (1. - limited_v / 10.), 20.), atol=1e-12)
        np.testing.assert_allclose(lo, np.maximum(30. * (-1. - limited_v / 10.), -20.), atol=1e-12)
        self.assertTrue(np.all(lo <= up))
        self.assertLess(envelope(11., 30., 10., 20.)[0], 0.)

    def test_actuation_derivatives(self):
        a, st = self.motor, self.state
        eps = 1e-6
        for speed in (-30., -11., -5., 0., 5., 11., 30.):
            for demand in (-60., -4., 4., 60.):
                x = self.x.copy(); x[-1] = speed
                u = np.array([0., (demand + a.kd * speed) / a.kp])
                d = a.createData()
                a.calc(d, x, u); a.calcDiff(d, x, u)
                analytic = np.c_[d.dtau_dx, d.dtau_du]
                numeric = np.zeros_like(analytic)
                for j in range(st.ndx + 2):
                    dx = np.zeros(st.ndx); du = np.zeros(2)
                    if j < st.ndx:
                        dx[j] = eps
                    else:
                        du[j - st.ndx] = eps
                    dp, dm = a.createData(), a.createData()
                    a.calc(dp, st.integrate(x, dx), u + du)
                    a.calc(dm, st.integrate(x, -dx), u - du)
                    numeric[:, j] = (dp.tau - dm.tau) / (2. * eps)
                np.testing.assert_allclose(analytic, numeric, atol=1e-7)

    def test_reserve_derivatives_and_saturated_control_gradient(self):
        st = self.state
        r = TorqueReserveResidual(st, self.motor, .8)
        def data():
            return types.SimpleNamespace(r=np.zeros(4), Rx=np.zeros((4, st.ndx)), Ru=np.zeros((4, 2)))
        eps = 1e-6
        for speed in (-30., -11., -5., 0., 5., 11., 30.):
            for demand in (-60., -4., 4., 60.):
                x = self.x.copy(); x[-1] = speed
                u = np.array([0., (demand + 2. * speed) / 40.])
                d = data(); r.calc(d, x, u); r.calcDiff(d, x, u)
                analytic = np.c_[d.Rx, d.Ru]
                numeric = np.zeros_like(analytic)
                for j in range(st.ndx + 2):
                    dx = np.zeros(st.ndx); du = np.zeros(2)
                    if j < st.ndx:
                        dx[j] = eps
                    else:
                        du[j - st.ndx] = eps
                    dp, dm = data(), data()
                    r.calc(dp, st.integrate(x, dx), u + du)
                    r.calc(dm, st.integrate(x, -dx), u - du)
                    numeric[:, j] = (dp.r - dm.r) / (2. * eps)
                np.testing.assert_allclose(analytic, numeric, atol=1e-7)
                if abs(demand) == 60.:
                    self.assertGreater(np.linalg.norm(d.Ru), 0.)

    def test_coupled_actuation_and_reserve_derivatives(self):
        """A belt-coupled servo: Kp = kp J^T J with J = [[1, 0], [.5, 1]] (thigh, belt)."""
        st = self.state
        J = np.array([[1., 0.], [.5, 1.]])
        a = ServoActuation(st, 40., 2., [20., 20.], [30., 30.], [10., 10.], 2,
                           Kp=40. * J.T @ J, Kd=2. * J.T @ J)
        r = TorqueReserveResidual(st, a, .8)
        def rdata():
            return types.SimpleNamespace(r=np.zeros(4), Rx=np.zeros((4, st.ndx)), Ru=np.zeros((4, 2)))
        eps = 1e-6
        for speed in (-30., -5., 0., 5., 30.):
            for u in (np.array([.0, .02]), np.array([.3, -.4]), np.array([-1.5, 1.5])):
                x = self.x.copy(); x[-2:] = speed, -.5 * speed
                d = a.createData(); a.calc(d, x, u); a.calcDiff(d, x, u)
                rd = rdata(); r.calc(rd, x, u); r.calcDiff(rd, x, u)
                analytic = np.c_[d.dtau_dx, d.dtau_du]
                ranalytic = np.c_[rd.Rx, rd.Ru]
                numeric = np.zeros_like(analytic)
                rnumeric = np.zeros_like(ranalytic)
                for j in range(st.ndx + 2):
                    dx = np.zeros(st.ndx); du = np.zeros(2)
                    if j < st.ndx:
                        dx[j] = eps
                    else:
                        du[j - st.ndx] = eps
                    dp, dm = a.createData(), a.createData()
                    a.calc(dp, st.integrate(x, dx), u + du)
                    a.calc(dm, st.integrate(x, -dx), u - du)
                    numeric[:, j] = (dp.tau - dm.tau) / (2. * eps)
                    rp, rm = rdata(), rdata()
                    r.calc(rp, st.integrate(x, dx), u + du)
                    r.calc(rm, st.integrate(x, -dx), u - du)
                    rnumeric[:, j] = (rp.r - rm.r) / (2. * eps)
                np.testing.assert_allclose(analytic, numeric, atol=1e-6)
                np.testing.assert_allclose(ranalytic, rnumeric, atol=1e-6)
                # commands() inverts calc() wherever the servo law is unsaturated
                tau = np.asarray(d.tau)
                q_j, v_j = a._joint_state(x)
                if np.allclose(a.demand(q_j, v_j, u), tau[-2:], rtol=0, atol=1e-12):
                    cd = a.createData(); a.commands(cd, x, tau)
                    np.testing.assert_allclose(cd.u, u, atol=1e-12)

    def test_cone_normal_bound_and_outer_diagonal(self):
        outer = crocoddyl.FrictionCone(np.eye(3), .6, 4, False)
        inner = crocoddyl.FrictionCone(np.eye(3), .6, 4, True)
        np.testing.assert_allclose(outer.A[-1], [0., 0., 1.])
        self.assertEqual(outer.lb[-1], 0.)
        f = np.array([.55, .55, 1.])
        self.assertTrue(np.all(outer.A @ f <= outer.ub))
        self.assertTrue(np.any(inner.A @ f > inner.ub))


if __name__ == "__main__":
    unittest.main()

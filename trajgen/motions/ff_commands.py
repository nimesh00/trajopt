"""Feedforward servo commands for a kinematic clip: inverse dynamics instead of raw joint angles.

    .venv/bin/python -m trajgen.motions.ff_commands CLIP.states OUT.commands --config ROBOT.toml

WHY. The dog-clip tracking references used the clip's joint angles as the position commands. A
servo at the plant's kp 40 holding the robot's weight sags under them: the full-stride gallop policies crouched
to 0.19-0.21 m (clip 0.39) and learned nothing after 2M transitions, where the maneuver references,
whose commands were plant rollouts, trained. Here each frame's command is the one the servo law
tau = kp (u - q) - kd qdot needs to make the clip's own joint torques.

HOW. Per frame: the clip's acceleration by central differences of its velocities; the feet whose
sphere is within 4 mm of the floor stand (the track task's own contact rule); the minimum-norm
contact forces that balance the base rows of M a + h, then tau = (M a + h - Jc^T f) on the joints.
Friction cone and fz >= 0 are not enforced (a kinematic clip asks for some pulling); the servo
command is u = q + (tau + kd qdot) / kp, clipped to the joint range.
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np
import pinocchio as pin

from .. import adapter as am, config as cm

LEGS = ("FL", "FR", "RL", "RR")


def commands(clip: np.ndarray, ad, m: mujoco.MjModel, kp: float, kd: float, dt: float):
    nq = m.nq
    M = ad.pin
    data = M.createData()
    xs = [ad.x_from_mj(r[:nq], r[nq:]) for r in clip]
    qs = [x[:M.nq] for x in xs]
    vs = np.array([x[M.nq:] for x in xs])
    acc = np.gradient(vs, dt, axis=0)
    d = mujoco.MjData(m)
    gid = [int(ad.feet[leg].mj_geom) for leg in LEGS]
    jv = [M.joints[M.getJointId(m.joint(int(m.actuator_trnid[k, 0])).name)].idx_v for k in range(m.nu)]
    out, fz = [], []
    for k in range(len(clip) - 1):
        d.qpos[:] = clip[k, :nq]
        mujoco.mj_kinematics(m, d)
        down = [leg for leg, g in zip(LEGS, gid) if d.geom_xpos[g][2] - m.geom_size[g][0] < .004]
        q, v, a = qs[k], vs[k], acc[k]
        pin.computeAllTerms(M, data, q, v)
        rhs = data.M @ a + data.nle
        f = np.zeros(0)
        Jc = np.zeros((0, M.nv))
        if down:
            pin.updateFramePlacements(M, data)
            Jc = np.vstack([pin.getFrameJacobian(M, data, ad.feet[leg].frame, pin.LOCAL_WORLD_ALIGNED)[:3]
                            for leg in down])
            f = np.linalg.lstsq(Jc[:, :6].T, rhs[:6], rcond=None)[0]
        tau = rhs - Jc.T @ f
        u = np.array([clip[k, m.jnt_qposadr[m.actuator_trnid[i, 0]]] + (tau[jv[i]] + kd * v[jv[i]]) / kp
                      for i in range(m.nu)])
        lo, hi = m.jnt_range[m.actuator_trnid[:, 0], 0], m.jnt_range[m.actuator_trnid[:, 0], 1]
        out.append(np.clip(u, lo, hi))
        fz.append(float(sum(f[2::3])) if len(f) else 0.)
    return np.array(out), np.array(fz)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("out")
    ap.add_argument("--config", required=True, help="robot config, e.g. robots/dhav1.toml")
    ap.add_argument("--kp", type=float, default=None, help="servo gain (default: the config's kp)")
    ap.add_argument("--kd", type=float, default=None, help="servo damping (default: the config's kd)")
    a = ap.parse_args(argv)
    cfg = cm.load(a.config)
    m = mujoco.MjModel.from_binary_path(str(cfg.mjb))
    ad = am.build(cfg.mjb, cfg.urdf)
    clip = np.loadtxt(a.clip)
    kp = cfg.kp if a.kp is None else a.kp
    kd = cfg.kd if a.kd is None else a.kd
    u, fz = commands(clip, ad, m, kp, kd, cfg.control_dt)
    np.savetxt(a.out, u)
    weight = float(sum(m.body_mass)) * 9.81
    print(f"{a.out}: {len(u)} commands; |u - q| mean {np.abs(u - clip[:-1, 7:m.nq]).mean():.3f} rad, "
          f"vertical contact force mean {fz.mean() / weight:.2f} mg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

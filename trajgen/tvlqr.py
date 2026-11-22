"""Time-varying LQR about a reference, linearised on the MuJoCo plant itself.

Why this rather than the DDP's own Riccati gains:

  * Those gains come from the solver's model - one explicit Euler step of 20 ms - while
    the plant takes ten 2 ms steps. Measured, that mismatch is 0.015 rad per step at the
    command magnitudes a jump uses, and it compounds.
  * They are only stabilising if the solve CONVERGED. The backflip solve never does
    (stopping criterion 0.0043 to 0.17), and its gains actively destabilised execution:
    the command slammed between joint limits and open-loop replay beat closed loop.

Here the reference is taken as given - states and joint commands, whatever produced them -
and the feedback is rebuilt from finite differences of the true transition:

    x_{k+1} = F(x_k, u_k)      one control interval = `substeps` physics steps
    A_k = dF/dx,  B_k = dF/du  by central differences in the TANGENT space
    P_k from the backward Riccati recursion,  K_k = (R + B'PB)^-1 B'PA
    u_k = u_ref_k - K_k * (x - x_ref_k)

The state difference uses mj_differentiatePos, so the free joint's quaternion is handled
on the manifold rather than component-wise.
"""
from __future__ import annotations

import mujoco
import numpy as np


def state_diff(model: mujoco.MjModel, qpos_a, qvel_a, qpos_b, qvel_b) -> np.ndarray:
    """a - b in the tangent space: [dq(nv); dv(nv)]."""
    dq = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, dq, 1.0, np.asarray(qpos_b, float),
                               np.asarray(qpos_a, float))
    return np.concatenate([dq, np.asarray(qvel_a, float) - np.asarray(qvel_b, float)])


def state_add(model: mujoco.MjModel, qpos, qvel, dx):
    """Integrate a tangent perturbation onto a state."""
    q = np.array(qpos, float)
    mujoco.mj_integratePos(model, q, np.asarray(dx[: model.nv], float), 1.0)
    return q, np.asarray(qvel, float) + dx[model.nv:]


def _step(model, data, qpos, qvel, ctrl, substeps):
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl
    data.time = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(substeps):
        mujoco.mj_step(model, data)
    return data.qpos.copy(), data.qvel.copy()


def linearise(model: mujoco.MjModel, qpos, qvel, ctrl, substeps: int,
              eps_x: float = 1e-5, eps_u: float = 1e-5):
    """(A, B) of one control interval, by central differences in the tangent space."""
    nv, nu = model.nv, model.nu
    data = mujoco.MjData(model)
    q0, v0 = _step(model, data, qpos, qvel, ctrl, substeps)
    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, nu))
    for i in range(2 * nv):
        d = np.zeros(2 * nv); d[i] = eps_x
        qp, vp = state_add(model, qpos, qvel, d)
        qm, vm = state_add(model, qpos, qvel, -d)
        fp = _step(model, data, qp, vp, ctrl, substeps)
        fm = _step(model, data, qm, vm, ctrl, substeps)
        A[:, i] = state_diff(model, *fp, *fm) / (2 * eps_x)
    for j in range(nu):
        du = np.zeros(nu); du[j] = eps_u
        fp = _step(model, data, qpos, qvel, np.asarray(ctrl) + du, substeps)
        fm = _step(model, data, qpos, qvel, np.asarray(ctrl) - du, substeps)
        B[:, j] = state_diff(model, *fp, *fm) / (2 * eps_u)
    return A, B


def gains(model: mujoco.MjModel, qpos_ref, qvel_ref, ctrl_ref, substeps: int,
          Q: np.ndarray, R: np.ndarray, Qf: np.ndarray | None = None, progress=None):
    """K_k for every control interval of the reference, by backward Riccati."""
    steps = len(ctrl_ref)
    nv, nu = model.nv, model.nu
    As, Bs = [], []
    for k in range(steps):
        A, B = linearise(model, qpos_ref[k], qvel_ref[k], ctrl_ref[k], substeps)
        As.append(A); Bs.append(B)
        if progress and k % progress == 0:
            print("    linearised %d/%d" % (k, steps), flush=True)
    P = np.array(Qf if Qf is not None else Q, float)
    Ks = [None] * steps
    for k in range(steps - 1, -1, -1):
        A, B = As[k], Bs[k]
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
        P = .5 * (P + P.T)
        Ks[k] = K
    return Ks

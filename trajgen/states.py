"""Start states: settled falls and standing holds.

The fall generator reproduces the training getup reset exactly, so generated
references start from the same distribution the policy
will see. Diverging here would produce references for falls that never occur in
training.

The recipe: random unit quaternion, z = base_height + 0.6, joints at the
stand keyframe +- 0.1, command set to those joint targets, then settle by
stepping 0.5 s of physics.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .contract import Contract


@dataclass
class StartState:
    """A complete, reproducible starting point.

    `integration` is the full mjSTATE_INTEGRATION snapshot, which carries
    `qacc_warmstart` and the other fields a clip's qpos/qvel/ctrl omit. Measured
    consequence of restoring only qpos/qvel/ctrl: most episodes still replay to
    1e-10 m, but some jump to 5e-3 m at a single contact event and then plateau.
    The generator therefore keeps the snapshot and never round-trips a start
    state through the lossy clip fields.
    """

    qpos: np.ndarray
    qvel: np.ndarray
    command: np.ndarray      # the accumulated PD command in effect at t=0
    seed: int
    label: str
    integration: np.ndarray | None = None

    def upright(self) -> float:
        """World-z component of the torso z-axis; 1.0 is level, -1.0 inverted."""
        _, x, y, _ = self.qpos[3:7]
        return float(1.0 - 2.0 * (x * x + y * y))

    def height(self) -> float:
        return float(self.qpos[2])


def snapshot(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Full mjSTATE_INTEGRATION, including the solver warmstart."""
    state = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    return state


def restore(model: mujoco.MjModel, data: mujoco.MjData, state: np.ndarray) -> None:
    mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)


def settle(
    model: mujoco.MjModel,
    contract: Contract,
    qpos: np.ndarray,
    qvel: np.ndarray,
    command: np.ndarray,
    seconds: float,
    with_snapshot: bool = False,
):
    """Step physics with a fixed command until the state stops changing."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = command
    mujoco.mj_forward(model, data)
    for _ in range(int(round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
    q, v = np.array(data.qpos, float), np.array(data.qvel, float)
    if with_snapshot:
        return q, v, snapshot(model, data)
    return q, v


def random_fall(
    model: mujoco.MjModel,
    contract: Contract,
    base_height: float,
    seed: int,
    settle_seconds: float = 0.5,
) -> StartState:
    """One settled fall, matching the training getup reset distribution."""
    rng = np.random.default_rng(seed)
    qpos = np.zeros(model.nq)
    qvel = np.zeros(model.nv)

    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    qpos[3:7] = quaternion
    qpos[2] = base_height + 0.6

    joints = contract.nominal + rng.uniform(-0.1, 0.1, size=contract.nominal.size)
    joints = np.clip(joints, contract.joint_lower, contract.joint_upper)
    qpos[contract.qpos_adr] = joints

    # The trainer drives the drop with the joint targets themselves, so the legs are
    # held rather than limp while the robot lands.
    command = joints.copy()
    qpos, qvel, state = settle(
        model, contract, qpos, qvel, command, settle_seconds, with_snapshot=True
    )
    return StartState(
        qpos=qpos,
        qvel=qvel,
        command=command,
        seed=seed,
        label="random_fall",
        integration=state,
    )


def classify(state: StartState) -> str:
    """Which recovery family a fall belongs to.

    References do not generalize across these: a left-roll recovery is not a
    rotated right-roll one, so the library is keyed by family and they are never
    averaged together.
    """
    w, x, y, z = state.qpos[3:7]
    # Body y-axis expressed in world coordinates; its z component says which
    # side the robot is lying on.
    side_z = 2.0 * (y * z + w * x)
    up = state.upright()
    if up > 0.7:
        return "belly_down"
    if up < -0.7:
        return "back_down"
    return "left_side" if side_z > 0 else "right_side"


def standing(model: mujoco.MjModel, contract: Contract, settle_seconds: float = 1.0):
    """The settled standing pose under the servos.

    The stand keyframe is NOT the equilibrium: holding gravity with kp=40
    requires steady-state position error, so the pose the robot actually settles
    into differs from the keyframe. Terminal targets are measured from this, not
    assumed from the keyframe.
    """
    from .model import stand_keyframe

    qpos = np.array(model.key_qpos[stand_keyframe(model)], float)
    qvel = np.zeros(model.nv)
    command = qpos[contract.qpos_adr].copy()
    settled_q, settled_v = settle(model, contract, qpos, qvel, command, settle_seconds)
    return StartState(
        qpos=settled_q, qvel=settled_v, command=command, seed=0, label="standing"
    )

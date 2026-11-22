"""The trainer's action/command contract, replicated exactly.

Every number here mirrors the trainer. Getting any of it wrong produces references
that look fine in Python and are unreachable in training, so the contract is
asserted against the compiled model rather than trusted.

The command is STATE. The trainer's motion tasks integrate each action onto the
*previous* command and clamp to the joint range:

    u_k = clamp(u_{k-1} + action_scale * clip(a_k, -1, 1), jnt_lo, jnt_hi)

so a command sequence cannot be recovered from actions alone, and a reference
clip must carry both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The trainer's foot order; the trailing column of a clip's contact flags is the
# "some non-foot geom touched the ground" channel.
FOOT_ORDER = ("FL", "FR", "RL", "RR")
CONTACT_COLUMNS = FOOT_ORDER + ("other",)


@dataclass(frozen=True)
class Contract:
    """Resolved control contract for one task, read from the compiled model."""

    control_dt: float
    physics_timestep: float
    substeps: int
    kp: float
    kd: float
    action_scale: float
    joint_lower: np.ndarray          # [nu]
    joint_upper: np.ndarray          # [nu]
    actuator_names: tuple[str, ...]  # [nu]
    qpos_adr: np.ndarray             # [nu] qpos index per actuated joint
    dof_adr: np.ndarray              # [nu] qvel index per actuated joint
    nominal: np.ndarray              # [nu] stand keyframe joint pose

    def integrate(self, previous: np.ndarray, action: np.ndarray) -> np.ndarray:
        """One command update: clip the action, scale, accumulate, clamp."""
        step = self.action_scale * np.clip(np.asarray(action, float), -1.0, 1.0)
        return np.clip(previous + step, self.joint_lower, self.joint_upper)

    def action_for(self, previous: np.ndarray, command: np.ndarray) -> np.ndarray:
        """The action that realises `command` from `previous`, if reachable.

        Inverts the update above. The caller must verify reachability with
        `integrate`, because a command more than action_scale away in one step
        is simply not expressible and would be silently truncated.
        """
        return (np.asarray(command, float) - previous) / self.action_scale

    def reachable(self, previous: np.ndarray, command: np.ndarray) -> bool:
        action = self.action_for(previous, command)
        return bool(np.all(np.abs(action) <= 1.0 + 1e-9))

    def project(self, previous: np.ndarray, desired: np.ndarray) -> np.ndarray:
        """Nearest command the trainer can actually reach in one step.

        Optimizing over unreachable commands wastes the search on trajectories
        the consumer cannot execute, so candidates are projected before being
        evaluated, and the projected sequence is what gets scored.
        """
        return self.integrate(previous, self.action_for(previous, desired))


def assert_matches_config(contract: Contract, config: dict) -> None:
    """Fail loudly when the model and the YAML disagree."""
    expected = config["contract"]
    checks = (
        ("control_dt", contract.control_dt),
        ("physics_timestep", contract.physics_timestep),
        ("substeps", contract.substeps),
        ("kp", contract.kp),
        ("kd", contract.kd),
        ("action_scale", contract.action_scale),
    )
    for key, actual in checks:
        want = expected[key]
        if not np.isclose(actual, want, rtol=0, atol=1e-9):
            raise ValueError(f"contract mismatch: {key} model={actual!r} config={want!r}")
    if tuple(expected["foot_order"]) != FOOT_ORDER:
        raise ValueError(f"foot order must be {FOOT_ORDER}, got {expected['foot_order']}")
    if contract.control_dt != contract.physics_timestep * contract.substeps:
        raise ValueError("control_dt must equal physics_timestep * substeps")

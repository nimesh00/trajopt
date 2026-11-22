"""Offline reference-trajectory generation for RL motion tasks.

The one rule that matters most:
this pipeline plans against the trainer's COMPILED model (position servos, force
limits, task integrator), never the source MJCF, which has torque motors at a
different timestep.
"""
from __future__ import annotations

__all__ = ["contract", "model", "clips", "video"]

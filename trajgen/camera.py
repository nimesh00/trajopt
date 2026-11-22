"""Motion-following chase camera.

A chase camera that picks its pose from the robot's own vertical motion, so
the subject is never occluded by the terrain it is traversing.

For stair scenes it would switch between climb / descend / flat poses.
Getup is a single vertical transition over ~5 s on flat ground, so the state
machine collapses to two poses: GROUNDED (low and close, to show the limbs
working against the floor) and RISING (pulled back and higher, to show the
whole body come up). The mechanisms that make it stable are kept as-is:

  * low-pass on base vertical velocity, so per-step jitter does not flip it
  * a warm-up, because the spawn settling transient reads as a descent
  * a minimum dwell, so a thrashing robot cannot strobe the view
  * yaw-follow, so "behind-right" stays behind-right as the robot rolls
  * shortest-angular-path blending, so a swing never spins the long way round
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

VZ_TAU = 0.35      # seconds; rides through per-step oscillation
WARMUP = 0.40      # seconds; drops the settling transient at t=0
MIN_DWELL = 0.80   # seconds on one pose before another may be selected


class MotionChaseCam:
    """Tracking camera whose pose follows the robot's vertical motion."""

    def __init__(
        self,
        model: mujoco.MjModel,
        base_id: int,
        *,
        grounded=(125.0, -8.0, 1.5),
        rising=(145.0, -18.0, 2.2),
        vz_threshold: float = 0.08,
        blend_time: float = 0.6,
        yaw_follow: bool = True,
    ):
        self._base = base_id
        self._grounded = tuple(float(v) for v in grounded)
        self._rising = tuple(float(v) for v in rising)
        self._vz_thr = float(vz_threshold)
        self._blend = float(blend_time)
        self._yaw_follow = bool(yaw_follow)

        self._state = "grounded"
        self._vz_ema = 0.0
        self._t_prev: float | None = None
        self._t_flip = -MIN_DWELL
        self._az: float | None = None
        self._el = self._grounded[1]
        self._dist = self._grounded[2]
        self.flips = 0

        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = base_id

    @property
    def state(self) -> str:
        return self._state

    def update(self, data: mujoco.MjData, time: float) -> mujoco.MjvCamera:
        dt = 0.0 if self._t_prev is None else max(0.0, time - self._t_prev)
        self._t_prev = time

        # cvel[3:6] is the body's linear velocity in world axes.
        vz = float(data.cvel[self._base][5])
        if dt > 0.0:
            self._vz_ema += (1.0 - math.exp(-dt / VZ_TAU)) * (vz - self._vz_ema)
        else:
            self._vz_ema = vz

        want = "rising" if self._vz_ema > self._vz_thr else "grounded"
        can_flip = time >= WARMUP and (time - self._t_flip) >= MIN_DWELL
        if want != self._state and can_flip:
            self._state, self.flips, self._t_flip = want, self.flips + 1, time

        az_off, el_tgt, dist_tgt = (
            self._rising if self._state == "rising" else self._grounded
        )

        yaw_deg = 0.0
        if self._yaw_follow:
            rot = np.asarray(data.xmat[self._base]).reshape(3, 3)
            yaw_deg = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
        az_tgt = yaw_deg + az_off

        if self._az is None or self._blend <= 0.0 or dt <= 0.0:
            self._az, self._el, self._dist = az_tgt, el_tgt, dist_tgt
        else:
            b = 1.0 - math.exp(-dt / self._blend)
            # Shortest angular path, stable across the +-180 wrap.
            self._az += b * ((az_tgt - self._az + 180.0) % 360.0 - 180.0)
            self._el += b * (el_tgt - self._el)
            self._dist += b * (dist_tgt - self._dist)

        self.cam.azimuth = self._az
        self.cam.elevation = self._el
        self.cam.distance = self._dist
        return self.cam

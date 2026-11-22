"""Load one AI4Animation (MANN) dog capture and derive what gait analysis and retargeting need.

    c = load_clip("D1_009_KAN01_001")   # dict, see below

* markers: low-passed (4th-order Butterworth, 12 Hz, zero phase) 27-marker positions, Z up, metres.
* trunk: the robot-like trunk frame from the four LEG-ROOT joints (shoulder joints LeftArm /
  RightArm and hip joints LeftUpLeg / RightUpLeg) - origin at their centroid, x from the hip
  midpoint to the shoulder midpoint, z up. That is the part of the dog a quadruped robot's rigid
  base can copy; the spine flexion between the two girdles, the neck/head and the tail cannot.
* toes / roots: (T, 4, 3) in ROBOT leg order FL, FR, RL, RR (fore = "Arm"/"Hand", hind = "UpLeg"/"Foot").
* ground: per-foot floor level (the toe end-site marker does not sit at 0 in stance; front paws
  read about 2 cm, hind about 0.5 cm), contact: (T, 4) bool, from height AND horizontal speed with
  short-run cleanup.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.signal import butter, filtfilt

import bvh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# The unpacked MANN capture (51 BVH files, data/dog/README.md); MANN_DIR overrides it.
RAW = os.environ.get("MANN_DIR", os.path.join(REPO, "out", "assets", "ai4animation_mann", "extracted"))
LEGS = ("FL", "FR", "RL", "RR")
ROOT_IDS = [7, 12, 16, 20]   # LeftArm, RightArm, LeftUpLeg, RightUpLeg
TOE_IDS = [10, 15, 19, 23]   # LeftHand_end, RightHand_end, LeftFoot_end, RightFoot_end
PELVIS, NECK, HEAD_END, TAIL_END = 0, 3, 5, 26
CONTACT_HEIGHT = 0.025       # m above the foot's floor level
CONTACT_SPEED = 0.45         # m/s horizontal
MIN_RUN = 3                  # frames (50 ms at 60 Hz): shorter contact or swing runs are noise


def _clean_runs(c: np.ndarray, n: int) -> np.ndarray:
    c = c.copy()
    for value in (True, False):  # drop short contacts, then fill short swings
        k = 0
        while k < len(c):
            e = k
            while e < len(c) and c[e] == c[k]:
                e += 1
            if c[k] == value and e - k < n and k > 0 and e < len(c):
                c[k:e] = not value
            k = e
    return c


def trunk_frame(roots: np.ndarray):
    """roots (T, 4, 3) FL FR RL RR -> centre (T, 3), rotation (T, 3, 3) columns = x fwd, y left, z up."""
    centre = roots.mean(1)
    fwd = 0.5 * (roots[:, 0] + roots[:, 1]) - 0.5 * (roots[:, 2] + roots[:, 3])
    left = 0.5 * ((roots[:, 0] - roots[:, 1]) + (roots[:, 2] - roots[:, 3]))
    x = fwd / np.linalg.norm(fwd, axis=1, keepdims=True)
    z = np.cross(x, left)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    y = np.cross(z, x)
    return centre, np.stack([x, y, z], -1)


def euler_zyx(R: np.ndarray):
    """-> yaw, pitch, roll (MuJoCo/ROS convention: pitch > 0 is nose DOWN)."""
    yaw = np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    pitch = np.arcsin(np.clip(-R[:, 2, 0], -1, 1))
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    return yaw, pitch, roll


def load_clip(name: str, cutoff_hz: float = 12.0) -> dict:
    names, P, fps = bvh.load(os.path.join(RAW, name + ".bvh"))
    fps = round(fps)
    b, a = butter(4, cutoff_hz / (fps / 2))
    P = filtfilt(b, a, P, axis=0) if len(P) > 30 else P
    toes, roots = P[:, TOE_IDS], P[:, ROOT_IDS]
    vel = np.gradient(toes, 1.0 / fps, axis=0)
    speed = np.linalg.norm(vel[..., :2], axis=-1)
    ground = np.zeros(4)
    for f in range(4):
        slow = toes[speed[:, f] < 0.2, f, 2]
        ground[f] = np.percentile(slow if len(slow) > 10 else toes[:, f, 2], 20)
    height = toes[..., 2] - ground
    contact = (height < CONTACT_HEIGHT) & (speed < CONTACT_SPEED)
    contact = np.stack([_clean_runs(contact[:, f], MIN_RUN) for f in range(4)], 1)
    centre, R = trunk_frame(roots)
    yaw, pitch, roll = euler_zyx(R)
    floor = float(np.mean(ground))  # trunk height is measured from the mean foot floor level
    return dict(name=name, fps=fps, markers=P, marker_names=names, toes=toes, roots=roots,
                toe_height=height, toe_speed=speed, ground=ground, floor=floor, contact=contact,
                centre=centre, R=R, yaw=yaw, pitch=pitch, roll=roll)


def all_clips() -> list[str]:
    return sorted(f[:-4] for f in os.listdir(RAW) if f.endswith(".bvh"))

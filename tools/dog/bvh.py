"""Minimal BVH reader with vectorised forward kinematics (no external dependency).

    joints, pos, fps = load("clip.bvh")        # pos: (T, J, 3) metres, Z UP, includes End Sites

Built for the AI4Animation MANN dog capture (Zhang et al. 2018): centimetres, Y up, channel
order Zrotation Xrotation Yrotation, root with 6 channels. The marker list is the depth-first
joint order WITH end sites, named "<parent>_end" - the same 27-marker layout motion_imitation's
retarget_motion/data/*_joint_pos.txt uses (pelvis 0, neck 3, LeftShoulder 6, LeftArm 7,
LeftHand_end 10, ...), which makes the two directly comparable.
Axis change: BVH (x, y, z) -> world (x, -z, y), a proper rotation (det +1), so left stays left.
"""
from __future__ import annotations

import numpy as np


def _rot(axis: str, deg: np.ndarray) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    i = "XYZ".index(axis)
    j, k = (i + 1) % 3, (i + 2) % 3
    R[..., i, i] = 1
    R[..., j, j] = c
    R[..., k, k] = c
    R[..., j, k] = -s
    R[..., k, j] = s
    return R


def parse(path: str):
    tok = open(path).read().split()
    names, parents, offsets, channels = [], [], [], []
    stack, i = [], tok.index("HIERARCHY") + 1
    while tok[i] != "MOTION":
        t = tok[i]
        if t in ("ROOT", "JOINT"):
            names.append(tok[i + 1]); parents.append(stack[-1] if stack else -1)
            channels.append([]); offsets.append(None); i += 2
        elif t == "End":  # End Site
            names.append(names[stack[-1]] + "_end"); parents.append(stack[-1])
            channels.append([]); offsets.append(None); i += 2
        elif t == "{":
            stack.append(len(names) - 1); i += 1
        elif t == "}":
            stack.pop(); i += 1
        elif t == "OFFSET":
            offsets[stack[-1]] = [float(x) for x in tok[i + 1:i + 4]]
            i += 4
        elif t == "CHANNELS":
            n = int(tok[i + 1]); channels[stack[-1]] = tok[i + 2:i + 2 + n]; i += 2 + n
        else:
            raise ValueError(f"{path}: unexpected token {t!r}")
    nframes = int(tok[i + 2]); dt = float(tok[i + 5])
    data = np.array(tok[i + 6:], dtype=float)
    ncols = sum(len(c) for c in channels)
    data = data[: nframes * ncols].reshape(nframes, ncols)
    return names, np.array(parents), np.array(offsets, float), channels, data, dt


def load(path: str):
    """-> (marker names, positions (T, J, 3) in metres with Z up, frame rate)."""
    names, parents, offsets, channels, data, dt = parse(path)
    T, J = len(data), len(names)
    pos = np.zeros((T, J, 3)); rot = np.zeros((T, J, 3, 3))
    col = 0
    for j in range(J):
        ch = channels[j]
        local = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()
        p = None
        for c in ch:
            if c.endswith("position"):
                p = data[:, col] if p is None else p; col += 1
            else:
                local = local @ _rot(c[0], data[:, col]); col += 1
        if parents[j] < 0:
            k = [c for c in ch if c.endswith("position")]
            first = col - len(ch)
            pos[:, j] = np.stack([data[:, first + ch.index(a + "position")] for a in "XYZ"], -1) \
                if len(k) == 3 else offsets[j]
            rot[:, j] = local
        else:
            P = parents[j]
            pos[:, j] = pos[:, P] + np.einsum("tij,j->ti", rot[:, P], offsets[j])
            rot[:, j] = rot[:, P] @ local
    pos = pos / 100.0  # cm -> m
    world = np.stack([pos[..., 0], -pos[..., 2], pos[..., 1]], -1)
    return names, world, 1.0 / dt

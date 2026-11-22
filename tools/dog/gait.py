#!/usr/bin/env python3
"""Morphology-independent gait statistics of the MANN dog capture -> out/dog/gait_stats.json.

    .venv/bin/python tools/dog/gait.py [--table]

Stride = FL touchdown to the next FL touchdown. Per stride: duty factor per foot, footfall phase of
each foot after FL's touchdown (fraction of the stride), stride frequency, trunk speed, yaw rate,
Froude number v^2 / (g h), trunk heave and pitch peak-to-peak, and swing-foot clearance / h.
h is the dog's leg-root (shoulder + hip joint) height over the floor, the median over all walking
strides - the same quantity as a robot's hip height (base z).

Gait label from the footfalls (Hildebrand): L = fraction of the stride by which the left hind
touchdown precedes the left fore touchdown.
  asymmetric (the fore pair or the hind pair not ~half a stride apart) -> canter if a diagonal pair
  lands together, else gallop; symmetric: L < 0.1 or > 0.9 pace, 0.4..0.6 trot, else walk
  (lateral-sequence L < 0.5, diagonal-sequence otherwise). |yaw rate| > 0.6 rad/s adds "-turn".
Consecutive strides with the same label and speed within 30 % of their mean form a segment; a
segment needs 3 strides (2 for turns). Segments are what the retargeter uses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dog  # noqa: E402

G = 9.81
OUT = os.path.join(dog.REPO, "out", "dog")


def circ(a, b):
    d = (a - b) % 1.0
    return np.minimum(d, 1 - d)


def label(ph, yaw_rate):
    fr, rl, rr = ph[1], ph[2], ph[3]
    hind_split = (rr - rl) % 1.0
    if circ(fr, 0.5) > 0.15 or circ(hind_split, 0.5) > 0.15:
        sync = min(circ(rr, 0.0), circ(fr, rl))  # a diagonal pair landing together
        g = "canter" if sync < 0.1 else "gallop"
    else:
        L = (1.0 - rl) % 1.0
        if L < 0.1 or L > 0.9:
            g = "pace"
        elif 0.4 <= L <= 0.6:
            g = "trot"
        else:
            g = "walk"
    return g + ("-turn" if abs(yaw_rate) > 0.6 else "")


def strides(c: dict):
    fps, C = c["fps"], c["contact"]
    td = [np.flatnonzero(C[1:, f] & ~C[:-1, f]) + 1 for f in range(4)]
    out = []
    for a, b in zip(td[0][:-1], td[0][1:]):
        T = (b - a) / fps
        if not 0.2 <= T <= 1.5:
            continue
        ph = np.zeros(4)
        ok = True
        for f in range(1, 4):
            nxt = td[f][(td[f] >= a) & (td[f] < b)]
            if len(nxt) != 1:
                ok = False
                break
            ph[f] = (nxt[0] - a) / (b - a)
        if not ok:
            continue
        sl = slice(a, b)
        disp = c["centre"][b, :2] - c["centre"][a, :2]
        v = np.linalg.norm(disp) / T
        if v < 0.15:
            continue
        yaw_rate = (c["yaw"][b] - c["yaw"][a]) / T
        z = c["centre"][sl, 2] - c["floor"]
        clear = []
        for f in range(4):
            sw = ~C[sl, f]
            clear.append(float(c["toe_height"][sl, f][sw].max()) if sw.any() else 0.0)
        out.append(dict(a=int(a), b=int(b), T=T, v=v, yaw_rate=yaw_rate, phase=ph,
                        duty=C[sl].mean(0), z_mean=float(z.mean()), heave=float(np.ptp(z)),
                        pitch=float(np.ptp(c["pitch"][sl])), roll=float(np.ptp(c["roll"][sl])),
                        clear=np.array(clear), stride_len=float(np.linalg.norm(disp)),
                        label=label(ph, yaw_rate)))
    return out


def segments(st: list[dict], fps: int):
    segs, cur = [], []
    for s in st:
        if cur and (s["label"] == cur[-1]["label"] and s["a"] == cur[-1]["b"]
                    and abs(s["v"] - np.mean([x["v"] for x in cur])) < 0.3 * np.mean([x["v"] for x in cur])):
            cur.append(s)
        else:
            if cur:
                segs.append(cur)
            cur = [s]
    if cur:
        segs.append(cur)
    return [g for g in segs if len(g) >= (2 if g[0]["label"].endswith("turn") else 3)]


def circmean(x):
    return float(np.angle(np.mean(np.exp(2j * np.pi * np.asarray(x)))) / (2 * np.pi) % 1.0)


def summarise(seg: list[dict], h: float, clip: str, fps: int) -> dict:
    A = lambda k: np.array([s[k] for s in seg])  # noqa: E731
    v, T = A("v"), A("T")
    ph = A("phase")
    r = lambda x, n=3: round(float(x), n)  # noqa: E731
    return dict(
        clip=clip, gait=seg[0]["label"], frames=[seg[0]["a"], seg[-1]["b"]], fps=fps,
        seconds=r((seg[-1]["b"] - seg[0]["a"]) / fps, 2), strides=len(seg),
        duty_factor={leg: r(A("duty")[:, f].mean()) for f, leg in enumerate(dog.LEGS)},
        phase_after_FL={leg: r(circmean(ph[:, f])) for f, leg in enumerate(dog.LEGS)},
        lateral_L=r((1 - circmean(ph[:, 2])) % 1.0),
        stride_freq_hz=r(np.mean(1 / T)), stride_period_s=r(T.mean()),
        speed_mps=r(v.mean()), speed_std=r(v.std()), yaw_rate=r(A("yaw_rate").mean()),
        froude=r(np.mean(v ** 2) / (G * h)), speed_over_sqrt_gh=r(v.mean() / np.sqrt(G * h)),
        stride_len_over_h=r(A("stride_len").mean() / h),
        trunk_height_over_h=r(A("z_mean").mean() / h),
        heave_ptp_over_h=r(A("heave").mean() / h), heave_ptp_m=r(A("heave").mean(), 4),
        pitch_ptp_deg=r(np.degrees(A("pitch").mean()), 2), roll_ptp_deg=r(np.degrees(A("roll").mean()), 2),
        clearance_over_h={leg: r(A("clear")[:, f].mean() / h) for f, leg in enumerate(dog.LEGS)},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", action="store_true")
    a = p.parse_args()
    clips = {n: dog.load_clip(n) for n in dog.all_clips()}
    per = {n: strides(c) for n, c in clips.items()}
    walk_z = [s["z_mean"] for st in per.values() for s in st if s["label"] == "walk"]
    h = float(np.median(walk_z))
    segs = []
    for n, st in per.items():
        for g in segments(st, clips[n]["fps"]):
            segs.append(summarise(g, h, n, clips[n]["fps"]))
    total = sum(len(c["markers"]) / c["fps"] for c in clips.values())
    nstr = sum(len(s) for s in per.values())
    by = {}
    for s in segs:
        by.setdefault(s["gait"], []).append(s)
    agg = {}
    for g, ss in sorted(by.items()):
        w = np.array([s["strides"] for s in ss], float)
        m = lambda k: round(float(np.average([s[k] for s in ss], weights=w)), 3)  # noqa: E731
        agg[g] = dict(segments=len(ss), strides=int(w.sum()), seconds=round(sum(s["seconds"] for s in ss), 1),
                      duty_factor=round(float(np.average([np.mean(list(s["duty_factor"].values())) for s in ss], weights=w)), 3),
                      lateral_L=round(float(np.angle(np.sum(w * np.exp(2j * np.pi * np.array([s["lateral_L"] for s in ss]))))
                                            / (2 * np.pi) % 1.0), 3), stride_freq_hz=m("stride_freq_hz"), speed_mps=m("speed_mps"),
                      speed_range=[min(s["speed_mps"] for s in ss), max(s["speed_mps"] for s in ss)],
                      froude=m("froude"), froude_range=[min(s["froude"] for s in ss), max(s["froude"] for s in ss)],
                      heave_ptp_over_h=m("heave_ptp_over_h"), pitch_ptp_deg=m("pitch_ptp_deg"),
                      trunk_height_over_h=m("trunk_height_over_h"),
                      clearance_fore_over_h=round(float(np.average([(s["clearance_over_h"]["FL"] + s["clearance_over_h"]["FR"]) / 2 for s in ss], weights=w)), 3),
                      clearance_hind_over_h=round(float(np.average([(s["clearance_over_h"]["RL"] + s["clearance_over_h"]["RR"]) / 2 for s in ss], weights=w)), 3))
    out = dict(
        source="AI4Animation MANN dog capture (Zhang et al. 2018), 51 BVH clips, 60 Hz, CC BY-NC 4.0",
        generator="tools/dog/gait.py", total_seconds=round(total, 1), strides_detected=nstr,
        hip_height_m=round(h, 4),
        hip_height_definition="median leg-root (shoulder joint + hip joint centroid) height above the foot floor level over walking strides",
        contact_rule=f"toe end-site within {dog.CONTACT_HEIGHT} m of its floor level AND horizontal speed < {dog.CONTACT_SPEED} m/s; runs < {dog.MIN_RUN} frames removed",
        no_robot_analog=["spine flexion/extension between the shoulder and hip girdles (large in canter/gallop)",
                         "head and neck motion", "tail", "scapula sliding (shoulder joint moves relative to the trunk)",
                         "paw/digit and carpus/hock segments (dog legs have 4 segments, robot legs 2 + a point foot)",
                         "hind-leg knee-forward / hock-back geometry (robots use one knee direction)"],
        by_gait=agg, segments=segs)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gait_stats.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"{len(clips)} clips, {total / 60:.1f} min, {nstr} strides, {len(segs)} steady segments, h = {h:.3f} m")
    hdr = f"{'gait':12s} {'seg':>3s} {'str':>4s} {'sec':>5s} {'duty':>5s} {'L':>5s} {'f Hz':>5s} {'v m/s':>6s} {'Fr':>5s} {'heave/h':>7s} {'pitch°':>6s} {'z/h':>5s} {'clrF/h':>6s} {'clrH/h':>6s}"
    lines = [hdr]
    for g, s in agg.items():
        lines.append(f"{g:12s} {s['segments']:3d} {s['strides']:4d} {s['seconds']:5.1f} {s['duty_factor']:5.2f} {s['lateral_L']:5.2f} "
                     f"{s['stride_freq_hz']:5.2f} {s['speed_mps']:6.2f} {s['froude']:5.2f} {s['heave_ptp_over_h']:7.3f} "
                     f"{s['pitch_ptp_deg']:6.1f} {s['trunk_height_over_h']:5.2f} {s['clearance_fore_over_h']:6.3f} {s['clearance_hind_over_h']:6.3f}")
    table = "\n".join(lines)
    with open(os.path.join(OUT, "gait_stats.txt"), "w") as f:
        f.write(f"h = {h:.3f} m; per-gait stride-weighted means over steady segments (gait_stats.json has every segment)\n" + table + "\n")
    print(table)
    if a.table:
        for s in segs:
            print(f"{s['clip']:20s} {s['gait']:12s} {s['frames']} n{s['strides']} v{s['speed_mps']:.2f} f{s['stride_freq_hz']:.2f} "
                  f"duty{np.mean(list(s['duty_factor'].values())):.2f} ph{list(s['phase_after_FL'].values())} yaw{s['yaw_rate']:+.2f}")


if __name__ == "__main__":
    main()

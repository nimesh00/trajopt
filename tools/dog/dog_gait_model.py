#!/usr/bin/env python3
"""Fit the DOG GAIT MODEL: per-gait timing, foot paths and rigid-trunk waveforms from the MANN dog,
in dynamic-similarity units, for trajgen's dog-guided walk (trajgen/dogstyle.py).

    .venv/bin/python tools/dog/dog_gait_model.py [--out data/dog/dog-gait-model.json]

Data: every stride of every MANN capture (tools/dog/gait.py: FL touchdown to the next FL
touchdown, each other foot touching down exactly once, dog contact = toe within 2.5 cm of its floor
and slower than 0.45 m/s). NO MIRRORING anywhere: left and right turns and both gallop leads are
fitted from their own strides.

Steady strides: a stride is kept when a CONTIGUOUS neighbour (the previous or next stride) has the
same family label and a speed within 30 % (gait.py's segment rule over a pair; gait.py's 3-stride
segments keep only 7 straight walk strides of 41 minutes). Gallop and canter strides additionally need
a mean duty < 0.45 and Froude speed > 0.5 (slow asymmetric shuffles are labelled gallop too).

Families (labels from gait.py's Hildebrand rule):
    walk      walk + walk-turn              lateral-sequence walk
    pace      pace + pace-turn              lateral pairs
    trot      trot + trot-turn              diagonal pairs (the dog trots only fast, Fr 0.8-1.5)
    gallop-R  gallop with the fore lead FR  (lead = the fore foot landing second)
    gallop-L  gallop with the fore lead FL
    canter-R / canter-L
Per family, units: h = dog leg-root height 0.3446 m, s = v / sqrt(g h), f* = f sqrt(h / g),
w* = yaw rate sqrt(h / g); lengths / h.
    f*(s)                 Theil-Sen line over the straight strides (|w*| < 0.112, gait.py's 0.6 rad/s)
    duty_leg(s)           Theil-Sen line per leg (FL FR RL RR), straight strides
    offsets               circular median of each foot's touchdown phase after FL's, straight strides
    swing path (per leg)  over the swing (lift-off -> touchdown), 21 progress bins, medians:
                            along  = fraction of the lift-off -> touchdown horizontal chord covered
                            across = lateral deviation from the chord / h
                            up     = toe height over its floor / h, normalized to the swing APEX
                          and apex(s) = Theil-Sen line of the swing apex / h
    hip-relative path     foot minus its leg root in the heading frame, / h, 50 stride-phase bins
                          (reporting), and touchdown / lift-off fore-aft position vs the stance
                          travel: x_td / h = a + b L_st / h, L_st = v duty / f (Theil-Sen), per leg
    trunk waveform        heave (leg-root centroid height / h), pitch, roll (rad; the leg-root trunk
                          frame, pitch > 0 nose DOWN, roll > 0 left side up), sway and surge (the
                          centroid's lateral / fore-aft offset from the stride's straight chord in
                          the heading frame, / h, > 0 left / forward) per stride, mean
                          removed, harmonics 1..3 of the stride-phase waveform (phase 0 = FL
                          touchdown): complex coefficients, Theil-Sen line in s per real/imag part
                          (families with >= 30 straight strides), else the median
    turns                 left (w* > +0.112) and right (< -0.112) strides separately: offsets,
                          duty, f* residual against the straight line, trunk harmonics, mean roll
                          (lean) against the centripetal s w*; the per-leg stride-length ratio
                          outer / inner against the rigid-body prediction (v + w x r_hip) T
Coverage: strides per family and turn direction, Froude speed and yaw rate (5-95 %), in units of a robot
with leg-root height H0.

Dog data: AI4Animation MANN, CC BY-NC 4.0, non-commercial; the model is a derived work under the same
licence (data/dog/README.md).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dog  # noqa: E402
import gait  # noqa: E402

G = 9.81
H_DOG = 0.3446                 # out/dog/gait_stats.json hip_height_m (tools/dog/gait.py)
H0 = 0.4000356                 # m, reference robot leg-root height: the report's units and the model's default h0
TURN = 0.6 * np.sqrt(H_DOG / G)  # gait.py's |yaw rate| > 0.6 rad/s, dimensionless
BINS_SWING, BINS_PHASE, HARM = 21, 50, 3
OUT = os.path.join(dog.REPO, "data", "dog", "dog-gait-model.json")
LEGS = dog.LEGS


def theil_sen(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) < 1e-6:
        return float(np.median(y)), 0.0
    i, j = np.triu_indices(len(x), 1)
    ok = np.abs(x[j] - x[i]) > 1e-6 * max(1.0, np.ptp(x))
    slope = float(np.median((y[j] - y[i])[ok] / (x[j] - x[i])[ok]))
    return float(np.median(y - slope * x)), slope


def circmedian(ph):
    ph = np.asarray(ph) % 1.0
    d = lambda a: np.minimum(np.abs(ph - a) % 1.0, 1 - np.abs(ph - a) % 1.0).sum()  # noqa: E731
    return float(min(ph, key=d))


def circ(a, b=0.0):
    d = (np.asarray(a) - b) % 1.0
    return np.minimum(d, 1 - d)


def family(st):
    lab = st["label"]
    base = lab.removesuffix("-turn")
    if base in ("gallop", "canter"):
        lead = "R" if st["phase"][1] < 0.5 else "L"   # style/mann.py label_window's rule
        return f"{base}-{lead}"
    return base


def resample(y, n):
    """(T, ...) samples over [0, 1] inclusive -> n bins at 0, 1/(n-1), ..., 1."""
    t = np.linspace(0, 1, len(y))
    u = np.linspace(0, 1, n)
    y = np.asarray(y, float)
    return np.stack([np.interp(u, t, y[:, k]) for k in range(y.shape[1])], 1) if y.ndim == 2 else np.interp(u, t, y)


def periodic(y, n):
    """samples over one period [0, 1) (T samples at k/T) -> n bins at k/n."""
    T = len(y)
    t = np.arange(T + 1) / T
    yy = np.r_[y, y[:1]]
    return np.interp(np.arange(n) / n, t, yy)


def harmonics(y):
    c = np.fft.rfft(y) / len(y) * 2
    return c[1:HARM + 1]


def stride_records(name):
    c = dog.load_clip(name)
    fps, C = c["fps"], c["contact"]
    st = gait.strides(c)
    lo_all = [np.flatnonzero(~C[1:, f] & C[:-1, f]) + 1 for f in range(4)]
    td_all = [np.flatnonzero(C[1:, f] & ~C[:-1, f]) + 1 for f in range(4)]
    out = []
    for s in st:
        a, b = s["a"], s["b"]
        T = s["T"]
        yaw_a, yaw_b = c["yaw"][a], c["yaw"][b]
        yaw_m = 0.5 * (yaw_a + yaw_b)
        hd = np.array([np.cos(yaw_m), np.sin(yaw_m)])
        nl = np.array([-hd[1], hd[0]])
        disp = c["centre"][b, :2] - c["centre"][a, :2]
        vx, vy = float(disp @ hd / T), float(disp @ nl / T)
        rec = dict(clip=name, a=int(a), b=int(b), T=T, f=1 / T, v=s["v"], vx=vx, vy=vy, wz=s["yaw_rate"],
                   label=s["label"], duty=s["duty"].tolist(), phase=s["phase"].tolist())
        rec["family"] = family(s)
        # trunk waveform: leg-root centroid height, pitch, roll over the stride (b excluded), mean removed
        sl = slice(a, b)
        z = (c["centre"][sl, 2] - c["floor"]) / H_DOG
        wave = {}
        # sway / surge: the leg-root centroid off the stride's straight chord, heading frame, / h
        tt = (np.arange(a, b) - a) / (b - a)
        off = c["centre"][sl, :2] - c["centre"][a, :2] - tt[:, None] * disp
        sway, surge = off @ nl / H_DOG, off @ hd / H_DOG
        for key, y in (("heave", z), ("pitch", c["pitch"][sl]), ("roll", c["roll"][sl]),
                       ("sway", sway), ("surge", surge)):
            y = np.asarray(y, float)
            wave[key] = periodic(y - y.mean(), BINS_PHASE)
        rec["wave"] = {k: v.tolist() for k, v in wave.items()}
        rec["roll_mean"] = float(np.mean(c["roll"][sl]))
        rec["pitch_mean"] = float(np.mean(c["pitch"][sl]))
        rec["z_mean"] = float(z.mean())
        # hip-relative foot path in the heading frame (per frame yaw), stride phase bins
        yaw = c["yaw"][sl]
        cy, sy = np.cos(yaw), np.sin(yaw)
        rel = c["toes"][sl] - c["roots"][sl]
        rx = (cy[:, None] * rel[..., 0] + sy[:, None] * rel[..., 1]) / H_DOG
        ry = (-sy[:, None] * rel[..., 0] + cy[:, None] * rel[..., 1]) / H_DOG
        rz = c["toe_height"][sl] / H_DOG
        rec["hip_rel"] = {k: [periodic(arr[:, f], BINS_PHASE).tolist() for f in range(4)]
                          for k, arr in (("x", rx), ("y", ry), ("z", rz))}
        # per leg: the swing ending at this leg's touchdown inside [a, b] (FL: at b) and its stance
        legs = []
        ok = True
        for f in range(4):
            if f == 0:
                td = b
            else:
                cand = td_all[f][(td_all[f] >= a) & (td_all[f] < b)]
                td = int(cand[0])
            prev_lo = lo_all[f][lo_all[f] < td]
            prev_td = td_all[f][td_all[f] < td]
            if not len(prev_lo) or not len(prev_td) or prev_lo[-1] <= prev_td[-1]:
                ok = False
                break
            lo, td0 = int(prev_lo[-1]), int(prev_td[-1])   # the stance td0..lo, then the swing lo..td
            P = c["toes"][lo:td + 1, f]
            chord = P[-1, :2] - P[0, :2]
            L = float(np.linalg.norm(chord))
            if L < 1e-3:
                ok = False
                break
            u = chord / L
            n = np.array([-u[1], u[0]])
            along = (P[:, :2] - P[0, :2]) @ u / L
            across = (P[:, :2] - P[0, :2]) @ n / H_DOG
            up = c["toe_height"][lo:td + 1, f] / H_DOG
            apex = float(up.max())
            # hip-relative fore-aft at touchdown (td) and lift-off (lo), heading frame at that frame
            def hx(t):
                d = c["toes"][t, f, :2] - c["roots"][t, f, :2]
                return float((np.cos(c["yaw"][t]) * d[0] + np.sin(c["yaw"][t]) * d[1]) / H_DOG)
            legs.append(dict(swing_frac=(td - lo) / fps / T, stance_frac=(lo - td0) / fps / T,
                             along=resample(along, BINS_SWING).tolist(),
                             across=resample(across, BINS_SWING).tolist(),
                             up=(resample(up, BINS_SWING) / max(apex, 1e-6)).tolist(), apex=apex,
                             chord=L / H_DOG, x_td=hx(td), x_lo=hx(lo), x_td0=hx(td0)))
        if not ok:
            continue
        rec["legs"] = legs
        out.append(rec)
    return out


def steady(records):
    keep = []
    by_clip = collections.defaultdict(list)
    for r in records:
        by_clip[r["clip"]].append(r)
    for rs in by_clip.values():
        rs.sort(key=lambda r: r["a"])
        for i, r in enumerate(rs):
            nb = [rs[j] for j in (i - 1, i + 1) if 0 <= j < len(rs)]
            nb = [x for x in nb if (x["b"] == r["a"] or x["a"] == r["b"]) and x["family"] == r["family"]
                  and abs(x["v"] - r["v"]) < 0.3 * 0.5 * (x["v"] + r["v"])]
            if not nb:
                continue
            if r["family"].startswith(("gallop", "canter")):
                if np.mean(r["duty"]) >= 0.45 or r["v"] / np.sqrt(G * H_DOG) <= 0.5:
                    continue
            keep.append(r)
    return keep


def line_fit(xs, ys):
    a, b = theil_sen(xs, ys)
    resid = np.asarray(ys) - (a + b * np.asarray(xs))
    return dict(a=a, b=b, mad=float(np.median(np.abs(resid))))


def fit_group(rs, with_lines=True):
    """Timing, swing paths, footholds and trunk harmonics of one group of strides."""
    s = np.array([r["v"] for r in rs]) / np.sqrt(G * H_DOG)
    fs = np.array([r["f"] for r in rs]) * np.sqrt(H_DOG / G)
    duty = np.array([r["duty"] for r in rs])
    ph = np.array([r["phase"] for r in rs])
    out = dict(strides=len(rs), clips=len({r["clip"] for r in rs}),
               s_pct=[float(np.percentile(s, 5)), float(np.median(s)), float(np.percentile(s, 95))],
               wz_pct=[float(np.percentile([r["wz"] for r in rs], q) * np.sqrt(H_DOG / G)) for q in (5, 50, 95)],
               f_star=line_fit(s, fs) if with_lines else dict(a=float(np.median(fs)), b=0.0, mad=0.0),
               f_star_median=float(np.median(fs)),
               duty=[line_fit(s, duty[:, k]) if with_lines else dict(a=float(np.median(duty[:, k])), b=0.0, mad=0.0)
                     for k in range(4)],
               duty_median=[float(np.median(duty[:, k])) for k in range(4)],
               offsets=[0.0] + [circmedian(ph[:, k]) for k in range(1, 4)],
               offsets_spread=[0.0] + [float(np.median(circ(ph[:, k], circmedian(ph[:, k])))) for k in range(1, 4)])
    legs = []
    for f in range(4):
        L = [r["legs"][f] for r in rs]
        Lst = np.array([r["v"] * r["duty"][f] / r["f"] for r in rs]) / H_DOG      # stance travel / h
        legs.append(dict(
            along=np.median([x["along"] for x in L], 0).tolist(),
            across=np.median([x["across"] for x in L], 0).tolist(),
            up=np.median([x["up"] for x in L], 0).tolist(),
            apex=line_fit(s, [x["apex"] for x in L]) if with_lines else dict(a=float(np.median([x["apex"] for x in L])), b=0.0, mad=0.0),
            apex_median=float(np.median([x["apex"] for x in L])),
            x_td=line_fit(Lst, [x["x_td"] for x in L]), x_lo=line_fit(Lst, [x["x_lo"] for x in L]),
            x_td_median=float(np.median([x["x_td"] for x in L])), x_lo_median=float(np.median([x["x_lo"] for x in L])),
            hip_rel_x=np.median([r["hip_rel"]["x"][f] for r in rs], 0).tolist(),
            hip_rel_z=np.median([r["hip_rel"]["z"][f] for r in rs], 0).tolist()))
    out["legs"] = legs
    trunk = {}
    for key in ("heave", "pitch", "roll", "sway", "surge"):
        H = np.array([harmonics(np.array(r["wave"][key])) for r in rs])          # n, HARM complex
        coef = []
        for k in range(HARM):
            if with_lines and len(rs) >= 30:
                re, im = line_fit(s, H[:, k].real), line_fit(s, H[:, k].imag)
            else:
                re = dict(a=float(np.median(H[:, k].real)), b=0.0, mad=0.0)
                im = dict(a=float(np.median(H[:, k].imag)), b=0.0, mad=0.0)
            coef.append(dict(re=re, im=im))
        W = np.array([r["wave"][key] for r in rs])
        trunk[key] = dict(harmonics=coef, ptp_median=float(np.median(np.ptp(W, 1))),
                          wave_median=np.median(W, 0).tolist())
    out["trunk"] = trunk
    out["roll_mean"] = float(np.median([r["roll_mean"] for r in rs]))
    out["z_mean"] = float(np.median([r["z_mean"] for r in rs]))
    return out


def turn_geometry(rs):
    """Outer / inner stride-length ratio (each foot's lift-off -> touchdown chord, fore and hind
    pooled) against the rigid-body prediction |v + w x r| for hips at +-half the dog's track."""
    meas, pred = [], []
    half_track = 0.5 * 0.14 / H_DOG     # dog leg-root track ~0.14 m (shoulder/hip width)
    for r in rs:
        left = np.mean([r["legs"][0]["chord"], r["legs"][2]["chord"]])
        right = np.mean([r["legs"][1]["chord"], r["legs"][3]["chord"]])
        s = r["v"] / np.sqrt(G * H_DOG)
        w = r["wz"] * np.sqrt(H_DOG / G)
        outer, inner = (right, left) if w > 0 else (left, right)
        meas.append(outer / max(inner, 1e-6))
        pred.append((s + abs(w) * half_track) / max(s - abs(w) * half_track, 1e-3))
    return dict(ratio_measured=float(np.median(meas)), ratio_rigid_body=float(np.median(pred)), n=len(rs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--clip", default=None, metavar="NAME",
                    help="fit ONE capture only (a single-clip model, e.g. the retargeted pace's own strides)")
    ap.add_argument("--frames", type=int, nargs=2, default=None, metavar=("A", "B"),
                    help="with --clip: only strides inside these capture frames")
    a = ap.parse_args()
    allr = []
    for n in ([a.clip] if a.clip else dog.all_clips()):
        allr += stride_records(n)
    if a.frames:
        allr = [r for r in allr if r["a"] >= a.frames[0] and r["b"] <= a.frames[1]]
    st = steady(allr)
    fams = collections.defaultdict(list)
    for r in st:
        fams[r["family"]].append(r)
    model = dict(version=1, generator="tools/dog/dog_gait_model.py",
                 source="AI4Animation MANN dog capture (Zhang et al. 2018), CC BY-NC 4.0, research only",
                 h_dog=H_DOG, h0_robot=H0, g=G, turn_threshold=TURN, bins_swing=BINS_SWING,
                 bins_phase=BINS_PHASE, legs=list(LEGS), strides_all=len(allr), strides_steady=len(st),
                 conventions=dict(pitch="rad, > 0 nose DOWN (leg-root trunk frame)", roll="rad, > 0 left side up",
                                  heave="leg-root centroid height / h", offsets="touchdown phase after FL's touchdown",
                                  harmonics="y(phi) = sum_k Re(c_k exp(2 pi i k phi)), phi = stride phase from FL touchdown",
                                  lines="value = a + b x, x = Froude speed s (timing, apex, trunk) or stance travel / h (x_td, x_lo)"),
                 families={})
    lines = []
    lg = lambda s="": (print(s), lines.append(s))  # noqa: E731
    v0r, t0r = np.sqrt(G * H0), np.sqrt(H0 / G)
    lg(f"MANN dog: {len(allr)} strides, {len(st)} steady (contiguous same-family neighbour within 30 % speed)")
    for fam, rs in sorted(fams.items(), key=lambda x: -len(x[1])):
        straight = [r for r in rs if abs(r["wz"]) * np.sqrt(H_DOG / G) < TURN]
        left = [r for r in rs if r["wz"] * np.sqrt(H_DOG / G) >= TURN]
        right = [r for r in rs if r["wz"] * np.sqrt(H_DOG / G) <= -TURN]
        if len(straight) < 5:
            lg(f"{fam:9s} {len(rs)} steady strides, only {len(straight)} straight: skipped")
            continue
        entry = dict(straight=fit_group(straight))
        for side, grp in (("left", left), ("right", right)):
            if len(grp) >= 5:
                g = fit_group(grp, with_lines=False)
                sg = np.array([r["v"] for r in grp]) / np.sqrt(G * H_DOG)
                fl = entry["straight"]["f_star"]
                resid = np.array([r["f"] for r in grp]) * np.sqrt(H_DOG / G) - (fl["a"] + fl["b"] * sg)
                g["f_star_resid_median"] = float(np.median(resid))
                g["duty_resid_median"] = [float(np.median(np.array([r["duty"][k] for r in grp])
                                                          - (entry["straight"]["duty"][k]["a"] + entry["straight"]["duty"][k]["b"] * sg)))
                                          for k in range(4)]
                g["geometry"] = turn_geometry(grp)
                entry[side] = g
        # banking: mean roll against the centripetal acceleration s w* over all strides of the family
        cen = np.array([r["v"] / np.sqrt(G * H_DOG) * r["wz"] * np.sqrt(H_DOG / G) for r in rs])
        entry["lean"] = line_fit(cen, [r["roll_mean"] for r in rs])
        entry["lean"]["corr"] = float(np.corrcoef(cen, [r["roll_mean"] for r in rs])[0, 1]) if np.ptp(cen) > 0 else 0.0
        model["families"][fam] = entry
        S = entry["straight"]
        lo, md, hi = S["s_pct"]
        f = lambda s_: (S["f_star"]["a"] + S["f_star"]["b"] * s_) / t0r  # noqa: E731
        lg(f"\n{fam}: {len(rs)} steady strides ({len(straight)} straight, {len(left)} left, {len(right)} right) "
           f"from {S['clips']} captures; speed at H0 {lo * v0r:.2f}..{hi * v0r:.2f} m/s (median {md * v0r:.2f})")
        lg(f"  cadence f = {f(lo):.2f}..{f(hi):.2f} Hz (f* = {S['f_star']['a']:.3f} + {S['f_star']['b']:.3f} s, MAD {S['f_star']['mad']:.3f})")
        lg(f"  duty FL/FR/RL/RR at median speed " + "/".join(f"{d['a'] + d['b'] * md:.2f}" for d in S["duty"])
           + "  slope/s " + "/".join(f"{d['b']:+.2f}" for d in S["duty"]))
        lg(f"  touchdown offsets after FL: FR/RL/RR " + "/".join(f"{o:.3f}" for o in S["offsets"][1:])
           + " (spread " + "/".join(f"{o:.3f}" for o in S["offsets_spread"][1:]) + ")"
           + f"  L = {(1 - S['offsets'][2]) % 1:.2f}")
        lg(f"  swing apex / h FL/FR/RL/RR " + "/".join(f"{x['apex_median']:.3f}" for x in S["legs"])
           + f" (at H0 {np.mean([x['apex_median'] for x in S['legs'][:2]]) * H0 * 100:.1f} cm fore, "
           f"{np.mean([x['apex_median'] for x in S['legs'][2:]]) * H0 * 100:.1f} cm hind); along at mid-swing "
           + "/".join(f"{x['along'][BINS_SWING // 2]:.2f}" for x in S["legs"]))
        lg(f"  hip-relative touchdown x / h FL/FR/RL/RR " + "/".join(f"{x['x_td_median']:+.2f}" for x in S["legs"])
           + "  lift-off " + "/".join(f"{x['x_lo_median']:+.2f}" for x in S["legs"]))
        tr = S["trunk"]
        lg(f"  trunk ptp: heave {tr['heave']['ptp_median'] * H0 * 100:.1f} cm at H0, pitch {np.degrees(tr['pitch']['ptp_median']):.1f} deg, "
           f"roll {np.degrees(tr['roll']['ptp_median']):.1f} deg; |c1..3| heave/h "
           + "/".join(f"{abs(complex(c['re']['a'] + c['re']['b'] * md, c['im']['a'] + c['im']['b'] * md)):.4f}" for c in tr["heave"]["harmonics"])
           + f" sway ptp {tr['sway']['ptp_median'] * H0 * 100:.1f} cm, surge {tr['surge']['ptp_median'] * H0 * 100:.1f} cm"
           + " roll deg " + "/".join(f"{np.degrees(abs(complex(c['re']['a'] + c['re']['b'] * md, c['im']['a'] + c['im']['b'] * md))):.2f}" for c in tr["roll"]["harmonics"]))
        for side in ("left", "right"):
            if side in entry:
                g = entry[side]
                lg(f"  {side:5s} turns: {g['strides']} strides, yaw {g['wz_pct'][0] / t0r:+.2f}..{g['wz_pct'][2] / t0r:+.2f} rad/s at H0, "
                   f"speed {g['s_pct'][0] * v0r:.2f}..{g['s_pct'][2] * v0r:.2f}; offsets " + "/".join(f"{o:.3f}" for o in g["offsets"][1:])
                   + f"; f* resid {g['f_star_resid_median']:+.3f}; duty resid " + "/".join(f"{d:+.2f}" for d in g["duty_resid_median"])
                   + f"; roll ptp {np.degrees(g['trunk']['roll']['ptp_median']):.1f} deg; outer/inner stride {g['geometry']['ratio_measured']:.2f}"
                   f" (rigid body {g['geometry']['ratio_rigid_body']:.2f})")
        lg(f"  lean: mean roll = {entry['lean']['a']:+.3f} + {entry['lean']['b']:+.3f} (s w*) rad, corr {entry['lean']['corr']:+.2f}")
    # gait selection (units of H0) and coverage summary
    model["selection"] = dict(
        note="forward speed |c_xy| in m/s at h0_robot; first matching row; lateral/backward = walk, extrapolated",
        rows=[["walk", 0.0, 0.85], ["pace", 0.85, 1.25], ["trot", 1.25, 1.5], ["gallop-R", 1.5, 9.9]])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(model, open(a.out, "w"), indent=0)
    open(os.path.splitext(a.out)[0] + ".txt", "w").write("\n".join(lines) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

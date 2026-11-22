"""Dog-guided walking: targets synthesised from the fitted MANN dog gait model, and the scheduler
that plans to them (`trajgen.pipeline solve walk --dog-style MODEL`).

The model (tools/dog/dog_gait_model.py -> data/dog/dog-gait-model.json, fitted to the AI4Animation MANN
dog capture, CC BY-NC 4.0: data/dog/README.md) is per-gait and dimensionless (h = the dog's leg-root
height; s = v / sqrt(g h); f* = f sqrt(h / g)).
For a command (vx, vy, wz) and a gait family, DogGaitModel.targets() gives, Froude-scaled to the
robot's h0:

    timing      period 1 / f(s), per-leg duty(s), touchdown offsets (FL = 0). Turns blend toward the
                dog's own left- or right-turn strides by |wz| (never mirrored).
    swing       per leg: the fraction of the lift-off -> foothold chord covered against swing
                progress, the lateral deviation from the chord (x h0) and the height profile
                (zero at both ends, apex 1) times the dog's apex(s) x h0.
    footholds   Raibert about the dog's mid-stance point: the foot lands half the leg's own stance
                travel ahead of where the hip will be at mid-stance, the mid-stance point sitting
                at the dog's median (x_td + x_lo) / 2 fore-aft offset from the hip.
    trunk       heave (x h0), pitch, roll and lateral sway (x h0) waveforms over the stride (3 harmonics, phase 0 = FL
                touchdown), plus the dog's lean into a turn (mean roll ~ s w*). Gallop pitch is
                scaled by GALLOP_PITCH (most of the dog's gallop pitch is spine flexion).

Outside the dog's support (speed outside its 5-95 % range for the gait, any lateral or backward
command, turning in place) the timing is held at the nearest supported value and the targets are
LABELLED in `labels` (recorded in the bundle manifest).

DogScheduler is gait.Scheduler with the extension points overridden: per-leg stance and swing
durations, the dog's mid-stance offset, the dog swing path instead of the cycloid, and the trunk
waveform added to the CoM height and posture after the terrain-attitude filter. Each part can be
switched off (`parts`) to measure it alone. Everything fades with the commanded speed fraction, so
the ramp-up starts and the stop ends at the plain standing posture.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

import numpy as np

from .gait import DogPattern, Scheduler

G = 9.81
PARTS = ("timing", "swing", "foothold", "trunk", "vfilter")   # vfilter: DogScheduler.knots
GALLOP_PITCH = 0.25          # as style/stage1.py: the dog's gallop pitch is mostly spine flexion
REPLAN_SECONDS = 0.1         # walk.py's re-plan interval: the filter below runs once per re-plan
TURN_RADIUS = 0.25           # m, default hip distance from the base centre: in-place turn "speed"
LEGS = ("FL", "FR", "RL", "RR")


@dataclass
class DogTargets:
    family: str
    command: tuple
    period: float
    duties: dict
    offsets: dict
    swing: dict              # leg -> {grid, along, across (m), up (0..1), apex (m)}
    stance_centre: dict      # leg -> m, mid-stance fore-aft offset from the hip
    trunk: dict              # heave (m), pitch (ROS, nose down > 0), roll: complex harmonics 1..K
    lean: float              # rad, mean roll in the turn (> 0 left side up)
    labels: list = field(default_factory=list)
    support: dict = field(default_factory=dict)
    parts: tuple = PARTS
    trunk_scale: float = 1.0
    duty_ramp: float | None = None   # stance duty at standstill; the dog's at cruise (DogScheduler)
    support_sway: float = 0.0        # gain on the support-centre CoM sway (DogScheduler; 0 = off)

    def pattern(self, swing_height: float | None = None) -> DogPattern:
        apex = float(np.mean([self.swing[leg]["apex"] for leg in LEGS]))
        return DogPattern(f"dog-{self.family}", period=float(self.period),
                          duty=float(np.mean([self.duties[leg] for leg in LEGS])),
                          swing_height=float(swing_height or apex), offsets=dict(self.offsets),
                          duties=dict(self.duties))

    def trunk_at(self, phi: float):
        """(dz m, pitch ROS rad, roll rad, sway m) and their phase derivatives per cycle."""
        out, rate = [], []
        for key in ("heave", "pitch", "roll", "sway"):
            c = np.asarray(self.trunk.get(key, ()), complex)
            k = np.arange(1, len(c) + 1)
            e = np.exp(2j * np.pi * k * phi)
            out.append(float(np.real(c @ e)))
            rate.append(float(np.real((2j * np.pi * k * c) @ e)))
        return out, rate

    def as_dict(self) -> dict:
        tr = {k: [[float(np.real(x)), float(np.imag(x))] for x in v] for k, v in self.trunk.items()}
        sw = {leg: {"apex_m": round(v["apex"], 4), "across_max_m": round(float(np.max(np.abs(v["across"]))), 4),
                    "along_mid": round(float(np.interp(0.5, v["grid"], v["along"])), 3)}
              for leg, v in self.swing.items()}
        return {"family": self.family, "command": list(self.command), "period": round(self.period, 5),
                "frequency_hz": round(1.0 / self.period, 4),
                "duties": {k: round(v, 4) for k, v in self.duties.items()},
                "offsets": {k: round(v, 4) for k, v in self.offsets.items()},
                "swing": sw, "stance_centre_m": {k: round(v, 4) for k, v in self.stance_centre.items()},
                "trunk_harmonics": tr, "lean_rad": round(self.lean, 4), "labels": list(self.labels),
                "support": self.support, "parts": list(self.parts), "trunk_scale": self.trunk_scale,
                **({} if self.duty_ramp is None else {"duty_ramp": self.duty_ramp}),
                **({} if not self.support_sway else {"support_sway": self.support_sway}),
                "extrapolated": any(lab.startswith("extrapolated") for lab in self.labels)}


def _line(p: dict, x: float) -> float:
    return float(p["a"] + p["b"] * x)


def _blend_phase(a: float, b: float, t: float) -> float:
    d = ((b - a + 0.5) % 1.0) - 0.5
    return float((a + t * d) % 1.0)


def leg_root_scale(model, qpos) -> tuple[float, float]:
    """(h0, turn radius) of a robot standing at `qpos`: the leg roots' ({leg}_hip_joint) centroid
    height above the lowest foot sphere's bottom, the quantity the model's h is for the dog, and
    the hips' mean horizontal distance from the base."""
    import mujoco

    from .plantmap import foot_geom
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    roots = np.array([data.xanchor[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_hip_joint")]
                      for leg in LEGS])
    floor = min(data.geom_xpos[g][2] - model.geom_size[g][0] for g in (foot_geom(model, leg) for leg in LEGS))
    radius = np.linalg.norm(roots[:, :2] - data.qpos[:2], axis=1)
    return float(roots[:, 2].mean() - floor), float(radius.mean())


class DogGaitModel:
    def __init__(self, path, h0: float | None = None, turn_radius: float | None = None):
        """`h0` is the robot's leg-root height and `turn_radius` its hips' distance from the base
        centre (leg_root_scale); left None, the model's reference h0_robot and TURN_RADIUS."""
        self.path = str(path)
        self.m = json.load(open(path))
        self.h_dog = float(self.m["h_dog"])
        self.h0 = float(h0 if h0 is not None else self.m["h0_robot"])
        self.turn_radius = float(TURN_RADIUS if turn_radius is None else turn_radius)
        self.v0 = np.sqrt(G * self.h0)            # robot speed per unit Froude speed
        self.t0 = np.sqrt(self.h0 / G)            # robot seconds per unit normalized time
        self.turn = float(self.m["turn_threshold"])
        self.families = self.m["families"]

    def select(self, vx: float, vy: float, wz: float) -> str:
        speed = float(np.hypot(vx, vy))
        if vx < 0 or abs(vy) > 0.1 or speed < 0.2:
            return "walk"
        # The rows are m/s at the model's h0_robot: compare at the same Froude speed.
        froude = speed * np.sqrt(float(self.m["h0_robot"]) / self.h0)
        for fam, lo, hi in self.m["selection"]["rows"]:
            if lo <= froude < hi and fam in self.families:
                return fam
        return "walk"

    def targets(self, vx: float, vy: float, wz: float, family: str | None = None,
                parts=PARTS, trunk_scale: float = 1.0, duty_ramp: float | None = None,
                support_sway: float = 0.0) -> DogTargets:
        fam = family or self.select(vx, vy, wz)
        if fam not in self.families:
            raise ValueError(f"dog gait model has no family {fam!r} (has {sorted(self.families)})")
        E = self.families[fam]
        S = E["straight"]
        labels = []
        speed = float(np.hypot(vx, vy))
        s_cmd = speed / self.v0
        w_star = wz * self.t0
        s_eff = s_cmd
        if speed < 0.2 and abs(wz) > 0.2:
            s_eff = abs(wz) * self.turn_radius / self.v0
            labels.append("extrapolated: turning in place (no steady dog strides); timing at the hips' "
                          f"tangential speed {abs(wz) * self.turn_radius:.2f} m/s")
        if vx < -0.05:
            labels.append("extrapolated: backward (no dog data); forward timing and swing shapes")
        if abs(vy) > 0.1:
            labels.append("extrapolated: lateral (1 dog window); forward timing and swing shapes along "
                          "the travel direction")
        lo, _, hi = S["s_pct"]
        s_c = float(np.clip(s_eff, lo, hi))
        if s_eff < 0.95 * lo:
            labels.append(f"extrapolated: speed {s_eff * self.v0:.2f} m/s below the dog's {fam} support "
                          f"({lo * self.v0:.2f}-{hi * self.v0:.2f}); cadence, duty and shapes held at "
                          f"{lo * self.v0:.2f}, stride shortened (v / f)")
        elif s_eff > 1.05 * hi:
            labels.append(f"extrapolated: speed {s_eff * self.v0:.2f} m/s above the dog's {fam} support "
                          f"({lo * self.v0:.2f}-{hi * self.v0:.2f}); held at {hi * self.v0:.2f}, stride lengthened")
        fstar = _line(S["f_star"], s_c)
        duty = [_line(S["duty"][k], s_c) for k in range(4)]
        offs = list(S["offsets"])
        trunk = {}
        for key in ("heave", "pitch", "roll", "sway"):
            if key not in S["trunk"]:
                continue
            trunk[key] = np.array([complex(_line(c["re"], s_c), _line(c["im"], s_c))
                                   for c in S["trunk"][key]["harmonics"]])
        side = "left" if w_star > 0 else "right"
        t = 0.0
        if abs(w_star) > 0.25 * self.turn:
            if side in E:
                Tn = E[side]
                w_med = abs(Tn["wz_pct"][1]) or self.turn
                t = float(np.clip(abs(w_star) / w_med, 0.0, 1.0))
                fstar += t * Tn["f_star_resid_median"]
                duty = [d + t * r for d, r in zip(duty, Tn["duty_resid_median"])]
                offs = [_blend_phase(a, b, t) for a, b in zip(offs, Tn["offsets"])]
                for key in trunk:
                    ct = np.array([complex(c["re"]["a"], c["im"]["a"]) for c in Tn["trunk"][key]["harmonics"]])
                    trunk[key] = (1 - t) * trunk[key] + t * ct
                w_hi = max(abs(Tn["wz_pct"][0]), abs(Tn["wz_pct"][2]))
                if abs(w_star) > 1.1 * w_hi:
                    labels.append(f"extrapolated: yaw rate {wz:+.2f} beyond the dog's {fam} {side} turns "
                                  f"(to {np.sign(wz) * w_hi / self.t0:+.2f} rad/s); turn timing held")
            else:
                labels.append(f"extrapolated: no steady dog {fam} {side}-turn strides; straight timing")
        if fam.startswith("gallop") or fam.startswith("canter"):
            trunk["pitch"] = GALLOP_PITCH * trunk["pitch"]
            labels.append(f"gallop trunk pitch x{GALLOP_PITCH} (the dog's is mostly spine flexion)")
        trunk["heave"] = trunk["heave"] * self.h0
        if "sway" in trunk:
            trunk["sway"] = trunk["sway"] * self.h0
        for key in trunk:
            trunk[key] = trunk_scale * trunk[key]
        lean = float(E["lean"]["b"] * s_cmd * w_star) if "lean" in E else 0.0
        duty = [float(np.clip(d, 0.12, 0.88)) for d in duty]
        grid = np.linspace(0.0, 1.0, int(self.m["bins_swing"]))
        swing, centre = {}, {}
        for k, leg in enumerate(LEGS):
            L = S["legs"][k]
            along = np.asarray(L["along"], float)
            along = (along - along[0]) / max(along[-1] - along[0], 1e-6)
            across = np.asarray(L["across"], float)
            across = across - (across[0] + grid * (across[-1] - across[0]))
            up = np.asarray(L["up"], float)
            up = up - (up[0] + grid * (up[-1] - up[0]))
            up = np.maximum(up, 0.0) / max(up.max(), 1e-6)
            apex = max(0.02, _line(L["apex"], s_c)) * self.h0
            swing[leg] = {"grid": grid, "along": along, "across": across * self.h0, "up": up, "apex": float(apex)}
            centre[leg] = 0.5 * (L["x_td_median"] + L["x_lo_median"]) * self.h0
        support = {"family_strides": S["strides"], "captures": S["clips"],
                   "speed_mps": [round(lo * self.v0, 3), round(hi * self.v0, 3)],
                   "turn_blend": round(t, 3), "turn_side": side if t > 0 else None}
        return DogTargets(family=fam, command=(float(vx), float(vy), float(wz)),
                          period=float(self.t0 / fstar), duties=dict(zip(LEGS, duty)),
                          offsets=dict(zip(LEGS, [float(o) % 1.0 for o in offs])), swing=swing,
                          stance_centre=centre, trunk=trunk, lean=lean, labels=labels, support=support,
                          parts=tuple(parts), trunk_scale=float(trunk_scale),
                          duty_ramp=None if duty_ramp is None else float(duty_ramp),
                          support_sway=float(support_sway))


@dataclass
class DogScheduler(Scheduler):
    """gait.Scheduler planning to DogTargets (see the module docstring)."""

    dog: DogTargets | None = None
    cruise: tuple = (0.0, 0.0, 0.0)          # the full command (vx, vy, wz), for the fade

    def _has(self, part: str) -> bool:
        return self.dog is not None and part in self.dog.parts

    def _dog_pattern(self) -> bool:
        return isinstance(self.pattern, DogPattern)

    def fade(self) -> float:
        """Commanded fraction of the cruise command: 0 standing, 1 at cruise."""
        vx, vy, wz = self.cruise
        lin = float(np.hypot(vx, vy))
        f = []
        if lin > 1e-6:
            f.append(float(np.linalg.norm(self.velocity)) / lin)
        if abs(wz) > 1e-6:
            f.append(abs(float(self.yaw_rate)) / abs(wz))
        return float(np.clip(max(f) if f else 0.0, 0.0, 1.0))

    def knots(self, q, posture, steps, dt):
        """STRIDE-AVERAGED velocity for the footholds. The base scheduler carries each foothold by
        the base velocity measured at the re-plan; a dog-like gait sways, and the instantaneous
        lateral velocity points toward the stance side while the opposite feet swing - measured on
        the dog pace, every swing foot was pulled inward and the fore track closed from 0.30 m to
        0.05-0.17 m. The measured LATERAL velocity and the yaw rate are low-passed over one gait
        period in the HEADING frame (a world-frame filter would lag a turn's rotation)."""
        if self.dog is not None and self.dog.duty_ramp is not None and self._dog_pattern():
            # DUTY RAMP (gallop): a gallop's stance (the dog's 0.2-0.33) cannot hold the robot up at
            # walking speed - measured, every gallop from a stand collapsed within 0.4-0.8 s. Duties
            # blend from duty_ramp at standstill to the dog's at cruise with the command fraction;
            # offsets and the clock are untouched, so the phase stays continuous.
            a = self.fade()
            self.pattern = dataclasses.replace(self.pattern, duties={
                leg: (1 - a) * self.dog.duty_ramp + a * self.dog.duties[leg] for leg in LEGS})
        if self.measured_velocity is not None and self._dog_pattern() and self._has("vfilter"):
            R = np.asarray(q[3:7], float)
            x, y, z, w = R
            yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            c, s = np.cos(yaw), np.sin(yaw)
            v = np.asarray(self.measured_velocity, float)[:2]
            vh = np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])
            wz = float(self.measured_yaw_rate or 0.0)
            a = min(1.0, REPLAN_SECONDS / max(self.pattern.period, 1e-6))
            if getattr(self, "_vy", None) is None:
                self._vy, self._wz = vh[1], wz
            else:
                self._vy = (1 - a) * self._vy + a * vh[1]
                self._wz = (1 - a) * self._wz + a * wz
            # forward speed stays the measured one (the base scheduler's fix for a body cruising
            # faster than commanded); only the sway-dominated lateral part and the yaw rate are filtered
            vh[1] = self._vy
            self.measured_velocity = np.array([c * vh[0] - s * vh[1], s * vh[0] + c * vh[1]])
            self.measured_yaw_rate = self._wz
        if self._swaying():
            # SUPPORT SWAY. The CoM target is the measured CoM carried by the command, held at
            # 1e6, so a pace could not lean over its lateral pair: the planner instead pulled the
            # feet under the body (fore track 0.11 m, hip abduction 0.24 rad; retargeted dog 0.29 m,
            # 0.0), and full-width footholds alone rocked it (roll 11-14 deg, hips swinging 0.55
            # rad). The CoM now sways by the first stride harmonic of the support centre's lateral
            # position (a pendulum is nearest its support at mid-stance; zero for a trot's
            # diagonal pairs), relative to the current phase since the measured CoM already holds
            # the sway, and the footholds are placed from the sway-free path at the stand's track,
            # carried by the commanded lateral velocity rather than the swaying measured one.
            phi = np.arange(400) / 400.0
            centre = np.array([np.mean([self.nominal[leg][1] for leg in LEGS
                                        if self.pattern.stance(leg, p)] or [0.0]) for p in phi])
            self._support_c1 = 2.0 * np.mean(centre * np.exp(-2j * np.pi * phi))
            self._sway_now = self._support_sway(self.phase)
            x, y, z, w = np.asarray(q[3:7], float)
            yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
            c, s = np.cos(yaw), np.sin(yaw)
            v = np.asarray(self.measured_velocity, float)[:2]
            cmd = np.asarray(self.velocity, float)[:2]
            vh = np.array([c * v[0] + s * v[1], -s * cmd[0] + c * cmd[1]])
            self.measured_velocity = np.array([c * vh[0] - s * vh[1], s * vh[0] + c * vh[1]])
        return super().knots(q, posture, steps, dt)

    def _swaying(self) -> bool:
        return (self.dog is not None and bool(self.dog.support_sway) and self._dog_pattern()
                and self.measured_velocity is not None)

    def _support_sway(self, phase: float) -> float:
        """Lateral CoM offset (m, heading frame, + left) at a gait phase; see knots."""
        return float(self.dog.support_sway * self.fade()
                     * np.real(self._support_c1 * np.exp(2j * np.pi * float(phase))))

    def foothold(self, leg: str, q, seconds_to_land: float) -> np.ndarray:
        if not self._swaying():
            return super().foothold(leg, q, seconds_to_land)
        x, y, z, w = np.asarray(q[3:7], float)
        yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        q = np.array(q, float)
        q[:2] -= np.array([-np.sin(yaw), np.cos(yaw)]) * self._sway_now
        return super().foothold(leg, q, seconds_to_land)

    def _stance_seconds(self, leg: str) -> float:
        if self._dog_pattern():
            return self.pattern.leg_stance_seconds(leg)
        return self.pattern.stance_seconds

    def _swing_seconds(self, leg: str) -> float:
        if self._dog_pattern():
            return self.pattern.leg_swing_seconds(leg)
        return self.pattern.swing_seconds

    def _neutral(self, leg: str, neutral: np.ndarray) -> np.ndarray:
        if not self._has("foothold") or not self._dog_pattern():
            return neutral
        out = np.array(neutral, float)
        out[0] += self.dog.stance_centre[leg] * self.fade()
        return out

    def swing_target(self, leg, progress, remaining, start, foothold):
        if not self._has("swing") or not self._dog_pattern():
            return super().swing_target(leg, progress, remaining, start, foothold)
        sw = self.dog.swing[leg]
        p = float(np.clip(progress, 0.0, 1.0))
        along = float(np.interp(p, sw["grid"], sw["along"]))
        across = float(np.interp(p, sw["grid"], sw["across"])) * self.fade()
        up = float(np.interp(p, sw["grid"], sw["up"]))
        chord = np.asarray(foothold[:2], float) - np.asarray(start[:2], float)
        L = float(np.linalg.norm(chord))
        n = np.array([-chord[1], chord[0]]) / L if L > 1e-6 else np.zeros(2)
        xy = start[:2] + chord * along + n * across
        climb = max(0.0, float(foothold[2] - start[2]))
        z = start[2] + (foothold[2] - start[2]) * along + (sw["apex"] + climb) * up
        return np.array([xy[0], xy[1], z])

    def _trunk(self, phase: float):
        if not self._has("trunk") or not self._dog_pattern():
            return None
        (dz, pitch, roll, sway), (dz_r, pitch_r, roll_r, _) = self.dog.trunk_at(float(phase) % 1.0)
        a = self.fade()
        w = 1.0 / self.pattern.period          # cycles per second
        lean = self._support_sway(phase) - self._sway_now if self._swaying() else 0.0
        return {"dz": a * dz, "sway": a * sway + lean, "pitch": -a * pitch, "roll": a * (roll + self.dog.lean),
                "vz": a * dz_r * w, "pitch_rate": a * pitch_r * w, "roll_rate": a * roll_r * w}


def scheduler_from(base: Scheduler, dog: DogTargets, cruise) -> DogScheduler:
    """A DogScheduler with every field of an already-built Scheduler."""
    kw = {f.name: getattr(base, f.name) for f in dataclasses.fields(Scheduler)}
    return DogScheduler(**kw, dog=dog, cruise=tuple(float(c) for c in cruise))

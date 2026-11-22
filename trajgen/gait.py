"""Phase-clocked gait scheduler for walking.

Why this exists: Crocoddyl's quadruped helper builds a problem that always
STARTS in double support at the beginning of a cycle, so a plan can only ever
begin at a cycle boundary. Re-planning more often than that just restarts the
gait -- measured, the robot stood stable and travelled -0.026 m in 6 s -- while
re-planning only once per cycle lets the error grow for a whole cycle, and the
robot sank from 0.40 m to 0.25 m and fell by the fifth cycle.

A phase clock fixes that. The gait is a periodic contact pattern evaluated at a
continuous phase, so a horizon can be built starting ANYWHERE in the cycle:

    pattern     per leg, a phase offset and a stance duty fraction
    footholds   where a swinging foot will land (Raibert-style: under the hip,
                offset by half a stance period of desired velocity)
    swing       liftoff -> foothold, with a sine arc that LANDS at the ground
                (upstream's discretisation stops 2*height/knots short, 6 mm
                with the settings used here, so every touchdown is planned in
                mid-air)
    horizon     knots from the current phase forward, contacts and swing
                targets per knot, impulses where a foot lands

The scheduler holds only what a clock needs: the phase. Foot positions,
terrain heights and the CoM are measured from the robot each time it re-plans.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import pinocchio as pin

from .adapter import LEGS, Adapter
from .croc import Knot


class Terrain:
    """Ground height under a point, read from the model's static geoms.

    Analytic ray-geom intersection rather than `mj_ray`, because the robot's
    own geoms are in the way of a downward ray and only the static world is
    terrain. Flat ground gives 0 everywhere, so the same scheduler serves both.
    """

    def __init__(self, model: mujoco.MjModel, top: float = 10.0):
        self.top = float(top)
        # The world is static, so a height is a pure function of (x, y). The
        # slope fit below samples it a few dozen times per knot; without the
        # cache that is millions of ray-geom tests per solve.
        self._cache: dict = {}
        self.geoms = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != 0 or not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            mat = np.empty(9)
            mujoco.mju_quat2Mat(mat, model.geom_quat[g])
            self.geoms.append((np.array(model.geom_pos[g]), mat, np.array(model.geom_size[g]),
                               int(model.geom_type[g])))

    def height(self, x: float, y: float) -> float:
        key = (round(float(x), 4), round(float(y), 4))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        value = self._height(*key)
        self._cache[key] = value
        return value

    def _height(self, x: float, y: float) -> float:
        pnt = np.array([float(x), float(y), self.top])
        vec = np.array([0.0, 0.0, -1.0])
        best = 0.0
        for pos, mat, size, kind in self.geoms:
            distance = mujoco.mju_rayGeom(pos, mat, size, pnt, vec, kind)
            if distance >= 0:
                best = max(best, self.top - distance)
        return best

    def plane(self, points, at):
        """Fit a ground plane through the terrain under `points` (x, y pairs)."""
        A = np.array([[px, py, 1.0] for px, py in points])
        z = np.array([self.height(px, py) for px, py in points])
        (a, b, c), *_ = np.linalg.lstsq(A, z, rcond=None)
        return float(a * at[0] + b * at[1] + c), float(np.arctan(a)), float(np.arctan(b))

    def choose_foothold(self, x: float, y: float, foot_z: float, *, max_rise: float = 0.22,
                        max_drop: float = 0.25, reach: float = 0.30, patch: float = 0.05,
                        tolerance: float = 0.005, step: float = 0.02, hip=None,
                        leg_reach: float):
        """Pick a foothold near (x, y) that the leg can actually step onto.

        Two things disqualify the nominal placement on stairs: it can be on the
        lip of a step (the tread is only 24 cm deep), and it can be a whole
        step too far. Measured, a 0.2 m stride put the front feet at x = 1.08,
        which is the SECOND stair - a 0.4 m rise - and the robot stalled
        against the first riser instead of climbing.

        Candidates are searched OUTWARD from the nominal placement, nearest
        first, and the first one that is level and within one riser of the
        foot's current height wins. Searching backwards only is what stranded
        the rear feet: their nominal placement lands near a step edge, the
        patch test rejects it, and the nearest level ground behind is the floor
        - so they stepped down onto the floor instead of up onto the tread just
        ahead, the body climbed away from them, and the robot pitched over
        backwards. On flat ground the nominal placement qualifies and is
        returned unchanged.

        Returns (x, y, ground height, moved).
        """
        steps = int(reach / step)
        offsets = [0.0]
        for i in range(1, steps + 1):
            offsets.extend((i * step, -i * step))
        for offset in offsets:
            cx = x + offset
            heights = [self.height(cx + dx, y + dy)
                       for dx, dy in ((0, 0), (patch, 0), (-patch, 0), (0, patch), (0, -patch))]
            if max(heights) - min(heights) > tolerance:
                continue                      # on an edge, not a tread
            height = heights[0]
            if height - foot_z > max_rise or foot_z - height > max_drop:
                continue                      # too big a step up or down
            if hip is not None:
                span = np.linalg.norm(np.array([cx, y, height]) - np.asarray(hip, float))
                if span > 0.9 * leg_reach:
                    continue                  # the leg does not reach that far
            return cx, y, height, abs(offset) > 1e-9
        return x, y, self.height(x, y), False


@dataclass(frozen=True)
class Pattern:
    """A periodic contact pattern: when each leg is on the ground."""

    name: str
    period: float                     # seconds for one full cycle
    duty: float                       # fraction of the cycle each leg is in stance
    offsets: dict                     # leg -> phase offset in [0, 1)
    swing_height: float = 0.06

    def stance(self, leg: str, phase: float) -> bool:
        return self.leg_phase(leg, phase) < self.duty

    def leg_phase(self, leg: str, phase: float) -> float:
        return (phase - self.offsets[leg]) % 1.0

    def swing_progress(self, leg: str, phase: float) -> float | None:
        """0 at liftoff, 1 at touchdown; None while the leg is in stance."""
        p = self.leg_phase(leg, phase)
        if p < self.duty:
            return None
        return (p - self.duty) / (1.0 - self.duty)

    @property
    def swing_seconds(self) -> float:
        return self.period * (1.0 - self.duty)

    @property
    def stance_seconds(self) -> float:
        return self.period * self.duty


# Diagonal pairs move together; duty > 0.5 keeps a double-support overlap.
# The period and swing height are set by what the servos can track: a 0.7 s
# trot swings fast enough that the foot overshoots its foothold and is dragged
# back through the air before landing (25 backward samples, worst 26 mm), and
# the knee crosses the calf motor's no-load speed. At 0.9 s and a
# 5 cm swing the same gait passes every gate with 4 such samples.
TROT = Pattern("trot", period=0.9, duty=0.6, swing_height=0.05,
               offsets={"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5})
# One foot at a time: three feet always down, the most forgiving gait.
CRAWL = Pattern("crawl", period=1.6, duty=0.8,
                offsets={"FL": 0.0, "RR": 0.25, "FR": 0.5, "RL": 0.75})
# Front pair together, then rear pair together. Left-right symmetric, and the
# rear feet follow the front ones onto the same tread half a cycle later, which
# is how a quadruped clears stairs: a trot puts one front and one diagonal rear
# foot on different steps and rolls the body.
PAIRS = Pattern("pairs", period=1.1, duty=0.6, swing_height=0.06,
                offsets={"FL": 0.0, "FR": 0.0, "RL": 0.5, "RR": 0.5})
# All four feet down: the same scheduler, used to stand still at the end of a
# walk so the closing hold is part of one continuous motion.
STAND = Pattern("stand", period=1.0, duty=1.0,
                offsets={"FL": 0.0, "FR": 0.0, "RL": 0.0, "RR": 0.0})
# THE MANN DOG'S SYMMETRICAL GAITS (AI4Animation MANN dog capture, CC BY-NC 4.0, see
# data/dog/README.md; Froude-scaled to a 0.40 m leg-root height; the offsets are each foot's
# touchdown phase after FL's, projected onto the left/right mirror).
# Lateral-sequence walk: RL, FL, RR, FR, each 0.36-0.5 of a cycle after the last but one,
# three feet down most of the time (L = 0.14). Defaults are the dog at 0.56 m/s (0.99 Hz).
WALK = Pattern("walk", period=1.011, duty=0.63, swing_height=0.06,
               offsets={"FL": 0.0, "FR": 0.5, "RL": 0.858, "RR": 0.358})
# Pace: the lateral pairs swing together (hind lands 0.045 of a cycle before its fore foot).
# Defaults are the dog at 1.0 m/s (1.29 Hz).
PACE = Pattern("pace", period=0.777, duty=0.55, swing_height=0.06,
               offsets={"FL": 0.0, "FR": 0.5, "RL": 0.955, "RR": 0.455})
PATTERNS = {p.name: p for p in (TROT, CRAWL, PAIRS, STAND, WALK, PACE)}


@dataclass(frozen=True)
class DogPattern(Pattern):
    """A contact pattern with a stance duty PER LEG (trajgen/dogstyle.py): the dog's fore and hind
    feet keep different duties (walk 0.65/0.61, gallop 0.26/0.17), and the asymmetric gaits (gallop
    and canter leads, the lateral-sequence walk) need their own touchdown offsets per leg. `duty`
    holds the mean of `duties` for reporting; every timing query below uses the leg's own duty."""

    duties: dict = field(default_factory=dict)       # leg -> stance fraction

    def leg_duty(self, leg: str) -> float:
        return float(self.duties.get(leg, self.duty))

    def stance(self, leg: str, phase: float) -> bool:
        return self.leg_phase(leg, phase) < self.leg_duty(leg)

    def swing_progress(self, leg: str, phase: float) -> float | None:
        p = self.leg_phase(leg, phase)
        d = self.leg_duty(leg)
        if p < d:
            return None
        return (p - d) / (1.0 - d)

    def leg_swing_seconds(self, leg: str) -> float:
        return self.period * (1.0 - self.leg_duty(leg))

    def leg_stance_seconds(self, leg: str) -> float:
        return self.period * self.leg_duty(leg)


# THE MANN DOG'S FAST GAITS (the dog gait model, tools/dog/dog_gait_model.py ->
# data/dog/dog-gait-model.json; Froude-scaled to a 0.40 m leg-root height, NOT mirrored;
# offsets = each foot's touchdown phase after FL's). The dog trots and gallops only above ~2 m/s
# on that scale; these defaults are its slowest steady strides. trajgen/dogstyle.py builds the
# per-command versions from the model; these are for `--gait` without a model.
DOG_TROT = DogPattern("dogtrot", period=0.50, duty=0.34, swing_height=0.08,
                      offsets={"FL": 0.0, "FR": 0.500, "RL": 0.536, "RR": 0.065},
                      duties={"FL": 0.38, "FR": 0.39, "RL": 0.31, "RR": 0.27})
# Rotary gallop, fore lead FR, hind lead RL (the dog's commoner lead, 30 straight strides).
GALLOP_R = DogPattern("gallop-R", period=0.49, duty=0.21, swing_height=0.10,
                      offsets={"FL": 0.0, "FR": 0.185, "RL": 0.821, "RR": 0.667},
                      duties={"FL": 0.26, "FR": 0.22, "RL": 0.20, "RR": 0.17})
# Rotary gallop, fore lead FL, hind lead RR (12 straight strides).
GALLOP_L = DogPattern("gallop-L", period=0.49, duty=0.27, swing_height=0.10,
                      offsets={"FL": 0.0, "FR": 0.815, "RL": 0.519, "RR": 0.630},
                      duties={"FL": 0.30, "FR": 0.35, "RL": 0.24, "RR": 0.20})
for _p in (DOG_TROT, GALLOP_R, GALLOP_L):
    PATTERNS[_p.name] = _p


def _smooth(t: float) -> float:
    """Smoothstep: zero velocity at both ends of a segment."""
    t = float(np.clip(t, 0.0, 1.0))
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _plateau(p: float, ramp: float) -> float:
    """Swing advance 0 -> 1 whose speed rises over `ramp` of the swing as (1 - cos), holds, and
    falls symmetrically: zero speed at both ends, peak 1 / (1 - ramp) times the mean."""
    peak = 1.0 / (1.0 - ramp)
    if p > 0.5:
        return 1.0 - _plateau(1.0 - p, ramp)
    if p < ramp:
        return 0.5 * peak * (p - ramp / np.pi * np.sin(np.pi * p / ramp))
    return peak * (0.5 * ramp + (p - ramp))


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw_rotation(quat_xyzw) -> np.ndarray:
    """Rotation about world z only: footholds follow heading, not body tilt."""
    R = pin.Quaternion(np.asarray(quat_xyzw, float)).matrix()
    yaw = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class Scheduler:
    """Phase clock plus the per-leg liftoff positions the swing arcs need."""

    ad: Adapter
    pattern: Pattern
    velocity: np.ndarray              # desired world velocity [vx, vy]
    clearance: float                   # CoM height ABOVE the ground under it
    nominal: dict                     # leg -> foot offset from base, in base yaw frame
    yaw_rate: float = 0.0             # desired turn rate about world z, rad/s
    terrain: Terrain | None = None    # None means a flat plane at z = 0
    phase: float = 0.0
    snap_footholds: bool = False      # choose footholds the leg can step onto
    max_lead: float = 0.08            # how far the CoM target may lead the support
    last_attitude: tuple = (0.0, 0.0)  # filtered (pitch, roll)
    attitude_tau: float = 0.60        # seconds; first-order lag on the reference
    attitude_rate: float = 0.35       # rad/s; hard slew limit on top of the lag
    attitude_source: str = "contacts"  # contacts | terrain | support | flat
    slope_span_x: float = 0.80        # terrain window along the heading, metres
    slope_span_y: float = 0.30        # and across it
    slope_samples: int = 17           # along-heading samples in that window
    raibert_lead: bool = True         # carry the foothold forward by the gait
    memory_span: float = 0.90         # metres of contact history the slope is fitted to
    memory_max: int = 96
    memory: list = field(default_factory=list)   # world stance points, oldest first
    leg_reach: float | None = None    # hip-to-foot extension, metres: the robot config's [gait].leg_reach
    latched: dict = field(default_factory=dict)   # foothold fixed at liftoff, per leg
    hips: dict = field(default_factory=dict)      # hip offsets from base, in the yaw frame
    measured_velocity: np.ndarray | None = None      # base xy velocity (world), set per re-plan
    measured_yaw_rate: float | None = None           # base yaw rate, set per re-plan
    last_stance: dict = field(default_factory=dict)  # last measured contact point per leg
    feet_cache: dict = field(default_factory=dict)   # last measured foot positions
    # ROLLING feet (wheel-legged plant, trajgen/rolling.py): a stance wheel rolls along its heading,
    # so the foothold keeps only the part of the gait lead ACROSS the wheel; along it the foot
    # lands under where its hip will be at touchdown and the rolling carries it from there.
    rolling: bool = False
    # RUNNING SWING: the fraction of the swing the foot spends at its peak horizontal speed. 0 is
    # the cycloid, whose world speed peaks at TWICE its mean: at speed v, duty d, the foot then
    # passes the hip at v (1 + d) / (1 - d), which outruns the thigh's no-load speed long before
    # the stance does. A plateau over a fraction F of the swing, with (1 - cos) ramps on either
    # side, peaks at 2 / (1 + F) of the mean; the foot still lifts and lands at rest.
    swing_flat: float = 0.0
    # LATERAL FOOT PATHS (running). At a long stride the rear foot swings forward past the
    # same-side front foot; at the stand's track the two paths meet (run-v1 at 2.0-2.5 m/s: foot
    # centres 4.6-6.1 cm apart against two 2.5 cm spheres, closest as the rear foot lands, 0.9 of
    # its swing, just after the front foot lifted). Front footholds move IN toward the midline and
    # rear footholds OUT; the front swing bows IN early (`front_bow_peak` of its horizontal advance,
    # while the foot is still moving back past the body) and the rear swing swells OUT late
    # (`rear_bow_peak` of its swing time, where it passes the front foot), so the rear foot passes
    # outside the front one. Metres at the cruise command (`lateral_speed`, set by the walk loop);
    # they scale with the commanded speed below it, so the ramp, the stop and the closing hold keep
    # the stand's track. All 0 changes nothing.
    front_inward: float = 0.0
    rear_outward: float = 0.0
    front_bow: float = 0.0
    front_bow_peak: float = 0.5
    rear_bow: float = 0.0
    rear_bow_peak: float = 0.5
    lateral_speed: float = 0.0

    def _lateral_scale(self) -> float:
        if self.lateral_speed <= 0.0:
            return 1.0
        return float(np.clip(np.linalg.norm(self.velocity) / self.lateral_speed, 0.0, 1.0))

    def ground(self, x: float, y: float) -> float:
        return 0.0 if self.terrain is None else self.terrain.height(x, y)

    @classmethod
    def from_stand(cls, ad: Adapter, pattern: Pattern, stand_qpos, velocity=(0.0, 0.0),
                   terrain: Terrain | None = None, snap_footholds: bool = False):
        """Nominal foot placement measured from the settled standing pose."""
        q = ad.q_from_mj(stand_qpos)
        data = ad.pin.createData()
        pin.framesForwardKinematics(ad.pin, data, q)
        R = _yaw_rotation(q[3:7])
        base = np.asarray(q[:3], float)
        nominal = {leg: R.T @ (np.array(data.oMf[ad.feet[leg].frame].translation) - base)
                   for leg in LEGS}
        # The target is the CoM, which sits below the base: 0.367 m versus
        # 0.404 m standing. Using the base height made the robot walk 37 mm
        # taller than it stands.
        hips = {}
        for leg in LEGS:
            name = f"{leg}_hip_link"
            if ad.pin.existFrame(name):
                hips[leg] = R.T @ (np.array(data.oMf[ad.pin.getFrameId(name)].translation) - base)
            # THE RAIBERT PIVOT IS THE THIGH JOINT, the leg's sagittal axis. On a point-foot plant the
            # abduction and thigh axes meet at the same x (0.2235 m both), so the hip
            # link is that point and is kept exactly. A geared plant's thigh joint sits 43.8 mm
            # OUTBOARD of its hip joint fore-aft (0.1908 vs 0.147 m), and placing about
            # the hip (0.34 m stance) landed front feet 2.1-2.7 cm and rear feet 9.4-9.6 cm ahead of the
            # thigh at 0.5 m/s against a Raibert 8.6 cm for all four.
            thigh = f"{leg}_thigh_link"
            if leg in hips and ad.pin.existFrame(thigh):
                pivot = R.T @ (np.array(data.oMf[ad.pin.getFrameId(thigh)].translation) - base)
                if abs(float(pivot[0] - hips[leg][0])) > 1e-3:
                    hips[leg] = pivot
        com = pin.centerOfMass(ad.pin, data, q)
        under = float(np.mean([data.oMf[ad.feet[leg].frame].translation[2] - ad.feet[leg].radius
                               for leg in LEGS]))
        return cls(ad=ad, pattern=pattern, velocity=np.asarray(velocity, float)[:2],
                   clearance=float(com[2] - under), nominal=nominal, terrain=terrain,
                   snap_footholds=snap_footholds, hips=hips)

    # ---------------------------------------------------------------- state

    def feet_now(self, q) -> dict:
        data = self.ad.pin.createData()
        pin.framesForwardKinematics(self.ad.pin, data, q)
        return {leg: np.array(data.oMf[self.ad.feet[leg].frame].translation) for leg in LEGS}

    def foothold(self, leg: str, q, seconds_to_land: float) -> np.ndarray:
        """Where `leg` should land: under its hip, carried forward by the gait.

        The half-stance offset is the Raibert heuristic: a foot placed under
        where the hip WILL be at mid-stance keeps the body moving at the
        commanded velocity instead of braking against its own footprint.

        `raibert_lead=False` drops that offset and places the foot under the
        hip. On STAIRS the lead is the wrong quantity to place a foot with: the
        landing surface is a 238 mm tread at a discrete height, not a continuum,
        so a placement chosen from velocity has to be snapped onto a tread
        afterwards anyway -- and the further ahead it starts, the more often it
        snaps a whole step too far. Measured on the 24-riser plant, the lead put
        latched front footholds 0.37-0.45 m ahead of the base and two to three
        treads up while the rear feet were still on the floor, which is the
        stretched posture the robot then toppled out of. Under the hip, the
        terrain decides the foothold and the body's own advance decides when.
        """
        base = np.asarray(q[:3], float)
        R = _yaw_rotation(q[3:7])
        # The rotational half of the Raibert lead. A turning body carries its
        # hips around with it, so the foot goes under where the hip WILL be at
        # mid-stance, not where it is now -- the same argument as the linear
        # lead, about the other degree of freedom. Without it the stance feet
        # are left behind the turn and the robot yaws against its own footprint.
        lead_seconds = seconds_to_land + 0.5 * self.pattern.stance_seconds
        if self.yaw_rate and self.raibert_lead:
            # As for the linear lead: the hip turns with the body's MEASURED rate until
            # touchdown (turns cruised ~10% above command and stance feet landed short
            # tangentially), and the half-stance offset uses the commanded rate.
            turning = (self.yaw_rate if self.measured_yaw_rate is None
                       else float(self.measured_yaw_rate))
            R = _rotation_z(turning * seconds_to_land
                            + self.yaw_rate * 0.5 * self._stance_seconds(leg)) @ R
        # THE NEUTRAL POINT IS UNDER THE HIP. The standing foot sits ~5 cm behind the hip
        # joint, and placing the Raibert offset from it landed every step that much short:
        # measured at 0.5 m/s, feet touched down 2 cm ahead of the hip against the
        # heuristic's 8, with execution within 3 mm of the plan. Fore-aft comes from the
        # hip, the stance width from the stand.
        neutral = np.array(self.nominal[leg], float)
        if self.raibert_lead and leg in self.hips:
            neutral[0] = self.hips[leg][0]
        neutral = self._neutral(leg, neutral)
        lateral = self.rear_outward if leg[0] == "R" else -self.front_inward
        if lateral:
            neutral[1] += np.sign(neutral[1]) * lateral * self._lateral_scale()
        offset = R @ neutral
        # The hip's own travel until touchdown, plus the Raibert half-stance offset ahead of
        # it. A foot cannot be placed further from its hip than the leg reaches at this body
        # height (sqrt(0.95 L^2 - h^2)), so the reach limits the OFFSET FROM THE HIP,
        # not the hip's travel: capping their sum at the reach landed feet behind the hips
        # at speed (1.3-1.6 m/s: -12 cm against +12-16 cm wanted) and the body ran ahead.
        height = max(0.05, float(base[2]) - self.ground(base[0], base[1]))
        if self.leg_reach is None:
            raise ValueError("Scheduler.leg_reach is not set: give the robot config [gait].leg_reach")
        limit = np.sqrt(max(0.01, 0.95 * self.leg_reach ** 2 - height ** 2))
        if self.raibert_lead:
            raibert = self.velocity * 0.5 * self._stance_seconds(leg)
            if np.linalg.norm(raibert) > limit:
                raibert = raibert / np.linalg.norm(raibert) * limit
            # Where the hip will be at touchdown follows the body's MEASURED velocity: it
            # cruised 0.56 m/s under a 0.5 command and, predicted from the command, feet
            # still landed 3.5 cm short. The half-stance offset ahead of the hip stays the
            # commanded one (the Raibert target).
            hip_velocity = (self.velocity if self.measured_velocity is None
                            else np.asarray(self.measured_velocity, float)[:2])
            travel = hip_velocity * seconds_to_land + raibert
        else:
            travel = np.zeros(2)
        target = base + offset
        target[:2] += travel
        if self.rolling:
            # Across/along the wheel at touchdown: the heading turned by the measured rate
            # until then. The lead along the wheel (the rolling part of the turn's tangential
            # Raibert offset, and of any linear lead) is dropped; across it is kept.
            turning = (self.yaw_rate if self.measured_yaw_rate is None
                       else float(self.measured_yaw_rate))
            R_td = _rotation_z(turning * seconds_to_land) @ _yaw_rotation(q[3:7])
            hip_velocity = (self.velocity if self.measured_velocity is None
                            else np.asarray(self.measured_velocity, float)[:2])
            under = base + R_td @ neutral
            under[:2] += hip_velocity * seconds_to_land
            across = R_td[:2, 1]
            target[:2] = under[:2] + across * float((target[:2] - under[:2]) @ across)
        x, y = target[0], target[1]
        if self.snap_footholds and self.terrain is not None:
            foot_z = self.ground(*self.feet_cache[leg][:2]) if leg in self.feet_cache else 0.0
            hip = base + R @ self.hips[leg] if leg in self.hips else None
            x, y, height, _ = self.terrain.choose_foothold(x, y, foot_z, hip=hip,
                                                           leg_reach=self.leg_reach)
        else:
            height = self.ground(x, y)
        return np.array([x, y, height + self.ad.feet[leg].radius])

    def swing_target(self, leg: str, progress: float, remaining: float, start: np.ndarray,
                     foothold: np.ndarray) -> np.ndarray:
        """Cycloid swing from where the foot is to where it lands.

        Horizontal motion follows (phi - sin phi) / 2pi and height follows
        (1 - cos phi) / 2, so the foot leaves and meets the ground with zero
        velocity and bounded acceleration. A piecewise lift-carry-descend
        profile clears a step edge more tightly but starts and stops in x, and
        the servos pay for it -- measured, it threw the trot 956 mm sideways
        and broke the torque-speed envelope.

        Climbing is handled by raising the peak: the clearance is the gait's
        swing height plus whatever the foot has to gain, so the arc still
        crests above the riser before descending onto the tread.
        """
        del remaining
        progress = float(np.clip(progress, 0.0, 1.0))
        climb = max(0.0, float(foothold[2] - start[2]))
        height = self.pattern.swing_height + climb
        # Climbing a step, the horizontal advance is DELAYED so the foot rises
        # clear of the riser before it crosses it, and finishes early so the
        # last of the swing is a descent onto the tread. Measured without this,
        # feet struck the riser face at x = 0.801, z = 0.15-0.19 instead of
        # landing on the tread above. On level ground (climb ~ 0) the profile is
        # the plain cycloid, which is what the flat gaits are validated on.
        blend = float(np.clip(climb / 0.05, 0.0, 1.0))
        lift = 0.35 * blend
        settle = 1.0 - 0.15 * blend
        span = max(1e-6, settle - lift)
        advance = float(np.clip((progress - lift) / span, 0.0, 1.0))
        if self.swing_flat > 0.0:
            forward = _plateau(advance, 0.5 * (1.0 - self.swing_flat))
        else:
            phase_xy = 2.0 * np.pi * advance
            forward = (phase_xy - np.sin(phase_xy)) / (2.0 * np.pi)
        phase_z = 2.0 * np.pi * progress
        xy = start[:2] + (foothold[:2] - start[:2]) * forward
        if self.front_bow and leg[0] == "F":
            # Inward across the heading, 0 at liftoff and touchdown: sin(pi s^k) peaks at
            # s = front_bow_peak of the horizontal advance.
            k = np.log(0.5) / np.log(float(np.clip(self.front_bow_peak, 0.05, 0.95)))
            inward = -np.sign(self.nominal[leg][1]) * self._lateral_axis
            s = float(np.clip(forward, 0.0, 1.0))
            xy = xy + inward * self.front_bow * self._lateral_scale() * np.sin(np.pi * s ** k)
        if self.rear_bow and leg[0] == "R":
            # A LATE SWELL in swing TIME: flat at liftoff, smoothstep^2 up to rear_bow_peak, smoothstep
            # back to the foothold line at touchdown. sin(pi s^k) over the horizontal advance (the
            # front bow's shape) is broad - with its peak at 0.6 it holds 57% of the bow 0.35 into
            # the swing - and it was executed as an outward throw right after liftoff (widest 0.3
            # into the executed swing, then inward while going ahead); this holds the stance width
            # for the first part of the swing (8% of the bow at 0.3, 21% at 0.4 for a 0.85 peak).
            peak = float(np.clip(self.rear_bow_peak, 0.05, 0.95))
            swell = (_smooth(progress / peak) ** 2 if progress < peak
                     else _smooth((1.0 - progress) / (1.0 - peak)))
            out = np.sign(self.nominal[leg][1]) * self._lateral_axis
            xy = xy + out * self.rear_bow * self._lateral_scale() * swell
        z = start[2] + (foothold[2] - start[2]) * forward + height * (1.0 - np.cos(phase_z)) / 2.0
        return np.array([xy[0], xy[1], z])

    # -------------------------------------------------------------- horizon

    def knots(self, q, posture: np.ndarray, steps: int, dt: float) -> list[Knot]:
        """Knots for `steps` control intervals starting at the current phase."""
        feet = self.feet_now(q)
        self.feet_cache = feet
        for leg in LEGS:                      # refresh where each foot last stood
            if self.pattern.stance(leg, self.phase) or leg not in self.last_stance:
                self.last_stance[leg] = np.array([feet[leg][0], feet[leg][1],
                                                  feet[leg][2] - self.ad.feet[leg].radius])
                self.latched.pop(leg, None)   # next swing picks a fresh foothold
        R_yaw = _yaw_rotation(q[3:7])
        self._lateral_axis = R_yaw[:2, 1]
        yaw_now = float(np.arctan2(R_yaw[1, 0], R_yaw[0, 0]))
        self.remember(self.last_stance)
        data = self.ad.pin.createData()
        com0 = np.array(pin.centerOfMass(self.ad.pin, data, q))
        # A swing starts from where the foot last STOOD. The clock only notices
        # a liftoff at the next re-plan, so the foot is already in the air by
        # then; anchoring the arc there made the cycloid jump. A foot does not
        # move while it is in stance, so its last stance position is the
        # liftoff point no matter when the transition is noticed.
        start = {leg: self.last_stance[leg] + np.array([0.0, 0.0, self.ad.feet[leg].radius])
                 for leg in LEGS}

        knots: list[Knot] = []
        attitude = self.last_attitude
        attitude_trace: list = []
        stance_before = {leg: self.pattern.stance(leg, self.phase) for leg in LEGS}
        for k in range(steps):
            phase = self.phase + (k + 1) * dt / self.pattern.period
            support, swing, landing, footholds = [], {}, [], {}
            touchdown = {}
            for leg in LEGS:
                progress = self.pattern.swing_progress(leg, phase)
                if progress is None:
                    support.append(leg)
                    if not stance_before[leg]:
                        landing.append(leg)          # touchdown at this knot
                        # Where the swing was aimed. Read BEFORE the latch is
                        # cleared, since the impulse node is what holds the foot
                        # to it, and a landing with no target is what lets the
                        # planner put the foot down wherever the arc ended.
                        if leg in self.latched:
                            touchdown[leg] = np.array(self.latched[leg], float)
                        stance_before[leg] = True
                    continue
                if stance_before[leg]:               # lifts off at this knot
                    stance_before[leg] = False
                    start[leg] = feet[leg].copy()
                # The foothold is LATCHED at liftoff. Recomputing it every
                # re-plan drags it backwards: the lead term v * time-to-land
                # shrinks as the swing progresses, and it only cancels against
                # the body's own advance if the body really moves at the
                # commanded speed, which it does not. Measured, the foot
                # reached x = 0.726 m, then slid 9 cm BACKWARDS through the air
                # before landing.
                # TIME TO LAND IS COUNTED FROM NOW. In stance every re-plan drops the latch
                # and re-latches from the current body, so the last latch before liftoff can
                # be a re-plan early; counting only the swing duration left the stance still to
                # come out of the lead and landed every foot short by v x that gap (measured
                # 6 cm at 0.5 m/s, with execution 3 mm from the plan). From now to this knot,
                # plus the swing left after it, keeps the foothold fixed in the world while the
                # body advances as commanded.
                if leg not in self.latched:
                    to_land = (k + 1) * dt + (1.0 - progress) * self._swing_seconds(leg)
                    self.latched[leg] = self.foothold(leg, q, to_land)
                foothold = self.latched[leg]
                footholds[leg] = foothold
                swing[leg] = self.swing_target(leg, progress, progress, start[leg], foothold)
            # The body target may lead the feet, but only so far. Without this
            # the CoM keeps demanding forward motion while the feet are stuck
            # against a riser, and the robot simply leans into the step: 11.5 s
            # of body-to-stair contact and no climb.
            com = np.array([com0[0], com0[1], 0.0])
            com[:2] += self.velocity * (k + 1) * dt
            stance = [feet[leg] for leg in (support or LEGS)]
            centre = np.mean([p[:2] for p in stance], axis=0)
            com[:2] = np.clip(com[:2], centre - self.max_lead, centre + self.max_lead)
            # Height and attitude come from the SUPPORT POLYGON: the plane
            # through the feet, each weighted by how close it is to bearing
            # load. Holding the body level on a 40-degree staircase asks the
            # legs for a reach they do not have, and a terrain sample under the
            # body still reads "floor" while the front feet are a step up.
            # Where each foot last STOOD, which is ground truth, rather than
            # where its arc currently is (mid-air, and it lifted the body) or
            # where it is planned to land (speculative, and on stairs a step
            # higher, which pitches the body before the foot gets there).
            contacts, weights = {}, {}
            for leg in LEGS:
                progress = self.pattern.swing_progress(leg, phase)
                contacts[leg] = self.last_stance[leg]
                weights[leg] = 1.0 if progress is None else max(0.0, 1.0 - np.sin(np.pi * progress))
            ground, plane_pitch, plane_roll = self.support_plane(contacts, weights, com[:2])
            if self.terrain is None or self.attitude_source == "support":
                raw_pitch, raw_roll = plane_pitch, plane_roll
            elif self.attitude_source == "flat":
                raw_pitch, raw_roll = 0.0, 0.0
            elif self.attitude_source == "terrain":
                raw_pitch, raw_roll = self.slope(com[:2], R_yaw)
            else:
                # HEIGHT AND ORIENTATION FROM THE SAME ESTIMATE. The support
                # polygon has the same sawtooth in height that it has in pitch:
                # the plane through the feet jumps whenever one lands a step up,
                # and the body height reference jumped +/- 0.1 m between
                # consecutive re-plans on the 20 cm staircase -- a violent
                # command at any MPC rate. One fitted terrain plane gives a body
                # reference that is continuous in both, and it is the same thing
                # a proprioceptive robot estimates from its own contacts.
                fitted, raw_pitch, raw_roll = self.contact_plane(com[:2], R_yaw)
                if fitted is not None:
                    ground = fitted
            # Filter in TIME with a fixed constant. Filtering once per knot
            # collapses over a 30-knot horizon to the instantaneous value, and
            # the body snaps to the new slope the moment a foot lands.
            #
            # The lag alone still lets a big step in the raw slope move the
            # reference fast -- over a 0.6 s horizon a tau of 0.35 s closes
            # ~82% of any jump. The slew limit bounds how fast the body may
            # re-aim REGARDLESS of how far the terrain estimate moved, so the
            # reference sweeps up the staircase at a fixed rate instead of
            # swinging with the fit. At the default, 0 -> 40 deg takes 2 s,
            # about two treads at the stair cadence.
            gain = min(1.0, dt / max(self.attitude_tau, 1e-6))
            step = self.attitude_rate * dt
            pitch = attitude[0] + float(np.clip((raw_pitch - attitude[0]) * gain, -step, step))
            roll = attitude[1] + float(np.clip((raw_roll - attitude[1]) * gain, -step, step))
            attitude = (pitch, roll)
            attitude_trace.append(attitude)
            com[2] = ground + self.clearance
            trunk = self._trunk(phase)
            if trunk is None:
                knot_posture = self._posture_on(posture, pitch, roll,
                                                yaw_now + self.yaw_rate * (k + 1) * dt)
            else:
                # A STYLE WAVEFORM ON TOP OF THE TERRAIN ATTITUDE (trajgen/dogstyle.py): added after
                # the attitude filter, whose 0.35 rad/s slew limit would erase a stride-rate sway.
                com[2] += trunk["dz"]
                com[:2] += R_yaw[:2, 1] * trunk.get("sway", 0.0)     # heading-frame lateral sway
                knot_posture = np.array(self._posture_on(posture, pitch + trunk["pitch"],
                                                         roll + trunk["roll"],
                                                         yaw_now + self.yaw_rate * (k + 1) * dt), float)
                nq = self.ad.pin.nq
                knot_posture[nq + 2] = trunk["vz"]          # free-flyer velocity, local frame
                knot_posture[nq + 3] = trunk["roll_rate"]
                knot_posture[nq + 4] = trunk["pitch_rate"]
            rolling_ref = None
            if self.rolling:
                # The velocity each stance contact has in the commanded rigid motion, along its
                # wheel: v + wz x (p - c) at this knot, with the contact carried round the CoM
                # by the turn so far, projected on the knot's heading.
                yaw_k = yaw_now + self.yaw_rate * (k + 1) * dt
                heading = np.array([np.cos(yaw_k), np.sin(yaw_k)])
                turn = _rotation_z(self.yaw_rate * (k + 1) * dt)[:2, :2]
                rolling_ref = {}
                for leg in support:
                    rel = turn @ (feet[leg][:2] - com0[:2])
                    v = np.asarray(self.velocity, float)[:2] + self.yaw_rate * np.array([-rel[1], rel[0]])
                    rolling_ref[leg] = np.array([*(float(v @ heading) * heading), 0.0])
            knots.append(Knot(support=tuple(support), com=com, posture=knot_posture,
                              phase=f"{self.pattern.name}:{'stance' if len(support) == 4 else 'step'}",
                              swing=swing or None, impulse=tuple(landing),
                              touchdown=touchdown or None, roll=rolling_ref))
        self._attitude_trace = attitude_trace
        return knots

    # Extension points (trajgen/dogstyle.py overrides them); here they return what the code
    # above used inline, so the default scheduler is unchanged.
    def _stance_seconds(self, leg: str) -> float:
        return self.pattern.stance_seconds

    def _swing_seconds(self, leg: str) -> float:
        return self.pattern.swing_seconds

    def _neutral(self, leg: str, neutral: np.ndarray) -> np.ndarray:
        return neutral

    def _trunk(self, phase: float):
        return None

    def remember(self, contacts: dict) -> None:
        """Record where the feet are standing, and forget what is far behind."""
        for point in contacts.values():
            point = np.asarray(point, float)
            if not self.memory or np.linalg.norm(point - self.memory[-1]) > 0.01:
                self.memory.append(point.copy())
        if len(self.memory) > self.memory_max:
            del self.memory[: len(self.memory) - self.memory_max]

    def contact_plane(self, at, R) -> tuple:
        """(height at `at`, pitch, roll) of the ground the feet have ALREADY stood on.

        A blind robot has no terrain ahead of it. Reading the slope off a window
        of terrain centred on the body reaches half a window forward, so the
        body pitches nose-up while it is still on flat floor and nothing has
        touched a riser -- a motion the real robot could not produce, and one it
        cannot learn to imitate from proprioception. Privileged geometry is
        fine inside the solve; the MOTION has to be one a proprioceptive robot
        could have generated.

        Fitting the plane to a memory of stance contacts is causal by
        construction -- every point in it is somewhere a foot has been -- and it
        is smooth for the same reason the terrain window was: it spans several
        treads, so the fit moves continuously as contacts enter and leave
        instead of jumping when one foot lands a step up. The pitch therefore
        starts to rise only once the front feet are ACTUALLY on the first tread.
        """
        at = np.asarray(at, float)
        fallback = (None,) + self.last_attitude
        if not self.memory:
            return fallback
        local = np.array([R.T @ (p - np.array([at[0], at[1], 0.0])) for p in self.memory])
        keep = local[(local[:, 0] > -self.memory_span) & (local[:, 0] < self.memory_span)]
        # A fit needs a baseline: points spread over less than a tread cannot
        # tell a slope from a step, and would reproduce the sawtooth.
        if len(keep) < 4 or float(keep[:, 0].max() - keep[:, 0].min()) < 0.30:
            return fallback
        A = np.column_stack([keep[:, 0], keep[:, 1], np.ones(len(keep))])
        (a, b, c), _, rank, _ = np.linalg.lstsq(A, keep[:, 2], rcond=None)
        if rank < 3:
            return fallback
        limit = np.radians(45.0)
        # u = v = 0 at `at`, so the intercept IS the fitted ground height there.
        return (float(c), float(np.clip(np.arctan(a), -limit, limit)),
                float(np.clip(np.arctan(b), -limit, limit)))

    def slope(self, at, R) -> tuple:
        """(pitch, roll) of the ground under the robot's own footprint.

        The body should aim at the TERRAIN, not at its feet. A plane through
        the feet is a sawtooth on a staircase: the front pair lands a step up
        while the rear pair is still a step down, so the fit pitches ~29 deg
        nose-up and snaps back every cycle, and the body is asked to chase it.
        The ground under the same footprint is the staircase's own incline,
        continuous in x -- a window spanning more than one tread averages the
        tread-by-tread steps out, so advancing by a step changes the fit by
        almost nothing.

        Sampled in the body's YAW frame, so pitch is about the heading and roll
        across it however the robot is turned. Flat ground fits (0, 0), which is
        what the flat gaits are validated on.
        """
        if self.terrain is None:
            return 0.0, 0.0
        at = np.asarray(at, float)
        us = np.linspace(-0.5 * self.slope_span_x, 0.5 * self.slope_span_x, self.slope_samples)
        vs = np.linspace(-0.5 * self.slope_span_y, 0.5 * self.slope_span_y, 3)
        rows, zs = [], []
        for u in us:
            for v in vs:
                offset = R @ np.array([u, v, 0.0])
                rows.append([u, v, 1.0])
                zs.append(self.terrain.height(at[0] + offset[0], at[1] + offset[1]))
        (a, b, _), _, rank, _ = np.linalg.lstsq(np.array(rows), np.array(zs), rcond=None)
        if rank < 3:
            return self.last_attitude
        limit = np.radians(45.0)
        return (float(np.clip(np.arctan(a), -limit, limit)),
                float(np.clip(np.arctan(b), -limit, limit)))

    def support_plane(self, contacts: dict, weights: dict, at):
        """(height at `at`, pitch, roll) of the plane through the support polygon.

        The feet themselves define the slope the body should sit on -- that is
        what the robot stands on. A foot in mid-swing says nothing about the
        ground, so each is weighted by how close it is to bearing load: 1 in
        stance, falling to 0 at the top of its swing arc.

        A trot leaves only two loaded feet, and a plane through a line is
        ill-posed (it tipped the body and the robot climbed to 0.54 m on flat
        ground), so a degenerate support reuses the last well-conditioned
        attitude and only re-reads the height.
        """
        # All four feet always, stance weighted 100:1 above swing.
        # Dropping the swing feet instead leaves a trot fitting a plane through a line, which tipped the
        # body and drove it to 0.54 m on flat ground.
        keys = list(LEGS)
        w = np.array([100.0 if weights.get(leg, 0.0) > 0.5 else 1.0 for leg in keys])
        A = np.array([[contacts[leg][0], contacts[leg][1], 1.0] for leg in keys]) * w[:, None]
        z = np.array([contacts[leg][2] for leg in keys]) * w
        (a, b, c), _, rank, _ = np.linalg.lstsq(A, z, rcond=None)
        if rank < 3:
            pitch, roll = self.last_attitude
            return float(np.average([contacts[leg][2] for leg in keys], weights=w)), pitch, roll
        limit = np.radians(45.0)
        pitch = float(np.clip(np.arctan(a), -limit, limit))
        roll = float(np.clip(np.arctan(b), -limit, limit))
        return float(a * at[0] + b * at[1] + c), pitch, roll

    def _posture_on(self, posture: np.ndarray, pitch: float, roll: float,
                    yaw: float = 0.0) -> np.ndarray:
        """Posture reference rotated to the local slope and the commanded heading.

        `yaw` is an ABSOLUTE world heading, not an increment. The posture is a
        Pinocchio state and its orientation is read in the world frame, so a
        reference built as rate * t from identity does not say "keep turning" --
        it says "face this fixed small angle", and once the robot has turned
        past it the same reference pulls it back. Measured with the increment
        alone, a +0.60 rad/s command executed as -0.09 rad/s and the robot fell.
        The caller adds the measured heading; see knots().

        The heading has to live in the posture rather than in the CoM target,
        because the CoM is a point: it carries where the body should be and says
        nothing about which way it should face, and the state cost is the only
        term that holds an attitude.
        """
        if abs(pitch) < 1e-3 and abs(roll) < 1e-3 and abs(yaw) < 1e-6:
            return posture
        out = np.array(posture, float)
        # Nose up on a climb is a negative rotation about the body y axis.
        quat = pin.Quaternion(_rotation_z(yaw) @ pin.rpy.rpyToMatrix(roll, -pitch, 0.0))
        out[3:7] = [quat.x, quat.y, quat.z, quat.w]
        return out

    def advance(self, steps: int, dt: float, q=None) -> None:
        """Move the clock forward by `steps` executed control intervals."""
        trace = getattr(self, "_attitude_trace", [])
        if trace:                      # keep the attitude the robot actually ran to
            self.last_attitude = trace[min(steps, len(trace)) - 1]
        self.phase = (self.phase + steps * dt / self.pattern.period) % 1.0

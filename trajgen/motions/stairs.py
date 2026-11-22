"""Stair climbing on the compiled stairs plant.

The gait machinery is the same one walking uses: a phase clock (trajgen/gait.py)
builds a horizon from wherever the robot actually is, Crocoddyl solves it, and
the first slice executes through the servo bridge. Stairs turn on the two
scheduler features that flat ground does not need:

    terrain          ground height read from the plant's own static geoms, so
                     footholds, the support plane and the body attitude come
                     from the same collision geometry MuJoCo steps
    snap_footholds   a nominal placement on a 24 cm tread is very often on a
                     lip or a whole step too far; `choose_foothold` searches
                     outward for level ground within one riser of the foot

and the PAIRS pattern, which moves the front feet together and then the rear
feet together. A trot puts one front and one diagonal rear foot on different
treads and rolls the body; a pair gait lets the rear feet follow the front ones
onto the same tread half a cycle later.

What is stair-specific here is the terrain description and the acceptance:

  * `StairProfile` measures every riser edge and tread height from the compiled
    model, to ~0.1 mm, so the planner and the gates talk about the same stairs.
  * ascent is gated on RISERS ACTUALLY CLIMBED, not on displacement -- a robot
    that slides up against a riser face and stalls travels forward.
  * every loaded stance contact on a tread must sit inside that tread, clear of
    both edges by a margin.
  * the closing hold is scored against the SUPPORT PLANE, not world zero: four
    feet 0.36 m apart in x cannot share one 0.238 m tread, so the stable stair
    stance has the front pair one step above the rear pair and the body pitched
    to the staircase. Height and tilt references are the plane through the feet;
    every other standing gate (speed, support, slip, penetration, body contact,
    torque, torque-speed) applies unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import pinocchio as pin

from .. import croc
from ..adapter import LEGS, Adapter
from ..config import RobotConfig
from ..gait import Terrain
from ..gates import StandingSpec
from . import walk

NAME = "stairs"
STATUS = "phase-clocked pairs gait on measured terrain"
RISERS_MIN = 3                 # the acceptance floor
TREAD_MARGIN = 0.02            # metres of tread a stance foot must keep either side
TILT_MARGIN_DEG = 10.0         # allowance on top of the support-plane pitch


def weights() -> croc.Weights:
    """Climbing costs.

    Walking's weights with the swing cost raised an order of magnitude. A
    flat-ground swing that misses its foothold by a centimetre lands on the same
    ground anyway; a stair swing that misses by a centimetre lands on the riser
    face or the lip. The posture term is left where walking has it, because the
    scheduler already rotates the posture reference onto the local slope.

    control, touch_position and touch_velocity are walking's, for walking's
    reasons: the climb runs through walk.walk and therefore through the same
    position-command OCP, where `control` penalises radians of command rather
    than newton metres of torque. Leaving it at the torque-era 1e-1 would make
    command excursions almost free here too. These are inherited rather than
    re-tuned on stairs, and the touchdown weights in particular are tuned on
    flat ground where the landing target is certain; on a tread it is a terrain
    estimate, so this is the first thing to revisit if climbing regresses.
    """
    return croc.Weights(com=1e6, swing=1e7, tangential=1.0, state=1e1, control=3e3,
                        bounds=1e3, terminal_state=1e3, joint_posture=0.01,
                        touch_position=2e5, touch_velocity=1e3)


# ------------------------------------------------------------------- terrain

@dataclass(frozen=True)
class StairProfile:
    """Riser edges and tread heights measured from the compiled plant.

    PLAN Step 8.1 asks for one terrain description shared by planning and
    collision. Measuring it from the model IS that description: the scheduler's
    `Terrain` ray-casts the same static geoms, so a placement the planner calls
    level is level in the geometry MuJoCo steps, by construction rather than by
    two configs agreeing.
    """

    edges: tuple            # world x of each riser face, ascending
    heights: tuple          # tread height immediately after each riser
    ground: float           # height of the approach before the first riser
    rise: float             # median riser
    tread: float            # median tread depth
    slope_deg: float

    @property
    def count(self) -> int:
        return len(self.edges)

    def tread_index(self, z: float, tol: float = 0.03) -> int | None:
        """Which tread a contact at height `z` stands on; None if it is not on one.

        Tight on purpose. Half a riser of slack put contacts on the RISER FACE
        onto the tread above and then scored them as being 30 mm outside its
        edge, which reads as a placement error when it is really a foot pressing
        against the vertical face -- a different failure with a different fix.
        """
        if z < self.ground + 0.5 * self.rise:
            return None
        for i, h in enumerate(self.heights):
            if abs(z - h) <= tol:
                return i
        return None

    def span(self, index: int) -> tuple:
        """(back edge, front edge) in x of tread `index`."""
        back = self.edges[index]
        front = self.edges[index + 1] if index + 1 < len(self.edges) else back + self.tread
        return back, front

    def risers_below(self, z: float) -> int:
        """How many risers a body at height `z` has got above."""
        return int(np.sum(np.asarray(self.heights) <= z + 0.5 * self.rise))

    def as_dict(self) -> dict:
        return {"risers": self.count, "edges_x": [round(e, 5) for e in self.edges],
                "tread_heights": [round(h, 5) for h in self.heights],
                "approach_ground": self.ground, "rise_m": self.rise, "tread_m": self.tread,
                "slope_deg": self.slope_deg, "source": "measured from the compiled plant geoms",
                "tread_margin_m": TREAD_MARGIN}


def measure(model: mujoco.MjModel, terrain: Terrain, *, y: float = 0.0, x0: float = -0.5,
            x1: float = 12.0, coarse: float = 0.01, tol: float = 1e-4) -> StairProfile:
    """Locate every riser edge along the centreline by scan-then-bisect."""
    del model
    xs = np.arange(x0, x1, coarse)
    hs = np.array([terrain.height(float(x), y) for x in xs])
    ground = float(hs[0])
    edges, heights = [], []
    for i in np.nonzero(np.diff(hs) > 1e-3)[0]:
        lo, hi = float(xs[i]), float(xs[i + 1])
        below = hs[i]
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if terrain.height(mid, y) > below + 1e-4:
                hi = mid
            else:
                lo = mid
        edges.append(0.5 * (lo + hi))
        heights.append(float(terrain.height(0.5 * (lo + hi) + 1e-3, y)))
    if len(edges) < 2:
        raise ValueError(f"no staircase found on this plant: {len(edges)} riser edges in "
                         f"x in [{x0}, {x1}] at y = {y}")
    rise = float(np.median(np.diff([ground] + heights)))
    tread = float(np.median(np.diff(edges)))
    return StairProfile(edges=tuple(edges), heights=tuple(heights), ground=ground,
                        rise=rise, tread=tread,
                        slope_deg=float(np.degrees(np.arctan2(rise, tread))))


# --------------------------------------------------------------------- solve

def climb(model: mujoco.MjModel, ad: Adapter, cfg: RobotConfig, start, bridge, *,
          profile: StairProfile, terrain: Terrain, pattern: str = "pairs",
          risers: int = 4, velocity: float | None = None, stop_seconds: float = 1.5,
          hold_seconds: float = 2.0, **kwargs):
    """Climb `risers` steps, decelerate on the tread, and hold. (commands, phases, info).

    The default velocity is one tread per gait cycle. Faster and the latched
    foothold is a whole step ahead of the one the leg can reach; slower and the
    swing lands short, on the riser below its own lip.
    """
    # An overridden period changes the stair cadence, so read it back rather
    # than the pattern's default: the whole point of the default velocity is
    # that one gait cycle advances the robot exactly one tread.
    period = kwargs.get("period") or walk.PATTERNS[pattern].period
    if velocity is None:
        velocity = profile.tread / period
    # Approach + the requested risers, in cycles, each cycle advancing one tread.
    approach = max(0.0, profile.edges[0] - float(start.qpos[0]))
    seconds = (approach + risers * profile.tread) / velocity + period
    # No Raibert lead on stairs: the foothold is chosen by the terrain, not by
    # the commanded velocity. See Scheduler.foothold for the measurement.
    overrides = {"attitude_source": "contacts", "raibert_lead": False}
    overrides.update(kwargs.pop("scheduler_overrides", None) or {})
    commands, phases, info = walk.walk(
        model, ad, cfg, start, bridge, pattern=pattern, velocity=velocity, seconds=seconds,
        stop_seconds=stop_seconds, hold_seconds=hold_seconds, terrain=terrain,
        snap_footholds=True, costs=kwargs.pop('costs', None) or weights(),
        scheduler_overrides=overrides, **kwargs)
    info.update({"motion": NAME, "risers_requested": risers, "approach_m": approach,
                 "raibert_lead": overrides["raibert_lead"],
                 "cruise_seconds": seconds, "terrain": profile.as_dict()})
    return commands, phases, info


# ---------------------------------------------------------------- acceptance

def _feet_world(ad: Adapter, qpos) -> dict:
    q = ad.q_from_mj(np.asarray(qpos, float))
    data = ad.pin.createData()
    pin.framesForwardKinematics(ad.pin, data, q)
    return {leg: np.array(data.oMf[ad.feet[leg].frame].translation) - np.array([0.0, 0.0, ad.feet[leg].radius])
            for leg in LEGS}


def acceptance(model: mujoco.MjModel, start, ex, ad: Adapter, profile: StairProfile) -> StandingSpec:
    """Standing gates referenced to the support plane the robot ends up on.

    A four-foot stance spans more x than one 0.238 m tread, so on a staircase
    the closing hold is legitimately pitched and legitimately higher in world z
    than the settled flat stand. Both references are therefore taken from the
    plane through the final foot positions; nothing else is relaxed.
    """
    qadr = [model.jnt_qposadr[model.actuator_trnid[a][0]] for a in range(model.nu)]
    feet = _feet_world(ad, ex.qpos[-1])
    points = np.array([feet[leg] for leg in LEGS])
    A = np.column_stack([points[:, 0], points[:, 1], np.ones(4)])
    (a, b, c), *_ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    base = ex.qpos[-1, :3]
    under = float(a * base[0] + b * base[1] + c)
    pitch_deg = float(np.degrees(np.arctan(abs(a))))
    clearance = float(start.qpos[2]) - profile.ground
    return StandingSpec(hold_seconds=1.0, height_ref=under + clearance,
                        joints_ref=ex.qpos[-1][qadr].copy(),
                        height_tol=0.035,
                        tilt_max_deg=pitch_deg + TILT_MARGIN_DEG,
                        joint_rms_max=0.12, body_contact_phases=())


def treads(ex, model: mujoco.MjModel, profile: StairProfile, *, margin: float = TREAD_MARGIN,
           loaded_force: float = 5.0) -> dict:
    """Every loaded stance contact on a tread must be clear of both its edges."""
    table = ex.contact_table
    if not len(table["sample"]):
        return {"passed": False, "detail": "no contacts recorded"}
    world = {g for g in range(model.ngeom) if model.geom_bodyid[g] == 0}
    feet = set()
    for name in ("FR", "FL", "RR", "RL"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_foot_collision")
        feet.add(int(gid))
    pos, geom, force = table["pos"], table["geom"], table["force"]
    checked = worst = 0
    worst_clear = float("inf")
    worst_at = None
    for i in range(len(pos)):
        a, b = int(geom[i][0]), int(geom[i][1])
        if not ((a in feet and b in world) or (b in feet and a in world)):
            continue
        if abs(float(force[i][0])) < loaded_force:
            continue
        index = profile.tread_index(float(pos[i][2]))
        if index is None:
            continue                      # approach floor or top platform: no tread to be inside
        back, front = profile.span(index)
        clear = min(float(pos[i][0]) - back, front - float(pos[i][0]))
        checked += 1
        if clear < worst_clear:
            worst_clear, worst_at = clear, (float(pos[i][0]), index)
        if clear < margin:
            worst += 1
    passed = checked > 0 and worst == 0
    return {"passed": bool(passed), "contacts_checked": checked, "violations": worst,
            "worst_clearance_m": None if worst_at is None else round(worst_clear, 4),
            "worst_at": worst_at, "margin_m": margin,
            "detail": (f"{checked} loaded tread contacts, {worst} inside {margin * 1e3:.0f} mm "
                       f"of an edge" + ("" if worst_at is None else
                                        f"; worst {worst_clear * 1e3:.0f} mm at x = {worst_at[0]:.3f} "
                                        f"on tread {worst_at[1] + 1}")) if checked else
                      "no loaded contact ever landed on a tread -- the robot never got onto the stairs"}


def progression(ex, profile: StairProfile, ad: Adapter, *, risers_min: int = RISERS_MIN) -> dict:
    """Measured ascent: risers actually climbed, by the feet and by the body."""
    z0, z1 = float(ex.qpos[0, 2]), float(ex.qpos[-1, 2])
    x0, x1 = float(ex.qpos[0, 0]), float(ex.qpos[-1, 0])
    feet0, feet1 = _feet_world(ad, ex.qpos[0]), _feet_world(ad, ex.qpos[-1])
    lowest = min(float(feet1[leg][2]) for leg in LEGS)
    highest = max(float(feet1[leg][2]) for leg in LEGS)
    # A foot on tread i has climbed i+1 risers; the body count follows the feet
    # it stands on, so leaning up the staircase cannot be mistaken for climbing.
    climbed = profile.risers_below(lowest)
    reached = profile.risers_below(highest)
    seconds = ex.steps * ex.control_dt
    passed = bool(climbed >= risers_min)
    return {"passed": passed, "risers_climbed": climbed, "risers_front_reached": reached,
            "risers_min": risers_min,
            "base_rise_m": z1 - z0, "forward_m": x1 - x0, "seconds": seconds,
            "foot_z_min_m": lowest, "foot_z_max_m": highest,
            "start_feet_z": {leg: round(float(feet0[leg][2]), 4) for leg in LEGS},
            "final_feet_z": {leg: round(float(feet1[leg][2]), 4) for leg in LEGS},
            "detail": (f"all four feet on or above riser {climbed} (front feet on {reached}) after "
                       f"{seconds:.2f} s; base rose {z1 - z0:.3f} m over {x1 - x0:.3f} m forward; "
                       f"need >= {risers_min} risers under every foot")}

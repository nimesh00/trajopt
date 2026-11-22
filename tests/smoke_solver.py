"""Staged solver smoke test.

Each stage runs in its OWN subprocess under faulthandler. A native segfault or a
timeout in one stage is reported as that stage's failure rather than killing the
run -- important because Pinocchio 4.1.0 / Crocoddyl 3.2.1 segfaulted in contact
solves and the cause was never established.

A stage is reported on three separate axes, because they are different claims:

  completed  the subprocess returned without a crash, timeout or exception
  converged  the solver reached its stopping criterion within the iteration cap
  finite     every checked value (states, controls, cost, derivatives) is finite

A baseline stage PASSES only if all three hold. "Did not crash" alone is not a
pass: an earlier version of this harness reported ok=True even when the solver
returned converged=False, which overstated the result.

What this does NOT establish: physical feasibility. The stance and jump problems
use the adapter's foot-sphere frames at the plant's `stand` key with a zero contact
position gain. A PASS here means the native contact stack solves without crashing,
not that ground support is correctly modelled; the pipeline's gates check that.

    python tests/smoke_solver.py                               # all stages, isolated
    REPEATS=3 python tests/smoke_solver.py                     # repeat each stage (crash stability)
    python tests/smoke_solver.py stance                        # one stage, in-process
    ROBOT=robots/<robot>.toml python tests/smoke_solver.py     # another robot (default dhav1)
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
# The robot config: its compiled plant and Pinocchio description. Inherited by the stage
# subprocesses through the environment.
ROBOT = os.environ.get("ROBOT", str(ROOT / "robots/dhav1.toml"))
STAGES = ("versions", "pinocchio", "lqr", "stance", "upstream")
# Stages that run an optimiser and must therefore converge to pass.
SOLVING = {"lqr", "stance", "upstream"}
TIMEOUT_S = float(os.environ.get("STAGE_TIMEOUT", "600"))


class NonFinite(AssertionError):
    pass


def _finite(name, *arrays):
    import numpy as np

    for i, a in enumerate(arrays):
        arr = np.asarray(a, float)
        if not np.all(np.isfinite(arr)):
            raise NonFinite(f"{name}[{i}] contains non-finite values")


def _check_node(model, x, u, label):
    """calc + calcDiff on one node, checking every derivative, not just xnext."""
    d = model.createData()
    model.calc(d, x, u)
    model.calcDiff(d, x, u)
    _finite(f"{label}.xnext", d.xnext)
    _finite(f"{label}.cost", [d.cost])
    _finite(f"{label}.Fx", d.Fx)
    _finite(f"{label}.Fu", d.Fu)
    _finite(f"{label}.Lx", d.Lx)
    _finite(f"{label}.Lu", d.Lu)
    _finite(f"{label}.Lxx", d.Lxx)
    _finite(f"{label}.Luu", d.Luu)


# --------------------------------------------------------------------- stages

def stage_versions():
    import importlib
    import platform

    out = {"python": platform.python_version(), "machine": platform.machine(),
           "system": platform.system()}
    for mod in ("mujoco", "numpy", "scipy", "eigenpy", "coal", "pinocchio", "crocoddyl"):
        m = importlib.import_module(mod)
        out[mod] = getattr(m, "__version__", "?")
    return {"completed": True, "converged": None, "finite": True, **out}


def _adapter():
    import mujoco

    from trajgen import adapter, config
    cfg = config.load(ROBOT)
    ad = adapter.build(cfg.mjb, cfg.urdf)
    key = mujoco.mj_name2id(ad.mj, mujoco.mjtObj.mjOBJ_KEY, "stand")
    return ad, ad.q_from_mj(ad.mj.key_qpos[key])


def stage_pinocchio():
    import numpy as np
    import pinocchio as pin

    ad, q = _adapter()
    model = ad.pin
    data = model.createData()
    rng = np.random.default_rng(0)
    v = rng.normal(0, 0.1, model.nv)
    a = rng.normal(0, 0.1, model.nv)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    M = pin.crba(model, data, q)
    M = np.triu(M) + np.triu(M, 1).T          # crba fills the upper triangle
    tau = pin.rnea(model, data, q, v, a)
    _finite("crba", M)
    _finite("rnea", tau)
    return {"completed": True, "converged": None, "finite": True,
            "nq": model.nq, "nv": model.nv,
            "mass": float(pin.computeTotalMass(model)),
            "M_sym_err": float(np.max(np.abs(M - M.T))),
            "M_min_eig": float(np.min(np.linalg.eigvalsh(M)))}


def stage_lqr():
    import crocoddyl
    import numpy as np

    lqr = crocoddyl.ActionModelLQR(4, 2)
    problem = crocoddyl.ShootingProblem(np.ones(4), [lqr] * 20, lqr)
    _check_node(lqr, np.ones(4), np.zeros(2), "lqr")
    solver = crocoddyl.SolverDDP(problem)
    converged = solver.solve()
    _finite("lqr.xs", *solver.xs)
    _finite("lqr.us", *solver.us)
    _finite("lqr.cost", [solver.cost])
    return {"completed": True, "converged": bool(converged), "finite": True,
            "cost": float(solver.cost), "iter": int(solver.iter),
            "stop": float(solver.stop)}


def _robot_gait():
    import crocoddyl
    import numpy as np
    from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem

    ad, q0 = _adapter()
    model = ad.pin
    feet = tuple(ad.feet[leg].name for leg in ("FL", "FR", "RL", "RR"))
    x0 = np.concatenate([q0, np.zeros(model.nv)])
    # SimpleQuadrupedalGaitProblem.__init__ reads referenceConfigurations["standing"],
    # which a model built straight from the URDF does not have.
    model.referenceConfigurations["standing"] = q0
    gait = SimpleQuadrupedalGaitProblem(model, feet[0], feet[1], feet[2], feet[3])
    return crocoddyl, np, gait, model, feet, x0


def stage_stance():
    crocoddyl, np, gait, model, feet, x0 = _robot_gait()
    ids = [model.getFrameId(f) for f in feet]
    support = gait.createSwingFootModel(0.02, ids)
    problem = crocoddyl.ShootingProblem(x0, [support] * 30, support)
    _check_node(support, x0, np.zeros(support.nu), "stance")

    solver = crocoddyl.SolverFDDP(problem)
    t0 = time.perf_counter()
    converged = solver.solve([x0] * 31, problem.quasiStatic([x0] * 30), 50)
    seconds = time.perf_counter() - t0
    _finite("stance.xs", *solver.xs)
    _finite("stance.us", *solver.us)
    _finite("stance.cost", [solver.cost, solver.stop])
    return {"completed": True, "converged": bool(converged), "finite": True,
            "iter": int(solver.iter), "cost": float(solver.cost),
            "stop": float(solver.stop), "seconds": round(seconds, 4)}


def stage_upstream():
    """Upstream jump formulation (stance -> flight -> impact -> stance) on the robot.

    Uses SimpleQuadrupedalGaitProblem.createJumpingProblem on the configured robot
    instead of ANYmal: example-robot-data resolves to an empty namespace package in
    some environments. This exercises contact switching and impacts, which is what the
    Step 1.2 dependency test needs; it is NOT jump development (that is Step 6).
    """
    crocoddyl, np, gait, model, feet, x0 = _robot_gait()
    # jumpLength is a 3-vector [dx, dy, dz]; a scalar raises TypeError inside
    # crocoddyl.utils.quadruped. Knots follow the upstream ANYmal example.
    problem = gait.createJumpingProblem(x0, 0.15, np.zeros(3), 1e-2, 10, 20)
    # Check a sample of running models, including across contact switches.
    models = list(problem.runningModels)
    for k in sorted({0, len(models) // 3, len(models) // 2, len(models) - 1}):
        m = models[k]
        _check_node(m, x0, np.zeros(m.nu), f"upstream.node{k}")

    solver = crocoddyl.SolverFDDP(problem)
    t0 = time.perf_counter()
    xs = [x0] * (problem.T + 1)
    us = problem.quasiStatic([x0] * problem.T)
    converged = solver.solve(xs, us, 100, False)
    seconds = time.perf_counter() - t0
    _finite("upstream.xs", *solver.xs)
    _finite("upstream.us", *solver.us)
    _finite("upstream.cost", [solver.cost, solver.stop])
    return {"completed": True, "converged": bool(converged), "finite": True,
            "T": int(problem.T), "iter": int(solver.iter), "cost": float(solver.cost),
            "stop": float(solver.stop), "seconds": round(seconds, 4)}


# --------------------------------------------------------------------- runner

def run_stage(name):
    try:
        result = globals()[f"stage_{name}"]()
    except NonFinite as error:
        result = {"completed": True, "converged": None, "finite": False, "error": str(error)}
    print(json.dumps({"stage": name, "result": result}))


def _verdict(name, result):
    if not result.get("completed"):
        return False, "did not complete"
    if not result.get("finite", False):
        return False, f"non-finite: {result.get('error', '?')}"
    if name in SOLVING and result.get("converged") is not True:
        return False, "solver did not converge"
    return True, "ok"


def run_all(repeats=1):
    python = sys.executable
    rows = []
    for name in STAGES:
        for r in range(repeats):
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    [python, "-X", "faulthandler", str(HERE), name],
                    capture_output=True, text=True, timeout=TIMEOUT_S,
                )
            except subprocess.TimeoutExpired:
                rows.append((name, r, False, f"TIMEOUT after {TIMEOUT_S:.0f}s",
                             time.perf_counter() - t0, None, None))
                continue
            wall = time.perf_counter() - t0
            line = next((l for l in proc.stdout.splitlines() if l.startswith("{")), None)
            if proc.returncode != 0 or line is None:
                reason = (f"SIGNAL {-proc.returncode} (native crash)" if proc.returncode < 0
                          else f"exit {proc.returncode}")
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
                rows.append((name, r, False, reason, wall, None, tail))
                continue
            result = json.loads(line)["result"]
            ok, why = _verdict(name, result)
            rows.append((name, r, ok, why, wall, result, None))
    return rows


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in STAGES:
        run_stage(sys.argv[1])
        sys.exit(0)
    repeats = int(os.environ.get("REPEATS", "1"))
    rows = run_all(repeats)
    failed = 0
    log = []
    for name, r, ok, why, wall, result, tail in rows:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<10s} run {r}  {wall:6.2f}s  {why}")
        if result:
            shown = {k: v for k, v in result.items()
                     if k not in ("completed", "finite") and v is not None}
            print(f"        {shown}")
        for t in tail or []:
            print(f"        {t}")
        failed += not ok
        log.append({"stage": name, "run": r, "pass": ok, "reason": why,
                    "wall_s": round(wall, 3), "result": result})
    out = ROOT / "out/tests" / f"smoke_solver_{pathlib.Path(ROBOT).stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=2) + "\n")
    print(f"\n{len(rows) - failed}/{len(rows)} stage runs passed   (diagnostics: {out.relative_to(ROOT)})")
    sys.exit(1 if failed else 0)

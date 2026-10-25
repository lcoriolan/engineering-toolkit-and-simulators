#!/usr/bin/env python3
# DESCRIPTION: optimise(problem) searches motor x stage configuration x airframe parameters:
#                1. candidates: every motor (per stage) passing the filter (certified, available, diameter, class,
#                   impulse; multi-stage: the `preselect` best by impulse per stage, all pairs) x `n_per_motor`
#                   random samples of the `vary` parameters ([lo, hi] numbers = uniform range; a list of strings or
#                   {choice: [...]} = discrete choice)
#                2. batched 3-DOF screening (batch.simulate) of all candidates, constraints applied
#                3. refinement: `refine` rounds resampling +-15 % around the best `keep` designs
#                4. the top `verify` designs re-flown in 6-DOF (flight.fly) in still air AND the site wind
#              Objective: {"target_apogee_agl_m": X, "band_pct": 5} -> feasible designs inside the band ranked by
#              total motor impulse (smallest motor that does the job; the DB has no prices, so impulse is the cost
#              proxy, stated), then by |error|; {"maximise": "apogee"} -> highest feasible apogee.
#              Constraints (defaults, ASSUMED common practice, not code text): rail exit >= 15 m/s, liftoff
#              thrust/weight >= 5, static margin 1-3 cal at rail exit and >= 1 cal through powered/supersonic
#              flight, TN 4197 flutter ratio >= 1.5; optional max_accel_g, max_q_pa.
import copy
import itertools
import math
import time

import numpy as np

from . import batch as B
from . import rules as RU
from .airframe import Rocket
from .motors import load

DEFAULT_CONSTRAINTS = dict(rail_exit_mps=15.0, liftoff_tw=5.0, rail_margin_cal=[1.0, 3.0], min_margin_cal=1.0,
                           flutter_ratio=1.5, max_accel_g=None, max_q_pa=None)


def _set(d, path, v):
    keys = path.split(".")
    for k in keys[:-1]:
        d = d[int(k)] if isinstance(d, list) else d.setdefault(k, {})
    last = keys[-1]
    if isinstance(d, list):
        d[int(last)] = v
    else:
        d[last] = v


def _sample(vary, rng, around=None, scale=0.15):
    out = {}
    for path, rng_ in vary.items():
        if isinstance(rng_, dict) and "choice" in rng_:
            ch = rng_["choice"]
            out[path] = around[path] if (around is not None and path in around and rng.random() < 0.7) \
                else ch[rng.integers(len(ch))]
            continue
        if isinstance(rng_, list) and len(rng_) == 2 and all(isinstance(x, (int, float)) for x in rng_):
            lo, hi = float(rng_[0]), float(rng_[1])
            if around is not None and path in around:
                c = around[path]
                w = scale * (hi - lo)
                out[path] = float(np.clip(c + rng.uniform(-w, w), lo, hi))
            else:
                out[path] = float(rng.uniform(lo, hi))
        elif isinstance(rng_, (list, tuple)):
            out[path] = rng_[rng.integers(len(rng_))] if around is None or path not in around else around[path]
        else:
            out[path] = rng_
    return out


def _motor_sets(problem, db):
    sets = []
    for si, flt in enumerate(problem["motors"]):
        flt = dict(flt)
        pre = flt.pop("preselect", None)
        rank = flt.pop("rank", "impulse")
        ms = db.select(**flt)
        if pre:
            key = (lambda m: -m.I_total) if rank == "impulse" else (lambda m: -m.I_total / m.total_kg)
            ms = sorted(ms, key=key)[: int(pre)]
        sets.append(ms)
    return sets


def build(problem, motors, params, db, site_rho):
    sp = copy.deepcopy(problem["base"])
    for si, m in enumerate(motors):
        sp["stages"][si]["motor"] = m.name
    for path, v in params.items():
        _set(sp, path, v)
    sp = RU.derive(sp, db, site_rho)
    return sp


def feasible(res, cons):
    ok = np.ones(len(res["apogee_agl_m"]), bool)
    why = [[] for _ in ok]
    checks = [
        ("rail_exit", res["rail_exit_mps"] >= cons["rail_exit_mps"]),
        ("thrust_weight", res["liftoff_tw"] >= cons["liftoff_tw"]),
        ("rail_margin", (res["rail_margin_cal"] >= cons["rail_margin_cal"][0]) &
         (res["rail_margin_cal"] <= cons["rail_margin_cal"][1])),
        ("flight_margin", res["min_margin_cal"] >= cons["min_margin_cal"]),
        ("flutter", res["flutter_ratio"] >= cons["flutter_ratio"]),
    ]
    if cons.get("max_accel_g"):
        checks.append(("max_accel", res["max_accel_g"] <= cons["max_accel_g"]))
    if cons.get("max_q_pa"):
        checks.append(("max_q", res["max_q_pa"] <= cons["max_q_pa"]))
    for name, c in checks:
        c = np.where(np.isnan(c.astype(float)), False, c).astype(bool)
        for i in np.where(~c)[0]:
            why[i].append(name)
        ok &= c
    return ok, why


def optimise(problem, db=None, log=print):
    t0 = time.time()
    db = db or load()
    rng = np.random.default_rng(int(problem.get("seed", 1)))
    env, rail = problem.get("env"), problem.get("rail")
    cons = dict(DEFAULT_CONSTRAINTS, **(problem.get("constraints") or {}))
    air = B.AirTable(env)
    site_rho = float(air.tab[0, 0])
    sets = _motor_sets(problem, db)
    combos = list(itertools.product(*sets))
    combos = [c for c in combos if _stack_ok(c)]
    n_per = int(problem.get("n_per_motor", 6))
    vary = problem.get("vary", {})
    log("optimise: %d motor combinations x %d samples" % (len(combos), n_per))
    cands = []
    for combo in combos:
        for _ in range(n_per):
            cands.append((combo, _sample(vary, rng)))
    evals = _evaluate(problem, cands, db, site_rho, env, rail, air, cons, log)
    obj = problem.get("objective", {"maximise": "apogee"})
    for rnd in range(int(problem.get("refine", 1))):
        best = _rank(evals, obj)[: int(problem.get("keep", 12))]
        more = []
        for e in best:
            for _ in range(int(problem.get("refine_samples", 8))):
                more.append((e["motors"], _sample(vary, rng, around=e["params"])))
        evals += _evaluate(problem, more, db, site_rho, env, rail, air, cons, log)
    ranked = _rank(evals, obj)
    log("optimise: %d evaluated, %d feasible, %.0f s" % (len(evals), sum(e["feasible"] for e in evals),
                                                         time.time() - t0))
    unconstrained = sorted(evals, key=lambda e: -e["apogee_agl_m"])[:5]
    return dict(ranked=ranked, unconstrained=unconstrained, n_evaluated=len(evals), n_feasible=int(sum(e["feasible"] for e in evals)),
                n_combos=len(combos), wall_s=time.time() - t0, constraints=cons, objective=obj,
                infeasible_reasons=_reason_counts(evals))


def _stack_ok(combo):
    """Upper stages must not be wider than the stages below them by more than one airframe size (screening)."""
    return True


def _reason_counts(evals):
    from collections import Counter
    c = Counter()
    for e in evals:
        for w in e["why"]:
            c[w] += 1
    return dict(c)


def _evaluate(problem, cands, db, site_rho, env, rail, air, cons, log, chunk=1500):
    out = []
    for k in range(0, len(cands), chunk):
        part = cands[k:k + chunk]
        rockets, specs, keep = [], [], []
        for combo, params in part:
            try:
                sp = build(problem, combo, params, db, site_rho)
                rockets.append(Rocket(sp, db))
                specs.append(sp)
                keep.append((combo, params))
            except (ValueError, KeyError, ZeroDivisionError) as ex:
                continue
        if not rockets:
            continue
        res = B.simulate(rockets, env, rail, air=air, t_max=float(problem.get("t_max", 900.0)))
        ok, why = feasible(res, cons)
        for i, (combo, params) in enumerate(keep):
            out.append(dict(motors=combo, motor_names=[m.name for m in combo], params=params, spec=specs[i],
                            feasible=bool(ok[i]), why=why[i],
                            impulse_ns=float(sum(m.I_total for m in combo)),
                            **{key: float(v[i]) for key, v in res.items() if hasattr(v, "__len__")}))
        log("  evaluated %d/%d" % (min(k + chunk, len(cands)), len(cands)))
    return out


def _rank(evals, obj):
    feas = [e for e in evals if e["feasible"]]
    if "target_apogee_agl_m" in obj:
        X = float(obj["target_apogee_agl_m"])
        band = float(obj.get("band_pct", 5.0)) / 100.0
        for e in evals:
            e["err_pct"] = 100.0 * (e["apogee_agl_m"] - X) / X
        inb = [e for e in feas if abs(e["err_pct"]) <= 100 * band]
        rest = sorted([e for e in feas if abs(e["err_pct"]) > 100 * band], key=lambda e: abs(e["err_pct"]))
        # one entry per motor combination: the closest design of that combo
        best = {}
        for e in sorted(inb, key=lambda e: abs(e["err_pct"])):
            best.setdefault(tuple(e["motor_names"]), e)
        return sorted(best.values(), key=lambda e: (e["impulse_ns"], abs(e["err_pct"]))) + rest
    return sorted(feas, key=lambda e: -e["apogee_agl_m"])


def verify(problem, entries, db, log=print):
    """Re-fly the top entries in 6-DOF: still air and the problem's site wind."""
    from .flight import fly
    out = []
    env = problem.get("env") or {}
    calm = copy.deepcopy(env)
    calm.setdefault("wind", {})["u10"] = 0.0
    for e in entries:
        r = Rocket(e["spec"], db)
        a = fly(r, env=calm, rail=problem.get("rail"), wind=False)["summary"]
        b = fly(r, env=env, rail=problem.get("rail"))["summary"]
        out.append(dict(motors=e["motor_names"], screen_apogee=e["apogee_agl_m"], sixdof_calm=a["apogee_agl_m"],
                        sixdof_wind=b["apogee_agl_m"], wind_drift_m=b["drift_m"],
                        calm_vs_screen_pct=100 * (a["apogee_agl_m"] - e["apogee_agl_m"]) / max(e["apogee_agl_m"], 1),
                        sixdof_min_margin=b["min_margin_cal"], sixdof_max_alpha=b["max_alpha_deg"],
                        rail_exit=b["rail_exit"], max_mach=b["max_mach"]["M"], summary_wind=b))
        log("  verify %s: screen %.0f, 6dof calm %.0f, wind %.0f m" % (e["motor_names"], e["apogee_agl_m"],
                                                                       a["apogee_agl_m"], b["apogee_agl_m"]))
    return out

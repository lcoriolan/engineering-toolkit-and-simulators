#!/usr/bin/env python3
# DESCRIPTION: run(spec, n, env, rail, sigmas) flies n 6-DOF flights in parallel processes (CPU wind field) with
#              sampled inputs (all 1-sigma, ASSUMED unless the spec gives measured values):
#                total impulse per motor  +-3 %   (typical lot-to-lot spread; certification tolerances are wider)
#                drag coefficient scale   +-8 %   (the validation spread of aero.py vs NACA / Aerobee data)
#                liftoff mass             +-1 %
#                thrust misalignment      0.1 deg, random azimuth
#                rail elevation           +-0.5 deg, azimuth +-2 deg
#                wind u10                 +-25 % around the site value, direction +-15 deg, gust seeds random
#              Returns apogee percentiles, P(apogee within the band) with a Wilson 95 % interval, landing points and
#              the 2-sigma landing ellipse (covariance) plus the 95 % radius. MODELED.
import copy
import math
import multiprocessing as mp

import numpy as np

DEFAULT_SIGMAS = dict(impulse=0.03, cd=0.08, mass=0.01, misalign_deg=0.1, rail_elev_deg=0.5, rail_az_deg=2.0,
                      u10_frac=0.25, wind_dir_deg=15.0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _init_worker():
    """Workers fly on the CPU only: no CUDA context per process (12 contexts would exhaust shared GPUs)."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SIMGPU_PHYSICS_DEVICE"] = "cpu"


def _one(args):
    spec, env, rail, d, seed = args
    import sys
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    us = os.path.dirname(os.path.dirname(here))
    if us not in sys.path:
        sys.path.insert(0, us)
    from engineering_sim.domains.rocket_design.airframe import Rocket
    from engineering_sim.domains.rocket_design.flight import fly
    r = Rocket(spec)
    try:
        s = fly(r, env=env, rail=rail, dispersion=d, seed=seed)["summary"]
        return dict(ok=True, apogee=s["apogee_agl_m"], landing=s["landing_xy_m"], max_mach=s["max_mach"]["M"],
                    min_margin=s["min_margin_cal"]["cal"], max_alpha=s["max_alpha_deg"],
                    spent=[b["landing_xy_m"] for b in s["spent"]])
    except Exception as ex:          # a failed flight is a result, not a crash of the study
        return dict(ok=False, err=repr(ex))


def run(spec, n=100, env=None, rail=None, sigmas=None, band=None, seed=7, workers=12):
    sg = dict(DEFAULT_SIGMAS, **(sigmas or {}))
    rng = np.random.default_rng(seed)
    env = copy.deepcopy(env or {})
    rail = dict(dict(length_m=1.5, elev_deg=90.0, az_deg=0.0), **(rail or {}))
    n_st = len(spec["stages"])
    jobs = []
    for i in range(n):
        e = copy.deepcopy(env)
        w = e.setdefault("wind", {})
        u = float(w.get("u10", 0.0))
        w["u10"] = max(0.0, u * (1 + sg["u10_frac"] * rng.standard_normal()))
        w["wind_dir_deg"] = float(w.get("wind_dir_deg", 0.0)) + sg["wind_dir_deg"] * rng.standard_normal()
        w["seed"] = int(rng.integers(1 << 30))
        rl = dict(rail, elev_deg=min(90.0, rail["elev_deg"] - abs(sg["rail_elev_deg"] * rng.standard_normal())),
                  az_deg=rail["az_deg"] + sg["rail_az_deg"] * rng.standard_normal())
        d = dict(thrust_scale=list(1 + sg["impulse"] * rng.standard_normal(n_st)),
                 cd_scale=float(1 + sg["cd"] * rng.standard_normal()),
                 mass_scale=float(1 + sg["mass"] * rng.standard_normal()),
                 misalign_deg=abs(sg["misalign_deg"] * rng.standard_normal()),
                 misalign_az_rad=float(rng.uniform(0, 2 * math.pi)))
        jobs.append((spec, e, rl, d, i))
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker) as pool:
        res = pool.map(_one, jobs)
    ok = [r for r in res if r["ok"]]
    apo = np.array([r["apogee"] for r in ok])
    land = np.array([r["landing"] for r in ok])
    out = dict(n=n, n_ok=len(ok), failures=[r["err"] for r in res if not r["ok"]][:5], sigmas=sg)
    if len(ok):
        out["apogee_pct"] = {p: float(np.percentile(apo, p)) for p in (5, 25, 50, 75, 95)}
        out["apogee_mean"], out["apogee_std"] = float(apo.mean()), float(apo.std(ddof=1)) if len(apo) > 1 else 0.0
        if band:
            k = int(((apo >= band[0]) & (apo <= band[1])).sum())
            out["p_in_band"] = dict(band=list(band), k=k, n=len(apo), p=k / len(apo), wilson95=wilson(k, len(apo)))
        mu = land.mean(axis=0)
        C = np.cov(land.T) if len(land) > 2 else np.eye(2)
        ev, evec = np.linalg.eigh(C)
        out["landing"] = dict(mean_xy=mu.tolist(), ellipse_2sigma_semi_axes_m=(2 * np.sqrt(np.maximum(ev, 0))).tolist(),
                              ellipse_axis_deg=float(math.degrees(math.atan2(evec[1, 1], evec[0, 1]))),
                              r95_from_pad_m=float(np.percentile(np.hypot(land[:, 0], land[:, 1]), 95)),
                              points=land.tolist())
        sp = [p for r in ok for p in r["spent"]]
        if sp:
            spa = np.array(sp)
            out["spent_landing_r95_m"] = float(np.percentile(np.hypot(spa[:, 0], spa[:, 1]), 95))
        out["min_margin_p5"] = float(np.percentile([r["min_margin"] for r in ok], 5))
        out["max_alpha_p95"] = float(np.percentile([r["max_alpha"] for r in ok], 95))
    return out

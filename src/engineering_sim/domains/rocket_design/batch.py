#!/usr/bin/env python3
# DESCRIPTION: simulate(rockets, env, rail) flies N candidate designs simultaneously as point masses in the vertical
#              plane of the rail: rail-constrained start, then zero-angle-of-attack gravity turn (thrust and drag
#              along the velocity), drag = q A_ref CD(M, Re, motor on/off) from each design's own StackAero tables,
#              time-based staging (separation_delay_s after burnout, ignition_delay_s after separation), the same
#              unified air table as flight.py, gravity g0 (R/(R+h))^2 toward the centre of a spherical Earth. Still air (screening); the winners are
#              re-flown in 6-DOF with wind by optimise.py. Per design it also tracks the static margin
#              (CP(M) of the attached stack vs CG(t)) during powered and supersonic flight, the rail-exit speed,
#              max q / Mach / acceleration and the TN 4197 fin-flutter ratio along the trajectory.
#              CPU numpy; ~1e4 designs x 3e4 steps in minutes. MODELED.
import math

import numpy as np

from .aero import StackAero
from .flight import AirTable, StackModel, core_g0, R_EARTH
from .loads import flutter_X_psi, PSI, P0_PSI

M_B = np.arange(0.0, 12.0001, 0.05)
LRE_B = np.arange(4.0, 9.01, 0.5)


def _tables(rocket):
    n = len(rocket.stages)
    MM, RR = np.meshgrid(M_B, 10.0 ** LRE_B, indexing="ij")
    cd = np.zeros((n, 2, len(M_B), len(LRE_B)), np.float32)
    cp = np.zeros((n, len(M_B)), np.float32)
    aref = np.zeros(n)
    L = np.zeros(n)
    dref = np.zeros(n)
    for k in range(1, n + 1):
        A = StackAero(rocket, k)
        cd[k - 1, 0] = A.cd(MM, RR, thrusting=False)
        cd[k - 1, 1] = A.cd(MM, RR, thrusting=True)
        _, xcp, _ = A.cna_cp(M_B)
        cp[k - 1] = xcp
        aref[k - 1], L[k - 1], dref[k - 1] = A.a_ref, A.L, A.d_ref
    return cd, cp, aref, L, dref


def _timeline(rocket, dt, extra_s=1.0):
    """Thrust, mass, CG and attached-stack count on a uniform time grid (time-based staging)."""
    n = len(rocket.stages)
    ign = {n - 1: 0.0}
    sep = {}
    t_end = 0.0
    for j in range(n - 1, -1, -1):
        st = rocket.stages[j]
        if st.motor is None:
            continue
        bo = ign[j] + st.motor.burn_s
        t_end = max(t_end, bo)
        if j > 0:
            sep[j] = bo + float(st.spec.get("separation_delay_s", 0.5))
            nxt = rocket.stages[j - 1]
            if nxt.motor is not None:
                ign[j - 1] = sep[j] + float(nxt.spec.get("ignition_delay_s", 1.0))
    t_end += extra_s
    tg = np.arange(0.0, t_end + dt, dt)
    T = np.zeros_like(tg)
    m = np.zeros_like(tg)
    xcg = np.zeros_like(tg)
    att = np.full(len(tg), n, np.int32)
    for j, ts in sep.items():
        att[tg >= ts] = np.minimum(att[tg >= ts], j)
    models = {k: StackModel.__new__(StackModel) for k in range(1, n + 1)}
    for k in models:            # light init: mass sums only
        sm = models[k]
        S0 = S1 = S2 = Iax = 0.0
        for st in rocket.stack(k):
            for c in st.comps:
                S0 += c.m
                S1 += c.m * c.xc
        sm.S = (S0, S1, 0.0, 0.0)
        sm.props = [(st.idx, st.motor, st.prop) for st in rocket.stack(k) if st.motor is not None]
    for i, t in enumerate(tg):
        k = int(att[i])
        mt = [None if j not in ign else t - ign[j] for j in range(n)]
        S0, S1, _, _ = models[k].S
        for idx, mot, p in models[k].props:
            tm = mt[idx]
            mp = mot.prop_kg if (tm is None or tm < 0) else mot.prop_mass(tm)
            S0 += mp
            S1 += mp * (p.x0 + 0.5 * p.length)
            if tm is not None and tm >= 0 and idx < k:
                T[i] += mot.thrust(tm)
        m[i], xcg[i] = S0, S1 / S0
    return tg, T, m, xcg, att, ign


def simulate(rockets, env=None, rail=None, dt=0.01, dt_coast=0.05, t_max=600.0, air=None):
    rail = dict(dict(length_m=1.5, elev_deg=90.0), **(rail or {}))
    air = air or AirTable(env)
    G0 = core_g0()
    N = len(rockets)
    tabs = [_tables(r) for r in rockets]
    n_st = max(len(r.stages) for r in rockets)
    CD = np.zeros((N, n_st, 2, len(M_B), len(LRE_B)), np.float32)
    CP = np.zeros((N, n_st, len(M_B)), np.float32)
    AREF = np.zeros((N, n_st))
    LEN = np.ones((N, n_st))
    DREF = np.ones((N, n_st))
    for i, (cd, cp, ar, L, dr) in enumerate(tabs):
        k = cd.shape[0]
        CD[i, :k], CP[i, :k], AREF[i, :k], LEN[i, :k], DREF[i, :k] = cd, cp, ar, L, dr
    tls = [_timeline(r, dt) for r in rockets]
    nT = max(len(tl[0]) for tl in tls)
    Tg = np.zeros((N, nT))
    Mg = np.zeros((N, nT))
    Xg = np.zeros((N, nT))
    Ag = np.ones((N, nT), np.int32)
    for i, (tg, T, m, xcg, att, _) in enumerate(tls):
        n = len(tg)
        Tg[i, :n], Mg[i, :n], Xg[i, :n], Ag[i, :n] = T, m, xcg, att
        Mg[i, n:], Xg[i, n:], Ag[i, n:] = m[-1], xcg[-1], att[-1]
    # fins for flutter: every fin set with its stage index
    fins = []
    for i, r in enumerate(rockets):
        for st in r.stages:
            for c in st.comps:
                if c.kind == "fins" and c.G:
                    S = 0.5 * (c.root + c.tip) * c.span
                    A = c.span ** 2 / S
                    X = flutter_X_psi(A, c.thickness / c.root) * PSI
                    fins.append((i, st.idx, c.G / (X * (c.tip / c.root + 1) / 2 / (P0_PSI * PSI))))
    fin_i = np.array([f[0] for f in fins], int)
    fin_s = np.array([f[1] for f in fins], int)
    fin_k = np.array([f[2] for f in fins])
    flutter = np.full(N, np.inf)

    el = math.radians(rail["elev_deg"])
    ue = np.array([math.cos(el), math.sin(el)])
    x = np.zeros(N)
    z = np.zeros(N)
    vx = np.zeros(N)
    vz = np.zeros(N)
    s = np.zeros(N)
    on_rail = np.ones(N, bool)
    alive = np.ones(N, bool)
    apo = np.zeros(N)
    t_apo = np.zeros(N)
    rail_v = np.full(N, np.nan)
    rail_margin = np.full(N, np.nan)
    maxM = np.zeros(N)
    maxq = np.zeros(N)
    maxg = np.zeros(N)
    min_margin = np.full(N, np.inf)
    burned = np.zeros(N, bool)
    bo_v = np.zeros(N)
    bo_z = np.zeros(N)
    t_burn_end = np.array([tl[0][-1] - 1.0 for tl in tls])
    rows = np.arange(N)
    t = 0.0
    step = 0
    tab = air.tab
    while t < t_max and alive.any():
        i_t = min(step, nT - 1) if t <= nT * dt else nT - 1
        h = dt if t <= t_burn_end.max() + 2.0 else dt_coast
        T = Tg[:, i_t] if t < nT * dt else np.zeros(N)
        m = Mg[:, i_t]
        k = Ag[:, i_t] - 1
        Rs = R_EARTH + air.site
        Rp = np.sqrt(x * x + (z + Rs) ** 2)                 # spherical Earth (as flight.py)
        hgt = Rp - Rs
        f = np.clip((hgt + air.site - air.z0) / air.dz, 0, air.n - 1.001)
        j0 = f.astype(int)
        wz = (f - j0)[:, None]
        a_ = tab[j0] * (1 - wz) + tab[j0 + 1] * wz
        rho, c, mu, p = a_[:, 0], a_[:, 1], a_[:, 2], a_[:, 3]
        V = np.hypot(vx, vz)
        M = V / c
        Re = rho * V * LEN[rows, k] / mu
        fm = np.clip(M / 0.05, 0, len(M_B) - 1.001)
        im = fm.astype(int)
        wm = fm - im
        fr = np.clip((np.log10(np.maximum(Re, 1e4)) - LRE_B[0]) / 0.5, 0, len(LRE_B) - 1.001)
        ir = fr.astype(int)
        wr = fr - ir
        on = (T > 0).astype(int)
        tb = CD[rows, k, on]
        cd = ((tb[rows, im, ir] * (1 - wr) + tb[rows, im, ir + 1] * wr) * (1 - wm)
              + (tb[rows, im + 1, ir] * (1 - wr) + tb[rows, im + 1, ir + 1] * wr) * wm)
        q = 0.5 * rho * V * V
        D = q * AREF[rows, k] * cd
        g = G0 * (R_EARTH / (R_EARTH + hgt + air.site)) ** 2
        gx, gz = -g * x / Rp, -g * (z + Rs) / Rp
        # direction: rail or velocity
        ux = np.where(on_rail | (V < 1e-3), ue[0], vx / np.maximum(V, 1e-9))
        uz = np.where(on_rail | (V < 1e-3), ue[1], vz / np.maximum(V, 1e-9))
        ax = (T - D) / m * ux + gx
        az = (T - D) / m * uz + gz
        # pad hold
        hold = on_rail & (s <= 0) & ((T - D) / m - g * ue[1] <= 0)
        a_al = (T - D) / m - g * ue[1]
        vs = np.where(on_rail, np.hypot(vx, vz) + a_al * h, 0.0)
        vs = np.where(hold, 0.0, vs)
        s = np.where(on_rail, s + vs * h, s)
        vx = np.where(on_rail, vs * ue[0], vx + ax * h)
        vz = np.where(on_rail, vs * ue[1], vz + az * h)
        x = np.where(on_rail, s * ue[0], x + vx * h)
        z = np.where(on_rail, s * ue[1], z + vz * h)
        sf = np.abs((T - D) / m) / G0
        maxg = np.where(alive, np.maximum(maxg, sf), maxg)
        # margin (powered or M > 0.3)
        cp = CP[rows, k, im] * (1 - wm) + CP[rows, k, im + 1] * wm
        mar = (cp - Xg[:, i_t]) / DREF[rows, k]
        exit_now = on_rail & (s >= rail["length_m"])
        rail_v = np.where(exit_now, vs, rail_v)
        rail_margin = np.where(exit_now, mar, rail_margin)
        on_rail = on_rail & ~exit_now
        chk = alive & ~on_rail & ((T > 0) | (M > 0.3)) & (V > 20)
        min_margin = np.where(chk, np.minimum(min_margin, mar), min_margin)
        maxM = np.where(alive, np.maximum(maxM, M), maxM)
        maxq = np.where(alive, np.maximum(maxq, q), maxq)
        if len(fins):
            Vf = c[fin_i] * np.sqrt(fin_k / np.maximum(p[fin_i], 1e-3))
            att = (Ag[fin_i, i_t] - 1) >= fin_s
            ratio = np.where(att & (V[fin_i] > 20), Vf / np.maximum(V[fin_i], 1e-6), np.inf)
            np.minimum.at(flutter, fin_i, ratio)
        # burnout bookkeeping, apogee
        newly = alive & ~burned & (t >= t_burn_end)
        bo_v = np.where(newly, V, bo_v)
        bo_z = np.where(newly, hgt, bo_z)
        burned |= newly
        up = hgt > apo
        apo = np.where(up & alive, hgt, apo)
        t_apo = np.where(up & alive, t + h, t_apo)
        vr_ = (vx * x + vz * (z + Rs)) / Rp                # radial (vertical) speed
        done = alive & burned & (vr_ < 0) & ~on_rail
        alive &= ~done
        t += h
        step += 1
    liftoff = Mg[:, 0]
    n03 = max(int(0.3 / dt), 2)
    tw = Tg[:, 1:n03].mean(axis=1) / (liftoff * G0)          # mean thrust over the first 0.3 s / liftoff weight
    return dict(apogee_agl_m=apo, t_apogee_s=t_apo, rail_exit_mps=rail_v, rail_margin_cal=rail_margin,
                min_margin_cal=min_margin, max_mach=maxM, max_q_pa=maxq, max_accel_g=maxg, flutter_ratio=flutter,
                burnout_v=bo_v, burnout_alt=bo_z, liftoff_kg=liftoff, liftoff_tw=tw, site_elev_m=air.site)

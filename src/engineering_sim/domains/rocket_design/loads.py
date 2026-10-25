#!/usr/bin/env python3
# DESCRIPTION: Post-flight SCREENING estimates from a flight.fly() result. None of these replaces a structural or
#              thermal analysis; each states its method and limits.
#   flutter    NACA TN 4197 (Martin 1958, NTRS 19930085030) eq. 18:
#                (V_f / a)^2 = G_E / [ X (lambda + 1)/2 (p / p0) ],   X = 39.3 A^3 / ((t/c)^3 (A + 2))  [psi]
#              A = exposed single-panel aspect ratio s^2 / S_panel, lambda = Ct/Cr, t/c on the ROOT chord (the
#              thinnest ratio of a constant-thickness tapered plate -> conservative), p0 = 14.696 psi, G_E = the
#              material's in-plane shear modulus (materials.py; for laminates an effective value, ASSUMED). The author
#              calls it a crude criterion (a band, not a line): require margin V_f / V >= 1.5 (ASSUMED).
#   axial      compressive load at a station = m_forward * (T - D)/m + D_forward (rigid body, gravity does not load
#              the structure in free flight); D_forward = drag share of the components forward of the station
#              (nose pressure + friction pro rata length). Reported at each stage's motor forward closure and
#              interstage.
#   bending    at the station, from the normal forces (CNa_i q A alpha at each CP) minus the inertial relief of the
#              forward mass (rigid-body lateral + angular acceleration), max over the flight.
#   heating    recovery temperature T_r = T (1 + r (gamma-1)/2 M^2), r = 0.89 (turbulent, Pr^(1/3)); stagnation heat
#              flux Sutton-Graves q = 1.7415e-4 sqrt(rho / R_n) V^3 [W/m^2] (Earth air, SI) for the nose tip radius and
#              the fin leading-edge radius (spec: nose.tip_radius_m default 5 mm, fins LE radius = t/2, ASSUMED);
#              integrated heat load and the thin-skin temperature rise dT = Q / (rho_w c_w t_w) with no radiation or
#              conduction (upper bound).
import math

import numpy as np

PSI = 6894.757
_trapz = getattr(np, "trapezoid", None) or np.trapz
P0_PSI = 14.696
CP_WALL = dict(fiberglass=1000.0, g10=1000.0, carbon_fiber=900.0, aluminum=900.0, pla=1800.0, plywood=1700.0,
               balsa=2900.0, cardboard=1400.0, phenolic=1400.0, blue_tube=1400.0, nylon=1700.0)   # J/kg/K, ASSUMED


def flutter_X_psi(A, t_over_c):
    return 39.3 * A ** 3 / (t_over_c ** 3 * (A + 2.0))


def flutter_velocity(fin, a, p):
    """V_f [m/s] for a fins Comp at local speed of sound a [m/s] and pressure p [Pa]. G in Pa."""
    S = 0.5 * (fin.root + fin.tip) * fin.span
    A = fin.span ** 2 / S
    lam = fin.tip / fin.root
    X = flutter_X_psi(A, fin.thickness / fin.root) * PSI
    G = fin.G if fin.G else float("nan")
    return a * np.sqrt(G / (X * (lam + 1.0) / 2.0 * (np.asarray(p) / (P0_PSI * PSI))))


def flutter_report(rocket, res, margin_req=1.5):
    from .flight import AirTable  # noqa: F401  (air already sampled in the trajectory)
    tr, cols = res["traj"], res["cols"]
    ix = {c: i for i, c in enumerate(cols)}
    z, p = tr[:, ix["z"]], tr[:, ix["p_amb"]]
    V = np.linalg.norm(tr[:, [ix["vx"], ix["vy"], ix["vz"]]], axis=1)
    a = V / np.maximum(tr[:, ix["mach"]], 1e-6)
    out = []
    for st in rocket.stages:
        for c in st.comps:
            if c.kind != "fins":
                continue
            attached = tr[:, ix["n_attached"]] > st.idx
            Vf = flutter_velocity(c, a, p)
            ok = attached & (V > 20.0) & np.isfinite(Vf)
            if not ok.any():
                continue
            ratio = Vf[ok] / V[ok]
            k = int(np.argmin(ratio))
            out.append(dict(stage=st.name, material=c.material, G_pa=c.G, thickness_m=c.thickness,
                            min_margin=float(ratio[k]), at_t_s=float(tr[ok][k, ix["t"]]), at_alt_m=float(z[ok][k]),
                            V_mps=float(V[ok][k]), Vf_mps=float(Vf[ok][k]), pass_=bool(ratio[k] >= margin_req),
                            required=margin_req, method="NACA TN 4197 eq. 18 (screening)"))
    return out


def heating_report(rocket, res, nose_tip_radius_m=None):
    tr, cols = res["traj"], res["cols"]
    ix = {c: i for i, c in enumerate(cols)}
    M = tr[:, ix["mach"]]
    V = np.linalg.norm(tr[:, [ix["vx"], ix["vy"], ix["vz"]]], axis=1)
    p = tr[:, ix["p_amb"]]
    a = V / np.maximum(M, 1e-6)
    T = a * a / (1.4 * 287.05287)
    rho = p / (287.05287 * np.maximum(T, 1.0))
    Tr = T * (1.0 + 0.89 * 0.2 * M * M)
    ns = rocket.spec["stages"][0]["nose"]
    Rn = float(nose_tip_radius_m or ns.get("tip_radius_m", 0.005))
    qdot = 1.7415e-4 * np.sqrt(rho / Rn) * V ** 3
    t = tr[:, ix["t"]]
    asc = (tr[:, ix["vz"]] > 0) & (M > 0.3)            # ascent, and M large enough for T = (V/M)^2/(gamma R)
    Q = float(_trapz(np.where(asc, qdot, 0.0), t)) if len(t) > 1 else 0.0
    mat = ns.get("material", "fiberglass")
    from .materials import MATERIALS
    rho_w = MATERIALS[mat]["rho"]
    tw = float(ns.get("thickness", 0.002))
    dT = Q / (rho_w * CP_WALL.get(mat, 1000.0) * tw)
    k = int(np.argmax(np.where(asc, Tr, 0)))
    fin_out = []
    for st in rocket.stages:
        for c in st.comps:
            if c.kind == "fins":
                R_le = 0.5 * c.thickness
                att = tr[:, ix["n_attached"]] > st.idx
                qf = np.where(att & asc, 1.7415e-4 * np.sqrt(rho / R_le) * V ** 3, 0.0)
                fin_out.append(dict(stage=st.name, le_radius_m=R_le, peak_qdot_w_m2=float(qf.max()),
                                    Q_j_m2=float(_trapz(qf, t)) if len(t) > 1 else 0.0))
    return dict(peak_recovery_temp_K=float(Tr[k]), at_mach=float(M[k]), at_alt_m=float(tr[k, ix["z"]]),
                nose_tip_radius_m=Rn, nose_peak_qdot_w_m2=float(np.max(np.where(asc, qdot, 0.0))),
                nose_heat_load_j_m2=Q, nose_skin_dT_upper_bound_K=dT, nose_material=mat, fins=fin_out,
                method="recovery temperature r = 0.89; Sutton-Graves stagnation flux (a hypersonic correlation, "
                       "conservative below M ~3); thin-skin no-loss bound (screening)")


def axial_bending_report(rocket, res):
    """Max compressive axial load and bending moment at each stage's forward end and motor forward closure."""
    from .flight import StackModel
    tr, cols = res["traj"], res["cols"]
    ix = {c: i for i, c in enumerate(cols)}
    out = []
    models = {}
    for row in tr[::2]:
        n_top = int(row[ix["n_attached"]])
        if n_top not in models:
            models[n_top] = StackModel(rocket, n_top)
    for n_top, SM in models.items():
        rows = tr[tr[:, ix["n_attached"]] == n_top]
        if not len(rows):
            continue
        comps = [c for st in rocket.stack(n_top) for c in st.comps]
        stations = []
        for st in rocket.stack(n_top):
            if st.idx > 0:
                stations.append(("%s forward end" % st.name, st.x_top))
            if st.motor is not None:
                stations.append(("%s motor forward closure" % st.name, st.motor_x0))
        for name, xs in stations:
            fwd_m = sum(c.m for c in comps if c.xc < xs)
            fwd_x = [(c.m, c.xc) for c in comps if c.xc < xs]
            N_max = (0.0, 0.0)
            Mb_max = (0.0, 0.0)
            for row in rows:
                m, T, cd, qd, M = row[ix["mass"]], row[ix["thrust"]], row[ix["cd"]], row[ix["q"]], row[ix["mach"]]
                D = qd * SM.a_ref * cd
                share = min(xs / SM.L, 1.0)
                N = fwd_m * (T - D) / max(m, 1e-9) + D * share
                if N > N_max[0]:
                    N_max = (N, row[ix["t"]])
                al = math.radians(row[ix["alpha_deg"]])
                if qd > 0 and al > 0:
                    cna, xc, _, _ = SM.lookup(M)
                    Fn = cna * qd * SM.a_ref * al
                    Ftot = float(Fn.sum())
                    a_lat = Ftot / m
                    Mb = float(sum(f * (xs - x) for f, x in zip(Fn, xc) if x < xs))
                    Mb -= sum(mi * a_lat * (xs - xi) for mi, xi in fwd_x)
                    if abs(Mb) > Mb_max[0]:
                        Mb_max = (abs(Mb), row[ix["t"]])
            out.append(dict(station=name, x_m=xs, max_compression_n=float(N_max[0]), at_t=float(N_max[1]),
                            max_bending_nm=float(Mb_max[0]), bending_at_t=float(Mb_max[1]),
                            method="rigid-body screening; translational inertial relief only"))
    return out

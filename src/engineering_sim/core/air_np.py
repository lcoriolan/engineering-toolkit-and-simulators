#!/usr/bin/env python3
# DESCRIPTION: The ONE host implementation of the core air state and mean-wind profile. core/atmos.py and
#              core/wind.py use it as their reference twin (their GPU functions mirror it line for line and the
#              tests hold GPU == this module), and CPU-only containers (legacy service, which has no warp) import it
#              directly:
#                  from engineering_sim.core.air_np import ref_state, air_density, sound_speed, mean_wind, bl_setup
#              Physics, citations and limits: see the headers of core/atmos.py and core/wind.py.
import math
import os

import numpy as np

from .constants import (G0, RD, RV, R_UNIVERSAL, M_DRY, M_H2O, MU_REF, T_MU_REF, S_SUTH, KAPPA_VK, OMEGA_EARTH,
                        T0_K, P0_PA, LAPSE_K_PER_M, H_TROPO_M, C_REF_MS)

KAPPA = KAPPA_VK
DRAG_LAW_A, DRAG_LAW_B = 1.8, 4.5   # neutral geostrophic drag-law constants (Tennekes 1973; Garratt 1992)
LAT_MIN_DEG = 5.0                   # |f| floor: the drag law and h = c u*/f are singular at the equator
# U.S. Standard Atmosphere 1976 (= ICAO/ISO 2533 to 32 km), layers in GEOPOTENTIAL height H [m]:
#   (base H, base T [K], lapse [K/m]); top of the model H = 84852 m  (= 86 km geometric)
US76_LAYERS = ((0.0, 288.15, -0.0065), (11000.0, 216.65, 0.0), (20000.0, 216.65, 0.001),
               (32000.0, 228.65, 0.0028), (47000.0, 270.65, 0.0), (51000.0, 270.65, -0.0028),
               (71000.0, 214.65, -0.002))
H_TOP_GEOPOT_M = 84852.0
R0_GEOPOT_M = 6356766.0             # US76 effective Earth radius for geopotential height
Z_TOP_M = 86000.0                   # model top, GEOMETRIC [m] above MSL; the state is held above it
H_MAX_M = Z_TOP_M                   # (name kept from v0.1, where the ceiling was 20 km)


# ------------------------------------------------------------------------------------------------- air
def geopotential(z_msl):
    """Geometric height above MSL -> geopotential height H = r0 z/(r0 + z) (US76 eq. 18)."""
    z = np.asarray(z_msl, float)
    return R0_GEOPOT_M * z / (R0_GEOPOT_M + z)


def T_std(H):
    """US76 standard temperature [K] at geopotential height H (clamped to the model top)."""
    H = np.minimum(np.asarray(H, float), H_TOP_GEOPOT_M)
    T = np.full_like(H, US76_LAYERS[0][1])
    for Hb, Tb, L in US76_LAYERS:
        T = np.where(H >= Hb, Tb + L * (H - Hb), T)
    return T


def column_P(H, dT):
    """Relative hydrostatic pressure P(H) = p(H)/p(0) of the US76 column with every layer shifted by dT [K]
    (H geopotential, clamped to the model top). dT=0 reproduces the US76 pressure ratio."""
    H = np.minimum(np.asarray(H, float), H_TOP_GEOPOT_M)
    lnP = np.zeros_like(H)
    tops = [l[0] for l in US76_LAYERS[1:]] + [H_TOP_GEOPOT_M]
    for (Hb, Tb, L), Ht in zip(US76_LAYERS, tops):
        dz = np.clip(H, Hb, Ht) - Hb
        T1 = Tb + dT
        if L != 0.0:
            lnP = lnP + (-G0 / (L * RD)) * np.log((T1 + L * dz) / T1)
        else:
            lnP = lnP - G0 * dz / (RD * T1)
    return np.exp(lnP)


def buck_es_pa(T_K):
    tc = np.asarray(T_K, float) - 273.15
    return 611.21 * np.exp((18.678 - tc / 234.5) * (tc / (257.14 + tc)))


def vapour_e_pa(H, T, p, Hs, dT_K, p0_pa, rh):
    """Water-vapour partial pressure [Pa]. rh applies up to the tropopause (H_TROPO_M, geopotential); above it the
    vapour MOLE FRACTION is held at its tropopause value (cold-trap; stratospheric H2O is ~constant, few ppmv).
    v0.2.0 applied rh at all heights: e exceeded p near 51 km (rho < 0, NaN sound speed) and rho was off by
    tens of % above ~20 km. Bit-identical to v0.2.0 at H <= H_TROPO_M. MODELED."""
    H = np.asarray(H, float)
    e = rh * buck_es_pa(T)
    if rh == 0.0 or not np.any(H > H_TROPO_M):
        return e
    T_tr = float(T_std(H_TROPO_M)) + dT_K
    p_tr = p0_pa * float(column_P(Hs, 0.0)) * float(column_P(H_TROPO_M, dT_K)) / float(column_P(Hs, dT_K))
    x_tr = rh * float(buck_es_pa(T_tr)) / p_tr
    return np.where(H > H_TROPO_M, x_tr * p, e)


def sutherland_mu(T_K):
    T = np.asarray(T_K, float)
    return MU_REF * (T / T_MU_REF) ** 1.5 * (T_MU_REF + S_SUTH) / (T + S_SUTH)


def c_ideal(T_K, x_w):
    M = (1 - x_w) * M_DRY + x_w * M_H2O
    cv = (1 - x_w) * 2.5 * R_UNIVERSAL + x_w * 3.0 * R_UNIVERSAL
    return np.sqrt((1 + R_UNIVERSAL / cv) * R_UNIVERSAL * np.asarray(T_K, float) / M)


def _H(h_msl, geometric):
    h = np.minimum(np.asarray(h_msl, float), Z_TOP_M)
    return geopotential(h) if geometric else np.minimum(h, H_TOP_GEOPOT_M)


def ref_state(z_world, site_elev_m=0.0, dT_K=0.0, p0_pa=P0_PA, rh=0.0, geometric=True, column=None):
    """Air state at heights z_world [m] above the site (world z=0 is the ground at site_elev_m MSL).
    geometric=True (v0.2 default): heights are geometric and converted to geopotential (US76);
    False: v0.1 convention (height used directly as geopotential; differs by z^2/r0, 0.4 m at 1.6 km).
    column: optional measured / forecast column (see column_nodes); None = the ISA + dT column (unchanged).
    Returns dict of arrays: T_K, p_pa, rho, c (dry, legacy sensor form), c_moist, e_pa, mu, nu."""
    if column is not None:
        return column_state(z_world, column_nodes(column, site_elev_m, dT_K, p0_pa, rh, geometric), site_elev_m,
                            geometric)
    H = _H(np.asarray(z_world, float) + site_elev_m, geometric)
    Hs = _H(float(site_elev_m), geometric)
    T = T_std(H) + dT_K
    p = p0_pa * column_P(Hs, 0.0) * column_P(H, dT_K) / column_P(Hs, dT_K)
    tc = T - 273.15
    e = vapour_e_pa(H, T, p, Hs, dT_K, p0_pa, rh)
    rho = (p - e) / (RD * T) + e / (RV * T)
    c = C_REF_MS * np.sqrt(1.0 + tc / 273.15)
    c_moist = C_REF_MS * c_ideal(T, e / p) / c_ideal(273.15, 0.0)
    mu = sutherland_mu(T)
    return dict(T_K=T, p_pa=p, rho=rho, c=c, c_moist=c_moist, e_pa=e, mu=mu, nu=mu / rho)


# ------------------------------------------------------------------------------------------------- column

#     dict(z_msl_m | z_agl_m | H_gpm  (heights; geometric MSL, geometric AGL, or geopotential MSL),
#          T_K | T_c, rh (0..1, optional), p_pa (optional), blend_m (default 3000))
# replaces the single ISA + dT column. The table is turned into NODES (column_nodes): T is piecewise linear in
# geopotential height H between nodes (the US76 convention), and ln p is integrated hydrostatically node to node
# with the exact layer form (d ln p/dH = -g0/(Rd T)), so an inversion (T rising with height) is exact, not
# special-cased. Pressure anchor: the station pressure p0 P_std(H_site) (the same QNH-like definition as the ISA
# column, so surface_to_atmos still sets it); when p_pa is given at the data nodes, ln p comes from the data and
# the hydrostatic shape only fills in between them. Vapour: x_w = rh e_s(T)/p at each node (rh from the table, else
# the scalar rh), held at its tropopause value above H_TROPO_M (cold trap, as the ISA column). Moisture enters the
# density only (as the ISA column: the hydrostatic integral uses Rd T; virtual-temperature effect < 0.3 %).
# Below the data: ISA lapse from the lowest node (and a site node T_std(H_site) + dT is added if the data start
# above the site). Above the data top: T = T_std(H) + dT + D(s), s = (H - H_top)/blend_m, D a cubic Hermite with
# D(0) = T_top - T_std(H_top) - dT, D'(0) = (top data lapse - US76 lapse) blend_m, D(1) = D'(1) = 0: T AND its lapse
# are continuous at the data top and the column is exactly US76 + dT from H_top + blend_m up; nodes every 250 m
# through the blend, at every US76 layer base and every 2 km to the model top.
# Node array [N, 6]: H, T_K, ln p, lapse to the next node [K/m], x_w, ln-p step to the next node (hydrostatic).
# The GPU (core/atmos.col_state) evaluates the SAME nodes (float32); tests/test_real.py holds GPU == host.
COLUMN_BLEND_M = 3000.0
# GPU code switch, read ONCE at import: UNIFIED_COLUMN=1 compiles the column (core/atmos) and wind-table
# (core/wind) paths into the kernels; unset/0 removes them at codegen (wp.static), so every kernel is the v0.2 code
# exactly. sim.py sets it for weather scenarios before importing the core; the host functions here always work.
COLUMN_CODE = os.environ.get("UNIFIED_COLUMN", "0") == "1"


def _seg_lnP(dz, T1, L):
    dz = np.asarray(dz, float)
    T1 = np.asarray(T1, float)
    L = np.asarray(L, float)
    flat = np.abs(L) <= 1e-7
    Ls = np.where(flat, 1.0, L)
    with np.errstate(divide="ignore", invalid="ignore"):
        curved = (-G0 / (Ls * RD)) * np.log((T1 + Ls * dz) / T1)
    return np.where(flat, -G0 * dz / (RD * T1), curved)


def _column_key(column, site_elev_m, dT_K, p0_pa, rh, geometric):
    def t(v):
        return tuple(float(x) for x in np.atleast_1d(v)) if v is not None and not isinstance(v, str) else v
    return (tuple(sorted((k, t(v)) for k, v in column.items())), float(site_elev_m), float(dT_K), float(p0_pa),
            float(rh), bool(geometric))


_NODE_CACHE = {}


def column_nodes(column, site_elev_m=0.0, dT_K=0.0, p0_pa=P0_PA, rh=0.0, geometric=True):
    """Column table -> node array [N, 6] (see the block comment above). Cached per (column, atmos)."""
    if isinstance(column, np.ndarray):
        return column
    key = _column_key(column, site_elev_m, dT_K, p0_pa, rh, geometric)
    if key in _NODE_CACHE:
        return _NODE_CACHE[key]
    c = column
    if "H_gpm" in c:
        Hd = np.asarray(c["H_gpm"], float)
        if not geometric:
            Hd = Hd.copy()
    else:
        z = np.asarray(c["z_msl_m"], float) if "z_msl_m" in c else np.asarray(c["z_agl_m"], float) + site_elev_m
        Hd = _H(z, geometric)
    Td = np.asarray(c["T_K"], float) if "T_K" in c else np.asarray(c["T_c"], float) + 273.15
    rhd = np.asarray(c["rh"], float) if c.get("rh") is not None else None
    pd = np.asarray(c["p_pa"], float) if c.get("p_pa") is not None else None
    o = np.argsort(Hd, kind="stable")
    Hd, Td = Hd[o], Td[o]
    rhd = rhd[o] if rhd is not None else None
    pd = pd[o] if pd is not None else None
    keep = np.concatenate([[True], np.diff(Hd) >= 1.0])
    Hd, Td = Hd[keep], Td[keep]
    rhd = rhd[keep] if rhd is not None else None
    pd = pd[keep] if pd is not None else None
    if np.any(Td <= 100.0) or np.any(Hd > H_TOP_GEOPOT_M):
        raise ValueError("column: T must be in K (> 100) and heights below the model top")
    Hs = float(_H(float(site_elev_m), geometric))
    H, T, RH, P, data = list(Hd), list(Td), list(rhd if rhd is not None else [rh] * len(Hd)), \
        list(pd if pd is not None else [None] * len(Hd)), [True] * len(Hd)
    if H[0] > Hs + 1.0:                                   # data start above the ground: site node from dT
        H.insert(0, Hs); T.insert(0, float(T_std(Hs)) + dT_K); RH.insert(0, rh)
        P.insert(0, None); data.insert(0, False)
    H0 = min(H[0], Hs) - 5000.0                           # ISA lapse below the lowest node
    T.insert(0, T[0] + float(LAPSE_K_PER_M) * (H0 - H[0])); H.insert(0, H0); RH.insert(0, RH[0]); P.insert(0, None)
    data.insert(0, False)
    Ht, dTt, blend = H[-1], T[-1] - float(T_std(H[-1])), float(c.get("blend_m", COLUMN_BLEND_M))
    L_dat = (T[-1] - T[-2]) / (H[-1] - H[-2]) if len(H) >= 2 else 0.0
    L_std = float(T_std(min(Ht + 1.0, H_TOP_GEOPOT_M)) - T_std(Ht)) / max(min(1.0, H_TOP_GEOPOT_M - Ht), 1e-9)
    D0, M0 = dTt - dT_K, (L_dat - L_std) * blend
    syn = list(np.arange(Ht + 250.0, Ht + blend, 250.0)) + [Ht + blend]
    syn += [l[0] for l in US76_LAYERS if l[0] > Ht + blend] + list(np.arange(np.ceil((Ht + blend) / 2000.0) * 2000.0,
                                                                             H_TOP_GEOPOT_M, 2000.0)) + [H_TOP_GEOPOT_M]
    syn = sorted(set(round(h, 6) for h in syn if Ht + 1.0 <= h <= H_TOP_GEOPOT_M))
    for h in syn:
        sx = min(max((h - Ht) / blend, 0.0), 1.0)
        D = (2.0 * sx ** 3 - 3.0 * sx * sx + 1.0) * D0 + (sx ** 3 - 2.0 * sx * sx + sx) * M0
        H.append(h); T.append(float(T_std(h)) + dT_K + D); RH.append(RH[-1]); P.append(None)
        data.append(False)
    H, T = np.array(H), np.array(T)
    L = np.zeros_like(H)
    L[:-1] = np.diff(T) / np.diff(H)
    g = np.zeros_like(H)
    g[:-1] = _seg_lnP(np.diff(H), T[:-1], L[:-1])
    lnp = np.zeros_like(H)
    have_p = [i for i in range(len(H)) if P[i] is not None]
    if have_p:                                            # data pressures; hydrostatic outward from the nearest
        for i in have_p:
            lnp[i] = math.log(P[i])
        a0, a1 = have_p[0], have_p[-1]
        for i in range(a0 - 1, -1, -1):
            lnp[i] = lnp[i + 1] - g[i]
        for i in range(a0 + 1, len(H)):
            if P[i] is None:
                lnp[i] = lnp[i - 1] + g[i - 1]
        del a1
    else:                                                 # anchored at the station pressure (QNH definition)
        k = int(np.searchsorted(H, Hs, side="right") - 1)
        lnp_s = math.log(p0_pa * float(column_P(Hs, 0.0)))
        lnp[k] = lnp_s - float(_seg_lnP(Hs - H[k], T[k], L[k]))
        for i in range(k - 1, -1, -1):
            lnp[i] = lnp[i + 1] - g[i]
        for i in range(k + 1, len(H)):
            lnp[i] = lnp[i - 1] + g[i - 1]
    g[:-1] = np.diff(lnp)                                 # ln-p step actually spanned (== hydrostatic without p data)
    p = np.exp(lnp)
    x = np.minimum(np.array(RH, float) * buck_es_pa(T) / p, 0.2)
    if np.any(H > H_TROPO_M) and H[0] <= H_TROPO_M:      # cold trap above the tropopause
        x_tr = float(np.interp(H_TROPO_M, H, x))
        x = np.where(H > H_TROPO_M, x_tr, x)
    nodes = np.stack([H, T, lnp, L, x, g], axis=1)
    _NODE_CACHE[key] = nodes
    return nodes


def column_eval(H, nodes):
    """(T_K, p_pa, x_w) at geopotential heights H from column nodes (host twin of core/atmos.col_state)."""
    H = np.asarray(H, float)
    Hn = nodes[:, 0]
    i = np.clip(np.searchsorted(Hn, H, side="right") - 1, 0, len(Hn) - 2)
    dz = H - Hn[i]
    T1, L = nodes[i, 1], nodes[i, 3]
    T = T1 + L * dz
    gD = _seg_lnP(Hn[i + 1] - Hn[i], T1, L)
    lnp = nodes[i, 2] + nodes[i, 5] * _seg_lnP(dz, T1, L) / gD
    x = nodes[i, 4] + (nodes[i + 1, 4] - nodes[i, 4]) * dz / (Hn[i + 1] - Hn[i])
    return T, np.exp(lnp), x


def column_state(z_world, nodes, site_elev_m=0.0, geometric=True):
    """ref_state() for a column (same keys)."""
    H = _H(np.asarray(z_world, float) + site_elev_m, geometric)
    T, p, xw = column_eval(H, nodes)
    e = xw * p
    tc = T - 273.15
    rho = (p - e) / (RD * T) + e / (RV * T)
    c = C_REF_MS * np.sqrt(1.0 + tc / 273.15)
    c_moist = C_REF_MS * c_ideal(T, xw) / c_ideal(273.15, 0.0)
    mu = sutherland_mu(T)
    return dict(T_K=T, p_pa=p, rho=rho, c=c, c_moist=c_moist, e_pa=e, mu=mu, nu=mu / rho)


def surface_to_atmos(site_elev_m, temp_c, pressure_pa, rh=0.0, geometric=True):
    """Inverse of ref_state at the ground: the atmos dict (site_elev_m, dT_K, p0_pa, rh) whose z=0 state is
    exactly temp_c / pressure_pa. Use this instead of hand-mirroring the ISA (e.g. legacy sensor flags -> core)."""
    Hs = _H(float(site_elev_m), geometric)
    dT = (temp_c + 273.15) - float(T_std(Hs))
    return dict(site_elev_m=float(site_elev_m), dT_K=dT, p0_pa=float(pressure_pa) / float(column_P(Hs, 0.0)),
                rh=float(rh), geometric=bool(geometric))


def air_density(z_world, **atmos):
    return ref_state(z_world, **atmos)["rho"]


def sound_speed(z_world, moist=False, **atmos):
    return ref_state(z_world, **atmos)["c_moist" if moist else "c"]


def reynolds_np(speed, length, z_world=0.0, **atmos):
    return float(speed) * float(length) / float(ref_state([z_world], **atmos)["nu"][0])


# ------------------------------------------------------------------------------------------------- wind
BH_A, BH_B, BH_C, BH_D = 1.0, 0.667, 5.0, 0.35   # Beljaars & Holtslag (1991) stable constants


def psi_m(zL, bh91=False):
    """Momentum stability function. Unstable: Paulson 1970 / Dyer 1974. Stable: Businger-Dyer -5 z/L (default,
    same as legacy sensor env_atmos; valid only to z/L ~ 1) or, with bh91=True, Beljaars & Holtslag (1991) eq. 28,
    which stays physical in strongly stable air (z/L >> 1)."""
    if zL < 0:
        x = (1 - 16 * zL) ** 0.25
        return 2 * math.log((1 + x) / 2) + math.log((1 + x * x) / 2) - 2 * math.atan(x) + math.pi / 2
    if bh91:
        return -(BH_A * zL + BH_B * (zL - BH_C / BH_D) * math.exp(-BH_D * zL) + BH_B * BH_C / BH_D)
    return -5 * zL


def phi_m(zL, bh91=False):
    """Dimensionless shear phi_m = 1 - zL dpsi/dzL (consistent with psi_m)."""
    if zL < 0:
        return (1 - 16 * zL) ** -0.25
    if bh91:
        return 1 + zL * (BH_A + BH_B * math.exp(-BH_D * zL) * (1 + BH_C - BH_D * zL))
    return 1 + 5 * zL


def u_star_from_u10(u10, z0, inv_L, bh91=False):
    return KAPPA * u10 / (math.log(10.0 / z0) - psi_m(10.0 * inv_L, bh91)) if u10 > 0 else 0.0


def mos_speed(z, u_star, z0, inv_L, z_cap, bh91=False):
    """Surface-layer MOS speed, held at u(z_cap) above z_cap (the v0.1 'cap' profile)."""
    zz = min(max(z, z0 * 1.0001), z_cap)
    return max((u_star / KAPPA) * (math.log(zz / z0) - psi_m(zz * inv_L, bh91)), 0.0)


def bl_setup(u10=0.0, wind_dir_deg=270.0, z0=0.05, obukhov_L=float("inf"), z_cap=100.0, profile="bl",
             lat_deg=40.0, bl_height_m=None, geostrophic_speed=None, geostrophic_dir_deg=None,
             free_shear_per_km=0.0, free_veer_deg_per_km=0.0, uniform_speed=None, stable_psi="bd", table=None,
             **_ignored):
    """All derived mean-wind parameters (dict) from the Env wind config. Extra keys (turbulence, seed, ...)
    are ignored so an Env 'wind' dict can be passed as **cfg."""
    if profile not in ("bl", "cap"):
        raise ValueError("profile must be 'bl' (core v0.2) or 'cap' (v0.1 legacy)")
    if stable_psi not in ("bd", "bh91"):
        raise ValueError("stable_psi must be 'bd' (Businger-Dyer, legacy sensor parity) or 'bh91' (Beljaars-Holtslag)")
    bh = stable_psi == "bh91"
    inv_L = 0.0 if not math.isfinite(obukhov_L) else 1.0 / obukhov_L
    u_star = u_star_from_u10(u10, z0, inv_L, bh)
    th = math.radians(wind_dir_deg)
    lat = math.copysign(max(abs(lat_deg), LAT_MIN_DEG), lat_deg if lat_deg != 0 else 1.0)
    f = 2.0 * OMEGA_EARTH * math.sin(math.radians(lat))
    fa, hemi = abs(f), (1.0 if f > 0 else -1.0)
    us = max(u_star, 1e-3)
    z_s = z_cap
    h_src = "parameter"
    if bl_height_m is not None:
        h = float(bl_height_m)
    elif inv_L < 0:
        h, h_src = 1500.0, "ASSUMED unstable default"
    else:
        h, h_src = 0.25 * us / fa, "neutral 0.25 u*/|f|"
        if inv_L > 0:
            hs = 0.4 * math.sqrt(us / inv_L / fa)
            if hs < h:
                h, h_src = hs, "stable 0.4 sqrt(u* L/|f|)"
    clamped = h < 1.5 * z_s
    h = max(h, 1.5 * z_s)
    u_s = mos_speed(z_s, u_star, z0, inv_L, z_s, bh)
    if geostrophic_speed is not None:
        G = float(geostrophic_speed)
    elif u_star > 0:
        G = (u_star / KAPPA) * math.sqrt((math.log(us / (fa * z0)) - DRAG_LAW_A) ** 2 + DRAG_LAW_B ** 2)
    else:
        G = 0.0
    G = max(G, u_s)
    if geostrophic_dir_deg is not None:   # geostrophic FROM minus surface FROM, wrapped to (-180, 180]
        alpha = math.radians((geostrophic_dir_deg - wind_dir_deg + 180.0) % 360.0 - 180.0)
    elif G > 0 and u_star > 0:
        alpha = hemi * math.asin(min(1.0, DRAG_LAW_B * u_star / (KAPPA * G)))
    else:
        alpha = 0.0
    phi = phi_m(z_s * inv_L, bh)
    m0H = (u_star * phi / (KAPPA * z_s)) * (h - z_s)
    m0H = min(max(m0H, 0.0), 3.0 * (G - u_s)) if G > u_s else 0.0
    return dict(profile=profile, stable_psi=stable_psi, bh91=bh, uniform=uniform_speed is not None, u_uniform=float(uniform_speed or 0.0),
                u10=u10, wind_dir_deg=wind_dir_deg, dir_x=-math.sin(th), dir_y=-math.cos(th), z0=z0, inv_L=inv_L,
                u_star=u_star, z_s=z_s, lat_deg=lat_deg, f=f, f_abs=fa, stable=inv_L > 0, h_bl=h,
                h_source=h_src, bl_clamped=clamped, u_s=u_s, G=G, alpha=alpha, alpha_deg=math.degrees(alpha),
                m0H=m0H, free_shear=free_shear_per_km / 1000.0, free_veer=math.radians(free_veer_deg_per_km) / 1000.0,
                free_shear_per_km=free_shear_per_km, free_veer_deg_per_km=free_veer_deg_per_km,
                geostrophic_dir_deg=(wind_dir_deg + math.degrees(alpha)) % 360.0,
                table=wind_table(table, wind_dir_deg, z_s))


def wind_table(table, wind_dir_deg, z_s):
    """MEASURED / FORECAST wind profile (REAL 2026-10-16): dict(z_agl_m, speed [m/s], dir_from_deg) -> dict(z, s, b)
    (b = turning [rad, + clockwise] from the surface FROM direction, unwrapped with height), nodes above z_s only.
    When present it REPLACES the boundary-layer law above z_s: linear from the surface-layer top (u(z_s), turning 0)
    to the first node, linear between nodes, held above the last. The surface layer (MOS) and the turbulence are
    unchanged. None -> None."""
    if table is None:
        return None
    z = np.asarray(table["z_agl_m"], float)
    o = np.argsort(z)
    z, s, d = z[o], np.asarray(table["speed"], float)[o], np.asarray(table["dir_from_deg"], float)[o]
    k = z > z_s + 1.0
    z, s, d = z[k], s[k], d[k]
    if len(z) == 0:
        return None
    rel = (d - wind_dir_deg + 180.0) % 360.0 - 180.0
    b = np.radians(np.unwrap(rel, period=360.0))
    return dict(z=[float(v) for v in z], s=[float(v) for v in s], b=[float(v) for v in b])


def table_wind(z, b):
    """(speed, turning) from bl_setup()['table'] at z > z_s (host twin of core/wind._table_wind)."""
    T = b["table"]
    zt, st, bt = T["z"], T["s"], T["b"]
    if z <= zt[0]:
        e = (z - b["z_s"]) / (zt[0] - b["z_s"])
        return b["u_s"] + (st[0] - b["u_s"]) * e, bt[0] * e
    if z >= zt[-1]:
        return st[-1], bt[-1]
    i = int(np.searchsorted(zt, z, side="right") - 1)
    e = (z - zt[i]) / (zt[i + 1] - zt[i])
    return st[i] + (st[i + 1] - st[i]) * e, bt[i] + (bt[i + 1] - bt[i]) * e


def mean_wind(z, b):
    """(speed [m/s], turning [rad, + clockwise from the surface direction]) at height z for bl_setup() dict b."""
    if b["uniform"]:
        return b["u_uniform"], 0.0
    if b.get("table") is not None and z > b["z_s"]:
        return table_wind(z, b)
    s = mos_speed(z, b["u_star"], b["z0"], b["inv_L"], b["z_s"], b["bh91"])
    if b["profile"] == "cap" or z <= b["z_s"]:
        return s, 0.0
    if z < b["h_bl"]:
        e = (z - b["z_s"]) / (b["h_bl"] - b["z_s"])
        h01 = 3 * e * e - 2 * e ** 3
        return b["u_s"] + b["m0H"] * (e - 2 * e * e + e ** 3) + (b["G"] - b["u_s"]) * h01, b["alpha"] * h01
    return max(b["G"] + b["free_shear"] * (z - b["h_bl"]), 0.0), b["alpha"] + b["free_veer"] * (z - b["h_bl"])


def mean_wind_enu(z, b):
    """Mean wind vector (E, N) [m/s] at height z (no gusts)."""
    s, beta = mean_wind(z, b)
    cb, sb = math.cos(beta), math.sin(beta)
    return s * (b["dir_x"] * cb + b["dir_y"] * sb), s * (-b["dir_x"] * sb + b["dir_y"] * cb)


def turb_decay(z, b):
    """Turbulence sigma multiplier: 1 in the surface layer, Hanna (1982) BL decay above, quiet free troposphere."""
    if b["profile"] != "bl" or b["uniform"] or z <= b["z_s"]:
        return 1.0
    zz, h = min(z, b["h_bl"]), b["h_bl"]
    if b["stable"]:
        d = max(1 - zz / h, 0.0) / max(1 - b["z_s"] / h, 1e-3)
    else:
        d = math.exp(-2 * b["f_abs"] * (zz - b["z_s"]) / max(b["u_star"], 1e-3))
    if z > h:
        d *= math.exp(-(z - h) / (0.1 * h))
    return d

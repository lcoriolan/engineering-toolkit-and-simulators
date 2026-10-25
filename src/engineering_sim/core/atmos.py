#!/usr/bin/env python3
# DESCRIPTION: Position-dependent air state for every domain on the unified core. GPU (Warp) functions plus a
#              numpy reference twin used by the tests.
#
#              PHYSICS (cited):
#                U.S. Standard Atmosphere 1976 (identical to ICAO / ISO 2533 below 32 km), ALL seven layers to the
#                model top 86 km geometric (v0.2; v0.1 had the troposphere + 11-20 km only and held p above 20 km):
#                    layers (geopotential base H, T_b, lapse): 0 288.15 -6.5 | 11k 216.65 0 | 20k 216.65 +1.0 |
#                    32k 228.65 +2.8 | 47k 270.65 0 | 51k 270.65 -2.8 | 71k 214.65 -2.0 K/km; top H = 84852 m.
#                    Above Z_TOP_M = 86 km the 86 km state is held (domains/space blends its own model above).
#                    T = T_std(H) + dT (molecular temperature; the 80-86 km M/M0 correction, < 0.04 %, is ignored).
#                GEOMETRIC -> GEOPOTENTIAL (v0.2 fix): world z is geometric; H = r0 z/(r0 + z), r0 = 6356766 m
#                    (US76 eq. 18). v0.1 used z as H directly: -0.04 % rho at 5 km, -0.15 % at 9 km vs tables.
#                    AtmosParams.geometric = 0 restores the v0.1 convention for parity runs.
#                STATION pressure p_s = p0 P_std(H_site) (p0 is a QNH-like sea-level setting; by definition QNH
#                    reduces to station pressure through the standard column, so p_s does NOT depend on dT).
#                    ABOVE the station the column is hydrostatic in the ACTUAL temperature T_std + dT
#                    (v0.2 fix; v0.1 used the ISA column everywhere: -0.02 % p at +100 m, ~-0.3 % at +1.6 km, ISA+5):
#                    p(H) = p_s P_dT(H)/P_dT(H_site),  P_dT = layered US76 column with every T_b shifted by dT.
#                    surface_to_atmos(site, T, p, rh) inverts this exactly (use it; do not hand-mirror the ISA).
#                Viscosity: Sutherland's law mu = mu_ref (T/T_ref)^1.5 (T_ref+S)/(T+S)
#                    (mu_ref 1.716e-5 Pa s, T_ref 273.15 K, S 110.4 K; White, Viscous Fluid Flow, Table 1-2);
#                    nu = mu/rho; Re = rho V L / mu.  Humidity effect on mu (< 1 %) is neglected.
#                Moist air: vapour pressure e = RH * e_s(T), e_s from Buck (1981) over water,
#                    e_s[hPa] = 6.1121 exp((18.678 - Tc/234.5) (Tc/(257.14 + Tc)))
#                    rho = (p - e)/(Rd T) + e/(Rv T)           (ideal-gas partial pressures)
#                Sound speed: c = 331.3 sqrt(1 + Tc/273.15) — SAME dry-air form as legacy sensor physx/env_atmos.py
#                (ISO 9613-1 / CRC) so the acoustic domain stays in parity. DEFAULT.
#                Moist option (sound_speed_moist): ideal-gas mixture c = sqrt(gamma_m R T / M_m),
#                    x_w = e/p, M_m = (1-x_w) M_d + x_w M_w, Cv_m = (1-x_w) 5/2 R + x_w 3 R, gamma_m = 1 + R/Cv_m,
#                    scaled by 331.3/c_ideal(0 C, dry) so the dry limit equals the default form exactly.
#                    +0.19 % at 20 C RH 50 %, ~+0.37 % saturated; consistent with the humidity increments of
#                    Bohn (1988, JASA 83:2122) and Cramer (1993, JASA 93:2510). MODELED.
#
#              PARAMETERS (all in AtmosParams; nothing hard-coded in kernels):
#                site_elev_m   ground elevation above MSL (world z=0 sits at this elevation)
#                dT_K          temperature offset from ISA ("ISA+dT" hot/cold day)
#                p0_pa         sea-level pressure (QNH-like), default ISA 101325
#                rh            relative humidity 0..1 up to the tropopause; above it the vapour mole fraction is held
#                              at its tropopause value (cold trap). ASSUMED uniform rh below 11 km.
#                geometric     1 (default) geometric heights; 0 = v0.1 convention (heights used as geopotential)
#

#                              core/air_np.column_nodes. col_n = 0 (default) = the ISA + dT column, bit-identical
#                              to v0.2.1; col_n > 0 replaces T, p and the vapour fraction everywhere (density, sound
#                              speed, viscosity) by the column nodes, blended into US76 + dT above the data top.
#
#              VALIDITY: 0..86 km geometric (continuum, well-mixed air). MODELED, not measured. Near-surface diurnal temperature structure lives in
#              env_atmos.DiurnalAtmosphere (acoustic refraction); its effect on density is < ~1 % and is not coupled.
import math

import numpy as np
import warp as wp

from . import physics_device
from .constants import (G0, RD, RV, R_UNIVERSAL, M_DRY, M_H2O, MU_REF, T_MU_REF, S_SUTH,  # noqa: F401
                        T0_K, P0_PA, LAPSE_K_PER_M, H_TROPO_M, C_REF_MS)
# host reference twin = the pure-numpy module (also used directly by CPU-only containers)
from .air_np import (ref_state, sutherland_mu, reynolds_np, surface_to_atmos, Z_TOP_M,  # noqa: F401,E402
                     H_TOP_GEOPOT_M, R0_GEOPOT_M, US76_LAYERS, column_nodes, COLUMN_CODE)


@wp.struct
class AtmosParams:
    site_elev_m: float
    dT_K: float
    p0_pa: float
    rh: float
    geometric: int        # 1 = heights are geometric (converted to geopotential, v0.2); 0 = v0.1 convention
    if COLUMN_CODE:       # only with UNIFIED_COLUMN=1 (air_np.COLUMN_CODE): the v0.2 struct is unchanged otherwise
        col_n: int
        col: wp.array2d(dtype=float)   # [col_n, 6] H, T, ln p, lapse, x_w, ln-p step (air_np.column_nodes)


def make_params(site_elev_m=0.0, dT_K=0.0, p0_pa=P0_PA, rh=0.0, geometric=True, column=None, device=None):
    a = AtmosParams()
    a.geometric = 1 if geometric else 0
    a.site_elev_m = float(site_elev_m)
    a.dT_K = float(dT_K)
    a.p0_pa = float(p0_pa)
    a.rh = float(rh)
    if COLUMN_CODE:
        a.col_n = 0
    if column is not None:
        if not COLUMN_CODE:
            raise RuntimeError("atmos column needs UNIFIED_COLUMN=1 in the environment before core is imported "
                               "(sim.py sets it for weather scenarios)")
        nodes = column_nodes(column, site_elev_m, dT_K, p0_pa, rh, geometric)
        a.col = wp.array(np.asarray(nodes, np.float32), dtype=float, device=physics_device(device))
        a.col_n = int(nodes.shape[0])
    return a


# ---------------------------------------------------------------- GPU (Warp) functions
@wp.func
def _layer_lnP(H: float, dT: float, Hb: float, Ht: float, Tb: float, L: float):
    dz = wp.clamp(H, Hb, Ht) - Hb
    T1 = Tb + dT
    out = -G0 * dz / (RD * T1)
    if L != 0.0:
        out = (-G0 / (L * RD)) * wp.log((T1 + L * dz) / T1)
    return out


@wp.func
def _column_P(H: float, dT: float):
    """US76 layered hydrostatic column, every layer shifted by dT (mirror of air_np.column_P)."""
    h = wp.min(H, 84852.0)
    lnP = _layer_lnP(h, dT, 0.0, 11000.0, 288.15, -0.0065)
    lnP = lnP + _layer_lnP(h, dT, 11000.0, 20000.0, 216.65, 0.0)
    lnP = lnP + _layer_lnP(h, dT, 20000.0, 32000.0, 216.65, 0.001)
    lnP = lnP + _layer_lnP(h, dT, 32000.0, 47000.0, 228.65, 0.0028)
    lnP = lnP + _layer_lnP(h, dT, 47000.0, 51000.0, 270.65, 0.0)
    lnP = lnP + _layer_lnP(h, dT, 51000.0, 71000.0, 270.65, -0.0028)
    lnP = lnP + _layer_lnP(h, dT, 71000.0, 84852.0, 214.65, -0.002)
    return wp.exp(lnP)


@wp.func
def _T_std(H: float):
    h = wp.min(H, 84852.0)
    T = 288.15 - 0.0065 * h
    if h >= 11000.0:
        T = 216.65
    if h >= 20000.0:
        T = 216.65 + 0.001 * (h - 20000.0)
    if h >= 32000.0:
        T = 228.65 + 0.0028 * (h - 32000.0)
    if h >= 47000.0:
        T = 270.65
    if h >= 51000.0:
        T = 270.65 - 0.0028 * (h - 51000.0)
    if h >= 71000.0:
        T = 214.65 - 0.002 * (h - 71000.0)
    return T


@wp.func
def _geopot(h_msl: float, geometric: int):
    h = wp.min(h_msl, 86000.0)
    out = wp.min(h, 84852.0)
    if geometric == 1:
        out = 6356766.0 * h / (6356766.0 + h)
    return out


@wp.func
def isa_T_p(h_msl: float, dT: float, p0: float, h_site: float, geometric: int):
    """(T, p) at h_msl for a site at h_site: station p via the standard column (QNH definition), hydrostatic in
    the actual (dT-shifted) column above it. Heights converted geometric -> geopotential when geometric=1."""
    H = _geopot(h_msl, geometric)
    Hs = _geopot(h_site, geometric)
    T = _T_std(H) + dT
    p = p0 * _column_P(Hs, 0.0) * _column_P(H, dT) / _column_P(Hs, dT)
    return wp.vec2(T, p)


@wp.func
def _seg_lnP(dz: float, T1: float, L: float):
    out = -G0 * dz / (RD * T1)
    if wp.abs(L) > 1.0e-7:
        out = (-G0 / (L * RD)) * wp.log((T1 + L * dz) / T1)
    return out


@wp.func
def col_state(H: float, a: AtmosParams):
    """(T, p, x_w) at geopotential H from the measured/forecast column nodes (mirror of air_np.column_eval)."""
    lo = int(0)
    hi = a.col_n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a.col[mid, 0] <= H:
            lo = mid
        else:
            hi = mid
    i = lo
    dz = H - a.col[i, 0]
    T1 = a.col[i, 1]
    L = a.col[i, 3]
    dH = a.col[i + 1, 0] - a.col[i, 0]
    lnp = a.col[i, 2] + a.col[i, 5] * _seg_lnP(dz, T1, L) / _seg_lnP(dH, T1, L)
    x = a.col[i, 4] + (a.col[i + 1, 4] - a.col[i, 4]) * dz / dH
    return wp.vec3(T1 + L * dz, wp.exp(lnp), x)


@wp.func
def buck_es_pa(T_k: float):
    tc = T_k - 273.15
    return 611.21 * wp.exp((18.678 - tc / 234.5) * (tc / (257.14 + tc)))


@wp.func
def vapour_x(z_world: float, a: AtmosParams, T: float, p: float):
    """Vapour mole fraction e/p: rh below the tropopause, frozen at the tropopause value above (see air_np)."""
    H = _geopot(a.site_elev_m + z_world, a.geometric)
    x = a.rh * buck_es_pa(T) / p
    if H > H_TROPO_M and a.rh > 0.0:
        Hs = _geopot(a.site_elev_m, a.geometric)
        T_tr = _T_std(H_TROPO_M) + a.dT_K
        p_tr = a.p0_pa * _column_P(Hs, 0.0) * _column_P(H_TROPO_M, a.dT_K) / _column_P(Hs, a.dT_K)
        x = a.rh * buck_es_pa(T_tr) / p_tr
    return x


@wp.func
def air_density(z_world: float, a: AtmosParams):
    tp = isa_T_p(a.site_elev_m + z_world, a.dT_K, a.p0_pa, a.site_elev_m, a.geometric)
    T = tp[0]
    p = tp[1]
    e = vapour_x(z_world, a, T, p) * p
    out = (p - e) / (RD * T) + e / (RV * T)
    if wp.static(COLUMN_CODE):       # column path compiled only with UNIFIED_COLUMN=1
        if a.col_n > 0:
            s = col_state(_geopot(a.site_elev_m + z_world, a.geometric), a)
            ec = s[2] * s[1]
            out = (s[1] - ec) / (RD * s[0]) + ec / (RV * s[0])
    return out


@wp.func
def sound_speed(z_world: float, a: AtmosParams):
    tp = isa_T_p(a.site_elev_m + z_world, a.dT_K, a.p0_pa, a.site_elev_m, a.geometric)
    T = tp[0]
    if wp.static(COLUMN_CODE):       # column path compiled only with UNIFIED_COLUMN=1
        if a.col_n > 0:
            T = col_state(_geopot(a.site_elev_m + z_world, a.geometric), a)[0]
    return C_REF_MS * wp.sqrt(1.0 + (T - 273.15) / 273.15)


@wp.func
def _c_ideal(T: float, xw: float):
    M = (1.0 - xw) * M_DRY + xw * M_H2O
    cv = (1.0 - xw) * 2.5 * R_UNIVERSAL + xw * 3.0 * R_UNIVERSAL
    return wp.sqrt((1.0 + R_UNIVERSAL / cv) * R_UNIVERSAL * T / M)


@wp.func
def sound_speed_moist(z_world: float, a: AtmosParams):
    tp = isa_T_p(a.site_elev_m + z_world, a.dT_K, a.p0_pa, a.site_elev_m, a.geometric)
    xw = vapour_x(z_world, a, tp[0], tp[1])
    T = tp[0]
    if wp.static(COLUMN_CODE):       # column path compiled only with UNIFIED_COLUMN=1
        if a.col_n > 0:
            s = col_state(_geopot(a.site_elev_m + z_world, a.geometric), a)
            T = s[0]
            xw = s[2]
    return C_REF_MS * _c_ideal(T, xw) / _c_ideal(273.15, 0.0)


@wp.func
def dyn_viscosity(z_world: float, a: AtmosParams):
    """Sutherland dynamic viscosity [Pa s]."""
    T = _T_std(_geopot(a.site_elev_m + z_world, a.geometric)) + a.dT_K
    if wp.static(COLUMN_CODE):       # column path compiled only with UNIFIED_COLUMN=1
        if a.col_n > 0:
            T = col_state(_geopot(a.site_elev_m + z_world, a.geometric), a)[0]
    return MU_REF * wp.pow(T / T_MU_REF, 1.5) * (T_MU_REF + S_SUTH) / (T + S_SUTH)


@wp.func
def kin_viscosity(z_world: float, a: AtmosParams):
    return dyn_viscosity(z_world, a) / air_density(z_world, a)


@wp.func
def reynolds(speed: float, length: float, z_world: float, a: AtmosParams):
    return speed * length / kin_viscosity(z_world, a)


# ---------------------------------------------------------------- numpy reference twin: core/air_np.py


@wp.kernel
def _probe_kernel(z: wp.array(dtype=float), a: AtmosParams, rho: wp.array(dtype=float), c: wp.array(dtype=float)):
    i = wp.tid()
    rho[i] = air_density(z[i], a)
    c[i] = sound_speed(z[i], a)


@wp.kernel
def _probe_ext_kernel(z: wp.array(dtype=float), a: AtmosParams, out: wp.array2d(dtype=float)):
    i = wp.tid()
    out[i, 0] = dyn_viscosity(z[i], a)
    out[i, 1] = kin_viscosity(z[i], a)
    out[i, 2] = sound_speed_moist(z[i], a)
    out[i, 3] = reynolds(10.0, 1.0, z[i], a)


def probe(z_world, params, device=None):
    """GPU evaluation of (rho, c) at heights z_world — used by tests and reports."""
    device = physics_device(device)
    z = wp.array(np.asarray(z_world, np.float32), dtype=float, device=device)
    rho = wp.zeros(len(z_world), dtype=float, device=device)
    c = wp.zeros(len(z_world), dtype=float, device=device)
    wp.launch(_probe_kernel, len(z_world), [z, params, rho, c], device=device)
    return rho.numpy(), c.numpy()


def probe_ext(z_world, params, device=None):
    """GPU (mu, nu, c_moist, Re at V=10 m/s L=1 m) at heights z_world, columns of an [N,4] array (tests)."""
    device = physics_device(device)
    z = wp.array(np.asarray(z_world, np.float32), dtype=float, device=device)
    out = wp.zeros((len(z_world), 4), dtype=float, device=device)
    wp.launch(_probe_ext_kernel, len(z_world), [z, params, out], device=device)
    return out.numpy()

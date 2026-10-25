#!/usr/bin/env python3
# DESCRIPTION: The vertical profiles one source -> receiver vertical plane needs, ALL from the unified core host
#              twin (core/air_np: US76 + site offset + humidity; full mean-wind profile incl. surface layer, boundary
#              layer and veer). Nothing is re-derived here.
#                c(z)       adiabatic sound speed (core "c", the legacy sensor/site_air form; moist=True -> core "c_moist")
#                u_par(z)   mean wind component ALONG the horizontal propagation direction (+ = downwind)
#                u_vec(z)   mean wind (E, N) — the crosswind part is used only for the apparent arrival direction
#              z is height above the site datum (world z = 0 at the site elevation). Optional `extra_dTdz_K_per_m`
#              adds a linear near-ground temperature gradient (e.g. a night inversion: positive) up to
#              `extra_dT_top_m` — ASSUMED input, the core has no diurnal surface layer.
#              linear_profile(c0, g) = synthetic c(z) = c0 + g z, no wind (analytic validation only).
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from engineering_sim.core import air_np as AIR  # noqa: E402


class Profile:
    """c(z), u_par(z), dc/dz, du/dz on a uniform grid z_k = z0 + k dz (tables for the GPU kernel)."""

    def __init__(self, z, c, u_par, u_vec=None, meta=None):
        self.z = np.asarray(z, float)
        self.c = np.asarray(c, float)
        self.u = np.asarray(u_par, float)
        self.u_vec = None if u_vec is None else np.asarray(u_vec, float)
        self.dz = float(self.z[1] - self.z[0])
        self.z0 = float(self.z[0])
        self.dc = np.gradient(self.c, self.dz)
        self.du = np.gradient(self.u, self.dz)
        self.meta = dict(meta or {})

    def at(self, z):
        return np.interp(z, self.z, self.c), np.interp(z, self.z, self.u)

    def wind_vec_at(self, z):
        if self.u_vec is None:
            return np.zeros(2)
        return np.array([np.interp(z, self.z, self.u_vec[:, 0]), np.interp(z, self.z, self.u_vec[:, 1])])

    def ceff(self, z):
        c, u = self.at(z)
        return c + u


def grid(z_top, dz=0.25, z_bottom=0.0):
    n = int(math.ceil((z_top - z_bottom) / dz)) + 1
    return z_bottom + dz * np.arange(n)


def core_profile(atmos, wind=None, azimuth_deg=0.0, z_top=600.0, dz=0.25, z_ground=0.0, moist=False,
                 extra_dTdz_K_per_m=0.0, extra_dT_top_m=100.0, z=None):
    """atmos: core atmos dict (site_elev_m, dT_K, p0_pa, rh[, geometric]); wind: core Env wind dict (u10,
    wind_dir_deg FROM, z0, obukhov_L, ...) or None; azimuth_deg: propagation direction source -> receiver, compass
    (0 = north, 90 = east). z_ground: world height of the local ground (the wind profile is a function of height
    above it). Returns Profile on z in [z_ground - 2, z_top], or on the given uniform grid z."""
    z = grid(z_top, dz, z_ground - 2.0) if z is None else np.asarray(z, float)
    st = AIR.ref_state(np.maximum(z, -500.0), **{k: v for k, v in atmos.items()
                                                 if k in ("site_elev_m", "dT_K", "p0_pa", "rh", "geometric", "column")})
    c = st["c_moist" if moist else "c"].astype(float)
    if extra_dTdz_K_per_m:
        T = st["T_K"]
        dT = extra_dTdz_K_per_m * np.clip(z - z_ground, 0.0, extra_dT_top_m)
        c = c * np.sqrt((T + dT) / T)
    ex, ey = math.sin(math.radians(azimuth_deg)), math.cos(math.radians(azimuth_deg))
    uv = np.zeros((len(z), 2))
    if wind and float(wind.get("u10", 0.0) or 0.0) > 0.0:
        b = AIR.bl_setup(**wind)
        for i, zz in enumerate(z):
            h = zz - z_ground
            if h > b["z0"]:
                uv[i] = AIR.mean_wind_enu(float(h), b)
    u_par = uv[:, 0] * ex + uv[:, 1] * ey
    return Profile(z, c, u_par, uv, meta=dict(kind="core", atmos=dict(atmos), wind=dict(wind or {}),
                                              azimuth_deg=azimuth_deg, moist=moist,
                                              extra_dTdz_K_per_m=extra_dTdz_K_per_m))


def linear_profile(c0, g, z_top=600.0, dz=0.25, z_bottom=-2.0, z=None):
    z = grid(z_top, dz, z_bottom) if z is None else np.asarray(z, float)
    return Profile(z, c0 + g * z, np.zeros_like(z), None, meta=dict(kind="linear", c0=c0, g=g))


def homogeneous_profile(c0, z_top=600.0, dz=0.25, z_bottom=-2.0, z=None):
    return linear_profile(c0, 0.0, z_top, dz, z_bottom, z)

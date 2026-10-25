#!/usr/bin/env python3
# DESCRIPTION: The ONE way a sim that is not a Warp kernel gets its air and wind from the unified core.
#              Same AtmosParams/Wind objects the GPU domains use, so a ported CPU sim and the GPU world see the
#              identical environment. Batch your queries (points arrays) — each call is one GPU launch.
#
#                env = Env.from_dict(cfg)            # cfg = dict(atmos={...}, wind={...}) — JSON-serialisable
#                env.rho(z) / env.sound_speed(z[, moist]) / env.T_p(z)   (numpy, host reference = GPU to 1e-5)
#                env.mu(z) / env.nu(z) / env.reynolds(V, L, z)            (Sutherland; core v0.2)
#                env.mean_wind(z) -> (speed, dir_from_deg)                 (full BL profile, host twin)
#                env.wind(points[N,3], t)            (GPU, includes gusts)
#                env.describe()                      -> dict for the state-log manifest
#
#              Legacy constants: legacy sensor physx/env_atmos.py equations are reproduced exactly (tests/test_core.py);
#              sims that used a fixed rho (1.225 / 1.21) now get rho(z, site) — a port records the delta.
import json
import math

import numpy as np

from . import atmos as A
from . import wind as W
from . import physics_device

DEFAULT_WIND = dict(u10=0.0, wind_dir_deg=270.0, z0=0.05, obukhov_L=float("inf"), z_cap=100.0, turbulence=False,
                    L_turb=50.0, n_modes=128, seed=0, uniform_speed=None,
                    # core v0.2 boundary layer above z_cap ("cap" = v0.1 behaviour); see core/wind.py header
                    profile="bl", lat_deg=40.0, bl_height_m=None, geostrophic_speed=None,
                    geostrophic_dir_deg=None, free_shear_per_km=0.0, free_veer_deg_per_km=0.0,
                    stable_psi="bd")
DEFAULT_ATMOS = dict(site_elev_m=0.0, dT_K=0.0, p0_pa=A.P0_PA, rh=0.0)


class Env:
    def __init__(self, atmos=None, wind=None, device=None):
        self.device = physics_device(device)
        self.atmos_cfg = dict(DEFAULT_ATMOS, **(atmos or {}))
        self.wind_cfg = dict(DEFAULT_WIND, **(wind or {}))
        self.atm = A.make_params(device=self.device, **self.atmos_cfg)   # atmos may carry a `column` (REAL)
        self.wind_obj = W.Wind(device=self.device, **self.wind_cfg)

    @classmethod
    def from_dict(cls, cfg, device=None):
        if isinstance(cfg, str):
            cfg = json.load(open(cfg))
        return cls(cfg.get("atmos"), cfg.get("wind"), device)

    def T_p(self, z):
        s = A.ref_state(z, **self.atmos_cfg)
        return s["T_K"], s["p_pa"]

    def rho(self, z):
        return A.ref_state(z, **self.atmos_cfg)["rho"]

    def sound_speed(self, z, moist=False):
        return A.ref_state(z, **self.atmos_cfg)["c_moist" if moist else "c"]

    def mu(self, z):
        return A.ref_state(z, **self.atmos_cfg)["mu"]

    def nu(self, z):
        return A.ref_state(z, **self.atmos_cfg)["nu"]

    def reynolds(self, speed, length, z=0.0):
        return A.reynolds_np(speed, length, z, **self.atmos_cfg)

    def mean_wind(self, z):
        s, b = self.wind_obj.mean_np(float(z))
        return s, (self.wind_cfg["wind_dir_deg"] + math.degrees(b)) % 360.0

    def wind(self, points, t=0.0):
        pts = np.atleast_2d(np.asarray(points, np.float32))
        return W.probe(self.wind_obj, pts, t, device=self.device)

    def describe(self):
        return dict(atmos=self.atmos_cfg, wind=self.wind_obj.describe())

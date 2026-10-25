# DESCRIPTION: rho(h_msl) for launch, re-entry and orbit, continuous with the core atmosphere:
#
#   0 .. core_top    core.env (ISA + site + dT + RH; the SAME model every other domain uses). core_top is detected,
#                    not hard-coded: the highest altitude where the core density still decreases (core v0.1 holds
#                    p above 20 km, so core_top = 20 km there; if core v0.2 extends the ISA, this follows it).
#   core_top .. 86   US Standard Atmosphere 1976 layers continued hydrostatically FROM the core state at core_top
#                    (geopotential lapse rates -6.5 / 0 / +1.0 / +2.8 / 0 / -2.8 / -2.0 K/km, base heights 0/11/20/32/47/51/71 km,
#                    dry air R = 287.05287). Continuous in T, p and rho at core_top by construction.
#   >= 86 km         Exponential model, Vallado 4th ed. Table 8-4 (US Std 1976 for 86..1000 km, CIRA-72 above):
#                    rho = rho0 exp(-(h - h0)/H) per band. Static (no solar/geomagnetic activity) — ASSUMED mean
#                    conditions; real density at 400 km varies ~x5-10 over the solar cycle.
#   84 .. 88 km      log-linear blend of the two (they differ by ~3 % at 86 km), so rho and d(rho)/dh are smooth.
#
#              GPU use: `table()` samples log(rho) on a 250 m grid (exact at every Vallado node) for the Warp kernel.
import numpy as np

from .constants import G0

# Vallado Table 8-4: base altitude [km], nominal density [kg/m^3], scale height [km]
_VALLADO = np.array([
    (0, 1.225, 7.249), (25, 3.899e-2, 6.349), (30, 1.774e-2, 6.682), (40, 3.972e-3, 7.554),
    (50, 1.057e-3, 8.382), (60, 3.206e-4, 7.714), (70, 8.770e-5, 6.549), (80, 1.905e-5, 5.799),
    (90, 3.396e-6, 5.382), (100, 5.297e-7, 5.877), (110, 9.661e-8, 7.263), (120, 2.438e-8, 9.473),
    (130, 8.484e-9, 12.636), (140, 3.845e-9, 16.149), (150, 2.070e-9, 22.523), (180, 5.464e-10, 29.740),
    (200, 2.789e-10, 37.105), (250, 7.248e-11, 45.546), (300, 2.418e-11, 53.628), (350, 9.518e-12, 53.298),
    (400, 3.725e-12, 58.515), (450, 1.585e-12, 60.828), (500, 6.967e-13, 63.822), (600, 1.454e-13, 71.835),
    (700, 3.614e-14, 88.667), (800, 1.170e-14, 124.64), (900, 5.245e-15, 181.05), (1000, 3.019e-15, 268.00)])

_US76_LAYERS = [(0.0, -6.5e-3), (11000.0, 0.0), (20000.0, 1.0e-3), (32000.0, 2.8e-3), (47000.0, 0.0), (51000.0, -2.8e-3), (71000.0, -2.0e-3),
                (84852.0, 0.0)]          # (geopotential base [m], lapse [K/m])
_R0 = 6356766.0                         # US76 effective Earth radius for geopotential [m]
_RD = 287.05287
BLEND_LO, BLEND_HI = 84000.0, 88000.0


def vallado_rho(h_m):
    hk = np.asarray(h_m, float) / 1000.0
    idx = np.clip(np.searchsorted(_VALLADO[:, 0], hk, side="right") - 1, 0, len(_VALLADO) - 1)
    h0, r0, H = _VALLADO[idx, 0], _VALLADO[idx, 1], _VALLADO[idx, 2]
    return r0 * np.exp(-(hk - h0) / H)


def _geopot(h):
    return _R0 * h / (_R0 + h)


class AirColumn:
    """Ground-to-space density bound to one core environment (the launch site's air)."""

    def __init__(self, env=None, site_elev_m=0.0):
        if env is None:
            from engineering_sim.core.env import Env
            env = Env(atmos=dict(site_elev_m=site_elev_m), device="cpu")
        self.env = env
        self.site = float(env.atmos_cfg.get("site_elev_m", 0.0))
        self.core_top = self._detect_core_top()
        T, p = self.env.T_p(np.array([self.core_top - self.site]))
        self.T_top, self.p_top = float(T[0]), float(p[0])

    def _detect_core_top(self):
        h = np.arange(11000.0, 86001.0, 500.0)
        r = self.env.rho(h - self.site)
        dec = np.diff(r) < -1e-12 * np.maximum(r[:-1], 1e-30)
        k = np.argmin(dec) if not dec.all() else len(dec)
        return float(h[k])

    def _us76_from_top(self, h):
        """Continue T,p hydrostatically above core_top through the US76 layers (geopotential)."""
        H = _geopot(h)
        Ht = _geopot(self.core_top)
        T, p = np.full_like(H, self.T_top), np.full_like(H, self.p_top)
        bases = [b for b, _ in _US76_LAYERS] + [1e9]
        Hcur = np.full_like(H, Ht)
        for k, (b, L) in enumerate(_US76_LAYERS):
            top = bases[k + 1]
            lo, hi = max(b, Ht), top
            if hi <= Ht:
                continue
            seg_end = np.clip(H, lo, hi)
            dH = np.maximum(seg_end - np.maximum(Hcur, lo), 0.0)
            if L == 0.0:
                p = p * np.exp(-G0 * dH / (_RD * T))
            else:
                T1 = T + L * dH
                p = p * (T1 / T) ** (-G0 / (L * _RD))
                T = T1
            Hcur = np.maximum(Hcur, seg_end)
        return T, p, p / (_RD * T)

    def rho(self, h_msl):
        h = np.atleast_1d(np.asarray(h_msl, float))
        out = np.empty_like(h)
        lo = h <= self.core_top
        if lo.any():
            out[lo] = self.env.rho(h[lo] - self.site)
        mid = (~lo) & (h < BLEND_HI)
        if mid.any():
            out[mid] = self._us76_from_top(h[mid])[2]
        hi = h >= BLEND_LO
        if hi.any():
            rv = vallado_rho(h[hi])
            w = np.clip((h[hi] - BLEND_LO) / (BLEND_HI - BLEND_LO), 0.0, 1.0)
            base = np.where(h[hi] < BLEND_HI, out[hi], rv)
            out[hi] = np.exp((1.0 - w) * np.log(np.maximum(base, 1e-300)) + w * np.log(rv))
        return out if np.ndim(h_msl) else out[0]

    def pressure(self, h_msl):
        """Ambient static pressure [Pa] on the same column (core below core_top, US76 continuation above;
        above 86 km p = rho R T with T = 186.87 K, ASSUMED — < 0.4 Pa there, only used for nozzle back-pressure)."""
        h = np.atleast_1d(np.asarray(h_msl, float))
        out = np.empty_like(h)
        lo = h <= self.core_top
        if lo.any():
            out[lo] = self.env.T_p(h[lo] - self.site)[1]
        mid = (~lo) & (h < 86000.0)
        if mid.any():
            out[mid] = self._us76_from_top(h[mid])[1]
        hi = (~lo) & (h >= 86000.0)
        if hi.any():
            out[hi] = self.rho(h[hi]) * _RD * 186.87
        return out if np.ndim(h_msl) else out[0]

    def table(self, h_max_m=1500e3, dh=250.0):
        h = np.arange(0.0, h_max_m + dh, dh)
        return dict(h0=0.0, dh=dh, log_rho=np.log(self.rho(h)))

    def describe(self):
        return dict(model="core<=%.0fm | US76 continued to 86 km | Vallado T8-4 exp >=86 km | blend 84-88 km"
                    % self.core_top, core_top_m=self.core_top, site_elev_m=self.site)

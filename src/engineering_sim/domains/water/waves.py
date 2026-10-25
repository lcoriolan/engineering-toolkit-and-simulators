#!/usr/bin/env python3
# DESCRIPTION: Free surface as a sum of linear (Airy) components drawn from a JONSWAP spectrum (gamma = 1 gives
#              Pierson-Moskowitz) with cos-2s directional spreading (Longuet-Higgins). Finite or infinite depth
#              dispersion w^2 = g k tanh(k h). Uniform current with Doppler shift. Everything is a parameter.
#
#                SeaState(Hs, Tp, from_deg, gamma, spread_s, n_freq, n_dir, depth, current, seed)
#                SeaState.from_wind(core Wind, fetch_m=None)  -> wind sea from the SAME core wind object
#                   fully developed: Pierson-Moskowitz (1964) Hs = 0.21 U19.5^2/g, w_p = 0.877 g/U19.5
#                   fetch-limited:   JONSWAP (Hasselmann 1973) g Hs/U10^2 = 1.6e-3 (gF/U10^2)^0.5,
#                                    f_p U10/g = 3.5 (gF/U10^2)^-0.33, capped at fully developed
#              Kernel API (wp.func):  wave_sample(x, y, z, t, ...) -> spatial_vector(eta, deta/dx, deta/dy, u, v, w)
#                velocities at depth z (z > eta clamps to z = 0: constant extrapolation above MWL, ASSUMED).
#              Discretisation: equal-dw bins over [f_lo, f_hi] x w_p with per-bin frequency jitter (no repeat period);
#              amplitudes rescaled so 4 sqrt(sum a^2/2) == Hs exactly.
#              Direction convention = core wind: from_deg is where waves come FROM (compass, 0 = N, 90 = E).
import math

import numpy as np
import warp as wp

from engineering_sim.core.constants import G0
from engineering_sim.core import physics_device


@wp.struct
class WaveParams:
    n: int
    depth: float      # <= 0 -> deep water
    g: float
    cur: wp.vec3      # uniform current [m/s] (world)


@wp.func
def wave_sample(x: float, y: float, z: float, t: float, wv: WaveParams, kv: wp.array(dtype=wp.vec3),
                om: wp.array(dtype=float), amp: wp.array(dtype=float), ph: wp.array(dtype=float)):
    eta = float(0.0)
    gx = float(0.0)
    gy = float(0.0)
    u = wp.vec3(0.0, 0.0, 0.0)
    zc = wp.min(z, 0.0)
    for i in range(wv.n):
        k = kv[i]
        omd = om[i] + k[0] * wv.cur[0] + k[1] * wv.cur[1]          # Doppler: absolute frequency
        th = k[0] * x + k[1] * y - omd * t + ph[i]
        c = wp.cos(th)
        s = wp.sin(th)
        a = amp[i]
        eta += a * c
        gx -= a * k[0] * s
        gy -= a * k[1] * s
        fh = float(0.0)
        fz = float(0.0)
        kh = k[2] * wv.depth
        if wv.depth <= 0.0 or kh > 20.0:
            fh = wp.exp(k[2] * zc)
            fz = fh
        else:
            sh = wp.sinh(kh)
            fh = wp.cosh(k[2] * (zc + wv.depth)) / sh
            fz = wp.sinh(k[2] * (zc + wv.depth)) / sh
        uh = a * om[i] * fh * c / k[2]
        u += wp.vec3(uh * k[0], uh * k[1], a * om[i] * fz * s)
    u += wv.cur
    return wp.spatial_vector(eta, gx, gy, u[0], u[1], u[2])


@wp.kernel
def _probe_time(x: float, y: float, z: float, t0: float, dt: float, wv: WaveParams, kv: wp.array(dtype=wp.vec3),
                om: wp.array(dtype=float), amp: wp.array(dtype=float), ph: wp.array(dtype=float),
                out: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    out[i] = wave_sample(x, y, z, t0 + float(i) * dt, wv, kv, om, amp, ph)


@wp.kernel
def _surface_grid(x0: float, y0: float, dx: float, nx: int, t: float, wv: WaveParams, kv: wp.array(dtype=wp.vec3),
                  om: wp.array(dtype=float), amp: wp.array(dtype=float), ph: wp.array(dtype=float),
                  out: wp.array(dtype=float)):
    i = wp.tid()
    ix = i % nx
    iy = i // nx
    s = wave_sample(x0 + float(ix) * dx, y0 + float(iy) * dx, 0.0, t, wv, kv, om, amp, ph)
    out[i] = s[0]


def dispersion_k(omega, depth, g=G0):
    """Solve w^2 = g k tanh(k h) for k (Newton); depth <= 0 -> deep water."""
    omega = np.asarray(omega, float)
    k = omega ** 2 / g
    if depth is None or depth <= 0:
        return k
    for _ in range(50):
        th = np.tanh(k * depth)
        f = g * k * th - omega ** 2
        df = g * th + g * k * depth * (1 - th ** 2)
        k = k - f / df
    return k


def jonswap(omega, Hs, Tp, gamma=3.3):
    """JONSWAP S(w) [m^2 s/rad], scaled numerically to Hs afterwards (alpha cancels)."""
    wp_ = 2 * math.pi / Tp
    sig = np.where(omega <= wp_, 0.07, 0.09)
    r = np.exp(-((omega - wp_) ** 2) / (2 * sig ** 2 * wp_ ** 2))
    return omega ** -5.0 * np.exp(-1.25 * (wp_ / omega) ** 4) * gamma ** r


class SeaState:
    """Host container: parameters + GPU component arrays. Hs = 0 gives flat water (current still applies)."""

    def __init__(self, Hs=0.0, Tp=4.0, from_deg=270.0, gamma=3.3, spread_s=10.0, n_freq=48, n_dir=12,
                 f_lo=0.5, f_hi=4.0, depth=0.0, current=(0.0, 0.0), seed=0, g=G0, device=None, source="explicit",
                 water=None):
        """water: None (vessels keep their own rho/nu — legacy PWC 1025) or a props.water() dict / kwargs dict
        (preset fresh|brackish|sea, S_gkg, T_c, air_T_c, rho, nu): the water body every vessel on this sea floats in."""
        from . import props
        self.water = None if water is None else (water if "rho" in water and "nu" in water and "S_gkg" in water
                                                 else props.water(**water))
        self.cfg = dict(Hs=float(Hs), Tp=float(Tp), from_deg=float(from_deg), gamma=float(gamma), spread_s=float(spread_s),
                        n_freq=int(n_freq), n_dir=int(n_dir), f_lo=f_lo, f_hi=f_hi, depth=float(depth),
                        current=[float(c) for c in current], seed=int(seed), source=source)
        self.g = g
        self.device = physics_device(device)
        rng = np.random.default_rng(seed)
        th0 = math.radians(from_deg)
        mean_dir = math.atan2(-math.cos(th0), -math.sin(th0))     # math angle of propagation (toward), x=E y=N
        if Hs > 0:
            wpk = 2 * math.pi / Tp
            edges = np.linspace(f_lo * wpk, f_hi * wpk, n_freq + 1)
            dw = np.diff(edges)
            if n_dir > 1:
                dth = np.linspace(-math.pi, math.pi, n_dir, endpoint=False) + math.pi / n_dir
                D = np.cos(dth / 2) ** (2 * spread_s)
                D = D / D.sum()
            else:
                dth, D = np.zeros(1), np.ones(1)
            # every (frequency, direction) component gets its OWN jittered frequency inside its bin, so no two
            # components are coherent at a point (single-point statistics then converge to the spectrum)
            W = edges[:-1, None] + dw[:, None] * rng.uniform(0.05, 0.95, (n_freq, len(dth)))
            TH = np.broadcast_to(dth[None, :], W.shape)
            a = np.sqrt(2 * jonswap(W, Hs, Tp, gamma) * dw[:, None] * D[None, :])
            a *= (Hs / 4.0) / math.sqrt(0.5 * np.sum(a ** 2))           # exact discrete Hs
            keep = a > 1e-6 * a.max()
            W, TH, a = W[keep], TH[keep], a[keep]
            k = dispersion_k(W, depth, g)
            ang = mean_dir + TH
            kv = np.c_[k * np.cos(ang), k * np.sin(ang), k]
            phs = rng.uniform(0, 2 * math.pi, len(a))
        else:
            W, a, kv, phs = np.ones(1), np.zeros(1), np.array([[1.0, 0.0, 1.0]]), np.zeros(1)
        self.n = len(a) if Hs > 0 else 0
        self.np_ = dict(omega=W.astype(np.float64), amp=a.astype(np.float64), k=kv.astype(np.float64), phase=phs)
        dev = self.device
        self.kv = wp.array(kv.astype(np.float32), dtype=wp.vec3, device=dev)
        self.om = wp.array(W.astype(np.float32), dtype=float, device=dev)
        self.amp = wp.array(a.astype(np.float32), dtype=float, device=dev)
        self.ph = wp.array(phs.astype(np.float32), dtype=float, device=dev)
        p = WaveParams()
        p.n, p.depth, p.g = self.n, float(depth), float(g)
        p.cur = wp.vec3(float(current[0]), float(current[1]), 0.0)
        self.params = p

    @classmethod
    def from_wind(cls, wind, fetch_m=None, gamma=3.3, **kw):
        """Wind sea driven by the SAME core wind object (mean profile; gusts do not build waves here)."""
        # mean wind from the core v0.2 profile (use stable_psi="bh91" on the Wind for stable nights)
        U195 = float(wind.mean_np(19.5)[0])
        U10 = float(wind.mean_np(10.0)[0])
        g = kw.get("g", G0)
        if U10 <= 0:
            return cls(Hs=0.0, source="wind: calm", **kw)
        Hs_pm, Tp_pm = 0.21 * U195 ** 2 / g, 2 * math.pi * U195 / (0.877 * g)
        if fetch_m:
            X = g * fetch_m / U10 ** 2
            Hs = min(1.6e-3 * X ** 0.5 * U10 ** 2 / g, Hs_pm)
            fp = 3.5 * X ** -0.33 * g / U10
            Tp = max(1.0 / fp, 0.0) if Hs < Hs_pm else Tp_pm
            src = "wind: JONSWAP fetch-limited F=%g m, U10=%.2f" % (fetch_m, U10)
        else:
            Hs, Tp, gamma = Hs_pm, Tp_pm, 1.0
            src = "wind: Pierson-Moskowitz fully developed, U19.5=%.2f" % U195
        return cls(Hs=Hs, Tp=Tp, from_deg=wind.dir_deg, gamma=gamma, source=src, **kw)

    def args(self):
        return [self.params, self.kv, self.om, self.amp, self.ph]

    def describe(self):
        return dict(self.cfg, n_components=self.n, g=self.g, water=self.water)

    def probe_time(self, x, y, z, t0, dt, n):
        out = wp.zeros(n, dtype=wp.spatial_vector, device=self.device)
        wp.launch(_probe_time, n, [float(x), float(y), float(z), float(t0), float(dt)] + self.args() + [out],
                  device=self.device)
        return out.numpy()

    def surface(self, x0, y0, dx, nx, t, out=None):
        out = out if out is not None else wp.zeros(nx * nx, dtype=float, device=self.device)
        wp.launch(_surface_grid, nx * nx, [float(x0), float(y0), float(dx), int(nx), float(t)] + self.args() + [out],
                  device=self.device)
        return out

    def eta_np(self, x, y, t):
        """Host reference (float64) for tests."""
        d = self.np_
        if self.n == 0:
            return np.zeros_like(np.asarray(t, float))
        cur = self.params.cur
        omd = d["omega"] + d["k"][:, 0] * cur[0] + d["k"][:, 1] * cur[1]
        th = d["k"][:, 0] * x + d["k"][:, 1] * y - np.multiply.outer(np.asarray(t, float), omd) + d["phase"]
        return (d["amp"] * np.cos(th)).sum(-1)

#!/usr/bin/env python3
# DESCRIPTION: A continuous wind VELOCITY field u(x, t) [m/s, ENU] evaluable at any point on the GPU, shared by
#              every domain (bodies, beams, acoustic propagation, tracers).
#
#              MEAN PROFILE — Monin-Obukhov similarity, the SAME equations and constants as legacy sensor
#              physx/env_atmos.py (Stull 1988 Ch.9; Businger-Dyer psi_m, Paulson 1970 / Dyer 1974):
#                  u(z) = (u*/kappa) [ ln(z/z0) - psi_m(z/L) ],   u* = kappa u10 / (ln(10/z0) - psi_m(10/L))
#              MOS is used up to the surface-layer top z_s = z_cap (default 100 m) — UNCHANGED from v0.1, so legacy sensor
#              parity below 100 m is exact.
#              ABOVE z_s (core v0.2, profile="bl"; v0.1 held u(z_cap) constant to any height, which was wrong for
#              rocket/BOP work at 300-3000 m):
#                boundary-layer height h [m]: parameter bl_height_m, else from stability —
#                    neutral  h = 0.25 u*/|f|                       (Garratt 1992 §6.1; Zilitinkevich 1989)
#                    stable   h = min(neutral, 0.4 sqrt(u* L/|f|))  (Zilitinkevich 1972)
#                    unstable h = 1500 m  ASSUMED (convective depth needs the surface heat-flux history)
#                    f = 2 Omega sin(lat_deg) (lat_deg parameter, default 40 ASSUMED; |lat| floored at 5 deg).
#                    h is floored at 1.5 z_s so the transition layer always exists (flagged bl_clamped).
#                geostrophic wind G and cross-isobaric turning alpha from the Rossby-number similarity
#                (geostrophic drag) law, neutral constants A = 1.8, B = 4.5 (Tennekes 1973; Garratt 1992 §6.1):
#                    G = (u*/kappa) sqrt((ln(u*/(|f| z0)) - A)^2 + B^2),  sin(alpha) = B u*/(kappa G)
#                    G is floored at u(z_s). geostrophic_speed / geostrophic_dir_deg override them.
#                    NH (f>0): the wind VEERS (turns clockwise seen from above) by alpha from z_s to h; SH backs.
#                transition z_s..h: monotone cubic Hermite in eta = (z-z_s)/(h-z_s) matching u(z_s) AND the MOS
#                    shear du/dz = u* phi_m/(kappa z_s) at z_s, and G with zero shear at h (slope limited to
#                    3(G-u_s) so it never overshoots — Fritsch & Carlson 1980). Turning follows 3eta^2-2eta^3.
#                    The SHAPE is an engineering interpolation (ASSUMED); the endpoints are the physics.
#                free atmosphere z > h: U = G + free_shear_per_km (z-h)/1000, turning + free_veer_deg_per_km
#                    (thermal-wind shear/veer, parameters, default 0).
#              STABLE psi_m option (v0.2): stable_psi="bd" (default) is Businger-Dyer -5 z/L, identical to legacy sensor
#              env_atmos but unphysical beyond z/L ~ 1 (strong nights: u(100 m) blows up); stable_psi="bh91" uses
#              Beljaars & Holtslag (1991, J. Appl. Meteor. 30:327, eq. 28). Non-legacy sensor domains should use "bh91".
#              profile="cap" reproduces v0.1 exactly (u held at u(z_cap), no turning, no turbulence decay).
#              Direction: wind_dir_deg is where the SURFACE wind comes FROM (meteorological), as in
#              physx_wiring._wind_enu.
#
#              TURBULENCE — random Fourier modes (Kraichnan 1970; Bechara et al. 1994), frozen and advected by the
#              mean wind at z_ref (Taylor 1938):
#                  u'(x,t) = sum_n 2 a_n cos(k_n . (x - U_adv t) + phi_n) s_n ,   s_n _|_ k_n  (divergence-free)
#                  a_n = sqrt(E(k_n) dk_n),  E(k) von Karman: E ~ (kL)^4 / (1+(kL)^2)^(17/6), int E dk = 1.5
#              giving unit variance per component; components are then scaled in the mean-wind frame by
#              (sigma_u, sigma_v, sigma_w) = (2.18, 1.7, 1.25) u* — 2.18 from the Kaimal 1972 variance integral used
#              in env_atmos (sigma_u^2 = 4.77 u*^2); 1.7/1.25 from Panofsky & Dutton 1984 neutral surface layer.
#              The per-axis scaling makes the field anisotropic and only approximately divergence-free (accepted).
#              L_turb (integral length) is a parameter; default 50 m is ASSUMED (MIL-F-8785C gives ~80 m at 10 m AGL).
#              HEIGHT DEPENDENCE (v0.2): sigmas are constant in the surface layer z <= z_s (constant-flux layer —
#              MOS theory; keeps legacy sensor parity), then decay as Hanna (1982) boundary-layer scaling relative to z_s:
#                  neutral/unstable  sigma ~ exp(-2 |f| z / u*);   stable  sigma ~ (1 - z/h)
#              and above h by a further exp(-(z-h)/(0.1 h)) (free troposphere ~quiet; clear-air turbulence NOT
#              modelled, ASSUMED). The integral length L_turb stays constant with height (limitation; ESDU 85020
#              grows it with z) and the frozen field is advected by the 10 m mean wind (Taylor) at all heights.
#

#              level winds from engineering_sim.domains/env_bridge/weather.py): REPLACES the boundary-layer law above z_s — linear from
#              (u(z_s), turning 0) to the first node, linear between nodes, held above the last (air_np.wind_table /
#              table_wind are the host twins). The surface layer and the turbulence model are unchanged; no table
#              (default) is bit-identical to v0.2.
#
#              ALL MODELED. Validated in tests/test_core.py: profile parity vs env_atmos.psi_m, sample variance.
import math

import numpy as np
import warp as wp

from . import physics_device
from . import air_np as AN
from .constants import KAPPA_VK

KAPPA = KAPPA_VK  # von Karman constant (env_atmos.KAPPA)
DRAG_LAW_A, DRAG_LAW_B = AN.DRAG_LAW_A, AN.DRAG_LAW_B
_psi_m_np = AN.psi_m        # host twins live in core/air_np.py (one formula; names kept for existing callers)
_mean_np = AN.mos_speed
SIG_RATIO = (2.18, 1.7, 1.25)   # sigma_(u,v,w)/u* (see header)


@wp.struct
class WindParams:
    u_star: float
    z0: float
    inv_L: float          # 1/Obukhov length [1/m]; 0 = neutral
    dir_x: float          # unit vector the mean wind blows TOWARD (E)
    dir_y: float          # (N)
    z_cap: float
    uniform: int          # 1 = height-independent mean wind u_uniform (tests / wind-tunnel style cases)
    u_uniform: float
    sig_u: float
    sig_v: float
    sig_w: float
    adv: wp.vec3          # Taylor advection velocity of the frozen turbulence
    n_modes: int
    profile: int          # 0 = v0.1 "cap" (u(z_cap) held aloft), 1 = boundary layer + free atmosphere
    h_bl: float           # boundary-layer height [m]
    u_s: float            # u(z_s), z_s = z_cap
    m0H: float            # limited Hermite start slope * (h - z_s) [m/s]
    G: float              # geostrophic speed [m/s]
    alpha: float          # total turning z_s -> h [rad], + = clockwise (veer)
    free_shear: float     # [1/s]
    free_veer: float      # [rad/m], + = clockwise
    f_abs: float          # |Coriolis parameter| [1/s]
    stable: int           # 1 if L > 0 (stable decay law)
    bh91: int             # 1 = Beljaars-Holtslag stable psi_m (else Businger-Dyer, legacy sensor parity)


@wp.func
def psi_m(zL: float):
    out = -5.0 * zL
    if zL < 0.0:
        x = wp.pow(1.0 - 16.0 * zL, 0.25)
        out = 2.0 * wp.log((1.0 + x) / 2.0) + wp.log((1.0 + x * x) / 2.0) - 2.0 * wp.atan(x) + 1.5707963267948966
    return out


@wp.func
def psi_m2(zL: float, bh91: int):
    out = psi_m(zL)
    if bh91 == 1 and zL > 0.0:   # Beljaars & Holtslag 1991 (a=1, b=0.667, c=5, d=0.35)
        out = -(zL + 0.667 * (zL - 5.0 / 0.35) * wp.exp(-0.35 * zL) + 0.667 * 5.0 / 0.35)
    return out


@wp.func
def mean_speed(z: float, w: WindParams):
    """Mean wind speed (surface layer MOS; v0.1-compatible). Use mean_speed_turn() for the full profile."""
    s = w.u_uniform
    if w.uniform == 0:
        zz = wp.min(wp.max(z, w.z0 * 1.0001), w.z_cap)
        s = (w.u_star / KAPPA) * (wp.log(zz / w.z0) - psi_m2(zz * w.inv_L, w.bh91))
        s = wp.max(s, 0.0)
    return s


@wp.func
def _table_wind(z: float, w: WindParams, tb: wp.array(dtype=float), off: int):
    """(speed, turning) from the wind table at z > z_cap (mirror of air_np.table_wind). The table rides at the tail
    of the mode-phase array (no WindParams field, so kernels without a table compile exactly as before):
    tb[off] = n, then n heights, n speeds, n turnings [rad]."""
    n = int(tb[off])
    z0 = off + 1
    s0 = z0 + n
    b0 = s0 + n
    s = tb[s0 + n - 1]
    b = tb[b0 + n - 1]
    if z <= tb[z0]:
        e = (z - w.z_cap) / (tb[z0] - w.z_cap)
        s = w.u_s + (tb[s0] - w.u_s) * e
        b = tb[b0] * e
    elif z < tb[z0 + n - 1]:
        lo = int(0)
        hi = n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if tb[z0 + mid] <= z:
                lo = mid
            else:
                hi = mid
        e = (z - tb[z0 + lo]) / (tb[z0 + lo + 1] - tb[z0 + lo])
        s = tb[s0 + lo] + (tb[s0 + lo + 1] - tb[s0 + lo]) * e
        b = tb[b0 + lo] + (tb[b0 + lo + 1] - tb[b0 + lo]) * e
    return wp.vec2(s, b)


@wp.func
def mean_speed_turn(z: float, w: WindParams):
    """(speed [m/s], turning angle [rad, + clockwise] relative to the surface direction) at height z."""
    s = mean_speed(z, w)
    b = float(0.0)
    if w.uniform == 0 and w.profile == 1 and z > w.z_cap:
        if z < w.h_bl:
            e = (z - w.z_cap) / (w.h_bl - w.z_cap)
            h01 = 3.0 * e * e - 2.0 * e * e * e
            s = w.u_s + w.m0H * (e - 2.0 * e * e + e * e * e) + (w.G - w.u_s) * h01
            b = w.alpha * h01
        else:
            s = wp.max(w.G + w.free_shear * (z - w.h_bl), 0.0)
            b = w.alpha + w.free_veer * (z - w.h_bl)
    return wp.vec2(s, b)


@wp.func
def turb_decay(z: float, w: WindParams):
    """Turbulence sigma multiplier: 1 in the surface layer, Hanna (1982) BL decay above, quiet free troposphere."""
    d = float(1.0)
    if w.profile == 1 and w.uniform == 0 and z > w.z_cap:
        zz = wp.min(z, w.h_bl)
        us = wp.max(w.u_star, 1e-3)
        if w.stable == 1:
            d = wp.max(1.0 - zz / w.h_bl, 0.0) / wp.max(1.0 - w.z_cap / w.h_bl, 1e-3)
        else:
            d = wp.exp(-2.0 * w.f_abs * (zz - w.z_cap) / us)
        if z > w.h_bl:
            d = d * wp.exp(-(z - w.h_bl) / (0.1 * w.h_bl))
    return d


@wp.func
def _wind_at_tab(p: wp.vec3, t: float, w: WindParams,
            mk: wp.array(dtype=wp.vec3), ma: wp.array(dtype=float),
            mph: wp.array(dtype=float), ms: wp.array(dtype=wp.vec3)):
    sb = mean_speed_turn(p[2], w)
    if w.uniform == 0 and p[2] > w.z_cap:
        sb = _table_wind(p[2], w, mph, wp.max(w.n_modes, 1))
    U = sb[0]
    cb = wp.cos(sb[1])
    sn = wp.sin(sb[1])
    # clockwise rotation (seen from above) of the surface direction by the turning angle
    ex = wp.vec3(w.dir_x * cb + w.dir_y * sn, -w.dir_x * sn + w.dir_y * cb, 0.0)
    ey = wp.vec3(-ex[1], ex[0], 0.0)
    ez = wp.vec3(0.0, 0.0, 1.0)
    turb = wp.vec3(0.0, 0.0, 0.0)
    if w.n_modes > 0:
        x = p - w.adv * t
        for n in range(w.n_modes):
            turb = turb + ms[n] * (2.0 * ma[n] * wp.cos(wp.dot(mk[n], x) + mph[n]))
    # no flow through the ground: taper turbulence to 0 at z=0 over ~z0*10 (keeps tracers/bodies sane at ground)
    g = wp.min(wp.max(p[2] / wp.max(10.0 * w.z0, 1e-3), 0.0), 1.0) * turb_decay(p[2], w)
    tu = wp.dot(turb, ex) * w.sig_u * g
    tv = wp.dot(turb, ey) * w.sig_v * g
    tw = turb[2] * w.sig_w * g
    return ex * (U + tu) + ey * tv + ez * tw


@wp.func
def wind_at(p: wp.vec3, t: float, w: WindParams,
            mk: wp.array(dtype=wp.vec3), ma: wp.array(dtype=float),
            mph: wp.array(dtype=float), ms: wp.array(dtype=wp.vec3)):
    if wp.static(AN.COLUMN_CODE):         # wind table (REAL) at the tail of mph; compiled only with UNIFIED_COLUMN=1
        if mph.shape[0] > wp.max(w.n_modes, 1):
            return _wind_at_tab(p, t, w, mk, ma, mph, ms)
    sb = mean_speed_turn(p[2], w)
    U = sb[0]
    cb = wp.cos(sb[1])
    sn = wp.sin(sb[1])
    # clockwise rotation (seen from above) of the surface direction by the turning angle
    ex = wp.vec3(w.dir_x * cb + w.dir_y * sn, -w.dir_x * sn + w.dir_y * cb, 0.0)
    ey = wp.vec3(-ex[1], ex[0], 0.0)
    ez = wp.vec3(0.0, 0.0, 1.0)
    turb = wp.vec3(0.0, 0.0, 0.0)
    if w.n_modes > 0:
        x = p - w.adv * t
        for n in range(w.n_modes):
            turb = turb + ms[n] * (2.0 * ma[n] * wp.cos(wp.dot(mk[n], x) + mph[n]))
    # no flow through the ground: taper turbulence to 0 at z=0 over ~z0*10 (keeps tracers/bodies sane at ground)
    g = wp.min(wp.max(p[2] / wp.max(10.0 * w.z0, 1e-3), 0.0), 1.0) * turb_decay(p[2], w)
    tu = wp.dot(turb, ex) * w.sig_u * g
    tv = wp.dot(turb, ey) * w.sig_v * g
    tw = turb[2] * w.sig_w * g
    return ex * (U + tu) + ey * tv + ez * tw


class Wind:
    """Host-side container: parameters + GPU mode arrays. All arguments are parameters (parametric-first)."""

    def __init__(self, u10=0.0, wind_dir_deg=270.0, z0=0.05, obukhov_L=float("inf"), z_cap=100.0,
                 turbulence=True, L_turb=50.0, n_modes=128, k_span=(0.02, 50.0), sig_ratio=SIG_RATIO,
                 uniform_speed=None, seed=0, device=None, profile="bl", lat_deg=40.0, bl_height_m=None,
                 geostrophic_speed=None, geostrophic_dir_deg=None, free_shear_per_km=0.0,
                 free_veer_deg_per_km=0.0, stable_psi="bd", table=None):
        if profile not in ("bl", "cap"):
            raise ValueError("profile must be 'bl' (core v0.2) or 'cap' (v0.1 legacy)")
        self.u10, self.dir_deg, self.z0, self.L, self.z_cap = u10, wind_dir_deg, z0, obukhov_L, z_cap
        self.profile, self.lat_deg = profile, lat_deg
        self.L_turb, self.seed = L_turb, seed
        device = physics_device(device)
        inv_L = 0.0 if not math.isfinite(obukhov_L) else 1.0 / obukhov_L
        bh = stable_psi == "bh91"
        u_star = AN.u_star_from_u10(u10, z0, inv_L, bh)
        th = math.radians(wind_dir_deg)
        dx, dy = -math.sin(th), -math.cos(th)          # FROM -> TOWARD
        w = WindParams()
        w.u_star, w.z0, w.inv_L, w.dir_x, w.dir_y, w.z_cap = u_star, z0, inv_L, dx, dy, z_cap
        w.uniform = 1 if uniform_speed is not None else 0
        w.u_uniform = float(uniform_speed or 0.0)
        su = sig_ratio[0] * u_star if turbulence else 0.0
        if uniform_speed is not None and turbulence:   # uniform cases: scale turbulence off u10-equivalent u*
            su = sig_ratio[0] * KAPPA * uniform_speed / math.log(10.0 / z0)
        scale = su / sig_ratio[0] if sig_ratio[0] else 0.0
        w.sig_u, w.sig_v, w.sig_w = su, sig_ratio[1] * scale, sig_ratio[2] * scale
        U_adv = (uniform_speed if uniform_speed is not None else _mean_np(10.0, u_star, z0, inv_L, z_cap, bh))
        w.adv = wp.vec3(dx * U_adv, dy * U_adv, 0.0)
        self._set_bl(w, dict(u10=u10, wind_dir_deg=wind_dir_deg, z0=z0, obukhov_L=obukhov_L, z_cap=z_cap,
                             profile=profile, lat_deg=lat_deg, bl_height_m=bl_height_m,
                             geostrophic_speed=geostrophic_speed, geostrophic_dir_deg=geostrophic_dir_deg,
                             free_shear_per_km=free_shear_per_km, free_veer_deg_per_km=free_veer_deg_per_km,
                             uniform_speed=uniform_speed, stable_psi=stable_psi, table=table))
        tb = self.bl_full.get("table")
        if tb is not None and not AN.COLUMN_CODE:
            raise RuntimeError("wind table needs UNIFIED_COLUMN=1 in the environment before core is imported "
                               "(sim.py sets it for weather scenarios)")
        n = n_modes if (turbulence and w.sig_u > 0) else 0
        w.n_modes = n
        self.params, self.u_star = w, u_star
        mk, ma, mph, ms = _modes(max(n, 1), L_turb, k_span, seed)
        self.mk = wp.array(mk, dtype=wp.vec3, device=device)
        self.ma = wp.array(ma, dtype=float, device=device)
        if tb is not None:                       # wind table rides at the tail of the phase array (see _table_wind)
            mph = np.concatenate([mph, np.array([len(tb["z"])] + tb["z"] + tb["s"] + tb["b"], np.float32)])
        self.mph = wp.array(mph, dtype=float, device=device)
        self.ms = wp.array(ms, dtype=wp.vec3, device=device)

    def _set_bl(self, w, cfg):
        b = AN.bl_setup(**cfg)
        w.profile = 1 if b["profile"] == "bl" else 0
        w.h_bl, w.u_s, w.m0H, w.G, w.alpha = b["h_bl"], b["u_s"], b["m0H"], b["G"], b["alpha"]
        w.free_shear, w.free_veer = b["free_shear"], b["free_veer"]
        w.f_abs, w.stable, w.bh91 = b["f_abs"], 1 if b["stable"] else 0, 1 if b["bh91"] else 0
        self.bl_full = b
        self.bl = {k: b[k] for k in ("profile", "lat_deg", "f", "h_bl", "h_source", "bl_clamped", "z_s", "u_s", "G",
                                     "alpha_deg", "m0H", "free_shear_per_km", "free_veer_deg_per_km",
                                     "geostrophic_dir_deg", "stable_psi")}

    def mean_np(self, z):
        """Host twin of mean_speed_turn: (speed [m/s], turning [rad, + clockwise]) for scalar z."""
        return AN.mean_wind(z, self.bl_full)

    def sigma_np(self, z):
        """Host twin of the turbulence sigmas (u, v, w) at height z, ground taper excluded."""
        d = AN.turb_decay(z, self.bl_full)
        p = self.params
        return p.sig_u * d, p.sig_v * d, p.sig_w * d

    def args(self):
        return [self.params, self.mk, self.ma, self.mph, self.ms]

    def describe(self):
        p = self.params
        return dict(u10=self.u10, wind_dir_deg=self.dir_deg, z0=self.z0, obukhov_L=self.L, z_cap=self.z_cap,
                    u_star=self.u_star, sigma_uvw=[p.sig_u, p.sig_v, p.sig_w], L_turb=self.L_turb,
                    n_modes=p.n_modes, seed=self.seed, uniform=bool(p.uniform), u_uniform=p.u_uniform,
                    boundary_layer=self.bl)


def _modes(n, L, k_span, seed):
    rng = np.random.default_rng(seed)
    kmin, kmax = k_span[0] / L, k_span[1] / L
    edges = np.geomspace(kmin, kmax, n + 1)
    k = np.sqrt(edges[:-1] * edges[1:])
    dk = np.diff(edges)
    kl = k * L
    E = kl ** 4 / (1 + kl ** 2) ** (17 / 6)
    # normalise so the discrete sum of E dk = 1.5 (unit variance per component; TKE = 1.5)
    E *= 1.5 / np.sum(E * dk)
    a = np.sqrt(E * dk)
    # isotropic random directions for k, random s perpendicular to k
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    kv = v * k[:, None]
    r = rng.normal(size=(n, 3))
    s = np.cross(v, r); s /= np.linalg.norm(s, axis=1, keepdims=True)
    # per-component variance = sum (2a)^2/2 * <s_i^2> = 2 sum a^2 / 3 = 1 because sum a^2 = int E dk = 1.5
    ph = rng.uniform(0, 2 * math.pi, n)
    return kv.astype(np.float32), a.astype(np.float32), ph.astype(np.float32), s.astype(np.float32)


@wp.kernel
def _probe_kernel(p: wp.array(dtype=wp.vec3), t: float, w: WindParams,
                  mk: wp.array(dtype=wp.vec3), ma: wp.array(dtype=float),
                  mph: wp.array(dtype=float), ms: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = wind_at(p[i], t, w, mk, ma, mph, ms)


def probe(wind, points, t=0.0, device=None):
    device = physics_device(device)
    pts = wp.array(np.asarray(points, np.float32), dtype=wp.vec3, device=device)
    out = wp.zeros(len(points), dtype=wp.vec3, device=device)
    wp.launch(_probe_kernel, len(points), [pts, float(t)] + wind.args() + [out], device=device)
    return out.numpy()

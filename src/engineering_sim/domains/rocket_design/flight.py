#!/usr/bin/env python3
# DESCRIPTION: fly(rocket, ...) flies a rocket_design.Rocket from the pad to landing:
#                rail       motion along the rail only (length / elevation / azimuth), held until net axial force > 0
#                boost      thrust from the motor's certified curve along body +X (optional misalignment), propellant
#                           mass + inertia burning off in proportion to delivered impulse (motors.py)
#                aero       StackAero tables of the ATTACHED stack: per-component CNa(M) at its CP(M) on the air-relative
#                           velocity (weathercocking), nonlinear body lift, zero-lift drag CD(M, Re, motor on/off) along
#                           -v_rel, pitch/yaw damping C2A (Mandell) + jet damping, fin-cant roll forcing / damping
#                           (core.aero.fin_roll strip theory)
#                staging    lower stage separates `separation_delay_s` after its burnout; the next motor lights
#                           `ignition_delay_s` after separation (optional tilt lock-out: no ignition if the nose is
#                           more than `tilt_lockout_deg` from vertical, a common amateur practice). The spent stage
#                           flies on as its own 3-DOF body (drag of its tube + fins + blunt forward face), with its own
#                           recovery if specified, so its landing point is reported.
#                recovery   the top stage's drogue / main: deploy at apogee or at an AGL altitude (ideal event logic),
#                           or, fc="rocket-fc", decided by the REAL rocket-fc flight computer at 100 Hz from its own
#                           baro altitude (core static pressure -> its ISA formula) and accelerometer, exactly as
#                           domains/rocket_fc. Under canopy: 3-DOF point mass, CdA inflating over inflate_s.
#              Environment: the unified core air (core.env.Env: US76 to 86 km, site elevation / dT / RH, boundary-
#              layer + geostrophic wind and gusts to 86 km) continued to space by domains/space/density.AirColumn (US76
#              above the core top, Vallado exponential >= 86 km; T held at 186.87 K above 86 km for c and mu, ASSUMED).
#              Gravity g0 (R/(R+h))^2 (core G0) toward the Earth's centre on a SPHERICAL Earth (height = |P| - R, so
#              long-range sounding-rocket flights get the curvature right); pad-fixed, non-rotating frame (no
#              Coriolis / centrifugal terms: < ~1 % on a 100-200 km apogee, MODELED estimate). Integrator: core.aero.
#              step_rigid_np (float64 mirror of THE rigid-body step), substeps 1-10 ms chosen from thrust and q.
#              ALL outputs MODELED. Inputs carry their own labels (motor curves DATASHEET, geometry/masses per spec).
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
US = os.path.dirname(os.path.dirname(HERE))
if US not in sys.path:
    sys.path.insert(0, US)

from engineering_sim.core import aero as AE          # noqa: E402

from .aero import StackAero          # noqa: E402

R_EARTH = 6371008.8
GAMMA, R_AIR = 1.4, 287.05287


# ------------------------------------------------------------------ scalar mirrors of core.aero (speed: one body at kHz)
def _qrot(q, v):
    w, x, y, z = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx))


def _qrot_inv(q, v):
    return _qrot((q[0], -q[1], -q[2], -q[3]), v)


def _step_rigid(x, q, v, w, F, tau, m, I, dt):
    """Same algorithm and order as core.aero.step_rigid_np / core.rigid6.step_rigid (semi-implicit Euler, exact
    body-rate rotation), scalar floats; tests/test_rocket_design.py checks it against step_rigid_np."""
    ax, ay, az = F[0] / m, F[1] / m, F[2] / m           # isotropic mass: world accel = F/m
    vn = (v[0] + ax * dt, v[1] + ay * dt, v[2] + az * dt)
    xn = (x[0] + vn[0] * dt, x[1] + vn[1] * dt, x[2] + vn[2] * dt)
    Ix, Iy, Iz = I
    gx = w[1] * Iz * w[2] - w[2] * Iy * w[1]
    gy = w[2] * Ix * w[0] - w[0] * Iz * w[2]
    gz = w[0] * Iy * w[1] - w[1] * Ix * w[0]
    wn = (w[0] + (tau[0] - gx) / Ix * dt, w[1] + (tau[1] - gy) / Iy * dt, w[2] + (tau[2] - gz) / Iz * dt)
    om = math.sqrt(wn[0] * wn[0] + wn[1] * wn[1] + wn[2] * wn[2])
    qn = q
    if om > 1.0e-9:
        h = 0.5 * om * dt
        sh = math.sin(h) / om
        bw, bx, by, bz = math.cos(h), wn[0] * sh, wn[1] * sh, wn[2] * sh
        aw, ax_, ay_, az_ = q
        qn = (aw * bw - ax_ * bx - ay_ * by - az_ * bz, aw * bx + ax_ * bw + ay_ * bz - az_ * by,
              aw * by - ax_ * bz + ay_ * bw + az_ * bx, aw * bz + ax_ * by - ay_ * bx + az_ * bw)
    n = math.sqrt(qn[0] ** 2 + qn[1] ** 2 + qn[2] ** 2 + qn[3] ** 2)
    return xn, (qn[0] / n, qn[1] / n, qn[2] / n, qn[3] / n), vn, wn


def _geo(r, site):
    """Spherical Earth: pad-ENU position r -> (height above the pad level [m], unit vector toward Earth's centre)."""
    Rs = 6371008.8 + site
    px, py, pz = r[0], r[1], r[2] + Rs
    Rp = math.sqrt(px * px + py * py + pz * pz)
    return Rp - Rs, (-px / Rp, -py / Rp, -pz / Rp)


def core_g0():
    try:
        from engineering_sim.core.constants import G0
        return float(G0)
    except ImportError:
        return 9.80665


# ------------------------------------------------------------------ air: ground to space on one table
class AirTable:
    """env_spec: core env dict; optional env_spec["profile"] = a measured/reanalysis sounding {z_m (MSL), p_pa, T_K,
    rh, u, v} (e.g. era5_profile.py output) which REPLACES the core air and mean wind inside its height range
    (log-pressure / linear interpolation, virtual temperature for density and sound speed); above its top the
    core/US76 column continues, scaled to match pressure and temperature at the top."""

    def __init__(self, env_spec=None, dz=50.0, z_top_msl=250e3, device="cpu"):
        from engineering_sim.core.env import Env
        spec = dict(env_spec or {})
        prof = spec.pop("profile", None)
        spec.setdefault("atmos", {})
        spec.setdefault("wind", {})
        spec["wind"] = dict(dict(profile="bl", stable_psi="bh91"), **spec["wind"])
        self.env = Env.from_dict(spec, device=device)
        self.site = float(self.env.atmos_cfg.get("site_elev_m", 0.0))
        from engineering_sim.domains.space.density import AirColumn
        col = AirColumn(env=self.env)
        self.core_top = col.core_top
        self.z0, self.dz = self.site, dz
        z = np.arange(self.site, z_top_msl + dz, dz)
        rho = np.empty_like(z)
        T = np.empty_like(z)
        p = np.empty_like(z)
        lo = z <= self.core_top
        h = z[lo] - self.site
        T[lo], p[lo] = self.env.T_p(h)
        rho[lo] = self.env.rho(h)
        c_lo = self.env.sound_speed(h)
        mu_lo = self.env.mu(h)
        hi = ~lo
        Tm, pm, _ = col._us76_from_top(np.minimum(z[hi], 86000.0))
        T[hi] = np.where(z[hi] < 86000.0, Tm, 186.87)
        rho[hi] = col.rho(z[hi])
        p[hi] = col.pressure(z[hi])
        c = np.sqrt(GAMMA * R_AIR * T)
        c[lo] = c_lo
        mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
        mu[lo] = mu_lo
        self.profile = None
        if prof:
            if isinstance(prof, str):
                import json as _json
                prof = _json.load(open(prof))
            zp = np.asarray(prof["z_m"], float)
            Tp, pp = np.asarray(prof["T_K"], float), np.asarray(prof["p_pa"], float)
            rh = np.clip(np.asarray(prof.get("rh", np.zeros_like(zp)), float), 0, 1)
            inr = z <= zp[-1]
            Ti = np.interp(z[inr], zp, Tp)
            lp = np.interp(z[inr], zp, np.log(pp))
            below = z[inr] < zp[0]                      # extrapolate hydrostatically below the lowest level
            lp[below] = np.log(pp[0]) + 9.80665 * (zp[0] - z[inr][below]) / (R_AIR * Tp[0])
            Ti[below] = Tp[0] + 0.0065 * (zp[0] - z[inr][below])
            pi_ = np.exp(lp)
            es = 611.2 * np.exp(17.67 * (Ti - 273.15) / (Ti - 29.65))
            e = np.interp(z[inr], zp, rh, left=rh[0]) * es
            Tv = Ti / (1.0 - 0.378 * e / pi_)
            ktop = int(np.argmax(~inr)) if (~inr).any() else None
            sp_, sT = (pi_[-1] / p[inr][-1], Ti[-1] / T[inr][-1]) if inr.any() else (1.0, 1.0)
            p[inr], T[inr] = pi_, Ti
            rho[inr] = pi_ / (R_AIR * Tv)
            c[inr] = np.sqrt(GAMMA * R_AIR * Tv)
            mu[inr] = 1.458e-6 * Ti ** 1.5 / (Ti + 110.4)
            if ktop is not None:
                p[~inr] *= sp_
                rho[~inr] *= sp_ / sT
                T[~inr] *= sT
                c[~inr] *= math.sqrt(sT)
            self.profile = dict(z=zp, u=np.asarray(prof.get("u", np.zeros_like(zp)), float),
                                v=np.asarray(prof.get("v", np.zeros_like(zp)), float), src=prof.get("source"))
        self.tab = np.stack([rho, c, mu, p, T], axis=1)
        self.n = len(z)
        self.wind_top = 86000.0

    def at(self, z_msl):
        f = (z_msl - self.z0) / self.dz
        if f <= 0.0:
            return self.tab[0]
        i = int(f)
        if i >= self.n - 1:
            return self.tab[-1]
        w = f - i
        return self.tab[i] * (1.0 - w) + self.tab[i + 1] * w

    def wind(self, xyz_agl, t):
        if self.profile is not None:
            zz = xyz_agl[2] + self.site
            P = self.profile
            if zz > P["z"][-1]:
                return np.zeros(3)
            return np.array([np.interp(zz, P["z"], P["u"]), np.interp(zz, P["z"], P["v"]), 0.0])
        if xyz_agl[2] + self.site > self.wind_top:
            return np.zeros(3)
        return self.env.wind([[xyz_agl[0], xyz_agl[1], max(xyz_agl[2], 0.5)]], t)[0].astype(float)


# ------------------------------------------------------------------ per-configuration tables
class StackModel:
    """Precomputed aero + dry-mass sums for one attached configuration (stages 0..n_top-1)."""
    M_GRID = np.concatenate([np.arange(0.0, 3.0, 0.01), np.arange(3.0, 15.01, 0.05)])
    LRE_GRID = np.arange(3.0, 9.01, 0.25)

    def __init__(self, rocket, n_top):
        self.n_top = n_top
        self.aero = A = StackAero(rocket, n_top)
        self.a_ref, self.d_ref, self.L = A.a_ref, A.d_ref, A.L
        Mg = self.M_GRID
        comps = A.components(Mg)
        self.cna = np.stack([c for c, _ in comps], axis=1)          # [nM, ncomp]
        self.xc = np.stack([x for _, x in comps], axis=1)
        MM, RR = np.meshgrid(Mg, 10.0 ** self.LRE_GRID, indexing="ij")
        self.cd_on = A.cd(MM, RR, thrusting=True)
        self.cd_off = A.cd(MM, RR, thrusting=False)
        self.A_plan, self.x_plan = A.A_plan, A.x_plan
        stages = rocket.stack(n_top)
        S0 = S1 = S2 = Iax = 0.0
        for st in stages:
            for c in st.comps:
                S0 += c.m
                S1 += c.m * c.xc
                S2 += c.I_tr_c + c.m * c.xc * c.xc
                Iax += c.I_ax
        self.S = (S0, S1, S2, Iax)
        self.props = [(st.idx, st.motor, st.prop) for st in stages if st.motor is not None]
        self.x_noz = stages[-1].x_aft
        # roll: lowest fin set of the stack
        fins = [c for st in stages for c in st.comps if c.kind == "fins"]
        self.fin = fins[-1] if fins else None
        if self.fin is not None:
            f = self.fin
            g = dict(radius_m=f.R, fin_root=f.root, fin_tip=f.tip, fin_span=f.span, fin_sweep=f.sweep)
            self.J1, self.J2, self.S_fin = AE.fin_roll(g)
        self.cna_tot0 = float(self.cna[0].sum())
        self.t_cn = self.cna.sum(axis=1).tolist()                       # sum CNa_i
        self.t_cnx = (self.cna * self.xc).sum(axis=1).tolist()          # sum CNa_i x_i
        self.t_cnxx = (self.cna * self.xc ** 2).sum(axis=1).tolist()    # sum CNa_i x_i^2
        self.t_cnfin = self.cna[:, -1].tolist()
        self.cd_on_l, self.cd_off_l = self.cd_on.tolist(), self.cd_off.tolist()
        self.Mg_l = self.M_GRID.tolist()

    def mass(self, motor_t):
        S0, S1, S2, Iax = self.S
        for idx, mot, p in self.props:
            tm = motor_t[idx]
            mp = mot.prop_kg if (tm is None or tm < 0) else mot.prop_mass(tm)
            xp = p.x0 + 0.5 * p.length
            S0 += mp
            S1 += mp * xp
            S2 += mp * (3 * p.r * p.r + p.length ** 2) / 12.0 + mp * xp * xp
            Iax += 0.5 * mp * p.r * p.r
        xcg = S1 / S0
        It = S2 - S0 * xcg * xcg
        return S0, xcg, (Iax, It, It)

    def lookup(self, M):
        Mg = self.M_GRID
        if M <= 0.0:
            return self.cna[0], self.xc[0], 0, 0.0
        if M < 3.0:
            i = min(int(M / 0.01), 299)
        else:
            i = min(300 + int((M - 3.0) / 0.05), len(Mg) - 2)
        w = (M - Mg[i]) / (Mg[i + 1] - Mg[i])
        w = min(max(w, 0.0), 1.0)
        return (self.cna[i] * (1 - w) + self.cna[i + 1] * w, self.xc[i] * (1 - w) + self.xc[i + 1] * w, i, w)

    def idx(self, M):
        Mg = self.Mg_l
        if M <= 0.0:
            return 0, 0.0
        i = min(int(M / 0.01), 299) if M < 3.0 else min(300 + int((M - 3.0) / 0.05), len(Mg) - 2)
        w = (M - Mg[i]) / (Mg[i + 1] - Mg[i])
        return i, (0.0 if w < 0 else (1.0 if w > 1 else w))

    def sums(self, i, w):
        a, b = 1.0 - w, w
        return (self.t_cn[i] * a + self.t_cn[i + 1] * b, self.t_cnx[i] * a + self.t_cnx[i + 1] * b,
                self.t_cnxx[i] * a + self.t_cnxx[i + 1] * b, self.t_cnfin[i] * a + self.t_cnfin[i + 1] * b)

    def cd(self, i, w, Re, on):
        tab = self.cd_on_l if on else self.cd_off_l
        lr = math.log10(max(Re, 1e3))
        j = (lr - self.LRE_GRID[0]) / 0.25
        j0 = min(max(int(j), 0), len(self.LRE_GRID) - 2)
        v = min(max(j - j0, 0.0), 1.0)
        r0, r1 = tab[i], tab[i + 1]
        a = r0[j0] * (1 - v) + r0[j0 + 1] * v
        b = r1[j0] * (1 - v) + r1[j0 + 1] * v
        return a * (1 - w) + b * w

    def margin(self, M, x_cg):
        cna, xc, _, _ = self.lookup(M)
        return (float((cna * xc).sum() / cna.sum()) - x_cg) / self.d_ref


def _spent_stage_cd(st, M):
    """Spent stage tumbling/coasting tail-first-ish: blunt forward face + base + friction (screening, ASSUMED)."""
    q_stag = 1 + M * M / 4 + M ** 4 / 40 if M < 1 else 1.84 - 0.76 / M ** 2 + 0.166 / M ** 4 + 0.035 / M ** 6
    base = 0.12 + 0.13 * M * M if M < 1 else 0.25 / M
    fric = 0.05 * (st.x_aft - st.x_top) / st.d
    return 0.85 * q_stag + base + fric


# ------------------------------------------------------------------ the flight
def _q_from_rail(el_deg, az_deg):
    el, az = math.radians(el_deg), math.radians(az_deg)
    u = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
    return u, AE.q_from_x_to(u)


class _Body3:
    """A separated stage or a vehicle under canopy: 3-DOF point mass."""

    def __init__(self, name, r, v, m, cd_fn, area, chutes=None, t0=0.0):
        self.name, self.r, self.v, self.m = name, np.array(r, float), np.array(v, float), m
        self.cd_fn, self.area, self.chutes = cd_fn, area, chutes or {}
        self.landed = False
        self.deployed = {}
        self.apogee = r[2]
        self.t0 = t0
        self.track = []

    def cda(self, M, t):
        c = self.cd_fn(M) * self.area
        for name, ch in self.chutes.items():
            if name in self.deployed:
                f = min(max((t - self.deployed[name]) / max(ch["inflate_s"], 1e-3), 0.0), 1.0)
                c = c + f * ch["cd"] * math.pi * ch["D"] ** 2 / 4.0
        return c


def fly(rocket, env=None, rail=None, t_max=3600.0, fc=None, dispersion=None, log_hz=20.0, statelog_dir=None,
        wind=True, device="cpu", seed=0, stop_at_apogee=False):
    """Fly one flight. env: core env dict ({atmos:{site_elev_m,dT_K,rh}, wind:{u10,wind_dir_deg,gusts..}}).
    rail: {length_m, elev_deg, az_deg}. dispersion: {thrust_scale: [k per stage], cd_scale, mass_scale,
    misalign_deg, ...} (applied by dispersion.py). Returns a result dict (summary + trajectory arrays)."""
    G0 = core_g0()
    air = AirTable(env, device=device)
    disp = dispersion or {}
    rail = dict(dict(length_m=1.5, elev_deg=90.0, az_deg=0.0), **(rail or {}))
    n_st = len(rocket.stages)
    models = {k: StackModel(rocket, k) for k in range(1, n_st + 1)}
    n_top = n_st
    SM = models[n_top]
    cd_scale = float(disp.get("cd_scale", 1.0))
    th_scale = list(disp.get("thrust_scale", [1.0] * n_st))
    misalign = math.radians(float(disp.get("misalign_deg", 0.0)))
    mis_az = float(disp.get("misalign_az_rad", 0.0))
    rng = np.random.default_rng(seed)

    u_rail, q = _q_from_rail(rail["elev_deg"], rail["az_deg"])
    q = tuple(float(v) for v in q)
    motor_t = [None] * n_st
    motor_t[n_st - 1] = 0.0
    ign_time = {n_st - 1: 0.0}
    m0, xcg0, _ = SM.mass(motor_t)
    r_cg0 = u_rail * (SM.L - xcg0)
    r = r_cg0.copy()
    vel = np.zeros(3)
    w = (0.0, 0.0, 0.0)
    s_rail, on_rail = 0.0, True
    t = 0.0
    wind_v, next_wind = np.zeros(3), 0.0
    events = [(0.0, "ignition", rocket.stages[n_st - 1].name)]
    spent = []
    burnout = {}
    sep_due = None
    ign_due = None
    canopy = None
    top = rocket.stages[0]
    chutes = getattr(top, "chutes", {})
    deployed = {}
    landed = False
    apogee = (0.0, -1e9)
    diag = dict(M=0.0, q=0.0, alpha=0.0, margin=float("nan"), cd=0.0, thrust=0.0, m=m0, x_cg=xcg0, ax=0.0)
    rail_exit = None
    traj = []
    next_log = 0.0
    maxes = dict(M=(0.0, 0.0), q=(0.0, 0.0, 0.0, 0.0), accel_g=(0.0, 0.0), alpha=(0.0, 0.0))
    min_margin = [float("inf"), None]
    fcs = None
    if fc == "rocket-fc":
        fcs = _FCBridge(air, r[2] + air.site)
    tilt_lock = rocket.spec.get("tilt_lockout_deg")
    lockouts = []
    slog = None
    if statelog_dir:
        from engineering_sim.core.statelog import StateLog
        slog = StateLog(statelog_dir, 30.0, env=air.env, meta=dict(scenario="rocket_design", name=rocket.name))
        slog.add_body(rocket.stages[0].name, "vehicle", "cylinder", [0.5 * SM.d_ref * 3, SM.L])
        for st in rocket.stages[1:]:
            slog.add_body(st.name, "vehicle", "cylinder", [0.5 * st.d * 3, st.x_aft - st.x_top])
        for s_, u_ in (("alt", "m"), ("mach", ""), ("q", "Pa"), ("margin", "cal")):
            slog.add_series(s_, u_)
    next_slog = 0.0
    tick = 0.01

    def thrust_now():
        T = 0.0
        on = False
        for idx in range(n_top):
            st = rocket.stages[idx]
            if st.motor is not None and motor_t[idx] is not None and motor_t[idx] >= 0:
                f = st.motor.thrust(motor_t[idx]) * th_scale[idx]
                if f > 0:
                    T += f
                    on = True
        return T, on

    while t < t_max and not landed:
        # --- substep size from the flight state
        T_now, on_now = thrust_now()
        if canopy is not None:
            nsub, tick = 1, (0.01 if fcs is not None else 0.05)
        elif on_rail or on_now or diag["q"] > 20e3:
            nsub = 10
        elif diag["q"] > 2e3:
            nsub = 5
        elif diag["q"] > 50.0:
            nsub = 2
        else:
            nsub = 1
        h = tick / nsub
        for _ in range(nsub):
            h_agl, gu = _geo(r, air.site)
            z_msl = h_agl + air.site
            rho, c_snd, mu, p_amb, T_amb = air.at(z_msl)
            if wind and t >= next_wind:
                wind_v = air.wind(r, t)
                next_wind = t + 0.05
            g = G0 * (R_EARTH / (R_EARTH + max(z_msl, 0.0))) ** 2
            T, on = thrust_now()
            m, x_cg, I = SM.mass(motor_t)
            m *= float(disp.get("mass_scale", 1.0))
            v_rel = vel - wind_v
            if canopy is not None:
                gw = np.array(gu) * (g * m)
                sp = math.sqrt(float(v_rel @ v_rel))
                Mn = sp / c_snd
                cda = canopy.cda(Mn, t)
                F = -0.5 * rho * cda * sp * v_rel + gw
                vel = vel + F / m * h
                r = r + vel * h
                if sp > 0.5:
                    q = tuple(float(v) for v in AE.q_from_x_to(-v_rel / sp))
                diag.update(M=Mn, q=0.5 * rho * sp * sp, thrust=0.0, m=m)
            else:
                vb = _qrot_inv(q, (float(v_rel[0]), float(v_rel[1]), float(v_rel[2])))
                Vb = math.sqrt(vb[0] * vb[0] + vb[1] * vb[1] + vb[2] * vb[2])
                Mn = Vb / c_snd
                i, wM = SM.idx(Mn)
                cn_sum, cn_x, cn_xx, cn_fin = SM.sums(i, wM)
                qd = 0.5 * rho * Vb * Vb
                Re = rho * Vb * SM.L / mu if mu > 0 else 0.0
                cdv = SM.cd(i, wM, Re, on) * cd_scale
                Fx = Fy = Fz = 0.0
                tx = ty = tz = 0.0
                alpha = 0.0
                if Vb > 1e-6:
                    kd = qd * SM.a_ref * cdv / Vb
                    Fx -= kd * vb[0]
                    Fy -= kd * vb[1]
                    Fz -= kd * vb[2]
                    k = 0.5 * rho * SM.a_ref * Vb
                    cn_arm = cn_sum * x_cg - cn_x                 # sum CNa_i (x_cg - x_i)
                    Fy -= k * cn_sum * vb[1]
                    Fz -= k * cn_sum * vb[2]
                    ty += k * vb[2] * cn_arm                      # r x F, r = (x_cg - x_i, 0, 0)
                    tz -= k * vb[1] * cn_arm
                    vperp = math.hypot(vb[1], vb[2])
                    if vperp > 1e-9:
                        sa = vperp / Vb
                        fb = qd * 1.1 * SM.A_plan * sa * sa / vperp
                        d_ = x_cg - SM.x_plan
                        Fy -= fb * vb[1]
                        Fz -= fb * vb[2]
                        ty += fb * vb[2] * d_
                        tz -= fb * vb[1] * d_
                    c2 = 0.5 * rho * Vb * SM.a_ref * (cn_xx - 2.0 * x_cg * cn_x + x_cg * x_cg * cn_sum)
                    if on:
                        mdot = sum(mt.prop_kg * mt.thrust(motor_t[ix]) / mt.I_total for ix, mt, _ in SM.props
                                   if motor_t[ix] is not None and motor_t[ix] >= 0)
                        c2 += mdot * (SM.x_noz - x_cg) ** 2
                    ty -= c2 * w[1]
                    tz -= c2 * w[2]
                    if SM.fin is not None:
                        f = SM.fin
                        a_eff = (cn_fin / f.n) * SM.a_ref / SM.S_fin if cn_fin > 0 else 0.0
                        tx += f.n * qd * a_eff * math.radians(f.cant_deg) * SM.J1
                        tx -= f.n * 0.5 * rho * Vb * a_eff * SM.J2 * w[0]
                    alpha = math.degrees(math.atan2(vperp, vb[0]))
                if T > 0:
                    ftx = T * math.cos(misalign)
                    fty = T * math.sin(misalign) * math.cos(mis_az)
                    ftz = T * math.sin(misalign) * math.sin(mis_az)
                    Fx += ftx
                    Fy += fty
                    Fz += ftz
                    d_ = x_cg - SM.x_noz
                    ty -= d_ * ftz
                    tz += d_ * fty
                Fw = _qrot(q, (Fx, Fy, Fz))
                gm = g * m
                F_world = np.array([Fw[0] + gu[0] * gm, Fw[1] + gu[1] * gm, Fw[2] + gu[2] * gm])
                ax = Fx / m
                margin = ((cn_x / cn_sum) - x_cg) / SM.d_ref
                diag.update(M=Mn, q=qd, alpha=alpha, margin=margin, cd=cdv, thrust=T, m=m, x_cg=x_cg, ax=ax)
                if on_rail:
                    a_along = float(np.dot(F_world, u_rail)) / m
                    vs = float(np.dot(vel, u_rail))
                    if not (s_rail <= 0.0 and vs <= 0.0 and a_along <= 0.0):
                        vs += a_along * h
                        s_rail += vs * h
                        vel = vs * u_rail
                        r = r_cg0 + s_rail * u_rail
                        if s_rail >= rail["length_m"]:
                            on_rail = False
                            rail_exit = dict(t=t + h, v=vs, margin=margin, accel_g=ax / G0)
                            events.append((t + h, "rail_exit", "%.1f m/s" % vs))
                else:
                    rn, q, vn, w = _step_rigid(r, q, vel, w, F_world, (tx, ty, tz), m, I, h)
                    r, vel = np.array(rn), np.array(vn)
                if (on or Mn > 0.3) and not on_rail and t > 0.05 and margin < min_margin[0] and Vb > 20.0:
                    min_margin[:] = [margin, dict(t=t, M=Mn, alt=h_agl)]
                if abs(ax / G0) > maxes["accel_g"][0]:
                    maxes["accel_g"] = (abs(ax / G0), t)
                if not on_rail and Vb > 30 and alpha > maxes["alpha"][0]:
                    maxes["alpha"] = (alpha, t)
            if diag["M"] > maxes["M"][0]:
                maxes["M"] = (diag["M"], t, h_agl)
            if diag["q"] > maxes["q"][0]:
                maxes["q"] = (diag["q"], t, h_agl, diag["M"])
            # --- motor clocks, staging
            for idx in range(n_top):
                if motor_t[idx] is not None:
                    motor_t[idx] += h
                    st = rocket.stages[idx]
                    if idx not in burnout and motor_t[idx] >= st.motor.burn_s:
                        burnout[idx] = dict(t=t + h, alt=h_agl, v=float(np.linalg.norm(vel)), M=diag["M"])
                        events.append((t + h, "burnout", st.name))
                        if idx > 0 and idx == n_top - 1:
                            sep_due = t + h + float(st.spec.get("separation_delay_s", 0.5))
            if sep_due is not None and t + h >= sep_due:
                st = rocket.stages[n_top - 1]
                ms = rocket.stage_dry_mass(st.idx)
                spent.append(_Body3(st.name, r.copy(), vel.copy(), ms, lambda Mx, s=st: _spent_stage_cd(s, Mx),
                                    math.pi * st.d ** 2 / 4, getattr(st, "chutes", {}), t))
                events.append((t + h, "separation", st.name))
                n_top -= 1
                SM = models[n_top]
                m_new, xcg_new, _ = SM.mass(motor_t)
                sep_due = None
                nxt = rocket.stages[n_top - 1]
                if nxt.motor is not None:
                    ign_due = (n_top - 1, t + h + float(nxt.spec.get("ignition_delay_s", 1.0)))
            if ign_due is not None and t + h >= ign_due[1]:
                idx = ign_due[0]
                nose = _qrot(q, (1.0, 0.0, 0.0))
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, nose[2]))))
                if tilt_lock is not None and tilt > float(tilt_lock):
                    lockouts.append(dict(stage=rocket.stages[idx].name, t=t + h, tilt_deg=tilt))
                    events.append((t + h, "ignition_inhibited", "%s tilt %.1f deg" % (rocket.stages[idx].name, tilt)))
                else:
                    motor_t[idx] = 0.0
                    ign_time[idx] = t + h
                    events.append((t + h, "ignition", rocket.stages[idx].name))
                ign_due = None
            # --- spent stages
            for b in spent:
                if b.landed:
                    continue
                hb, gub = _geo(b.r, air.site)
                zb = hb + air.site
                rb, cb, _, _, _ = air.at(zb)
                vr = b.v - (air.wind(b.r, t) if wind and zb < air.wind_top and int(t / h) % 20 == 0 else 0.0)
                sp = math.sqrt(float(vr @ vr))
                gb = G0 * (R_EARTH / (R_EARTH + max(zb, 0.0))) ** 2
                Fb_ = -0.5 * rb * b.cda(sp / cb, t) * sp * vr + np.array(gub) * (gb * b.m)
                b.v = b.v + Fb_ / b.m * h
                b.r = b.r + b.v * h
                if hb > b.apogee:
                    b.apogee = hb
                for nm, ch in b.chutes.items():
                    if nm in b.deployed:
                        continue
                    if (ch.get("deploy", "apogee") == "apogee" and b.v[2] < 0) or \
                            (ch.get("deploy") == "agl" and b.v[2] < 0 and hb <= float(ch.get("agl_m", 300))):
                        b.deployed[nm] = t
                if hb <= 0.0 and t - b.t0 > 1.0:
                    b.r[2] = 0.0
                    b.landed = True
                    events.append((t + h, "landed", b.name))
            t += h
            # --- recovery decisions (ideal event logic) / apogee
            if h_agl > apogee[1]:
                apogee = (t, h_agl)
            past_apogee = vel[2] < 0 and t > 1.0 and not on_rail and n_top == 1 and \
                all(i in burnout for i in range(len(rocket.stages)) if rocket.stages[i].motor is not None
                    and motor_t[i] is not None)
            if fcs is None:
                for nm, ch in chutes.items():
                    if nm in deployed:
                        continue
                    if past_apogee and (ch.get("deploy", "apogee") == "apogee" or
                                        (ch.get("deploy") == "agl" and h_agl - r_cg0[2] <= float(ch.get("agl_m", 300.0)))):
                        deployed[nm] = t
                        events.append((t, "%s_deploy" % nm, "%.0f m AGL" % h_agl))
            if stop_at_apogee and past_apogee:
                landed = True
                break
            if deployed and canopy is None:
                canopy = _Body3(top.name, r, vel, m, lambda Mx, s=SM: float(s.cd_off_l[min(int(Mx / 0.01), 299)][20]),
                                SM.a_ref, chutes, t)
                canopy.deployed = deployed
            if canopy is not None:
                canopy.deployed = deployed
            if h_agl < r_cg0[2] - 0.01 and t > 1.0 and not on_rail:
                r[2] = r_cg0[2]
                landed = True
                events.append((t, "landed", top.name))
                break
        if fcs is not None and not landed:
            dep = fcs.step(t, _geo(r, air.site)[0] + air.site, diag["ax"] if canopy is None else 0.0, rng)
            for nm in ("drogue", "main"):
                if dep.get(nm) and nm in chutes and nm not in deployed:
                    deployed[nm] = t
                    events.append((t, "%s_deploy" % nm, "rocket-fc at %.0f m AGL" % _geo(r, air.site)[0]))
                elif dep.get(nm) and nm not in chutes and nm == "drogue" and "main" in chutes and \
                        chutes["main"].get("deploy", "apogee") == "apogee" and "main" not in deployed:
                    deployed["main"] = t
                    events.append((t, "main_deploy", "rocket-fc apogee channel"))
        if t >= next_log:
            next_log += 1.0 / log_hz
            nose = _qrot(q, (1.0, 0.0, 0.0))
            h_log = _geo(r, air.site)[0]
            traj.append((t, r[0], r[1], h_log, vel[0], vel[1], vel[2], diag["M"], diag["q"], diag["alpha"],
                         diag["margin"], diag["m"], diag["thrust"], diag["cd"], n_top, diag["ax"],
                         math.degrees(math.acos(max(-1.0, min(1.0, nose[2])))), air.at(h_log + air.site)[3],
                         q[0], q[1], q[2], q[3]))
            for b in spent:
                b.track.append((t, b.r[0], b.r[1], b.r[2]))
        if slog is not None and t >= next_slog:
            next_slog += 1.0 / 30.0
            xs = [[r[0], r[1], r[2]]] + [[b.r[0], b.r[1], b.r[2]] if b else [0, 0, -100] for b in
                                         [next((s for s in spent if s.name == st.name), None)
                                          for st in rocket.stages[1:]]]
            qs = [list(q)] + [[1.0, 0, 0, 0]] * (len(rocket.stages) - 1)
            slog.frame(t, body_x=xs, body_q=qs, series=[r[2], diag["M"], diag["q"], diag["margin"]])

    # finish spent stages still airborne (ballistic, coarse step)
    for b in spent:
        tt = t
        while not b.landed and tt < t_max:
            hb, gub = _geo(b.r, air.site)
            zb = hb + air.site
            rb, cb, _, _, _ = air.at(zb)
            vr = b.v.copy()
            sp = math.sqrt(float(vr @ vr))
            gb = G0 * (R_EARTH / (R_EARTH + max(zb, 0.0))) ** 2
            for nm, ch in b.chutes.items():
                if nm not in b.deployed and ((ch.get("deploy", "apogee") == "apogee" and b.v[2] < 0) or
                                             (ch.get("deploy") == "agl" and b.v[2] < 0 and
                                              hb <= float(ch.get("agl_m", 300)))):
                    b.deployed[nm] = tt
            Fb_ = -0.5 * rb * b.cda(sp / cb, tt) * sp * vr + np.array(gub) * (gb * b.m)
            b.v = b.v + Fb_ / b.m * 0.05
            b.r = b.r + b.v * 0.05
            b.apogee = max(b.apogee, hb)
            tt += 0.05
            if int(tt / 0.05) % 4 == 0:
                b.track.append((tt, b.r[0], b.r[1], max(b.r[2], 0.0)))
            if hb <= 0:
                b.landed = True
                b.t_land = tt
    tr = np.array(traj)
    if slog is not None:
        slog.close(apogee_agl_m=apogee[1] - r_cg0[2])
    karman = None
    if apogee[1] + air.site >= 100e3 and len(tr):
        k = np.argmax(tr[:, 3] + air.site >= 100e3)          # z column ~ height (curvature < 1 % below apogee)
        karman = float(tr[k, 0])
    landing = (float(r[0]), float(r[1]))
    summary = dict(
        name=rocket.name, site_elev_m=air.site, apogee_agl_m=float(apogee[1] - r_cg0[2]),
        apogee_msl_m=float(apogee[1] - r_cg0[2] + air.site), t_apogee_s=float(apogee[0]),
        max_mach=dict(M=float(maxes["M"][0]), t=float(maxes["M"][1]), alt=float(maxes["M"][2]) if len(maxes["M"]) > 2 else 0.0),
        max_q=dict(pa=float(maxes["q"][0]), t=float(maxes["q"][1]), alt=float(maxes["q"][2]), M=float(maxes["q"][3])),
        max_accel_g=float(maxes["accel_g"][0]), max_alpha_deg=float(maxes["alpha"][0]),
        rail_exit={k: float(v) for k, v in (rail_exit or {}).items()} or None,
        min_margin_cal=dict(cal=float(min_margin[0]), **{k: float(v) for k, v in (min_margin[1] or {}).items()}),
        burnout={rocket.stages[k].name: v for k, v in burnout.items()},
        liftoff_mass_kg=float(m0), events=events, ignition_lockouts=lockouts,
        landing_xy_m=landing, drift_m=float(math.hypot(*landing)), flight_time_s=float(t),
        karman_crossing_t_s=karman, fc=fc or "ideal-events",
        spent=[dict(name=b.name, apogee_agl_m=float(b.apogee), landing_xy_m=(float(b.r[0]), float(b.r[1])),
                    drift_m=float(math.hypot(b.r[0], b.r[1]))) for b in spent],
        warnings=(["max Mach %.1f is outside the drag model's declared validity range (M <= 5)" % maxes["M"][0]]
                  if maxes["M"][0] > 5.0 else []),
        env=air.env.describe(), air_model="core<=%.0f m | US76 | Vallado >= 86 km" % air.core_top)
    cols = ["t", "x", "y", "z", "vx", "vy", "vz", "mach", "q", "alpha_deg", "margin_cal", "mass", "thrust", "cd",
            "n_attached", "accel_axial", "tilt_deg", "p_amb", "qw", "qx", "qy", "qz"]
    tracks = {b.name: np.array(b.track) for b in spent}
    return dict(summary=summary, traj=tr, cols=cols, spent_tracks=tracks, r_cg0=r_cg0)


class _FCBridge:
    def __init__(self,*args,**kwargs):
        raise ValueError("The hardware-specific flight-computer adapter is not part of this portable release; use the default ideal recovery events.")

#!/usr/bin/env python3
# DESCRIPTION: Every vehicle is a 6-DOF rigid chassis on N wheel stations, stepped in the SAME clock, air and wind as
#              every other domain (World plug-in: upload(device) / step(t, dt, atm, wind, device)).
#
#              Per wheel station (MODELED; every constant is a preset parameter, ASSUMED unless a source is given):
#                suspension  raycast strut along body -z from its mount; wheel centre at extension s in [0, s_max];
#                            F = k (L0 - s) - c ds/dt (+ bump stop 20 k below s = 0), never pulls; the ground
#                            reaction N = F acts along the terrain normal (raycast-vehicle convention: the linkage
#                            carries the strut-normal part, so a pitched body gets no spurious horizontal thrust).
#                            No anti-dive/anti-squat geometry, no unsprung mass, no tyre radial compliance.
#                tyre        Magic Formula + normalised combined slip (domains/land/tyre.py), surface coefficients
#                            from the terrain cell under the contact patch (core/terrain.SURFACES); transient slip
#                            from contact-patch deflection with relaxation length sigma (default 1 R, ASSUMED).
#                wheel types TYRE (steerable or fixed) | ROLLER (mecanum gamma=+-45 deg, omni gamma=0: transmits
#                            force only along a = cos(g) f + sin(g) l, rollers free across it with roller rolling
#                            resistance) | CASTER (free swivel: only rolling resistance, opposite contact velocity).
#                spin        I_w dw/dt = T_motor - F_a R cos(g); linearised-implicit in the tyre slope; brakes and
#                            rolling resistance (crr N R) are dissipative torques that can stop but never reverse the
#                            wheel (exact hold on a slope, no chatter).
#                motor       DC/BLDC equivalent per driven wheel: i = clamp((u V_bat - k_t G w) / R_m, +-i_max),
#                            T_wheel = eta G k_t i; battery V = V_oc(SOC) - R_int I, V_oc linear in SOC,
#                            I_bat = sum(u i) + I_q (PWM average, converter lossless: ASSUMED). SOC and energy logged.
#                control     body twist command (vx, vy, yaw rate) -> rate-limited by the profile accel limits ->
#                            per-wheel inverse kinematics (steer angle + wheel speed; exact Ackermann for steered
#                            tyres, exact mecanum/omni allocation for rollers) -> wheel-speed PI -> motor duty.
#                            Modes: 0 TWIST (cmd = vx, vy, r), 1 WAYPOINTS (pure pursuit, loop), 2 SPEED_STEER
#                            (cmd = v, bicycle steer delta: r = v tan(delta)/L, vy_cg = r b_rear).
#                            Hold (zero command): motor off, holding brake T_brake on every wheel (brake-to-stop).
#              Chassis: 8 box corners touch the terrain through domains/land/contact.contact_force (roll-over,
#              high-centring). Air drag 0.5 rho CdA |v_rel| v_rel at the CG; rho by world height, wind by AGL height.
#              Integration: semi-implicit Euler at dt/n_sub (n_sub = ceil(dt / dt_max), dt_max default 5e-4 s).
import math

import numpy as np
import warp as wp

from engineering_sim.core import atmos as A
from engineering_sim.core import terrain as T
from engineering_sim.core import wind as W
from . import contact as CT
from engineering_sim.core import rigid6 as RB   # THE shared rigid-body step (core/rigid6.step_rigid; UNIFY 2026-10-15)
from . import tyre as TY

TYRE, ROLLER, CASTER = 0, 1, 2
MODE_TWIST, MODE_WAYPOINTS, MODE_SPEED_STEER = 0, 1, 2
from engineering_sim.core.constants import G0  # noqa: E402  ONE gravity constant (core v0.2)
wp.set_module_options({"enable_backward": False})


# Packed per-element parameter vectors (keeps the kernel under the CUDA 4 KB parameter limit; layout = *_IDX below)
VP_KEYS = ("mass cda eta i_q Ixx Iyy Izz hx hy hz bx by bz vx_max vx_rev vy_max r_max ax_max ay_max rdot_max "
           "L b_rear cruise look kt R_m i_max gear V_full V_empty R_int cap_ah kp ki t_brake v_low kc_k kc_c").split()
WH_KEYS = "px py pz radius k c L0 s_max gamma I_w steer_max steer_rate mu_scale flags sigma".split()
vecVP = wp.types.vector(length=len(VP_KEYS), dtype=wp.float32)
vecWH = wp.types.vector(length=len(WH_KEYS), dtype=wp.float32)
vec5i = wp.types.vector(length=5, dtype=wp.int32)    # mode, wp_start, wp_count, w_start, w_count
vec5f = wp.types.vector(length=5, dtype=wp.float32)  # SOC, I_bat [A], V_bat [V], battery energy out [J], odometer [m]


@wp.struct
class VehP:
    pv: wp.array(dtype=vecVP)
    iv: wp.array(dtype=vec5i)
    wp_pts: wp.array(dtype=wp.vec3)


@wp.struct
class VehS:
    x: wp.array(dtype=wp.vec3)
    q: wp.array(dtype=wp.quat)
    v: wp.array(dtype=wp.vec3)
    wb: wp.array(dtype=wp.vec3)
    cmd: wp.array(dtype=wp.vec3)
    tw: wp.array(dtype=wp.vec3)         # rate-limited desired body twist
    wp_idx: wp.array(dtype=int)
    e5: wp.array(dtype=vec5f)


@wp.struct
class WhP:
    wv: wp.array(dtype=vecWH)


@wp.struct
class WhS:
    st4: wp.array(dtype=wp.vec4)        # omega, phi, steer, s
    integ: wp.array(dtype=float)
    rel: wp.array(dtype=wp.vec2)        # contact-patch deflection (along a, along b) [m]
    out4: wp.array(dtype=wp.vec4)       # Fz, kappa', tan(alpha'), drive torque
    centre: wp.array(dtype=wp.vec3)
    wq: wp.array(dtype=wp.quat)
    f: wp.array(dtype=wp.vec3)          # this substep: force on the chassis (world) [N]
    tq: wp.array(dtype=wp.vec3)         # this substep: torque about the CG (world) [N m]
    cur: wp.array(dtype=float)          # this substep: battery-side current [A]
    owner: wp.array(dtype=int)          # wheel -> vehicle index


@wp.struct
class VehX:   # per-vehicle scratch shared by the kernels of one substep
    aux: wp.array(dtype=wp.vec4)        # hold flag, V_bat, -, -
    uw: wp.array(dtype=wp.vec3)         # wind at the CG height AGL (refreshed every wind_every substeps)
    cf: wp.array(dtype=wp.vec3)         # [n*8] chassis-corner contact forces
    ct: wp.array(dtype=wp.vec3)         # [n*8] chassis-corner torques about the CG


@wp.func
def _rate(cur: float, tgt: float, step: float):
    return cur + wp.clamp(tgt - cur, -step, step)


# One substep = k_pre (per vehicle) -> k_wheel (per wheel) + k_corner (per corner) -> k_post (per vehicle), plus
# k_wind every wind_every substeps. Splitting by wheel/corner keeps each GPU thread's dependent-load chain short
# (the per-vehicle loop version was latency-bound at ~0.4 ms per substep on the P4).
@wp.kernel(enable_backward=False)
def k_wind(t_off: float, tarr: wp.array(dtype=float), tp: T.TerrainParams,
           w: W.WindParams, mk: wp.array(dtype=wp.vec3), ma: wp.array(dtype=float),
           mph: wp.array(dtype=float), ms: wp.array(dtype=wp.vec3), S: VehS, X: VehX):
    i = wp.tid()
    t = tarr[0] + t_off   # clock lives on the device so a whole frame can be one CUDA graph
    X.uw[i] = W.wind_at(T.agl_point(S.x[i], tp, 0.05), t, w, mk, ma, mph, ms)


@wp.kernel(enable_backward=False)
def k_pre(dt: float, P: VehP, S: VehS, X: VehX):
    i = wp.tid()
    pv = P.pv[i]
    iv = P.iv[i]
    p = S.x[i]
    rot = S.q[i]
    lim = wp.vec4(pv[13], pv[14], pv[15], pv[16])
    ac = wp.vec3(pv[17], pv[18], pv[19])
    g4 = wp.vec4(pv[20], pv[21], pv[22], pv[23])
    bat = wp.vec4(pv[28], pv[29], pv[30], pv[31])
    # ---------------- guidance -> target body twist ----------------
    mode = iv[0]
    tgt = S.cmd[i]
    if mode == 1:
        nwp = iv[2]
        j = S.wp_idx[i]
        tpt = P.wp_pts[iv[1] + j]
        d = wp.vec3(tpt[0] - p[0], tpt[1] - p[1], 0.0)
        if wp.length(d) < g4[3]:
            j = (j + 1) % nwp
            S.wp_idx[i] = j
            tpt = P.wp_pts[iv[1] + j]
            d = wp.vec3(tpt[0] - p[0], tpt[1] - p[1], 0.0)
        db = wp.quat_rotate_inv(rot, d)
        al = wp.atan2(db[1], db[0])
        ld = wp.max(wp.length(d), g4[3])
        vd = g4[2] * wp.max(wp.cos(al), 0.15)
        tgt = wp.vec3(vd, 0.0, 2.0 * vd * wp.sin(al) / ld)
    if mode == 2:
        rr = tgt[0] * wp.tan(tgt[1]) / g4[0]
        tgt = wp.vec3(tgt[0], rr * g4[1], rr)
    tgt = wp.vec3(wp.clamp(tgt[0], -lim[1], lim[0]), wp.clamp(tgt[1], -lim[2], lim[2]), wp.clamp(tgt[2], -lim[3], lim[3]))
    tw0 = S.tw[i]
    tw = wp.vec3(_rate(tw0[0], tgt[0], ac[0] * dt), _rate(tw0[1], tgt[1], ac[1] * dt), _rate(tw0[2], tgt[2], ac[2] * dt))
    S.tw[i] = tw
    hold = float(0.0)
    if wp.length(tgt) < 1.0e-4 and wp.length(tw) < 1.0e-4:
        hold = 1.0
    e5 = S.e5[i]
    vbat = wp.max(bat[1] + (bat[0] - bat[1]) * e5[0] - bat[2] * e5[1], 0.05 * bat[0])
    X.aux[i] = wp.vec4(hold, vbat, 0.0, 0.0)


@wp.kernel(enable_backward=False)
def k_wheel(dt: float, tp: T.TerrainParams, P: VehP, S: VehS, WP: WhP, WS: WhS, X: VehX):
    k = wp.tid()
    i = WS.owner[k]
    pv = P.pv[i]
    iv = P.iv[i]
    m = pv[0]
    eta = pv[2]
    mot = wp.vec4(pv[24], pv[25], pv[26], pv[27])
    pi = wp.vec4(pv[32], pv[33], pv[34], pv[35])
    vlow = pi[3]
    p = S.x[i]
    rot = S.q[i]
    vel = S.v[i]
    om_w = wp.quat_rotate(rot, S.wb[i])
    zb = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
    tw = S.tw[i]
    ax = X.aux[i]
    hold = int(ax[0])
    vbat = ax[1]
    share = m / float(wp.max(iv[4], 1))
    wv = WP.wv[k]
    pos = wp.vec3(wv[0], wv[1], wv[2])
    a4 = wp.vec4(wv[3], wv[4], wv[5], wv[6])
    b4 = wp.vec4(wv[7], wv[8], wv[9], wv[10])
    st4 = WS.st4[k]
    R = a4[0]
    fl = int(wv[13])
    typ = fl & 15
    gam = b4[1]
    cg = wp.cos(gam)
    # ---- wheel targets from the twist (inverse kinematics) ----
    ux = tw[0] - tw[2] * pos[1]
    uy = tw[1] + tw[2] * pos[0]
    st = st4[2]
    st_cmd = float(0.0)
    om_t = float(0.0)
    if typ == 0:
        if ((fl >> 4) & 1) == 1:
            st_cmd = st
            um = wp.sqrt(ux * ux + uy * uy)
            if um > 1.0e-5:
                sg = float(1.0)
                st_cmd = wp.atan2(uy, ux)
                if st_cmd > 0.5 * wp.pi:
                    st_cmd = st_cmd - wp.pi
                    sg = -1.0
                if st_cmd < -0.5 * wp.pi:
                    st_cmd = st_cmd + wp.pi
                    sg = -1.0
                om_t = sg * um / R
            st_cmd = wp.clamp(st_cmd, -b4[3], b4[3])
        else:
            om_t = ux / R
    if typ == 1:
        om_t = (ux * cg + uy * wp.sin(gam)) / (R * cg)
    st = _rate(st, st_cmd, wv[11] * dt)
    # ---- strut raycast onto the terrain ----
    mount = p + wp.quat_rotate(rot, pos)
    s_prev = st4[3]
    s = s_prev
    n = T.normal(mount[0], mount[1], tp)
    for it in range(3):
        pc = mount - zb * s - n * R
        hn = T.height_normal(pc[0], pc[1], tp)
        n = wp.vec3(hn[0], hn[1], hn[2])
        gap = (pc[2] - hn[3]) * n[2]
        s = s + gap / wp.max(wp.dot(zb, n), 0.3)
    smax = b4[0]
    contact = int(1)
    if s > smax:
        contact = 0
        s = smax
    s = wp.max(s, -0.5 * smax)
    sdot = (s - s_prev) / dt
    fs = float(0.0)
    if contact == 1:
        fs = a4[1] * (a4[3] - s) - a4[2] * sdot
        if s < 0.0:
            fs = fs - 20.0 * a4[1] * s
        fs = wp.max(fs, 0.0)
    N = fs
    ctr = mount - zb * s
    pc = ctr - n * R
    rc = pc - p
    vc = vel + wp.cross(om_w, rc)
    fw = wp.quat_rotate(rot, wp.vec3(wp.cos(st), wp.sin(st), 0.0))
    fdir = wp.normalize(fw - n * wp.dot(fw, n))
    ldir = wp.cross(n, fdir)
    mid = T.material(pc[0], pc[1], tp)
    coef = tp.mf[mid]
    coef = wp.vec4(coef[0], coef[1], coef[2] * wv[12], coef[3])
    crr = tp.crr[mid]
    om = st4[0]
    ft = wp.vec3(0.0, 0.0, 0.0)
    kap = float(0.0)
    alp = float(0.0)
    tq = float(0.0)
    cur = float(0.0)
    if typ == 2:
        vt = vc - n * wp.dot(vc, n)
        vts = wp.length(vt)
        if vts > 1.0e-9 and N > 0.0:
            ft = vt * (-wp.min(crr * N, share * vts / dt) / vts)
        om = wp.dot(vc, fdir) / R
    else:
        adir = fdir * cg + ldir * wp.sin(gam)
        bdir = wp.cross(n, adir)
        va = wp.dot(vc, adir)
        vb = wp.dot(vc, bdir)
        # transient slip (contact-patch deflection u, relaxation length sig): du/dt = v_slip - |va| u / sig,
        # integrated implicitly; kappa' = u_a / sig, tan(alpha') = u_b / sig (-> SAE slip at steady state, a
        # spring at standstill: no creep). Low-speed damping blended in below v_low.
        sig = wv[14]
        ava = wp.abs(va)
        rel = WS.rel[k]
        den = 1.0 + dt * ava / sig
        ua = (rel[0] + dt * (om * R * cg - va)) / den
        ub = float(0.0)
        if typ == 0:
            ub = (rel[1] + dt * vb) / den
        ua = wp.clamp(ua, -sig, sig)
        ub = wp.clamp(ub, -2.0 * sig, 2.0 * sig)
        kap = ua / sig
        alp = ub / sig
        fa = float(0.0)
        fbb = float(0.0)
        kslope = float(0.0)
        if N > 0.0:
            fxy = TY.combined(coef, kap, alp) * N
            fa = fxy[0]
            fbb = fxy[1]
            blend = wp.max(1.0 - ava / vlow, 0.0)
            cd = 2.0 * 0.7 * wp.sqrt(coef[0] * coef[1] * coef[2] * N / sig * share)   # ~0.7 critical
            fa = fa + blend * cd * (ua - rel[0]) / dt
            fbb = fbb - blend * cd * (ub - rel[1]) / dt
            fx2 = TY.combined(coef, kap + 1.0e-3, alp)[0] * N
            kslope = (wp.max((fx2 - fxy[0]) / 1.0e-3, 0.0) * dt / sig + blend * cd) * R * cg / den
            if typ == 1 and wp.abs(vb) > 1.0e-9:
                fbb = -wp.sign(vb) * wp.min(crr * N, share * wp.abs(vb) / dt)
        else:
            ua = 0.0
            ub = 0.0
        WS.rel[k] = wp.vec2(ua, ub)
        ft = adir * fa + bdir * fbb
        # ---- drive + spin ----
        tdrive = float(0.0)
        tbrk = float(0.0)
        if ((fl >> 5) & 1) == 1:
            tmax = eta * mot[3] * mot[0] * mot[2]
            integ = WS.integ[k]
            e = om_t - om
            tdes = float(0.0)
            if hold == 1:
                integ = 0.0
                tbrk = pi[2]
            else:
                i_new = wp.clamp(integ + pi[1] * e * dt, -tmax, tmax)
                t_un = pi[0] * e + i_new
                if wp.abs(t_un) <= tmax or t_un * e < 0.0:   # conditional integration (anti-windup)
                    integ = i_new
                tdes = wp.clamp(pi[0] * e + integ, -tmax, tmax)
            WS.integ[k] = integ
            wm = mot[3] * om
            ides = tdes / (eta * mot[3] * mot[0])
            u = wp.clamp((ides * mot[1] + mot[0] * wm) / vbat, -1.0, 1.0)
            if hold == 1:
                u = 0.0
            im = wp.clamp((u * vbat - mot[0] * wm) / mot[1], -mot[2], mot[2])
            if hold == 1:
                im = 0.0
            tdrive = eta * mot[3] * mot[0] * im
            cur = u * im
        elif hold == 1:
            tbrk = pi[2]
        tq = tdrive
        I_eff = b4[2] + R * cg * kslope * dt
        om_free = om + dt * (tdrive - fa * R * cg) / I_eff
        tdiss = tbrk + crr * N * R
        om = wp.sign(om_free) * wp.max(wp.abs(om_free) - dt * tdiss / I_eff, 0.0)
    phi = st4[1] + om * dt
    WS.st4[k] = wp.vec4(om, phi, st, s)
    WS.out4[k] = wp.vec4(N, kap, alp, tq)
    WS.centre[k] = ctr
    # wheel glyph attitude: chassis * yaw(steer) * spin about the axle (body y) * (cylinder +X -> axle +Y)
    WS.wq[k] = rot * wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), st) * \
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), phi) * \
        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * wp.pi)
    fsus = n * N   # ground reaction along the contact normal (linkage carries the strut-normal part)
    WS.f[k] = fsus + ft
    WS.tq[k] = wp.cross(ctr - p, fsus) + wp.cross(rc, ft)
    WS.cur[k] = cur


@wp.kernel(enable_backward=False)
def k_corner(dt: float, tp: T.TerrainParams, P: VehP, S: VehS, X: VehX):
    c = wp.tid()
    i = c // 8
    c8 = c % 8
    pv = P.pv[i]
    hb = wp.vec3(pv[7], pv[8], pv[9])
    bo = wp.vec3(pv[10], pv[11], pv[12])
    sx = float(1.0)
    sy = float(1.0)
    sz = float(1.0)
    if c8 % 2 == 1:
        sx = -1.0
    if (c8 // 2) % 2 == 1:
        sy = -1.0
    if c8 >= 4:
        sz = -1.0
    rot = S.q[i]
    r = wp.quat_rotate(rot, bo + wp.vec3(sx * hb[0], sy * hb[1], sz * hb[2]))
    om_w = wp.quat_rotate(rot, S.wb[i])
    fc = CT.contact_force(S.x[i] + r, S.v[i] + wp.cross(om_w, r), tp, pv[36], pv[37], 1.0, pv[0] / 8.0, dt)
    X.cf[c] = fc
    X.ct[c] = wp.cross(r, fc)


@wp.kernel(enable_backward=False)
def k_post(dt: float, atm: A.AtmosParams, P: VehP, S: VehS, WS: WhS, X: VehX):
    i = wp.tid()
    pv = P.pv[i]
    iv = P.iv[i]
    m = pv[0]
    p = S.x[i]
    vel = S.v[i]
    vr = X.uw[i] - vel
    f = wp.vec3(0.0, 0.0, -G0 * m) + vr * (0.5 * A.air_density(p[2], atm) * pv[1] * wp.length(vr))
    tau = wp.vec3(0.0, 0.0, 0.0)
    i_sum = pv[3]
    for jj in range(iv[4]):
        k = iv[3] + jj
        f = f + WS.f[k]
        tau = tau + WS.tq[k]
        i_sum = i_sum + WS.cur[k]
    for c8 in range(8):
        f = f + X.cf[i * 8 + c8]
        tau = tau + X.ct[i * 8 + c8]
    e5 = S.e5[i]
    vbat = X.aux[i][1]
    S.e5[i] = vec5f(wp.max(e5[0] - i_sum * dt / (3600.0 * pv[31]), 0.0), i_sum, vbat, e5[3] + vbat * i_sum * dt,
                    e5[4] + wp.length(vel) * dt)
    rot = S.q[i]
    Ib = wp.vec3(pv[4], pv[5], pv[6])
    RB.step_rigid(p, rot, vel, S.wb[i], f, wp.quat_rotate_inv(rot, tau), wp.vec3(m, m, m), Ib, Ib, dt, i,
                  S.x, S.q, S.v, S.wb)


# =====================================================================================================================
# host side: presets + batch
# =====================================================================================================================
def motor_from_specs(t_wheel_max, v_top, radius, v_bat, gear, eta=0.9, loss_frac=0.05):
    """Parametric DC/BLDC-equivalent from wheel-level specs: no-load wheel speed = 1.1 v_top / R at V_bat,
    controller current limit i_max gives t_wheel_max; winding resistance from the copper-loss fraction at the
    current limit, i_max^2 R_m = loss_frac V_bat i_max (ASSUMED: 0.05 traction motor, ~0.25 small DC gearmotor)."""
    wm_nl = gear * 1.1 * v_top / radius
    kt = v_bat / wm_nl
    i_max = t_wheel_max / (eta * gear * kt)
    return dict(kt=kt, R_m=loss_frac * v_bat / i_max, i_max=i_max, gear=gear, eta=eta)


def _box_inertia(m, lx, ly, lz):
    return [m * (ly ** 2 + lz ** 2) / 12, m * (lx ** 2 + lz ** 2) / 12, m * (lx ** 2 + ly ** 2) / 12]


def _strut(centre_b, radius, k, zeta, fz_static, sprung_share, travel, wtype=TYRE, steer=0, drive=0, gamma=0.0,
           I_w=None, steer_max=0.6, steer_rate=1.2, mu_scale=1.0, sigma=None):
    """Wheel station whose static wheel centre is `centre_b` (body frame) at mid travel."""
    s_static = 0.5 * travel
    mount = [centre_b[0], centre_b[1], centre_b[2] + s_static]
    return dict(pos=mount, radius=radius, k=k, c=2 * zeta * math.sqrt(k * sprung_share), L0=s_static + fz_static / k,
                s_max=travel, wtype=wtype, steer=steer, drive=drive, gamma=math.radians(gamma),
                I_w=I_w if I_w is not None else 0.6 * 10 * radius ** 2 * max(radius / 0.3, 0.05),
                steer_max=steer_max, steer_rate=steer_rate, mu_scale=mu_scale,
                sigma=sigma if sigma is not None else radius)   # relaxation length ~ 1 R (ASSUMED; Pacejka ch. 7)


PRESETS = {}


def preset(fn):
    PRESETS[fn.__name__] = fn
    return fn


@preset
def car(mass=1500.0, wheelbase=2.70, track=1.55, cg_from_rear=1.35, h_cg=0.55, radius=0.32, f_ride_hz=1.3,
        zeta=0.35, travel=0.20, drive="rwd", t_wheel_max=1200.0, v_top=45.0, v_bat=350.0, gear=9.0, cap_ah=150.0,
        length=4.5, width=1.8, height=1.4, cda=0.65, steer_max_deg=35.0, **kw):
    """Generic passenger car / EV (ASSUMED: every number). drive = rwd | fwd | awd."""
    a, b = wheelbase - cg_from_rear, cg_from_rear
    fz_f, fz_r = mass * G0 * b / wheelbase / 2, mass * G0 * a / wheelbase / 2
    k = mass / 4 * (2 * math.pi * f_ride_hz) ** 2
    z = -(h_cg - radius)
    W = []
    for x, fz, front in ((a, fz_f, True), (-b, fz_r, False)):
        for y in (track / 2, -track / 2):
            dr = (drive == "awd") or (drive == "fwd" and front) or (drive == "rwd" and not front)
            W.append(_strut([x, y, z], radius, k, zeta, fz, mass / 4, travel, TYRE, int(front), int(dr),
                            I_w=1.2, steer_max=math.radians(steer_max_deg), steer_rate=math.radians(60)))
    return dict(mass=mass, inertia=_box_inertia(mass, length, width, height * 0.6), cda=cda,
                half=[length / 2, width / 2, (height - 0.15) / 2], box_off=[0.0, 0.0, (height - 0.15) / 2 + 0.15 - h_cg],
                lim=[v_top, 8.0, 10.0, 1.2], acc=[4.0, 4.0, 1.5], geo=[wheelbase, cg_from_rear],
                motor=motor_from_specs(t_wheel_max, v_top, radius, v_bat, gear), bat=[v_bat * 1.1, v_bat * 0.85, 0.08, cap_ah],
                i_q=1.0, pi=[60.0, 400.0], t_brake=2500.0, v_low=1.0, wheels=W, h_cg=h_cg,
                shape=[length, width, height - 0.15], wheel_size=[radius, 0.22], **kw)


@preset
def diff_drive(mass=12.0, track=0.40, radius=0.08, h_cg=0.12, caster_x=0.22, f_ride_hz=4.0, zeta=0.5, travel=0.02,
               t_wheel_max=4.0, v_top=1.0, v_bat=24.0, gear=20.0, cap_ah=10.0, length=0.55, width=0.40, height=0.20,
               cda=0.08, **kw):
    """Differential-drive slow ground robot: 2 driven fixed tyres + front/rear casters (ASSUMED numbers; limits
    default to the combos diff-drive-slow-ground profile via combos_plant)."""
    z = -(h_cg - radius)
    k_d = mass * 0.35 / 1 * (2 * math.pi * f_ride_hz) ** 2
    zc = -(h_cg - 0.03)
    k_c = mass * 0.15 * (2 * math.pi * f_ride_hz) ** 2
    W = [_strut([0.0, track / 2, z], radius, k_d, zeta, mass * G0 * 0.35, mass * 0.35, travel, TYRE, 0, 1, I_w=0.004),
         _strut([0.0, -track / 2, z], radius, k_d, zeta, mass * G0 * 0.35, mass * 0.35, travel, TYRE, 0, 1, I_w=0.004),
         _strut([caster_x, 0.0, zc], 0.03, k_c, zeta, mass * G0 * 0.15, mass * 0.15, travel, CASTER, I_w=1e-4),
         _strut([-caster_x, 0.0, zc], 0.03, k_c, zeta, mass * G0 * 0.15, mass * 0.15, travel, CASTER, I_w=1e-4)]
    return dict(mass=mass, inertia=_box_inertia(mass, length, width, height), cda=cda,
                half=[length / 2, width / 2, height / 2 * 0.8], box_off=[0.0, 0.0, 0.10], lim=[v_top, 0.5, 0.0, 2.0],
                acc=[1.0, 0.0, 3.0], geo=[0.3, 0.15], motor=motor_from_specs(t_wheel_max, v_top, radius, v_bat, gear, loss_frac=0.25),
                bat=[25.2, 21.0, 0.05, cap_ah], i_q=0.3, pi=[0.4, 4.0], t_brake=6.0, v_low=0.3, wheels=W, h_cg=h_cg,
                shape=[length, width, height * 0.8], wheel_size=[radius, 0.04], **kw)


@preset
def mecanum(mass=15.0, wheelbase=0.40, track=0.42, radius=0.076, h_cg=0.12, f_ride_hz=4.0, zeta=0.5, travel=0.02,
            t_wheel_max=4.0, v_top=1.0, v_bat=24.0, gear=20.0, cap_ah=10.0, length=0.55, width=0.45, height=0.20,
            cda=0.09, **kw):
    """Holonomic robot on 4 mecanum wheels (45 deg rollers, X pattern seen from above). ASSUMED numbers."""
    z = -(h_cg - radius)
    k = mass / 4 * (2 * math.pi * f_ride_hz) ** 2
    W = []
    for x, y, g in ((wheelbase / 2, track / 2, -45), (wheelbase / 2, -track / 2, 45),
                    (-wheelbase / 2, track / 2, 45), (-wheelbase / 2, -track / 2, -45)):
        W.append(_strut([x, y, z], radius, k, zeta, mass * G0 / 4, mass / 4, travel, ROLLER, 0, 1, gamma=g,
                        I_w=0.004))
    return dict(mass=mass, inertia=_box_inertia(mass, length, width, height), cda=cda,
                half=[length / 2, width / 2, height / 2 * 0.8], box_off=[0.0, 0.0, 0.10], lim=[v_top, 0.5, 0.6, 2.0],
                acc=[1.0, 1.0, 3.0], geo=[wheelbase, wheelbase / 2], motor=motor_from_specs(t_wheel_max, v_top, radius, v_bat, gear, loss_frac=0.25),
                bat=[25.2, 21.0, 0.05, cap_ah], i_q=0.3, pi=[0.4, 4.0], t_brake=6.0, v_low=0.3, wheels=W, h_cg=h_cg,
                shape=[length, width, height * 0.8], wheel_size=[radius, 0.05], **kw)


@preset
def rover6(mass=60.0, wheelbase=0.9, track=0.7, radius=0.15, h_cg=0.35, f_ride_hz=2.5, zeta=0.45, travel=0.10,
           t_wheel_max=40.0, v_top=2.5, v_bat=48.0, gear=30.0, cap_ah=20.0, length=1.1, width=0.75, height=0.45,
           cda=0.4, **kw):
    """6-wheel skid-steer field rover (all wheels driven, none steered): rough-terrain test vehicle. ASSUMED."""
    z = -(h_cg - radius)
    k = mass / 6 * (2 * math.pi * f_ride_hz) ** 2
    W = [_strut([x, y, z], radius, k, zeta, mass * G0 / 6, mass / 6, travel, TYRE, 0, 1, I_w=0.05)
         for x in (wheelbase / 2, 0.0, -wheelbase / 2) for y in (track / 2, -track / 2)]
    return dict(mass=mass, inertia=_box_inertia(mass, length, width, height), cda=cda,
                half=[length / 2, width / 2, height / 2 * 0.7], box_off=[0.0, 0.0, 0.15], lim=[v_top, 1.0, 0.0, 1.2],
                acc=[1.5, 0.0, 2.0], geo=[wheelbase, wheelbase / 2], motor=motor_from_specs(t_wheel_max, v_top, radius, v_bat, gear, loss_frac=0.15),
                bat=[54.6, 42.0, 0.04, cap_ah], i_q=0.5, pi=[3.0, 30.0], t_brake=60.0, v_low=0.4, wheels=W, h_cg=h_cg,
                shape=[length, width, height * 0.7], wheel_size=[radius, 0.10], **kw)


class LandVehicles:
    """Batch of ground vehicles on one Terrain. add(preset_dict, pos, yaw) -> index."""

    def __init__(self, terrain, dt_max=5e-4, wind_every=10):
        """wind_every: substeps between wind samples at each CG (default 10 x 0.5 ms = 200 Hz; gust content
        above that is negligible at vehicle scale — ASSUMED)."""
        self.terrain, self.dt_max, self.rows, self.wind_every = terrain, float(dt_max), [], int(wind_every)

    def add(self, spec, pos_xy, yaw_deg=0.0, name=None, mode=MODE_TWIST, cmd=(0, 0, 0), waypoints=None, cruise=None,
            lookahead=None, soc=1.0, limits=None, accel=None, drop=0.0):
        s = dict(spec)
        if limits is not None:
            s["lim"] = list(limits)
        if accel is not None:
            s["acc"] = list(accel)
        x, y = pos_xy
        n = self.terrain.normal_np(x, y)          # start sitting on the local ground plane (no drop onto a slope)
        z = float(self.terrain.height_np(x, y))
        p0 = np.array([x, y, z]) + n * (s["h_cg"] + drop)
        s.update(x=p0.tolist(), q0=_quat_on_ground(n, math.radians(yaw_deg)), yaw=math.radians(yaw_deg),
                 name=name or "veh%d" % len(self.rows), mode=mode,
                 cmd=list(cmd), wps=[list(p) + [0.0] * (3 - len(p)) for p in (waypoints or [(x, y)])],
                 cruise=cruise if cruise is not None else 0.5 * s["lim"][0],
                 look=lookahead if lookahead is not None else max(2.0, 3 * s["geo"][0]), soc=soc)
        self.rows.append(s)
        return len(self.rows) - 1

    def upload(self, device):
        R = self.rows
        self.n = len(R)
        self.device = device
        F = lambda vals: wp.array(np.array(vals, np.float32), dtype=float, device=device)
        V3 = lambda vals: wp.array(np.array(vals, np.float32).reshape(-1, 3), dtype=wp.vec3, device=device)
        V4 = lambda vals: wp.array(np.array(vals, np.float32).reshape(-1, 4), dtype=wp.vec4, device=device)
        V2 = lambda vals: wp.array(np.array(vals, np.float32).reshape(-1, 2), dtype=wp.vec2, device=device)
        I = lambda vals: wp.array(np.array(vals, np.int32), dtype=int, device=device)
        ws, wc, Wl = [], [], []
        for r in R:
            ws.append(len(Wl)); wc.append(len(r["wheels"])); Wl.extend(r["wheels"])
        starts, counts, pts = [], [], []
        for r in R:
            starts.append(len(pts)); counts.append(len(r["wps"])); pts.extend(r["wps"])
        pv = []
        for r in R:
            kc_k = r["mass"] * G0 / 4 / 0.01
            d = dict(mass=r["mass"], cda=r["cda"], eta=r["motor"]["eta"], i_q=r["i_q"], kc_k=kc_k,
                     kc_c=2 * 0.7 * math.sqrt(kc_k * r["mass"] / 4), cruise=r["cruise"], look=r["look"])
            d.update(zip(("Ixx", "Iyy", "Izz"), r["inertia"]), **dict(zip(("hx", "hy", "hz"), r["half"])))
            d.update(zip(("bx", "by", "bz"), r["box_off"]), **dict(zip(("vx_max", "vx_rev", "vy_max", "r_max"), r["lim"])))
            d.update(zip(("ax_max", "ay_max", "rdot_max"), r["acc"]), **dict(zip(("L", "b_rear"), r["geo"])))
            d.update({k: r["motor"][k] for k in ("kt", "R_m", "i_max", "gear")})
            d.update(zip(("V_full", "V_empty", "R_int", "cap_ah"), r["bat"]), **dict(zip(("kp", "ki"), r["pi"])))
            d.update(t_brake=r["t_brake"], v_low=r["v_low"])
            pv.append([d[k] for k in VP_KEYS])
        P = VehP()
        P.pv = wp.array(np.array(pv, np.float32), dtype=vecVP, device=device)
        P.iv = wp.array(np.array([[r["mode"], starts[j], counts[j], ws[j], wc[j]] for j, r in enumerate(R)], np.int32),
                        dtype=vec5i, device=device)
        P.wp_pts = V3(pts)
        self.P, self.pv_np = P, np.array(pv, np.float32)
        S = VehS()
        S.x = V3([r["x"] for r in R])
        S.q = wp.array(np.array([r["q0"] for r in R], np.float32), dtype=wp.quat, device=device)
        S.v, S.wb = wp.zeros(self.n, dtype=wp.vec3, device=device), wp.zeros(self.n, dtype=wp.vec3, device=device)
        S.cmd, S.tw = V3([r["cmd"] for r in R]), wp.zeros(self.n, dtype=wp.vec3, device=device)
        S.wp_idx = wp.zeros(self.n, dtype=int, device=device)
        S.e5 = wp.array(np.array([[r["soc"], 0, r["bat"][0], 0, 0] for r in R], np.float32), dtype=vec5f, device=device)
        self.S = S
        self.nw = len(Wl)
        self.wheel_owner = np.repeat(np.arange(self.n), wc)
        Wp = WhP()
        Wp.wv = wp.array(np.array([list(w["pos"]) + [w[k] for k in WH_KEYS[3:13]] +
                                   [w["wtype"] | (w["steer"] << 4) | (w["drive"] << 5), w["sigma"]] for w in Wl],
                                  np.float32),
                         dtype=vecWH, device=device)
        self.WP = Wp
        Ws = WhS()
        Ws.st4 = V4([[0, 0, 0, 0.5 * w["s_max"]] for w in Wl])
        Ws.integ = wp.zeros(self.nw, dtype=float, device=device)
        Ws.rel = wp.zeros(self.nw, dtype=wp.vec2, device=device)
        Ws.out4 = wp.zeros(self.nw, dtype=wp.vec4, device=device)
        Ws.centre = wp.zeros(self.nw, dtype=wp.vec3, device=device)
        Ws.wq = wp.zeros(self.nw, dtype=wp.quat, device=device)
        Ws.f = wp.zeros(self.nw, dtype=wp.vec3, device=device)
        Ws.tq = wp.zeros(self.nw, dtype=wp.vec3, device=device)
        Ws.cur = wp.zeros(self.nw, dtype=float, device=device)
        Ws.owner = I(self.wheel_owner)
        self.WS = Ws
        X = VehX()
        X.aux = wp.zeros(self.n, dtype=wp.vec4, device=device)
        X.uw = wp.zeros(self.n, dtype=wp.vec3, device=device)
        X.cf = wp.zeros(self.n * 8, dtype=wp.vec3, device=device)
        X.ct = wp.zeros(self.n * 8, dtype=wp.vec3, device=device)
        self.X = X
        self._sub = 0
        self.tp = self.terrain.params(device)
        self.wheels = Wl
        if getattr(self, "tarr", None) is None:
            self.tarr = wp.zeros(1, dtype=float, device=device)

    def set_cmd(self, idx, cmd):
        c = self.S.cmd.numpy()
        c[idx] = cmd
        self.S.cmd.assign(c)

    def set_mode(self, idx, mode):
        iv = self.P.iv.numpy()
        iv[idx, 0] = mode
        self.P.iv.assign(iv)

    def step(self, t, dt, atm, wind, device):
        """World contract: advance by dt from time t."""
        if not self.n:
            return
        self.tarr.fill_(float(t))
        self.step_rel(0.0, dt, atm, wind, device)

    def step_rel(self, t_off, dt, atm, wind, device):
        """Advance by dt from time tarr[0] + t_off (no host sync: capturable into a CUDA graph)."""
        if not self.n:
            return
        nsub = max(1, int(math.ceil(dt / self.dt_max - 1e-9)))
        h = dt / nsub
        key = (h, id(atm), id(wind), str(device), id(self.tarr))
        if getattr(self, "_key", None) != key:   # record once; afterwards only the wind time offset changes
            P, S, WP, WS, X, tp = self.P, self.S, self.WP, self.WS, self.X, self.tp
            rc = dict(device=device, record_cmd=True)
            self._cmds = dict(
                wind=wp.launch(k_wind, self.n, [0.0, self.tarr, tp] + wind.args() + [S, X], **rc),
                pre=wp.launch(k_pre, self.n, [float(h), P, S, X], **rc),
                wheel=wp.launch(k_wheel, self.nw, [float(h), tp, P, S, WP, WS, X], **rc),
                corner=wp.launch(k_corner, self.n * 8, [float(h), tp, P, S, X], **rc),
                post=wp.launch(k_post, self.n, [float(h), atm, P, S, WS, X], **rc))
            self._key = key
        c = self._cmds
        for k in range(nsub):
            if self._sub % self.wind_every == 0:
                c["wind"].set_param_at_index(0, float(t_off + k * h))
                c["wind"].launch()
            self._sub += 1
            c["pre"].launch()
            c["wheel"].launch()
            c["corner"].launch()
            c["post"].launch()

    # ---------- state log ----------
    def log_rows(self):
        rows = [dict(name=r["name"], kind="vehicle", shape="box", size=r["shape"]) for r in self.rows]
        for j, w in enumerate(self.wheels):
            r = self.rows[self.wheel_owner[j]]
            rows.append(dict(name="%s.w%d" % (r["name"], j), kind="kinematic", shape="wheel",
                             size=[w["radius"], r["wheel_size"][1]]))
        return rows

    def frame(self):
        """-> body_x, body_v, body_q (w,x,y,z) for [chassis..., wheels...]; chassis position = CG (the logged
        visual mesh is in body axes about the CG)."""
        x, q, v = self.S.x.numpy(), self.S.q.numpy(), self.S.v.numpy()
        xc = x
        wc, wq = self.WS.centre.numpy(), self.WS.wq.numpy()
        wv = v[self.wheel_owner]
        return (np.concatenate([xc, wc]).astype(np.float32), np.concatenate([v, wv]).astype(np.float32),
                np.concatenate([q, wq])[:, [3, 0, 1, 2]].astype(np.float32))

    def telemetry(self):
        S = self.S
        e5, st4, o4 = S.e5.numpy(), self.WS.st4.numpy(), self.WS.out4.numpy()
        return dict(x=S.x.numpy(), v=S.v.numpy(), q=S.q.numpy(), wb=S.wb.numpy(), tw=S.tw.numpy(), soc=e5[:, 0],
                    i_bat=e5[:, 1], v_bat=e5[:, 2], e_used=e5[:, 3], odo=e5[:, 4],
                    omega=st4[:, 0], phi=st4[:, 1], steer=st4[:, 2], s=st4[:, 3],
                    fz=o4[:, 0], kappa=o4[:, 1], alpha=o4[:, 2], torque=o4[:, 3])


def _quat_on_ground(n, yaw):
    """Quaternion (x,y,z,w): yaw about world z, then tilt world z onto the ground normal n."""
    qy = np.array([0, 0, math.sin(yaw / 2), math.cos(yaw / 2)])
    n = np.asarray(n, float) / np.linalg.norm(n)
    ax = np.cross([0, 0, 1], n)
    sa = np.linalg.norm(ax)
    if sa < 1e-12:
        return qy.tolist()
    ang = math.atan2(sa, n[2])
    qt = np.r_[ax / sa * math.sin(ang / 2), math.cos(ang / 2)]
    x1, y1, z1, w1 = qt
    x2, y2, z2, w2 = qy
    return [w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2]


def _rotate(q_xyzw, v):
    """Rotate v[N,3] by quaternions q[N,4] (x,y,z,w)."""
    u, w = q_xyzw[:, :3], q_xyzw[:, 3:4]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def yaw_of(q_xyzw):
    x, y, z, w = q_xyzw.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

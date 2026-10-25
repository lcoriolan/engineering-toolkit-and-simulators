#!/usr/bin/env python3
# DESCRIPTION: A World domain (upload(device) / step(t, dt, atm, wind, device)) that flies any number of vessels on
#              core.rigid6 through the SAME clock, air (core.atmos) and wind (core.wind) as every other domain, on a
#              core water domain sea state (waves.SeaState). Per step, 6 GPU launches:
#                _pre      per vessel: clear accumulators, CG wave sample, planing weight w (Froude blend)
#                _columns  per hull column: hydrostatic/Froude-Krylov force under the LOCAL wave surface (any heel
#                          and trim), heave damping, cross-flow drag; wetted keel length/volume metrics
#                _reduce1/_reduce2  fixed-order two-pass segmented sum of the per-column forces, moments and metrics
#                          into each vessel (float64 partials; NO atomics -> bit-reproducible, WATER2 2026-10-15)
#                _vessel   per vessel: Savitsky planing normal force + friction (geometry-derived lambda, trim from
#                          the relative flow and local surface slope), displacement-mode resistance, hull side
#                          force, waterjets (momentum/energy, steering nozzle, intake ventilation), windage from the
#                          core wind + air density, guidance/speed control, telemetry
#                rigid6    integrate
#              Regime blend (MODELED/ASSUMED, documented in proof/WATER_DOMAIN_2026-10-15.md): w = smoothstep of the
#              length Froude number between fn_lo and fn_hi (the legacy modular-pwc band 0.45 -> 1.0). Hydrostatic
#              column force, its pitch/yaw moments and heave damping are scaled by (1-w); Savitsky's total lift
#              (which already contains its buoyant part) is scaled by w. Roll moment from the columns is kept at
#              full value (Savitsky has no roll model).
#              Every coefficient is a parameter in VesselSpec; defaults are ASSUMED and labelled there.
#              GAPS 2026-10-16: optional UPRIGHT-OCCUPANT couple (up_K = sum m g h of persons who keep their CG over
#              their support: torque up_K (z_body x z_world), faded out beyond the support slip angle up_slip_deg;
#              off by default) and a capsize monitor (largest tilt of body +Z from world up, time the hull passed
#              90 deg) in summary().
import math

import numpy as np
import warp as wp

from engineering_sim.core import atmos as A, wind as W, physics_device
from engineering_sim.core.rigid6 import Rigid6, euler_to_quat_wxyz
from . import waves as WV

NTEL = 24
TEL_NAMES = [("sog", "m/s"), ("u_water", "m/s"), ("trim", "deg"), ("heel", "deg"), ("tau_sav", "deg"),
             ("lambda", ""), ("w_plane", ""), ("thrust", "N"), ("P_elec", "kW"), ("energy", "Wh"), ("throttle", ""),
             ("steer", "deg"), ("R_disp", "N"), ("N_sav", "N"), ("Df_sav", "N"), ("V_sub", "m3"), ("wet_keel", "m"),
             ("eta_cg", "m"), ("v_jet", "m/s"), ("dist", "m"), ("heading", "deg"), ("wp_idx", ""), ("z_cg", "m"),
             ("F_air_x", "N")]
TEL = {n: i for i, (n, _) in enumerate(TEL_NAMES)}


@wp.struct
class VParams:
    rho: float
    nu: float
    g: float
    L_wl: float
    S_ref: float
    V_ref: float
    form_k: float
    cr0: float
    cr1: float
    fn0: float
    hump_ratio: float
    fn_lo: float
    fn_hi: float
    b: float
    beta_deg: float
    dCf: float
    x_transom: float
    z_keel: float
    c_heave: float
    cdy: float
    side_lift: float
    side_cp: float
    n_jet: int
    jet_area: float
    eta_pump: float
    eta_motor: float
    P_max: float
    jet_x: float
    jet_y: float
    jet_z: float
    steer_max: float
    thrust_mode: int
    T_max: float
    jet_fail: int
    cda_x: float
    cda_y: float
    z_air: float
    kp_v: float
    ki_v: float
    kp_psi: float
    kd_psi: float
    wp_radius: float
    steer_on: int
    speed_ctrl: int
    thr_fixed: float
    loop_wps: int
    wp_off: int
    wp_n: int
    mass: float
    pitch_damp: float
    steer_v_ref: float
    lean_max: float
    thr_min: float
    up_K: float
    up_slip: float


@wp.func
def _smooth(s: float):
    c = wp.clamp(s, 0.0, 1.0)
    return c * c * (3.0 - 2.0 * c)


@wp.kernel
def _pre(t: float, P: wp.array(dtype=VParams), x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.quat),
         v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3), tau: wp.array(dtype=wp.vec3),
         met: wp.array2d(dtype=float), cgw: wp.array(dtype=wp.spatial_vector), wpl: wp.array(dtype=float),
         wv: WV.WaveParams, kv: wp.array(dtype=wp.vec3), om: wp.array(dtype=float), amp: wp.array(dtype=float),
         ph: wp.array(dtype=float), fe: wp.array(dtype=wp.vec3), te: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    p = P[i]
    f[i] = fe[i]                            # external (contact) wrench for this step; zero unless contact set one
    tau[i] = te[i]
    fe[i] = wp.vec3(0.0, 0.0, 0.0)
    te[i] = wp.vec3(0.0, 0.0, 0.0)
    vprev = met[i, 0]                       # immersion of the previous step
    for k in range(6):
        met[i, k] = 0.0
    xi = x[i]
    qi = q[i]
    s = WV.wave_sample(xi[0], xi[1], 0.0, t, wv, kv, om, amp, ph)
    cgw[i] = s
    vr = wp.quat_rotate_inv(qi, v[i] - wp.vec3(s[3], s[4], s[5]))
    fn = wp.max(vr[0], 0.0) / wp.sqrt(p.g * p.L_wl)
    w_fn = _smooth((fn - p.fn_lo) / (p.fn_hi - p.fn_lo))
    # validity of the planing model (ASSUMED envelope): it hands the hull back to full hydrostatics when it digs in
    # (immersion > 1.0 -> 1.5 x static displacement), trims bow-down relative to the flow/surface (tau 1 -> 0 deg)
    # or heels beyond 15 -> 25 deg; inside the envelope (all calm steady planing) it is 1 and changes nothing
    hx = wp.quat_rotate(qi, wp.vec3(1.0, 0.0, 0.0))
    hy = wp.quat_rotate(qi, wp.vec3(0.0, 1.0, 0.0))
    hn = wp.normalize(wp.vec2(hx[0], hx[1]) + wp.vec2(1.0e-9, 0.0))
    tau_d = (wp.atan2(-vr[2], wp.max(vr[0], 0.1)) - wp.atan(s[1] * hn[0] + s[2] * hn[1])) * 57.2957795
    heel_d = wp.abs(wp.asin(wp.clamp(hy[2], -1.0, 1.0))) * 57.2957795
    r_imm = vprev / p.V_ref
    if vprev <= 0.0:
        r_imm = 0.0
    valid = _smooth((1.5 - r_imm) / 0.5) * _smooth(tau_d / 1.0) * _smooth((25.0 - heel_d) / 10.0)
    wpl[i] = w_fn * valid


NRED = 64          # pass-1 lanes per vessel: lane j sums columns off+j, off+j+NRED, ... in that fixed order
NROW = 12          # per-column row: 0-2 world force, 3-5 body moment, 6-11 metrics 0-5


# metrics columns: 0 V_sub, 1 wet keel length, 2 wet bottom area, 3 lateral area, 4 lateral area * x, 5 intake wet
@wp.kernel
def _columns(t: float, P: wp.array(dtype=VParams), cv: wp.array(dtype=int), cr: wp.array(dtype=wp.vec3),
             civ: wp.array2d(dtype=wp.vec2), cdA: wp.array(dtype=float), cdx: wp.array(dtype=float),
             ckeel: wp.array(dtype=int), cintake: wp.array(dtype=int),
             x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.quat), v: wp.array(dtype=wp.vec3),
             w: wp.array(dtype=wp.vec3), crow: wp.array2d(dtype=float), wpl: wp.array(dtype=float),
             wv: WV.WaveParams, kv: wp.array(dtype=wp.vec3), om: wp.array(dtype=float), amp: wp.array(dtype=float),
             ph: wp.array(dtype=float)):
    # per column (crow is [NROW, ncol], coalesced): 0-2 world force, 3-5 body moment (Mx full, My/Mz scaled),
    # 6-11 metrics 0-5
    c = wp.tid()
    vi = cv[c]
    p = P[vi]
    xi = x[vi]
    qi = q[vi]
    vb = wp.quat_rotate_inv(qi, v[vi])
    wi = w[vi]
    sc = 1.0 - wpl[vi]
    r0 = cr[c]
    dA = cdA[c]
    dx = cdx[c]
    Fs = wp.vec3(0.0, 0.0, 0.0)
    Mx = float(0.0)
    My = float(0.0)
    Mz = float(0.0)
    for r in range(6, NROW):            # metrics accumulate in this column's own row (one writer: no atomics)
        crow[r, c] = 0.0
    for k in range(civ.shape[1]):
        iv = civ[c, k]
        ln = iv[1] - iv[0]
        if ln > 0.0:
            pl = xi + wp.quat_rotate(qi, wp.vec3(r0[0], r0[1], iv[0]))
            pu = xi + wp.quat_rotate(qi, wp.vec3(r0[0], r0[1], iv[1]))
            lo = pl
            hi = pu
            if pu[2] < pl[2]:
                lo = pu
                hi = pl
            mid = 0.5 * (lo + hi)
            ws = WV.wave_sample(mid[0], mid[1], mid[2], t, wv, kv, om, amp, ph)
            eta = ws[0]
            dz = hi[2] - lo[2]
            frac = float(0.0)
            if dz > 1.0e-6:
                frac = wp.clamp((eta - lo[2]) / dz, 0.0, 1.0)
            elif eta > lo[2]:
                frac = 1.0
            if frac > 0.0:
                vol = dA * ln * frac
                cen = lo + (hi - lo) * (0.5 * frac)
                rb = wp.quat_rotate_inv(qi, cen - xi)
                Fb = wp.quat_rotate_inv(qi, wp.vec3(0.0, 0.0, p.rho * p.g * vol))
                vpb = vb + wp.cross(wi, rb)                                   # point velocity, body
                vr = vpb - wp.quat_rotate_inv(qi, wp.vec3(ws[3], ws[4], ws[5]))
                lat = vol / p.b      # this column's share of the projected LATERAL area (box: L T)
                Fd = wp.vec3(0.0, -0.5 * p.rho * p.cdy * lat * vr[1] * wp.abs(vr[1]), 0.0)
                if k == 0:
                    # heave/pitch/roll damping on the WORLD-vertical relative velocity of the wetted bottom (no
                    # spurious lift from the forward-speed component when trimmed; acts at every speed)
                    vz = wp.quat_rotate(qi, vpb)[2] - ws[5]
                    Fd += wp.quat_rotate_inv(qi, wp.vec3(0.0, 0.0, -p.c_heave * dA * vz))
                    crow[8, c] = crow[8, c] + dA
                    if ckeel[c] == 1:
                        # fraction of this station's dx under water along the inclined keel (continuous in trim)
                        kz = wp.max(wp.abs(wp.quat_rotate(qi, wp.vec3(1.0, 0.0, 0.0))[2]), 1.0e-3)
                        crow[7, c] = crow[7, c] + dx * wp.clamp(0.5 + (eta - lo[2]) / (dx * kz), 0.0, 1.0)
                    if cintake[c] == 1:
                        crow[11, c] = crow[11, c] + 1.0
                Ffull = Fb + Fd
                Fsc = Fb * sc + Fd
                Mf = wp.cross(rb, Ffull)
                Ms = wp.cross(rb, Fsc)
                Fs += Fsc
                Mx += Mf[0]
                My += Ms[1]
                Mz += Ms[2]
                crow[6, c] = crow[6, c] + vol
                crow[9, c] = crow[9, c] + lat
                crow[10, c] = crow[10, c] + lat * rb[0]
    Fw = wp.vec3(0.0, 0.0, 0.0)
    if wp.length(Fs) > 0.0:
        Fw = wp.quat_rotate(qi, Fs)
    crow[0, c] = Fw[0]
    crow[1, c] = Fw[1]
    crow[2, c] = Fw[2]
    crow[3, c] = Mx
    crow[4, c] = My
    crow[5, c] = Mz


@wp.kernel
def _reduce1(coff: wp.array(dtype=int), ccnt: wp.array(dtype=int), crow: wp.array2d(dtype=float),
             part: wp.array2d(dtype=wp.float64)):
    i, j, k = wp.tid()
    n = ccnt[i]
    o = coff[i]
    acc = wp.float64(0.0)
    c = j
    while c < n:
        acc += wp.float64(crow[k, o + c])
        c += NRED
    part[i * NRED + j, k] = acc


@wp.func
def _lanesum(part: wp.array2d(dtype=wp.float64), i: int, k: int):
    acc = wp.float64(0.0)
    for j in range(NRED):
        acc += part[i * NRED + j, k]
    return float(acc)


@wp.kernel
def _reduce2(part: wp.array2d(dtype=wp.float64), f: wp.array(dtype=wp.vec3), tau: wp.array(dtype=wp.vec3),
             met: wp.array2d(dtype=float)):
    i = wp.tid()
    f[i] = f[i] + wp.vec3(_lanesum(part, i, 0), _lanesum(part, i, 1), _lanesum(part, i, 2))
    tau[i] = tau[i] + wp.vec3(_lanesum(part, i, 3), _lanesum(part, i, 4), _lanesum(part, i, 5))
    for k in range(6):
        met[i, k] = _lanesum(part, i, 6 + k)


@wp.func
def _wrap(a: float):
    r = a
    if r > 3.14159265:
        r -= 6.28318531
    if r < -3.14159265:
        r += 6.28318531
    return r


@wp.kernel
def _vessel(t: float, dt: float, P: wp.array(dtype=VParams), atm: A.AtmosParams, wnd: W.WindParams,
            mk: wp.array(dtype=wp.vec3), ma: wp.array(dtype=float), mph: wp.array(dtype=float),
            ms: wp.array(dtype=wp.vec3),
            x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.quat), v: wp.array(dtype=wp.vec3),
            w: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3), tau: wp.array(dtype=wp.vec3),
            s_add: wp.array(dtype=float), met: wp.array2d(dtype=float), cgw: wp.array(dtype=wp.spatial_vector),
            wpl: wp.array(dtype=float), wps: wp.array(dtype=wp.vec3), ctl: wp.array2d(dtype=wp.float64),
            tel: wp.array2d(dtype=float), wv: WV.WaveParams, kv: wp.array(dtype=wp.vec3), om: wp.array(dtype=float),
            amp: wp.array(dtype=float), ph: wp.array(dtype=float), tmax: wp.array(dtype=float),
            tcap: wp.array(dtype=float)):
    i = wp.tid()
    p = P[i]
    xi = x[i]
    qi = q[i]
    vi = v[i]
    wi = w[i]
    ws = cgw[i]
    wl = wpl[i]
    uw = wp.vec3(ws[3], ws[4], ws[5])
    vb = wp.quat_rotate_inv(qi, vi - uw)
    u = vb[0]
    au = wp.abs(u)
    hx = wp.quat_rotate(qi, wp.vec3(1.0, 0.0, 0.0))
    hy = wp.quat_rotate(qi, wp.vec3(0.0, 1.0, 0.0))
    psi = wp.atan2(hx[1], hx[0])
    sog = wp.sqrt(vi[0] * vi[0] + vi[1] * vi[1])
    vsub = met[i, 0]
    wet_keel = met[i, 1]

    # ---- guidance + speed control (state in ctl: 0 integ, 1 wp_idx, 2 energy J, 3 dist m) -------------------
    wpi = int(ctl[i, 1])
    tgt = wps[p.wp_off + wpi]
    dxy = wp.vec2(tgt[0] - xi[0], tgt[1] - xi[1])
    if wp.length(dxy) < p.wp_radius:
        if wpi < p.wp_n - 1:
            wpi += 1
        elif p.loop_wps == 1:
            wpi = 0
        tgt = wps[p.wp_off + wpi]
        dxy = wp.vec2(tgt[0] - xi[0], tgt[1] - xi[1])
    ctl[i, 1] = wp.float64(wpi)
    delta = float(0.0)
    if p.steer_on == 1:
        err = _wrap(wp.atan2(dxy[1], dxy[0]) - psi)
        # steering authority reduced with speed (ASSUMED operator behaviour): full lock up to steer_v_ref,
        # then ~ 1/V, never below 30 %
        smax = p.steer_max * wp.clamp(p.steer_v_ref / wp.max(sog, 0.1), 0.3, 1.0)
        delta = wp.clamp(p.kp_psi * err - p.kd_psi * wi[2], -smax, smax)
    thr = p.thr_fixed
    if p.speed_ctrl == 1:
        e = tgt[2] - sog
        integ = float(ctl[i, 0])
        thr_raw = p.kp_v * e + p.ki_v * integ
        if (thr_raw < 1.0 or e < 0.0) and (thr_raw > 0.0 or e > 0.0):
            integ += e * dt                                          # conditional integration (anti-windup)
        ctl[i, 0] = wp.float64(integ)
        thr = wp.clamp(p.kp_v * e + p.ki_v * integ, p.thr_min, 1.0)

    # ---- displacement-mode resistance (legacy modular-pwc form: ITTC-57 x (1+k) + residuary, hump cap) ------
    S = p.S_ref * wp.pow(wp.max(vsub / p.V_ref, 0.0), 0.6666667)
    fn = au / wp.sqrt(p.g * p.L_wl)
    Re = wp.max(au * p.L_wl / p.nu, 1.0e5)
    lg = wp.log10(Re) - 2.0
    cf = 0.075 / (lg * lg)
    cr = p.cr0 + p.cr1 * wp.max(fn - p.fn0, 0.0) * wp.max(fn - p.fn0, 0.0)
    Rd = 0.5 * p.rho * u * u * S * (cf * (1.0 + p.form_k) + cr)
    if fn > p.fn_lo:
        Rd = wp.min(Rd, p.hump_ratio * p.mass * p.g)
    Rd_s = Rd * wp.sign(u) * (1.0 - wl)
    f[i] = f[i] + wp.quat_rotate(qi, wp.vec3(-Rd_s, 0.0, 0.0))

    # ---- Savitsky planing (normal force + friction), lambda from the wetted keel incl. spray pile-up ----------
    N = float(0.0)
    Df = float(0.0)
    lam = float(0.0)
    tau_deg = float(0.0)
    if wl > 0.0 and u > 0.5:
        hn = wp.normalize(wp.vec2(hx[0], hx[1]) + wp.vec2(1.0e-9, 0.0))
        slope = ws[1] * hn[0] + ws[2] * hn[1]
        br = p.beta_deg / 57.2957795
        Cv = u / wp.sqrt(p.g * p.b)
        # wetted keel length: columns (exact for any hull/wave) or, if larger, the prismatic extrapolation
        # depth-at-transom / sin(keel angle to the local surface) — keeps lift growing if immersion exceeds the hull
        ptr = xi + wp.quat_rotate(qi, wp.vec3(p.x_transom, 0.0, p.z_keel))
        wtr = WV.wave_sample(ptr[0], ptr[1], 0.0, t, wv, kv, om, amp, ph)
        d_tr = wtr[0] - ptr[2]
        tg = wp.asin(wp.clamp(hx[2], -1.0, 1.0)) - wp.atan(slope)
        if d_tr > 0.0 and tg > 0.005:
            wet_keel = wp.max(wet_keel, wp.min(d_tr / wp.sin(tg), 12.0 * p.b))
        # pass 1: trim from the CG relative flow -> lambda -> centre of pressure; pass 2: trim from the flow AT the
        # centre of pressure (adds pitch-rate damping), relative to the local wave slope along the heading
        tr0 = wp.clamp(wp.atan2(-vb[2], u) - wp.atan(slope), 0.005, 0.26)
        lam0 = wp.clamp(wet_keel / p.b - wp.tan(br) / (6.28318531 * wp.tan(tr0)), 0.05, 12.0)
        lp0 = lam0 * p.b * (0.75 - 1.0 / (5.21 * Cv * Cv / (lam0 * lam0) + 2.39))
        vcp = vb + wp.cross(wi, wp.vec3(p.x_transom + lp0, 0.0, p.z_keel))
        tau_r = wp.atan2(-vcp[2], vcp[0]) - wp.atan(slope)
        tau_deg = wp.clamp(tau_r * 57.2957795, 0.0, 15.0)
        if tau_deg > 0.3:
            tr = tau_deg / 57.2957795
            lam = wp.min(wet_keel / p.b - wp.tan(br) / (6.28318531 * wp.tan(tr)), 12.0)
            if lam > 0.0:
                CL0 = wp.pow(tau_deg, 1.1) * (0.012 * wp.sqrt(lam) + 0.0055 * wp.pow(lam, 2.5) / (Cv * Cv))
                CLb = wp.max(CL0 - 0.0065 * p.beta_deg * wp.pow(CL0, 0.6), 0.0)
                N = 0.5 * p.rho * u * u * p.b * p.b * CLb / wp.cos(tr)
                lp = lam * p.b * (0.75 - 1.0 / (5.21 * Cv * Cv / (lam * lam) + 2.39))
                Sw = lam * p.b * p.b / wp.cos(br)
                Re2 = wp.max(u * lam * p.b / p.nu, 1.0e5)
                lg2 = wp.log10(Re2) - 2.0
                Df = 0.5 * p.rho * u * u * Sw * (0.075 / (lg2 * lg2) + p.dCf)
                xcp = p.x_transom + lp
                rN = wp.vec3(xcp, 0.0, p.z_keel)
                rF = wp.vec3(xcp, 0.0, p.z_keel + 0.25 * p.b * wp.tan(br))
                FN = wp.vec3(0.0, 0.0, wl * N)
                FF = wp.vec3(-wl * Df, 0.0, 0.0)
                # quasi-steady pitch damping of the distributed planing lift (ASSUMED): incidence at a station x is
                # changed by q (x - x_cg)/u; Savitsky pressure ~ s^-1/2 from the spray root over the wetted length
                # gives second moment k2 = 4 (lam b)^2/45 + (x_cp)^2 about the CG; dN/dtau ~ 1.1 N/tau
                k2 = 4.0 * (lam * p.b) * (lam * p.b) / 45.0 + xcp * xcp
                Mq = -p.pitch_damp * wl * 1.1 * N / tr * k2 / u * wi[1]
                # same quasi-steady argument across the beam for roll rate (uniform lift over b: k2 = b^2/12)
                Mr = -p.pitch_damp * wl * 1.1 * N / tr * (p.b * p.b / 12.0) / u * wi[0]
                f[i] = f[i] + wp.quat_rotate(qi, FN + FF)
                tau[i] = tau[i] + wp.cross(rN, FN) + wp.cross(rF, FF) + wp.vec3(Mr, Mq, 0.0)

    # ---- hull side force (low-aspect lifting surface, Jones R.T. 1946: C_Y = pi/2 AR beta), ASSUMED -----------
    la = met[i, 3]
    if la > 1.0e-6 and au > 0.5:
        AR = 2.0 * la / (p.L_wl * p.L_wl)
        Y = -0.5 * p.rho * la * p.side_lift * 1.5707963 * AR * au * vb[1]
        xl = met[i, 4] / la + p.side_cp * p.L_wl
        rY = wp.vec3(xl, 0.0, 0.5 * p.z_keel)
        FY = wp.vec3(0.0, Y, 0.0)
        f[i] = f[i] + wp.quat_rotate(qi, FY)
        tau[i] = tau[i] + wp.cross(rY, FY)

    # ---- waterjets: ideal momentum/energy (legacy jet() form), nozzle steering, intake ventilation -----------
    Ttot = float(0.0)
    vj = float(0.0)
    Pel = float(0.0)
    if p.thrust_mode == 0:
        nwork = p.n_jet - p.jet_fail
        Vin = wp.max(u, 0.0)
        Ph = p.eta_pump * p.eta_motor * thr * p.P_max / float(p.n_jet)
        lo = Vin
        hi = Vin + 100.0
        for it in range(48):
            mid = 0.5 * (lo + hi)
            if 0.5 * p.rho * p.jet_area * mid * (mid * mid - Vin * Vin) < Ph:
                lo = mid
            else:
                hi = mid
        vj = 0.5 * (lo + hi)
        Tj = p.rho * p.jet_area * vj * (vj - Vin)
        if met[i, 5] < 0.5:
            Tj = 0.0                                                     # intake out of the water: ventilated
        d = wp.vec3(wp.cos(delta), -wp.sin(delta), 0.0)
        for j in range(p.n_jet):
            ys = p.jet_y
            if j % 2 == 1:
                ys = -p.jet_y
            on = 1.0
            if j >= nwork:
                on = 0.0
            Fj = d * (Tj * on)
            f[i] = f[i] + wp.quat_rotate(qi, Fj)
            tau[i] = tau[i] + wp.cross(wp.vec3(p.jet_x, ys, p.jet_z), Fj)
        Ttot = Tj * float(nwork)
        Pel = thr * p.P_max * float(nwork) / float(p.n_jet)
    else:
        Ttot = thr * p.T_max
        Fd = wp.quat_rotate(qi, wp.vec3(wp.cos(delta), -wp.sin(delta), 0.0))
        Fd = wp.normalize(wp.vec3(Fd[0], Fd[1], 0.0)) * Ttot                  # horizontal through CG (test rig)
        f[i] = f[i] + Fd
        # steering of the test rig: the yaw moment a deflected nozzle at the transom would make (no pitch moment)
        tau[i] = tau[i] + wp.vec3(0.0, 0.0, Ttot * wp.sin(delta) * wp.abs(p.x_transom))

    # ---- rider weight shift (ASSUMED, PWC): bounded roll moment opposing heel and roll rate -------------------
    if p.lean_max > 0.0:
        hr = wp.asin(wp.clamp(hy[2], -1.0, 1.0))          # > 0 = port side up
        Ml = wp.clamp(-p.lean_max * (hr / 0.1) - p.lean_max * (wi[0] / 0.5), -p.lean_max, p.lean_max)
        # a weight shift is a gravity couple: its moment is about the HORIZONTAL heading axis (WATER2 fix: about
        # body +X, a trimmed hull got a world-vertical component that nothing resists in still water -> yaw spin)
        ax = wp.normalize(wp.vec3(hx[0], hx[1], 0.0) + wp.vec3(1.0e-9, 0.0, 0.0))
        tau[i] = tau[i] + wp.quat_rotate_inv(qi, ax * Ml)

    # ---- upright occupants (GAPS, physical): persons keep their CG h above their support along WORLD up, a gravity
    # couple sum(m g h) (z_body x z_world) relative to the rigid-mass CG; faded out past the support slip angle
    zb = wp.quat_rotate(qi, wp.vec3(0.0, 0.0, 1.0))
    tilt = wp.acos(wp.clamp(zb[2], -1.0, 1.0))
    if p.up_K > 0.0:
        fade = _smooth((p.up_slip - tilt) / 0.0349066)
        tau[i] = tau[i] + wp.quat_rotate_inv(qi, wp.cross(zb, wp.vec3(0.0, 0.0, 1.0)) * (p.up_K * fade))
    # ---- capsize monitor (recording only)
    tilt_d = tilt * 57.2957795
    if tilt_d > tmax[i]:
        tmax[i] = tilt_d
    if zb[2] < 0.0 and tcap[i] < 0.0:
        tcap[i] = t

    # ---- windage from the core wind + air density ------------------------------------------------------------
    pa = xi + wp.vec3(0.0, 0.0, p.z_air)
    wa = W.wind_at(wp.vec3(pa[0], pa[1], wp.max(pa[2], 0.5)), t, wnd, mk, ma, mph, ms)
    rho_a = A.air_density(wp.max(pa[2], 0.0), atm)
    ra = wp.quat_rotate_inv(qi, vi - wa)
    Fa = wp.vec3(-0.5 * rho_a * p.cda_x * ra[0] * wp.abs(ra[0]), -0.5 * rho_a * p.cda_y * ra[1] * wp.abs(ra[1]), 0.0)
    f[i] = f[i] + wp.quat_rotate(qi, Fa)
    tau[i] = tau[i] + wp.cross(wp.vec3(0.0, 0.0, p.z_air), Fa)

    # added-mass scale follows immersion through a 0.5 s low-pass (an instantaneous immersion-dependent mass
    # pumps energy into heave/pitch oscillations)
    s_add[i] = s_add[i] + (wp.clamp(vsub / p.V_ref, 0.0, 1.0) - s_add[i]) * wp.min(dt / 0.5, 1.0)
    ctl[i, 2] = ctl[i, 2] + wp.float64(Pel * dt)
    ctl[i, 3] = ctl[i, 3] + wp.float64(sog * dt)
    tel[i, 0] = sog
    tel[i, 1] = u
    tel[i, 2] = wp.asin(wp.clamp(hx[2], -1.0, 1.0)) * 57.2957795
    tel[i, 3] = -wp.asin(wp.clamp(hy[2], -1.0, 1.0)) * 57.2957795
    tel[i, 4] = tau_deg
    tel[i, 5] = lam
    tel[i, 6] = wl
    tel[i, 7] = Ttot
    tel[i, 8] = Pel / 1000.0
    tel[i, 9] = float(ctl[i, 2] / wp.float64(3600.0))
    tel[i, 10] = thr
    tel[i, 11] = delta * 57.2957795
    tel[i, 12] = Rd_s
    tel[i, 13] = N * wl
    tel[i, 14] = Df * wl
    tel[i, 15] = vsub
    tel[i, 16] = wet_keel
    tel[i, 17] = ws[0]
    tel[i, 18] = vj
    tel[i, 19] = float(ctl[i, 3])
    tel[i, 20] = psi * 57.2957795
    tel[i, 21] = float(wpi)
    tel[i, 22] = xi[2]
    tel[i, 23] = Fa[0]


def _quat_R(q_wxyz):
    w, x, y, z = (float(c) for c in q_wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ------------------------------------------------------------------------------------------------------------------
DEFAULTS = dict(
    # water (MODELED via props unless overridden)
    rho=1025.0, nu=1.19e-6,
    # displacement resistance — legacy modular-pwc/pwc/performance.py constants (ASSUMED there, retained for parity)
    form_k=0.30, cr0=0.0015, cr1=0.02 / 0.06, fn0=0.15, hump_ratio=0.22, fn_lo=0.45, fn_hi=1.0,
    # Savitsky (1964) inputs: effective planing beam b and deadrise beta are geometry; dCf roughness allowance
    dCf=0.0004,
    # damping / cross-flow / side force (ASSUMED)
    zeta_heave=0.15, cdy=1.0, side_lift=1.0, side_cp=0.25,
    # added mass ratios (ASSUMED): surge, sway, heave of m; roll, pitch, yaw of I
    added_mass_ratio=(0.05, 0.8, 0.8), added_inertia_ratio=(0.2, 0.8, 0.5),
    # jets (legacy jet(): 2 units, exit dia 0.8 x bore; pump 0.70, motor 0.90 ASSUMED there)
    n_jet=2, jet_area=0.0, eta_pump=0.70, eta_motor=0.90, P_max=80e3, jet_x=0.0, jet_y=0.6449, jet_z=0.15,
    steer_max_deg=25.0, thrust_mode=0, T_max=0.0, jet_fail=0,
    # windage (ASSUMED: rider + craft frontal 1.2 m^2 x Cd 0.9; lateral 3.0 m^2 x Cd 1.0; centre 0.6 m above CG)
    cda_x=1.08, cda_y=3.0, z_air=0.6,
    # control (ASSUMED)
    kp_v=0.25, ki_v=0.04, kp_psi=1.2, kd_psi=0.6, wp_radius=25.0, steer_on=1, speed_ctrl=1, thr_fixed=0.0,
    loop_wps=0, pitch_damp=1.0, steer_v_ref=1.0e9, lean_max=0.0, thr_min=0.0,
    # upright-occupant couple (GAPS; 0 = off): sum m g h [N m], support slip angle [deg] (ASSUMED atan(mu))
    up_K=0.0, up_slip_deg=90.0)


class Vessels:
    """World domain: batch of surface vessels on one sea state. All arguments are parameters."""

    def __init__(self, sea):
        self.sea, self.specs, self.rb = sea, [], Rigid6()
        self.channels = []            # state-log series: (vessel index, TEL channel, unit)

    def add(self, name, cols, mass, cg, inertia, b, beta_deg, L_wl, S_ref, V_ref, pos_xy=(0.0, 0.0), z_cg=None,
            yaw_deg=0.0, trim_deg=0.0, heel_deg=0.0, vel=(0.0, 0.0, 0.0), waypoints=((0.0, 0.0, 0.0),), intake_x=None,
            mesh=None, color=None, **kw):
        """cols: hull.columns_from_mesh() in the hull frame; cg: CG in the hull frame [m]; waypoints (x, y, speed).
        mesh: optional (V, F) visual hull in the hull frame (logged true-scale about the CG); color: optional RGBA."""
        wat = getattr(self.sea, "water", None)
        if wat is not None:                  # the sea's water body, unless this vessel overrides rho / nu
            kw.setdefault("rho", wat["rho"])
            kw.setdefault("nu", wat["nu"])
        p = dict(DEFAULTS, **kw)
        cg = np.asarray(cg, float)
        r = np.c_[cols["x"] - cg[0], cols["y"] - cg[1], np.zeros(len(cols["x"]))]
        iv = cols["intervals"] - cg[2]
        iv[cols["intervals"][:, :, 1] - cols["intervals"][:, :, 0] <= 0] = 0.0
        x_tr = cols["bounds"][0][0] - cg[0]
        z_keel = cols["bounds"][0][2] - cg[2]
        ix = intake_x if intake_x is not None else cols["bounds"][0][0] + 0.2 * L_wl
        intake = ((np.abs(cols["x"] - ix) <= max(cols["dx"], 0.1)) & (cols["keel"] == 1)).astype(np.int32)
        if not intake.any():
            intake = cols["keel"].copy()
        Awp = len(cols["x"]) * cols["dA"] * 0.85
        a33 = p["added_mass_ratio"][2] * mass
        c_heave = p.get("c_heave") or 2 * p["zeta_heave"] * math.sqrt(p["rho"] * self.rb.g * Awp * (mass + a33)) / Awp
        if z_cg is None:
            z_cg = cg[2] - (cols["bounds"][0][2] + 0.5 * (cols["bounds"][1][2] - cols["bounds"][0][2]) * 0.6)
        # heel > 0 = port side down (roll about +X is port-up); trim > 0 = bow up (pitch about +Y is bow-down)
        quat = euler_to_quat_wxyz(-math.radians(heel_deg), -math.radians(trim_deg), math.radians(yaw_deg))
        bi = self.rb.add(name, mass, inertia, pos=(pos_xy[0], pos_xy[1], z_cg), quat_wxyz=quat,
                         vel=vel, added_mass=[r_ * mass for r_ in p["added_mass_ratio"]],
                         added_inertia=[r_ * I_ for r_, I_ in zip(p["added_inertia_ratio"], inertia)])
        # collision proxy: hull-column bounding box about the CG, body frame (contact hook)
        wet = (iv[:, :, 1] - iv[:, :, 0]) > 0
        hx_, hy_ = 0.5 * cols["dx"], 0.5 * cols["dA"] / cols["dx"]
        lo_b = np.array([r[:, 0].min() - hx_, r[:, 1].min() - hy_, iv[:, :, 0][wet].min()])
        hi_b = np.array([r[:, 0].max() + hx_, r[:, 1].max() + hy_, iv[:, :, 1][wet].max()])
        self.specs.append(dict(name=name, bi=bi, r=r, iv=iv, obb=(0.5 * (lo_b + hi_b), 0.5 * (hi_b - lo_b)), dA=cols["dA"], dx=cols["dx"], keel=cols["keel"],
                               intake=intake, p=p, mass=mass, b=b, beta_deg=beta_deg, L_wl=L_wl, S_ref=S_ref,
                               V_ref=V_ref, x_tr=x_tr, z_keel=z_keel, c_heave=c_heave, wps=np.asarray(waypoints, float),
                               mesh=None if mesh is None else (np.asarray(mesh[0], float) - cg, np.asarray(mesh[1])),
                               color=color))
        return bi

    def upload(self, device=None):
        dev = self.device = physics_device(device)
        self.rb.upload(dev)
        nv = self.nv = len(self.specs)
        R, IV, DA, DX, KE, IN, CV = [], [], [], [], [], [], []
        prm, wps = [], []
        for vi, s in enumerate(self.specs):
            n = len(s["r"])
            R.append(s["r"]); IV.append(s["iv"]); DA.append(np.full(n, s["dA"])); DX.append(np.full(n, s["dx"]))
            KE.append(s["keel"]); IN.append(s["intake"]); CV.append(np.full(n, vi, np.int32))
            p = s["p"]
            vp = VParams()
            vp.rho, vp.nu, vp.g = p["rho"], p["nu"], self.rb.g
            vp.L_wl, vp.S_ref, vp.V_ref = s["L_wl"], s["S_ref"], s["V_ref"]
            vp.form_k, vp.cr0, vp.cr1, vp.fn0 = p["form_k"], p["cr0"], p["cr1"], p["fn0"]
            vp.hump_ratio, vp.fn_lo, vp.fn_hi = p["hump_ratio"], p["fn_lo"], p["fn_hi"]
            vp.b, vp.beta_deg, vp.dCf = s["b"], s["beta_deg"], p["dCf"]
            vp.x_transom, vp.z_keel, vp.c_heave = s["x_tr"], s["z_keel"], s["c_heave"]
            vp.cdy, vp.side_lift, vp.side_cp = p["cdy"], p["side_lift"], p["side_cp"]
            vp.n_jet, vp.jet_area, vp.eta_pump, vp.eta_motor = int(p["n_jet"]), p["jet_area"], p["eta_pump"], p["eta_motor"]
            vp.P_max, vp.jet_x, vp.jet_y, vp.jet_z = p["P_max"], p["jet_x"], p["jet_y"], p["jet_z"]
            vp.steer_max, vp.thrust_mode, vp.T_max = math.radians(p["steer_max_deg"]), int(p["thrust_mode"]), p["T_max"]
            vp.jet_fail = int(p["jet_fail"])
            vp.cda_x, vp.cda_y, vp.z_air = p["cda_x"], p["cda_y"], p["z_air"]
            vp.kp_v, vp.ki_v, vp.kp_psi, vp.kd_psi = p["kp_v"], p["ki_v"], p["kp_psi"], p["kd_psi"]
            vp.wp_radius, vp.steer_on, vp.speed_ctrl = p["wp_radius"], int(p["steer_on"]), int(p["speed_ctrl"])
            vp.thr_fixed, vp.loop_wps = p["thr_fixed"], int(p["loop_wps"])
            vp.wp_off, vp.wp_n, vp.mass = len(wps), len(s["wps"]), s["mass"]
            vp.pitch_damp, vp.steer_v_ref, vp.lean_max = p["pitch_damp"], p["steer_v_ref"], p["lean_max"]
            vp.thr_min = p["thr_min"]
            vp.up_K, vp.up_slip = float(p["up_K"]), math.radians(p["up_slip_deg"])
            wps += [tuple(w) for w in s["wps"]]
            prm.append(vp)
        self._prm = prm
        self.P = wp.array(prm, dtype=VParams, device=dev)
        self.ncol = sum(len(r) for r in R)
        self.cv = wp.array(np.concatenate(CV), dtype=int, device=dev)
        self.cr = wp.array(np.concatenate(R).astype(np.float32), dtype=wp.vec3, device=dev)
        self.civ = wp.array(np.concatenate(IV).astype(np.float32), dtype=wp.vec2, device=dev)
        self.cdA = wp.array(np.concatenate(DA).astype(np.float32), dtype=float, device=dev)
        self.cdx = wp.array(np.concatenate(DX).astype(np.float32), dtype=float, device=dev)
        self.ckeel = wp.array(np.concatenate(KE).astype(np.int32), dtype=int, device=dev)
        self.cintake = wp.array(np.concatenate(IN).astype(np.int32), dtype=int, device=dev)
        self.wps = wp.array(np.array(wps, np.float32).reshape(-1, 3), dtype=wp.vec3, device=dev)
        cnt = np.array([len(r) for r in R], np.int32)
        self.coff = wp.array(np.r_[0, np.cumsum(cnt)[:-1]].astype(np.int32), dtype=int, device=dev)
        self.ccnt = wp.array(cnt, dtype=int, device=dev)
        self.crow = wp.zeros((NROW, self.ncol), dtype=float, device=dev)
        self.part = wp.zeros((nv * NRED, NROW), dtype=wp.float64, device=dev)
        self.fe = wp.zeros(nv, dtype=wp.vec3, device=dev)        # external (contact) wrench, consumed next step
        self.te = wp.zeros(nv, dtype=wp.vec3, device=dev)
        self.met = wp.zeros((nv, 6), dtype=float, device=dev)
        self.cgw = wp.zeros(nv, dtype=wp.spatial_vector, device=dev)
        self.wpl = wp.zeros(nv, dtype=float, device=dev)
        self.ctl = wp.zeros((nv, 4), dtype=wp.float64, device=dev)
        self.tel = wp.zeros((nv, NTEL), dtype=float, device=dev)
        self.tmax = wp.zeros(nv, dtype=float, device=dev)              # capsize monitor: largest tilt [deg]
        self.tcap = wp.full(nv, -1.0, dtype=float, device=dev)        # time body +Z first pointed below horizontal
        self.names = [s["name"] for s in self.specs]

    def step(self, t, dt, atm, wind, device=None):
        rb, sea, dev = self.rb, self.sea, self.device
        wa = sea.args()
        wp.launch(_pre, self.nv, [float(t), self.P, rb.x, rb.q, rb.v, rb.f, rb.tau, self.met, self.cgw, self.wpl] + wa +
                  [self.fe, self.te], device=dev)
        wp.launch(_columns, self.ncol, [float(t), self.P, self.cv, self.cr, self.civ, self.cdA, self.cdx, self.ckeel,
                                        self.cintake, rb.x, rb.q, rb.v, rb.w, self.crow, self.wpl] + wa,
                  device=dev)
        wp.launch(_reduce1, (self.nv, NRED, NROW), [self.coff, self.ccnt, self.crow, self.part], device=dev)
        wp.launch(_reduce2, self.nv, [self.part, rb.f, rb.tau, self.met], device=dev)
        wp.launch(_vessel, self.nv, [float(t), float(dt), self.P, atm] + wind.args() +
                  [rb.x, rb.q, rb.v, rb.w, rb.f, rb.tau, rb.s, self.met, self.cgw, self.wpl, self.wps, self.ctl,
                   self.tel] + wa + [self.tmax, self.tcap], device=dev)
        rb.integrate(dt)

    def control_state(self):
        """float64 [nv, 4]: speed-loop integral, waypoint index, electrical energy [J], distance [m]."""
        return self.ctl.numpy().copy()

    def snapshot(self):
        x, v, q, _ = self.rb.state()
        return dict(body_x=x, body_v=v, body_q=q, tel=self.tel.numpy().copy())

    def set(self, vi, **kw):
        """Change control parameters of vessel vi between steps (e.g. throttle schedule, target speed)."""
        for k, val in kw.items():
            setattr(self._prm[vi], k, val)
        self.P = wp.array(self._prm, dtype=VParams, device=self.device)

    # ---------- contact hook (cross-domain contact; core/world.py CONTACT protocol) ----------
    def contact_bind(self, cm):
        """core/contact.py protocol: one box per hull = the hull-column envelope (sealed hydrostatic hull, body frame
        about the CG) on the vessel's Rigid6 state. Collides with other bodies and the terrain (shore, lake bed ->
        grounding). NOT water-flagged: buoyancy/hydrodynamics stay in this domain. Per-vessel options from the
        scenario (self._contact_cfg[name] = dict(enabled, material, terrain)); default material 'hull'.
        Impact mass = rigid mass and inertia (ASSUMED: added mass not included in the impulse; it is in
        apply_contact() and in the integrator)."""
        cfg = getattr(self, "_contact_cfg", None) or {}
        prox, keep = [], []
        for k, sp in enumerate(self.specs):
            c = cfg.get(sp["name"]) or {}
            if not c.get("enabled", True):
                continue
            cen, half = sp["obb"]
            prox.append(dict(body=len(keep), shape="box", size=[float(h) for h in half],
                             offset=[float(o) for o in cen], material=c.get("material", "hull"),
                             terrain=c.get("terrain", True)))
            keep.append(k)
        if keep:
            rows = self.rb.rows
            bis = [self.specs[k]["bi"] for k in keep]
            cm.bind_rigid("vessels", self.rb.x, self.rb.q, self.rb.v, self.rb.w, bis, [rows[b]["m"] for b in bis],
                          [rows[b]["I"] for b in bis], prox, names=[self.specs[k]["name"] for k in keep], water=False)

    def collision_proxies(self):
        """One oriented box per vessel from its hull columns (sealed hydrostatic envelope), for broad/narrow phase.
        Returns a list of dicts: name, body (vessel index), kind 'obb', center_b / half [m] (body frame about the CG),
        x [m], q_wxyz, v [m/s] (world), w_b [rad/s] (body), mass [kg], inertia (diag, body) [kg m^2],
        m_eff / I_eff (rigid + added mass at the current immersion — what an impulse acts against)."""
        x, v, q, w = self.rb.state()
        s_ = self.rb.s.numpy()
        out = []
        for vi, sp in enumerate(self.specs):
            bi, r = sp["bi"], self.rb.rows[sp["bi"]]
            ma, Ia = np.array(r["ma"]), np.array(r["Ia"])
            out.append(dict(name=sp["name"], body=vi, kind="obb", center_b=sp["obb"][0].tolist(),
                            half=sp["obb"][1].tolist(), x=x[bi].tolist(), q_wxyz=q[bi].tolist(), v=v[bi].tolist(),
                            w_b=w[bi].tolist(), mass=r["m"], inertia=list(r["I"]),
                            m_eff=(r["m"] + ma * s_[bi]).tolist(), I_eff=(np.array(r["I"]) + Ia * s_[bi]).tolist()))
        return out

    def apply_contact(self, impulses=(), wrenches=()):
        """Apply cross-domain contact to the vessels, deterministically in the given order.
        impulses: iterable of (vessel index, J_world [N s], P_world [m]) — instantaneous velocity change now, against
                  the effective (rigid + added) mass/inertia the integrator uses;
        wrenches: iterable of (vessel index, F_world [N], P_world [m]) — penalty forces, added to the NEXT step only.
        Host-side (contacts are rare events); state arrays are read once and written once."""
        if impulses:
            x, v, q, w = self.rb.state()
            s_ = self.rb.s.numpy()
            for vi, J, P in impulses:
                bi, r = self.specs[vi]["bi"], self.rb.rows[self.specs[vi]["bi"]]
                R = _quat_R(q[bi])
                Jb = R.T @ np.asarray(J, float)
                rb_ = R.T @ (np.asarray(P, float) - x[bi])
                m_eff = r["m"] + np.array(r["ma"]) * s_[bi]
                I_eff = np.array(r["I"]) + np.array(r["Ia"]) * s_[bi]
                v[bi] = v[bi] + R @ (Jb / m_eff)
                w[bi] = w[bi] + np.cross(rb_, Jb) / I_eff
            self.rb.v.assign(v.astype(np.float32))
            self.rb.w.assign(w.astype(np.float32))
        if wrenches:
            x, _, q, _ = self.rb.state()
            fe, te = self.fe.numpy(), self.te.numpy()
            for vi, F, P in wrenches:
                bi = self.specs[vi]["bi"]
                R = _quat_R(q[bi])
                F = np.asarray(F, float)
                fe[vi] += F
                te[vi] += np.cross(R.T @ (np.asarray(P, float) - x[bi]), R.T @ F)
            self.fe.assign(fe.astype(np.float32))
            self.te.assign(te.astype(np.float32))

    # ---------- World logging protocol (core/world.py) ----------
    def log_channel(self, vi, channel, unit=""):
        """Log telemetry channel TEL[channel] of vessel vi as a state-log series."""
        self.channels.append((int(vi), channel, unit))

    def log_setup(self, log, run_dir):
        for s in self.specs:
            if s["mesh"] is not None:
                log.add_body(s["name"], kind="vehicle", shape="boat", size=[s["L_wl"]], V=s["mesh"][0], F=s["mesh"][1],
                             color=s["color"], domain="water")
            else:
                log.add_body(s["name"], kind="vehicle", shape="boat", size=s["L_wl"], color=s["color"], domain="water")
        for vi, ch, unit in self.channels:
            log.add_series("%s %s" % (self.specs[vi]["name"], ch), unit)

    def log_frame(self):
        s = self.snapshot()
        tel = s.pop("tel")
        if self.channels:
            s["series"] = [float(tel[vi][TEL[ch]]) for vi, ch, _ in self.channels]
        return s

    def capsize_state(self):
        """Per vessel: max_tilt_deg (largest angle of the deck normal from world up so far), capsized (passed 90
        deg: deck normal below horizontal), t_capsize_s (first time it did, None if never)."""
        tm, tc = self.tmax.numpy(), self.tcap.numpy()
        return [dict(max_tilt_deg=float(tm[i]), capsized=bool(tc[i] >= 0), t_capsize_s=float(tc[i]) if tc[i] >= 0
                     else None) for i in range(self.nv)]

    def summary(self):
        tel, ctl = self.tel.numpy(), self.control_state()
        cap = self.capsize_state()
        return {s["name"]: dict(energy_kWh=float(ctl[i][2] / 3.6e6), distance_m=float(ctl[i][3]),
                                **{k: float(tel[i][j]) for k, j in TEL.items()}, **cap[i])
                for i, s in enumerate(self.specs)}

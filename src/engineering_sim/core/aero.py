#!/usr/bin/env python3
# DESCRIPTION: Shared by every 6-DOF flight vehicle that is not a GPU batch (rocket_fc rocket, space launcher).
#
#   step_rigid_np()  float64 numpy mirror of core/rigid6.step_rigid (same algorithm, same order: semi-implicit Euler
#                    v -> x, Euler-equation body rate, attitude by the exact body-rate rotation over dt). Used where a
#                    single body is stepped in lock-step with CPU code (a real flight computer at 100 Hz) or needs
#                    float64 (orbital radii). tests/test_attitude.py checks it against Rigid6.integrate on the GPU.
#                    Quaternions here are (w, x, y, z), body -> world; body +X = vehicle nose / thrust axis.
#   barrowman()      Barrowman (1967) subsonic normal-force slope and CP for nose + one fin set (N fins), with
#                    body-fin interference K = 1 + R/(s+R) and Prandtl-Glauert 1/beta on the fins (M <= 0.9 clamp).
#                    Body (cylinder) lift neglected, as in Barrowman. Nose CN_alpha = 2 /rad; CP coefficients per
#                    shape: cone 2/3, ogive 0.466, von Karman / Haack 0.5, parabolic 0.5 of the nose length.
#   damping_c2a()    aerodynamic pitch/yaw damping moment coefficient C2A = 0.5 rho V A_ref sum CN_i (X_i - X_cg)^2
#                    (Mandell, "Topics in Advanced Model Rocketry", ch. 4) and jet damping C2R = mdot (X_noz - X_cg)^2.
#   fin_roll()       strip-theory roll forcing / damping for canted (or differentially deflected) fins:
#                    M_force = N q a_eff delta J1,  M_damp = -N (rho V / 2) a_eff p J2,  J1 = int c y dy, J2 = int c y^2 dy
#                    over the exposed span, a_eff = per-fin Barrowman slope on fin planform area. Steady roll rate
#                    p_ss = delta V J1 / J2.
#   short_period()   linear pitch oscillator I th'' + C2 th' + C1 th = 0, C1 = q A CN_alpha (X_cp - X_cg):
#                    omega_n = sqrt(C1/I), zeta = C2 / (2 sqrt(C1 I)).
# ALL MODELED. Geometry/coefficients are parameters supplied by the caller (labelled ASSUMED there).
import math

import numpy as np

NOSE_XCP = dict(cone=2.0 / 3.0, ogive=0.466, vonkarman=0.5, haack=0.5, parabolic=0.5)
M_PG_MAX = 0.9                  # Prandtl-Glauert clamp (subsonic method; transonic/supersonic not modelled)


# ---------------------------------------------------------------- quaternion helpers (w, x, y, z), body -> world
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def cross3(a, b):
    """3-vector cross product without numpy.cross overhead (hot path: one body stepped at kHz)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def qrot(q, v):
    """Rotate body vector v to world."""
    w, x, y, z = q
    u = (x, y, z)
    v = np.asarray(v, float)
    t = 2.0 * cross3(u, v)
    return v + w * t + cross3(u, t)


def qrot_inv(q, v):
    """Rotate world vector v to body."""
    return qrot(np.array([q[0], -q[1], -q[2], -q[3]]), v)


def q_axis_angle(axis, ang):
    s = math.sin(0.5 * ang)
    return np.array([math.cos(0.5 * ang), axis[0] * s, axis[1] * s, axis[2] * s])


def q_from_x_to(d):
    """Shortest rotation taking body +X onto world unit vector d (roll left at zero about that rotation)."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    c = float(np.clip(d[0], -1.0, 1.0))
    ax = np.array([0.0, -d[2], d[1]])                 # (1,0,0) x d
    s = math.hypot(d[1], d[2])
    if s < 1e-12:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 0, 0, 1.0])
    return q_axis_angle(ax / s, math.atan2(s, c))


def step_rigid_np(x, q, v, w, F_world, tau_body, m_eff, I_eff, I_rigid, dt):
    """float64 mirror of core/rigid6.step_rigid. Returns (x, q, v, w). F_world includes gravity."""
    q = np.asarray(q, float)
    Fb = qrot_inv(q, F_world)
    ab = Fb / np.asarray(m_eff, float)
    vn = np.asarray(v, float) + qrot(q, ab) * dt
    xn = np.asarray(x, float) + vn * dt
    w = np.asarray(w, float)
    I_rigid = np.asarray(I_rigid, float)
    T = np.asarray(tau_body, float) - cross3(w, I_rigid * w)
    wn = w + T / np.asarray(I_eff, float) * dt
    om = math.sqrt(wn[0] * wn[0] + wn[1] * wn[1] + wn[2] * wn[2])
    qn = q
    if om > 1.0e-9:
        qn = qmul(q, q_axis_angle(wn / om, om * dt))
    qn = qn / math.sqrt(qn[0] * qn[0] + qn[1] * qn[1] + qn[2] * qn[2] + qn[3] * qn[3])
    return xn, qn, vn, wn


# ---------------------------------------------------------------- Barrowman
def fin_geometry(g):
    """Derived fin quantities from a geometry dict (keys below, metres)."""
    Cr, Ct, s, Xs = g["fin_root"], g["fin_tip"], g["fin_span"], g["fin_sweep"]
    Lf = math.sqrt(s * s + (Xs + 0.5 * Ct - 0.5 * Cr) ** 2)           # mid-chord line length
    S = 0.5 * (Cr + Ct) * s                                            # one fin planform area
    return Cr, Ct, s, Xs, Lf, S


def barrowman(g, mach=0.0):
    """Normal-force slopes [1/rad, ref area pi R^2] and CP positions [m from nose tip].
    g: radius_m, nose_len, nose_shape, fin_n, fin_root, fin_tip, fin_span, fin_sweep, fin_x (root LE from nose)."""
    R = g["radius_m"]
    d = 2.0 * R
    cn_n = 2.0
    x_n = NOSE_XCP[g.get("nose_shape", "ogive")] * g["nose_len"]
    Cr, Ct, s, Xs, Lf, _ = fin_geometry(g)
    N = g["fin_n"]
    K = 1.0 + R / (s + R)
    cn_f0 = K * 4.0 * N * (s / d) ** 2 / (1.0 + math.sqrt(1.0 + (2.0 * Lf / (Cr + Ct)) ** 2))
    beta = math.sqrt(1.0 - min(abs(mach), M_PG_MAX) ** 2)
    cn_f = cn_f0 / beta
    x_f = g["fin_x"] + Xs * (Cr + 2.0 * Ct) / (3.0 * (Cr + Ct)) + ((Cr + Ct) - Cr * Ct / (Cr + Ct)) / 6.0
    cn = cn_n + cn_f
    return dict(cn_nose=cn_n, x_nose=x_n, cn_fins=cn_f, x_fins=x_f, cn=cn, x_cp=(cn_n * x_n + cn_f * x_f) / cn,
                K=K, beta=beta, a_ref=math.pi * R * R, d=d)


def damping_c2a(rho, V, a_ref, comps, x_cg):
    """Aerodynamic pitch/yaw damping coefficient [N m s]; comps = [(cn_i, x_i), ...]."""
    return 0.5 * rho * V * a_ref * sum(cn * (x - x_cg) ** 2 for cn, x in comps)


def fin_roll(g):
    """Span integrals J1 = int c y dy, J2 = int c y^2 dy over the exposed span (y from the body axis), linear chord
    c(y) = Cr + k (y - R), k = (Ct - Cr)/s, integrated analytically. Returns (J1, J2, one-fin area)."""
    R = g["radius_m"]
    Cr, Ct, s, _, _, S = fin_geometry(g)
    k = (Ct - Cr) / s
    a0 = Cr - k * R
    F1 = lambda y: a0 * y ** 2 / 2.0 + k * y ** 3 / 3.0      # noqa: E731
    F2 = lambda y: a0 * y ** 3 / 3.0 + k * y ** 4 / 4.0      # noqa: E731
    return F1(R + s) - F1(R), F2(R + s) - F2(R), S


def short_period(rho, V, a_ref, cn, x_cp, x_cg, I, c2):
    C1 = 0.5 * rho * V * V * a_ref * cn * (x_cp - x_cg)
    wn = math.sqrt(max(C1, 0.0) / I)
    zeta = c2 / (2.0 * math.sqrt(max(C1 * I, 1e-30)))
    return dict(C1=C1, omega_n=wn, zeta=zeta, omega_d=wn * math.sqrt(max(1.0 - zeta * zeta, 0.0)))

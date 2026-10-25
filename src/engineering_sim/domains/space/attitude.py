# DESCRIPTION: Rigid-body attitude coupled to the orbit:
#                q (w,x,y,z) body->inertial, omega [rad/s] body, reaction-wheel momenta h_w [N m s] on body axes.
#                I w' = tau_gg + tau_mag + tau_rw - w x (I w + h_w);   h_w' = -tau_rw   (Wie 2008, Ch. 7)
#              Torques:
#                gravity gradient  tau = 3 mu / r^5 (r_b x I r_b)                                (Wie eq. 6.62)
#                magnetic          tau = m x B; B = tilted dipole, B0 (R/r)^3 [3 (m.r) r - m], dipole fixed in
#                                  ECEF (IGRF-13 2025 pole, constants.py), rotated by GMST. ASSUMED adequate for
#                                  attitude-level torque/detumble studies (true field deviates up to ~30 % locally).
#                residual dipole   m_res [A m^2] (parameter)
#              Actuators (simple models, every limit a parameter):
#                magnetorquers     |m_i| <= m_max; controller "bdot" m = -k dB/dt (body), or desaturation
#                                  m = k_des (h_w x B)/|B|^2 (torque -k_des h_w,perp) while wheels hold attitude
#                reaction wheels   |tau_i| <= tau_max, |h_i| <= h_max (torque zeroed when saturated); PD law
#                                  tau = -Kp q_err_vec sign(q_err_w) - Kd (w - w_ref) toward an LVLH or inertial target
#              Orbit integrated alongside with forces.accel (same model as everything else in the domain).
#              RK4, fixed step; quaternion renormalised each step.
import numpy as np

from .constants import MU_EARTH, B0_DIPOLE, DIPOLE_POLE_LAT_DEG, DIPOLE_POLE_LON_DEG, WGS84_A
from .frames import gmst, lla_to_ecef
from . import forces as FZ


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_to_dcm(q):
    """DCM body->inertial."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def dcm_to_q(C):
    tr = np.trace(C)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (C[2, 1] - C[1, 2]) / s, (C[0, 2] - C[2, 0]) / s, (C[1, 0] - C[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(C)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + C[i, i] - C[j, j] - C[k, k])
        q = [0.0] * 4
        q[0] = (C[k, j] - C[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (C[j, i] + C[i, j]) / s
        q[1 + k] = (C[k, i] + C[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def lvlh_dcm(r, v):
    """LVLH->inertial: x along-track (completes the triad), y = -orbit normal, z = nadir."""
    z = -r / np.linalg.norm(r)
    y = -np.cross(r, v)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z], 1)


def dipole_axis_eci(jd):
    m_ecef = -lla_to_ecef(np.deg2rad(DIPOLE_POLE_LAT_DEG), np.deg2rad(DIPOLE_POLE_LON_DEG), 0.0)
    m_ecef /= np.linalg.norm(m_ecef)
    th = gmst(jd)
    c, s = np.cos(th), np.sin(th)
    return np.array([c * m_ecef[0] - s * m_ecef[1], s * m_ecef[0] + c * m_ecef[1], m_ecef[2]])


def b_field_eci(r, jd):
    m = dipole_axis_eci(jd)
    rn = np.linalg.norm(r)
    rh = r / rn
    return B0_DIPOLE * (WGS84_A / rn) ** 3 * (3.0 * np.dot(m, rh) * rh - m)


class Spacecraft:
    def __init__(self, inertia, m_res=(0, 0, 0), mtq_max=0.0, rw_tau_max=0.0, rw_h_max=0.0, cd_am=0.0, cr_am=0.0):
        self.I = np.asarray(inertia, float) if np.ndim(inertia) == 2 else np.diag(inertia)
        self.Iinv = np.linalg.inv(self.I)
        self.m_res = np.asarray(m_res, float)
        self.mtq_max, self.rw_tau_max, self.rw_h_max = mtq_max, rw_tau_max, rw_h_max
        self.cd_am, self.cr_am = cd_am, cr_am


def simulate(sc, r0, v0, q0, w0, jd0, h, n_steps, every=1, control=None, gains=None, orbit_terms=("j2",), air=None,
             hw0=(0, 0, 0), torques=("gg", "mag")):
    """control: None | "bdot" | "rw_lvlh" | "rw_lvlh+desat"; torques: subset of ("gg", "mag"). Returns dict of histories (t, r, v, q, w, hw, m, tau_*)."""
    g = dict(k_bdot=2e4, kp=2e-3, kd=4e-2, k_des=1e-2)
    g.update(gains or {})
    r, v = np.array(r0, float), np.array(v0, float)
    q, w, hw = np.array(q0, float), np.array(w0, float), np.array(hw0, float)
    out = {k: [] for k in ("t", "r", "v", "q", "w", "hw", "m", "tau_gg", "tau_mag")}
    B_prev = None

    def controls(t, r, v, q, w, hw):
        nonlocal B_prev
        C = q_to_dcm(q)
        B_b = C.T @ b_field_eci(r, jd0 + t / 86400.0)
        m_cmd, tau_rw = np.zeros(3), np.zeros(3)
        if control == "bdot":
            if B_prev is not None:
                m_cmd = -g["k_bdot"] * (B_b - B_prev[1]) / max(t - B_prev[0], 1e-9)
            B_prev = (t, B_b)
        elif control and control.startswith("rw_lvlh"):
            Ct = lvlh_dcm(r, v)
            qe = qmul(qconj(dcm_to_q(Ct)), q)                      # target->body error
            n = np.linalg.norm(np.cross(r, v)) / np.dot(r, r)    # orbit rate
            w_ref = (C.T @ Ct) @ np.array([0.0, -n, 0.0])
            tau_rw = -g["kp"] * np.sign(qe[0] if qe[0] != 0 else 1) * qe[1:] - g["kd"] * (w - w_ref)
            if control.endswith("desat"):
                m_cmd = g["k_des"] * np.cross(hw, B_b) / max(np.dot(B_b, B_b), 1e-30)
        if sc.mtq_max > 0:
            m_cmd = np.clip(m_cmd, -sc.mtq_max, sc.mtq_max)
        else:
            m_cmd = np.zeros(3)
        if sc.rw_tau_max > 0:
            tau_rw = np.clip(tau_rw, -sc.rw_tau_max, sc.rw_tau_max)
        else:
            tau_rw = np.zeros(3)
        return m_cmd, tau_rw

    def deriv(t, r, v, q, w, hw, m_cmd, tau_rw):
        C = q_to_dcm(q)
        rb = C.T @ r
        rn = np.linalg.norm(r)
        tau_gg = 3.0 * MU_EARTH / rn ** 5 * np.cross(rb, sc.I @ rb) if "gg" in torques else np.zeros(3)
        B_b = C.T @ b_field_eci(r, jd0 + t / 86400.0)
        tau_mag = np.cross(sc.m_res + m_cmd, B_b) if "mag" in torques else np.zeros(3)
        tr = tau_rw.copy()
        if sc.rw_h_max > 0:                                      # saturated wheel cannot absorb more momentum
            tr[(hw >= sc.rw_h_max) & (tr < 0)] = 0.0
            tr[(hw <= -sc.rw_h_max) & (tr > 0)] = 0.0
        wdot = sc.Iinv @ (tau_gg + tau_mag + tr - np.cross(w, sc.I @ w + hw))
        qdot = 0.5 * qmul(q, np.array([0.0, *w]))
        a = FZ.accel(r[None], v[None], jd0 + t / 86400.0, sc.cd_am, sc.cr_am, air, orbit_terms)[0]
        return v, a, qdot, wdot, -tr, tau_gg, tau_mag

    for k in range(n_steps + 1):
        t = k * h
        m_cmd, tau_rw = controls(t, r, v, q, w, hw)            # zero-order hold over the step
        if k % every == 0:
            d = deriv(t, r, v, q, w, hw, m_cmd, tau_rw)
            for key, val in (("t", t), ("r", r), ("v", v), ("q", q), ("w", w), ("hw", hw), ("m", m_cmd),
                             ("tau_gg", d[5]), ("tau_mag", d[6])):
                out[key].append(np.array(val, float))
        if k == n_steps:
            break
        s = (r, v, q, w, hw)
        k1 = deriv(t, *s, m_cmd, tau_rw)[:5]
        k2 = deriv(t + h / 2, *[x + h / 2 * dx for x, dx in zip(s, k1)], m_cmd, tau_rw)[:5]
        k3 = deriv(t + h / 2, *[x + h / 2 * dx for x, dx in zip(s, k2)], m_cmd, tau_rw)[:5]
        k4 = deriv(t + h, *[x + h * dx for x, dx in zip(s, k3)], m_cmd, tau_rw)[:5]
        r, v, q, w, hw = [x + h / 6 * (a + 2 * b + 2 * c + d) for x, a, b, c, d in zip(s, k1, k2, k3, k4)]
        q = q / np.linalg.norm(q)
        if sc.rw_h_max > 0:
            hw = np.clip(hw, -sc.rw_h_max, sc.rw_h_max)
    return {k: np.array(v) for k, v in out.items()}

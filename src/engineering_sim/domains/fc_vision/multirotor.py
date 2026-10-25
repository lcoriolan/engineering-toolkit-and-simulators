#!/usr/bin/env python3
# DESCRIPTION: The vehicle the legacy sensor terminal-autonomy camera rides on. Core bodies are point masses (no
#              attitude); a body-fixed camera needs attitude, so this domain adds the minimum: the thrust vector
#              tilts toward the commanded acceleration with a first-order lag (tau_att) — the standard
#              quasi-static multirotor approximation (Mahony, Kumar & Corke 2012, IEEE RAM: attitude loop much
#              faster than position loop). Yaw held at yaw0. NOT a 6-DOF rigid body: no rotor dynamics, no
#              body rates/inertia, no rotor drag. REQUESTED CORE CHANGE: 6-DOF rigid body domain.
#              Forces (MODELED): gravity; thrust along body +Z, |T| <= max_tw m g; drag 0.5 rho(z) CdA |v_rel| v_rel
#              with rho and the gusty wind from engineering_sim.core.env.Env at the vehicle position.
#              Parameters (ASSUMED, parametric): mass 1.2 kg, CdA 0.03 m^2, max_tw 2.5, tau_att 0.08 s,

import math

import numpy as np

try:
    from engineering_sim.core.constants import G0 as G      # the ONE core gravity constant (core v0.2)
except ImportError:                          # CPU container without unified_sim on sys.path: same value
    G = 9.80665
MIN_THRUST_FRAC = 0.1      # ASSUMED idle-thrust floor, fraction of hover thrust


def quat_from_z_yaw(zb, yaw):
    """Body->world quaternion whose body +Z = unit vector zb, heading yaw (rad)."""
    zb = zb / np.linalg.norm(zb)
    xc = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    yb = np.cross(zb, xc); yb /= np.linalg.norm(yb)
    xb = np.cross(yb, zb)
    R = np.c_[xb, yb, zb]
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


class Multirotor:
    def __init__(self, env, pos, mass=1.2, cda=0.03, max_tw=2.5, tau_att=0.08, max_tilt_deg=35.0, yaw0=0.0):
        self.env, self.m, self.cda, self.max_tw = env, mass, cda, max_tw
        self.tau, self.max_tilt, self.yaw = tau_att, math.radians(max_tilt_deg), yaw0
        self.x = np.array(pos, float)
        self.v = np.zeros(3)
        self.zb = np.array([0.0, 0.0, 1.0])
        self.T = mass * G
        self.t = 0.0
        self.last_wind = np.zeros(3)

    @property
    def q(self):
        return quat_from_z_yaw(self.zb, self.yaw)

    def tilt_deg(self):
        return math.degrees(math.acos(max(-1.0, min(1.0, self.zb[2]))))

    def step(self, a_cmd, dt):
        """a_cmd = desired acceleration (world, excluding gravity). Returns nothing; updates state."""
        f_des = self.m * (np.asarray(a_cmd, float) + np.array([0, 0, G]))
        # rotors cannot pull down: a demand below MIN_THRUST_FRAC of hover (steep descent) is floored there
        # (2026-10-15 fix: an all-vertical downward demand used to give a NaN tilt axis)
        f_des[2] = max(f_des[2], MIN_THRUST_FRAC * self.m * G)
        n = np.linalg.norm(f_des)
        zt = f_des / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        tilt = math.acos(max(-1.0, min(1.0, zt[2])))
        if tilt > self.max_tilt:            # clamp tilt, keep heading of the horizontal demand
            h = np.array([zt[0], zt[1], 0.0]); hn = np.linalg.norm(h)
            zt = np.array([h[0] / hn * math.sin(self.max_tilt), h[1] / hn * math.sin(self.max_tilt),
                           math.cos(self.max_tilt)])
        a = dt / (self.tau + dt)            # first-order attitude lag
        self.zb = (1 - a) * self.zb + a * zt
        self.zb /= np.linalg.norm(self.zb)
        self.T = min(n, self.max_tw * self.m * G)
        uw = self.env.wind([self.x], self.t)[0]
        self.last_wind = uw
        rho = float(self.env.rho([max(self.x[2], 0.0)])[0])
        vr = uw - self.v
        f = self.zb * self.T + np.array([0, 0, -self.m * G]) + 0.5 * rho * self.cda * np.linalg.norm(vr) * vr
        self.v = self.v + f / self.m * dt
        self.x = self.x + self.v * dt
        self.t += dt
        if self.x[2] <= 0.0:
            self.x[2] = 0.0
            self.v[:] = 0.0

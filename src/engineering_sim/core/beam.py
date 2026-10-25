#!/usr/bin/env python3
# DESCRIPTION: GPU batch of vertical uniform Euler-Bernoulli cantilevers, clamped at the base, with an optional
#              tip mass + tip "head" (sensor pod) of its own drag area, bending in BOTH horizontal planes under the
#              shared wind field.
#
#              METHOD — Rayleigh-Ritz on the first NM=3 clamped-free cantilever modes (Blevins 1979, "Formulas for
#              Natural Frequency and Mode Shape", Table 8-1):
#                  phi_r(xi) = cosh(b xi) - cos(b xi) - s_r (sinh(b xi) - sin(b xi)),  b = beta_r L,
#                  beta_r L = 1.87510, 4.69409, 7.85476,  s_r = (cosh b + cos b)/(sinh b + sin b),
#              normalised to int_0^1 phi^2 dxi = 1.
#                  M_ij = mu L delta_ij + m_tip phi_i(1) phi_j(1)
#                  K_ij = (beta_i L)^4 EI / L^3 delta_ij
#                  C    = M Phi diag(2 zeta omega) Phi^T M       (modal damping on the coupled eigenmodes)
#                  M qdd + C qd + K q = Q,   Q_r = int f(x) phi_r dx + F_tip phi_r(1)
#              Distributed drag per unit length uses the RELATIVE air velocity at the deflected station
#              (aerodynamic damping included):  f = 0.5 rho(z) Cd D |v_perp| v_perp,  v_perp = horizontal part of
#              (wind - structure velocity).  Tip:  F_tip = 0.5 rho Cd_h A_h |v_rel| v_rel (horizontal).
#              Root bending moment  M_root = EI/L^2 sum_r q_r phi_r''(0)  (host-side, from the logged q).
#
#              VALIDITY / LIMITS: small deflection, linear elastic, uniform section, vertical axis, no torsion,
#              no vortex shedding (lift / lock-in NOT modelled), no geometric stiffening by gravity. Validated vs
#              closed forms in tests/test_core.py (tip deflection qL^4/8EI, Rayleigh tip-mass frequency).
#              Stresses for qualification come from FEA, not from this model.
#              Integrator: semi-implicit Euler with sub-steps so dt_sub <= T_highest/40; station wind/density are
#              evaluated in parallel once per core step (kernel station_air) and held over the sub-steps.
import math

import numpy as np
import warp as wp

from . import atmos as A
from . import wind as W

NM = 3
NS = 17                      # quadrature / logging stations along each beam (incl. base and tip)
BETA_L = (1.8751040687, 4.6940911330, 7.8547574382)


def mode_tables(ns=NS):
    xi = np.linspace(0.0, 1.0, ns)
    phi, ddphi0 = [], []
    for b in BETA_L:
        s = (math.cosh(b) + math.cos(b)) / (math.sinh(b) + math.sin(b))
        f = lambda z: np.cosh(b * z) - np.cos(b * z) - s * (np.sinh(b * z) - np.sin(b * z))
        fine = np.linspace(0, 1, 20001)
        norm = math.sqrt(np.trapezoid(f(fine) ** 2, fine))
        phi.append(f(xi) / norm)
        ddphi0.append(2.0 * b * b / norm)          # d2phi/dxi2 at the root
    w = np.full(ns, 1.0 / (ns - 1)); w[0] = w[-1] = 0.5 / (ns - 1)   # trapezoid weights in xi
    return xi, np.array(phi), np.array(ddphi0), w


@wp.kernel
def station_air(t: float, atm: A.AtmosParams,
                w: W.WindParams, mk: wp.array(dtype=wp.vec3), ma: wp.array(dtype=float),
                mph: wp.array(dtype=float), ms: wp.array(dtype=wp.vec3),
                nodes: wp.array2d(dtype=wp.vec3), uw_out: wp.array2d(dtype=wp.vec3), rho_out: wp.array2d(dtype=float)):
    # one thread per (beam, station): wind + density at the station's CURRENT deflected position, evaluated once
    # per core step and held over the beam sub-steps (wind frozen over dt — dt << gust time scales).
    b, s = wp.tid()
    p = nodes[b, s]
    uw_out[b, s] = W.wind_at(p, t, w, mk, ma, mph, ms)
    rho_out[b, s] = A.air_density(p[2], atm)


@wp.kernel
def step_beams(dt: float, nsub: int,
               base: wp.array(dtype=wp.vec3), length: wp.array(dtype=float), diam: wp.array(dtype=float),
               cd: wp.array(dtype=float), cd_tip_a: wp.array(dtype=float),
               Minv: wp.array(dtype=wp.mat33), K: wp.array(dtype=wp.mat33), C: wp.array(dtype=wp.mat33),
               phi: wp.array2d(dtype=float), qw: wp.array(dtype=float), ns: int,
               uw_st: wp.array2d(dtype=wp.vec3), rho_st: wp.array2d(dtype=float),
               qx: wp.array(dtype=wp.vec3), qy: wp.array(dtype=wp.vec3),
               vx: wp.array(dtype=wp.vec3), vy: wp.array(dtype=wp.vec3),
               nodes: wp.array2d(dtype=wp.vec3)):
    b = wp.tid()
    L = length[b]
    hdt = dt / float(nsub)
    ax = qx[b]
    ay = qy[b]
    bx = vx[b]
    by = vy[b]
    for sub in range(nsub):
        Qx = wp.vec3(0.0, 0.0, 0.0)
        Qy = wp.vec3(0.0, 0.0, 0.0)
        for s in range(ns):
            ph = wp.vec3(phi[0, s], phi[1, s], phi[2, s])
            sx = wp.dot(bx, ph)
            sy = wp.dot(by, ph)
            uw = uw_st[b, s]
            rx = uw[0] - sx
            ry = uw[1] - sy
            rho = rho_st[b, s]
            vm = wp.sqrt(rx * rx + ry * ry)
            fq = 0.5 * rho * cd[b] * diam[b] * vm * qw[s] * L
            Qx = Qx + ph * (fq * rx)
            Qy = Qy + ph * (fq * ry)
            if s == ns - 1:
                ft = 0.5 * rho * cd_tip_a[b] * vm
                Qx = Qx + ph * (ft * rx)
                Qy = Qy + ph * (ft * ry)
        accx = Minv[b] * (Qx - C[b] * bx - K[b] * ax)
        accy = Minv[b] * (Qy - C[b] * by - K[b] * ay)
        bx = bx + accx * hdt
        by = by + accy * hdt
        ax = ax + bx * hdt
        ay = ay + by * hdt
    qx[b] = ax
    qy[b] = ay
    vx[b] = bx
    vy[b] = by
    for s in range(ns):
        ph = wp.vec3(phi[0, s], phi[1, s], phi[2, s])
        nodes[b, s] = base[b] + wp.vec3(wp.dot(ax, ph), wp.dot(ay, ph), L * float(s) / float(ns - 1))


def tube_section(od_m, wall_m):
    """Circular tube: returns (area m^2, I m^4)."""
    ri = od_m / 2 - wall_m
    return math.pi * ((od_m / 2) ** 2 - ri ** 2), math.pi / 4 * ((od_m / 2) ** 4 - ri ** 4)


class Beams:
    """Parametric cantilever set. Material/section/head are all arguments (parametric-first)."""

    def __init__(self):
        self.rows = []
        self.xi, self.phi, self.ddphi0, self.w = mode_tables()

    def add_cantilever(self, base, length_m, od_m, wall_m, E_pa=69e9, density=2700.0, cd=1.2,
                       tip_mass=0.0, tip_cd_area=0.0, zeta=0.02, name="mast"):
        A_, I = tube_section(od_m, wall_m)
        mu = density * A_
        EI = E_pa * I
        L = length_m
        ph1 = self.phi[:, -1]
        M = mu * L * np.eye(NM) + tip_mass * np.outer(ph1, ph1)
        K = np.diag([(b ** 4) * EI / L ** 3 for b in BETA_L])
        # coupled eigenmodes (generalised symmetric problem)
        Lc = np.linalg.cholesky(M)
        Li = np.linalg.inv(Lc)
        lam, U = np.linalg.eigh(Li @ K @ Li.T)
        Phi = Li.T @ U                                  # mass-normalised: Phi^T M Phi = I
        om = np.sqrt(lam)
        C = M @ Phi @ np.diag(2 * zeta * om) @ Phi.T @ M
        self.rows.append(dict(name=name, base=tuple(base), L=L, D=od_m, wall=wall_m, E=E_pa, rho_m=density,
                              cd=cd, tip_mass=tip_mass, tip_cda=tip_cd_area, zeta=zeta, mu=mu, EI=EI, I=I,
                              M=M, K=K, C=C, f_hz=(om / (2 * math.pi)).tolist()))
        return len(self.rows) - 1

    def upload(self, device):
        R = self.rows
        self.n = len(R)
        if self.n == 0:
            return
        m33 = lambda key: wp.array(np.array([r[key] for r in R], np.float32), dtype=wp.mat33, device=device)
        f32 = lambda key: wp.array(np.array([r[key] for r in R], np.float32), dtype=float, device=device)
        self.Minv = wp.array(np.array([np.linalg.inv(r["M"]) for r in R], np.float32), dtype=wp.mat33, device=device)
        self.K, self.C = m33("K"), m33("C")
        self.base = wp.array(np.array([r["base"] for r in R], np.float32), dtype=wp.vec3, device=device)
        self.length, self.diam, self.cd, self.cd_tip_a = f32("L"), f32("D"), f32("cd"), f32("tip_cda")
        self.phi_d = wp.array(self.phi.astype(np.float32), dtype=float, device=device)
        self.qw = wp.array(self.w.astype(np.float32), dtype=float, device=device)
        self.qx = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.qy = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.vx = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.vy = wp.zeros(self.n, dtype=wp.vec3, device=device)
        nd = np.zeros((self.n, NS, 3), np.float32)
        for i, r in enumerate(R):
            nd[i] = np.array(r["base"]) + np.c_[np.zeros(NS), np.zeros(NS), self.xi * r["L"]]
        self.nodes = wp.array(nd, dtype=wp.vec3, device=device)
        self.uw_st = wp.zeros((self.n, NS), dtype=wp.vec3, device=device)
        self.rho_st = wp.zeros((self.n, NS), dtype=float, device=device)
        f_max = max(max(r["f_hz"]) for r in R)
        self.T_min = 1.0 / f_max

    def step(self, t, dt, atm, wind, device):
        if self.n == 0:
            return
        nsub = max(1, int(math.ceil(dt / (self.T_min / 40.0))))
        wp.launch(station_air, (self.n, NS), [float(t), atm] + wind.args() + [self.nodes, self.uw_st, self.rho_st],
                  device=device)
        wp.launch(step_beams, self.n,
                  [float(dt), nsub, self.base, self.length, self.diam, self.cd, self.cd_tip_a, self.Minv, self.K,
                   self.C, self.phi_d, self.qw, NS, self.uw_st, self.rho_st, self.qx, self.qy, self.vx, self.vy,
                   self.nodes], device=device)

    def root_moment(self):
        """Root bending moment magnitude [N m] per beam from the current modal coordinates."""
        qx, qy = self.qx.numpy(), self.qy.numpy()
        out = []
        for i, r in enumerate(self.rows):
            k = r["EI"] / r["L"] ** 2
            mx, my = k * float(qx[i] @ self.ddphi0), k * float(qy[i] @ self.ddphi0)
            out.append(math.hypot(mx, my))
        return np.array(out)

    def root_stress(self):
        """Peak root bending stress [Pa] = M c / I."""
        return np.array([m * (r["D"] / 2) / r["I"] for m, r in zip(self.root_moment(), self.rows)])

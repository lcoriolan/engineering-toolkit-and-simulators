#!/usr/bin/env python3
# DESCRIPTION: contact_force(): penalty normal force + Coulomb friction for one contact point against core/terrain.
#                normal   Fn = max(0, -k d - c vn),  d = signed distance along the local terrain normal
#                friction |Ft| <= mu Fn, opposing the tangential velocity; velocity-capped so that one step can at
#                         most bring the point's share of mass (m_share) to rest — exact stick below the friction
#                         angle, no viscous creep and no explicit-integration chatter (a standard velocity-level
#                         Coulomb treatment; MODELED).
#              mu = surface D (peak Magic-Formula friction of the cell material) * mu_scale (per body: skids, feet,
#              hull — ASSUMED 1.0 default).
#              ContactBodies: GPU batch of 6-DOF rigid bodies with N contact points each (landing gear, skids,
#              crate corners): gravity, air drag in the core air + terrain-relative (AGL) core wind, optional
#              descent-rate autopilot (multirotor landing: thrust along body z, cut at touchdown).

import numpy as np
import warp as wp

from engineering_sim.core import atmos as A
from engineering_sim.core import terrain as T
from engineering_sim.core import wind as W
from engineering_sim.core import rigid6 as RB   # THE shared rigid-body step (core/rigid6.step_rigid; UNIFY 2026-10-15)

from engineering_sim.core.constants import G0  # noqa: E402  ONE gravity constant (core v0.2)
wp.set_module_options({"enable_backward": False})


@wp.func
def contact_force(p: wp.vec3, vp: wp.vec3, tp: T.TerrainParams, k: float, c: float, mu_scale: float,
                  m_share: float, dt: float):
    hn = T.height_normal(p[0], p[1], tp)
    n = wp.vec3(hn[0], hn[1], hn[2])
    d = (p[2] - hn[3]) * n[2]
    if d >= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    vn = wp.dot(vp, n)
    fn = wp.max(-k * d - c * vn, 0.0)
    vt = vp - n * vn
    vts = wp.length(vt)
    mu = tp.mf[T.material(p[0], p[1], tp)][2] * mu_scale
    ft = wp.vec3(0.0, 0.0, 0.0)
    if vts > 1.0e-9:
        ft = vt * (-wp.min(mu * fn, m_share * vts / dt) / vts)
    return n * fn + ft


@wp.kernel(enable_backward=False)
def step_contact_bodies(t_off: float, tarr: wp.array(dtype=float), dt: float, atm: A.AtmosParams, tp: T.TerrainParams,
                        w: W.WindParams, mk: wp.array(dtype=wp.vec3), ma: wp.array(dtype=float),
                        mph: wp.array(dtype=float), ms: wp.array(dtype=wp.vec3),
                        mass: wp.array(dtype=float), inertia: wp.array(dtype=wp.vec3), cda: wp.array(dtype=float),
                        k_c: wp.array(dtype=float), c_c: wp.array(dtype=float), mu_s: wp.array(dtype=float),
                        pts: wp.array(dtype=wp.vec3), p_start: wp.array(dtype=int), p_count: wp.array(dtype=int),
                        vz_cmd: wp.array(dtype=float), max_tw: wp.array(dtype=float), landed: wp.array(dtype=int),
                        x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.quat), v: wp.array(dtype=wp.vec3),
                        wb: wp.array(dtype=wp.vec3), fz_out: wp.array(dtype=float)):
    i = wp.tid()
    t = tarr[0] + t_off   # clock lives on the device so a whole frame can be one CUDA graph
    m = mass[i]
    p = x[i]
    rot = q[i]
    vel = v[i]
    om_b = wb[i]
    om_w = wp.quat_rotate(rot, om_b)
    uw = W.wind_at(T.agl_point(p, tp, 0.05), t, w, mk, ma, mph, ms)
    vr = uw - vel
    rho = A.air_density(p[2], atm)   # density by world height (MSL offset = site), wind by AGL
    f = vr * (0.5 * rho * cda[i] * wp.length(vr)) + wp.vec3(0.0, 0.0, -G0 * m)
    tau = wp.vec3(0.0, 0.0, 0.0)
    n = p_count[i]
    share = m / float(n)
    fz = float(0.0)
    for j in range(n):
        r = wp.quat_rotate(rot, pts[p_start[i] + j])
        pc = p + r
        fc = contact_force(pc, vel + wp.cross(om_w, r), tp, k_c[i], c_c[i], mu_s[i], share, dt)
        fz = fz + fc[2]
        f = f + fc
        tau = tau + wp.cross(r, fc)
    if fz > 0.0:
        landed[i] = 1
    if max_tw[i] > 0.0 and landed[i] == 0:
        # descent-rate hold: thrust along body z, magnitude for the commanded vertical speed (P on vz), capped
        zb = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
        thr = wp.clamp((2.0 * m * (vz_cmd[i] - vel[2]) - f[2]) / wp.max(zb[2], 0.3), 0.0, max_tw[i] * m * G0)
        f = f + zb * thr
    fz_out[i] = fz
    Ib = inertia[i]
    RB.step_rigid(p, rot, vel, om_b, f, wp.quat_rotate_inv(rot, tau), wp.vec3(m, m, m), Ib, Ib, dt, i, x, q, v, wb)


class ContactBodies:
    """Rigid bodies that touch the terrain at listed points (body frame, metres)."""

    def __init__(self, terrain):
        self.terrain, self.rows = terrain, []

    def add(self, pos, points, mass=1.0, inertia=None, cda=0.02, k=None, c=None, mu_scale=1.0, quat_wxyz=(1, 0, 0, 0),
            vel=(0, 0, 0), descent_rate=None, max_tw=0.0, name="lander", shape="box", size=None):
        pts = np.asarray(points, float)
        ext = pts.max(0) - pts.min(0) + 0.05
        if inertia is None:   # uniform box over the contact-point extent (ASSUMED)
            inertia = [mass * (ext[1] ** 2 + ext[2] ** 2) / 12, mass * (ext[0] ** 2 + ext[2] ** 2) / 12,
                       mass * (ext[0] ** 2 + ext[1] ** 2) / 12]
        # contact stiffness: ~1 % static sag of the per-point load at 1 cm, damping ~0.7 critical (ASSUMED)
        k = k or mass * G0 / len(pts) / 0.005
        c = c or 2 * 0.7 * np.sqrt(k * mass / len(pts))
        self.rows.append(dict(x=pos, q=quat_wxyz, v=vel, mass=mass, I=inertia, cda=cda, k=k, c=c, mu=mu_scale,
                              pts=pts, vz=(descent_rate if descent_rate is not None else 0.0), tw=max_tw, name=name,
                              shape=shape, size=size if size is not None else ext.tolist()))
        return len(self.rows) - 1

    def upload(self, device):
        R = self.rows
        self.n = len(R)
        f = lambda key: wp.array(np.array([r[key] for r in R], np.float32), dtype=float, device=device)
        v3 = lambda key: wp.array(np.array([r[key] for r in R], np.float32), dtype=wp.vec3, device=device)
        self.mass, self.cda, self.k, self.c, self.mu, self.vz, self.tw = (f(k) for k in
                                                                          ("mass", "cda", "k", "c", "mu", "vz", "tw"))
        self.I, self.x, self.v = v3("I"), v3("x"), v3("v")
        self.q = wp.array(np.array([[r["q"][1], r["q"][2], r["q"][3], r["q"][0]] for r in R], np.float32),
                          dtype=wp.quat, device=device)
        self.wb = wp.zeros(self.n, dtype=wp.vec3, device=device)
        starts, counts, pts = [], [], []
        for r in R:
            starts.append(len(pts)); counts.append(len(r["pts"])); pts.extend(r["pts"].tolist())
        self.pts = wp.array(np.array(pts, np.float32), dtype=wp.vec3, device=device)
        self.p_start = wp.array(np.array(starts, np.int32), dtype=int, device=device)
        self.p_count = wp.array(np.array(counts, np.int32), dtype=int, device=device)
        self.landed = wp.zeros(self.n, dtype=int, device=device)
        self.fz = wp.zeros(self.n, dtype=float, device=device)
        self.tp = self.terrain.params(device)
        if getattr(self, "tarr", None) is None:
            self.tarr = wp.zeros(1, dtype=float, device=device)

    def step(self, t, dt, atm, wind, device):
        if not self.n:
            return
        self.tarr.fill_(float(t))
        self.step_rel(0.0, dt, atm, wind, device)

    def step_rel(self, t_off, dt, atm, wind, device):
        if not self.n:
            return
        key = (float(dt), id(atm), id(wind), str(device), id(self.tarr))
        if getattr(self, "_key", None) != key:   # record once; per step only the time offset changes
            self._cmd = wp.launch(step_contact_bodies, self.n,
                                  [0.0, self.tarr, float(dt), atm, self.tp] + wind.args() +
                                  [self.mass, self.I, self.cda, self.k, self.c, self.mu, self.pts, self.p_start,
                                   self.p_count, self.vz, self.tw, self.landed, self.x, self.q, self.v, self.wb, self.fz],
                                  device=device, record_cmd=True)
            self._key = key
        self._cmd.set_param_at_index(0, float(t_off))
        self._cmd.launch()

    def log_rows(self):
        return [dict(name=r["name"], kind="vehicle", shape=r["shape"], size=r["size"]) for r in self.rows]

    def frame(self):
        q = self.q.numpy()
        return self.x.numpy(), self.v.numpy(), q[:, [3, 0, 1, 2]]

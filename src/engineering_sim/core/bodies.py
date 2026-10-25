#!/usr/bin/env python3
# DESCRIPTION: GPU batch of translational bodies stepped in the SAME atmosphere and wind field as every other
#              domain. Forces per body (MODELED):
#                  gravity      m g (g = 9.80665, -z)
#                  drag         0.5 rho(z) CdA |v_rel| v_rel,   v_rel = wind(x,t) - v      (rho from engineering_sim.core/atmos)
#                  thrust       GUIDED bodies only: velocity-tracking autopilot toward the current waypoint,
#                               a_cmd = kv (v_des - v) - g,  v_des = cruise * unit(wp - x) (slowing inside brake_m),
#                               |thrust| <= max_tw * m g. No integral term, so a steady crosswind leaves a
#                               tracking error — the drift the wind causes is visible, not hidden.
#              Kinds: 0 PASSIVE (gravity+drag), 1 GUIDED (waypoint loop), 2 TRACER (massless, moves with the wind;
#              wraps inside the tracer box so the gust field stays populated — visual + flow probe only).
#              Ground: z < 0 -> clamp to 0, zero velocity (bodies rest; tracers bounce to z0) — the v0.1 rule, kept
#              bit-identical for scenes WITHOUT terrain. With a terrain (set_terrain, 2026-10-15 CONTACT) the floor is
#              the terrain height under the body (tracers: 0.5 m above it), or — for bodies bound to the cross-domain
#              contact manager with terrain collision — no clamp at all: core/contact.py then resolves the ground
#              contact with restitution and friction like every other body.
#              Ground impacts (2026-10-16 GAPS): every body that reaches its floor with a normal approach speed >=
#              impact_speed (default 0.5 m/s, ASSUMED = the contact manager's event_speed) is recorded on the GPU:
#              first impact time, point, velocity and the TERRAIN NORMAL there (flat ground: +z), impact count and the
#              largest normal speed -> summary()["ground_impacts"]. Recording only: the state update is unchanged
#              (scenes without terrain stay bit-identical to v0.1).
#              Integrator: semi-implicit Euler at the core dt.
#              Attitude is NOT simulated in this version (point mass); renderers orient by velocity.
import numpy as np
import warp as wp

from . import atmos as A
from . import terrain as T
from . import wind as W
from .constants import G0

PASSIVE, GUIDED, TRACER = 0, 1, 2


@wp.func
def _record(i: int, t: float, p: wp.vec3, vel: wp.vec3, n: wp.vec3, v_imp: float, it: wp.array(dtype=float),
            ix: wp.array(dtype=wp.vec3), iv: wp.array(dtype=wp.vec3), inrm: wp.array(dtype=wp.vec3),
            icnt: wp.array(dtype=int), ivmax: wp.array(dtype=float)):
    """Ground impact bookkeeping (no effect on the state): an approach along the ground normal >= v_imp counts."""
    vn = -wp.dot(vel, n)
    if vn >= v_imp:
        if icnt[i] == 0:
            it[i] = t
            ix[i] = p
            iv[i] = vel
            inrm[i] = n
        icnt[i] = icnt[i] + 1
        if vn > ivmax[i]:
            ivmax[i] = vn


@wp.kernel
def step_bodies(t: float, dt: float, atm: A.AtmosParams,
                w: W.WindParams, mk: wp.array(dtype=wp.vec3), ma: wp.array(dtype=float),
                mph: wp.array(dtype=float), ms: wp.array(dtype=wp.vec3),
                kind: wp.array(dtype=int), mass: wp.array(dtype=float), cda: wp.array(dtype=float),
                cruise: wp.array(dtype=float), kv: wp.array(dtype=float), max_tw: wp.array(dtype=float),
                wp_pts: wp.array(dtype=wp.vec3), wp_start: wp.array(dtype=int), wp_count: wp.array(dtype=int),
                wp_idx: wp.array(dtype=int), wp_radius: float, brake_m: float,
                box_lo: wp.vec3, box_hi: wp.vec3,
                x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), v_imp: float,
                it: wp.array(dtype=float), ix: wp.array(dtype=wp.vec3), iv: wp.array(dtype=wp.vec3),
                inrm: wp.array(dtype=wp.vec3), icnt: wp.array(dtype=int), ivmax: wp.array(dtype=float)):
    i = wp.tid()
    p = x[i]
    vel = v[i]
    uw = W.wind_at(p, t, w, mk, ma, mph, ms)
    k = kind[i]
    if k == 2:
        p = p + uw * dt
        # periodic wrap inside the tracer box (keeps the flow visualisation populated)
        for a in range(3):
            span = box_hi[a] - box_lo[a]
            if p[a] > box_hi[a]:
                p[a] = p[a] - span
            if p[a] < box_lo[a]:
                p[a] = p[a] + span
        if p[2] < 0.5:
            p[2] = 0.5
        x[i] = p
        v[i] = uw
        return
    m = mass[i]
    rho = A.air_density(p[2], atm)
    vr = uw - vel
    f = vr * (0.5 * rho * cda[i] * wp.length(vr)) + wp.vec3(0.0, 0.0, -G0 * m)
    if k == 1:
        c = wp_count[i]
        j = wp_idx[i]
        tgt = wp_pts[wp_start[i] + j]
        d = tgt - p
        dist = wp.length(d)
        if dist < wp_radius:
            j = (j + 1) % c
            wp_idx[i] = j
            tgt = wp_pts[wp_start[i] + j]
            d = tgt - p
            dist = wp.length(d)
        spd = cruise[i] * wp.min(1.0, dist / brake_m)
        vdes = d * (spd / wp.max(dist, 1e-3))
        a_cmd = (vdes - vel) * kv[i] + wp.vec3(0.0, 0.0, G0)
        thrust = a_cmd * m
        tmax = max_tw[i] * m * G0
        tn = wp.length(thrust)
        if tn > tmax:
            thrust = thrust * (tmax / tn)
        f = f + thrust
    vel = vel + f * (dt / m)
    p = p + vel * dt
    if p[2] < 0.0:
        p[2] = 0.0
        _record(i, t + dt, p, vel, wp.vec3(0.0, 0.0, 1.0), v_imp, it, ix, iv, inrm, icnt, ivmax)
        vel = wp.vec3(0.0, 0.0, 0.0)
    x[i] = p
    v[i] = vel


@wp.kernel
def step_bodies_ground(t: float, dt: float, atm: A.AtmosParams,
                       w: W.WindParams, mk: wp.array(dtype=wp.vec3), ma: wp.array(dtype=float),
                       mph: wp.array(dtype=float), ms: wp.array(dtype=wp.vec3),
                       kind: wp.array(dtype=int), mass: wp.array(dtype=float), cda: wp.array(dtype=float),
                       cruise: wp.array(dtype=float), kv: wp.array(dtype=float), max_tw: wp.array(dtype=float),
                       wp_pts: wp.array(dtype=wp.vec3), wp_start: wp.array(dtype=int), wp_count: wp.array(dtype=int),
                       wp_idx: wp.array(dtype=int), wp_radius: float, brake_m: float,
                       box_lo: wp.vec3, box_hi: wp.vec3, tp: T.TerrainParams, gmode: wp.array(dtype=int),
                       x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), v_imp: float,
                       it: wp.array(dtype=float), ix: wp.array(dtype=wp.vec3), iv: wp.array(dtype=wp.vec3),
                       inrm: wp.array(dtype=wp.vec3), icnt: wp.array(dtype=int), ivmax: wp.array(dtype=float)):
    """step_bodies on a terrain: wind by height above the local ground, floor = terrain (gmode 1) or none (gmode 2:
    the contact manager owns the ground contact)."""
    i = wp.tid()
    p = x[i]
    vel = v[i]
    uw = W.wind_at(T.agl_point(p, tp, 0.05), t, w, mk, ma, mph, ms)
    k = kind[i]
    if k == 2:
        p = p + uw * dt
        for a in range(3):
            span = box_hi[a] - box_lo[a]
            if p[a] > box_hi[a]:
                p[a] = p[a] - span
            if p[a] < box_lo[a]:
                p[a] = p[a] + span
        hz = T.height(p[0], p[1], tp) + 0.5
        if p[2] < hz:
            p[2] = hz
        x[i] = p
        v[i] = uw
        return
    m = mass[i]
    rho = A.air_density(p[2], atm)
    vr = uw - vel
    f = vr * (0.5 * rho * cda[i] * wp.length(vr)) + wp.vec3(0.0, 0.0, -G0 * m)
    if k == 1:
        c = wp_count[i]
        j = wp_idx[i]
        tgt = wp_pts[wp_start[i] + j]
        d = tgt - p
        dist = wp.length(d)
        if dist < wp_radius:
            j = (j + 1) % c
            wp_idx[i] = j
            tgt = wp_pts[wp_start[i] + j]
            d = tgt - p
            dist = wp.length(d)
        spd = cruise[i] * wp.min(1.0, dist / brake_m)
        vdes = d * (spd / wp.max(dist, 1e-3))
        a_cmd = (vdes - vel) * kv[i] + wp.vec3(0.0, 0.0, G0)
        thrust = a_cmd * m
        tmax = max_tw[i] * m * G0
        tn = wp.length(thrust)
        if tn > tmax:
            thrust = thrust * (tmax / tn)
        f = f + thrust
    vel = vel + f * (dt / m)
    p = p + vel * dt
    if gmode[i] == 1:
        hz = T.height(p[0], p[1], tp)
        if p[2] < hz:
            p[2] = hz
            _record(i, t + dt, p, vel, T.normal(p[0], p[1], tp), v_imp, it, ix, iv, inrm, icnt, ivmax)
            vel = wp.vec3(0.0, 0.0, 0.0)
    x[i] = p
    v[i] = vel


class Bodies:
    """Parametric body set. add_* return the body index."""

    def __init__(self, impact_speed=0.5):
        self.rows, self.wps = [], []
        self.terrain = None
        self.impact_speed = float(impact_speed)     # [m/s] ASSUMED: ground-impact recording threshold

    def set_terrain(self, terrain):
        """Floor = this terrain (instead of z = 0) and wind by height above it (2026-10-15 CONTACT)."""
        self.terrain = terrain

    def add_passive(self, pos, vel=(0, 0, 0), mass=1.0, cda=0.05, name="passive", contact=None):
        """contact: optional dict for the cross-domain contact manager: shape (sphere), radius [m] (default 0.1),
        material, enabled (default True), terrain (default True), water (default True)."""
        self.rows.append(dict(kind=PASSIVE, x=pos, v=vel, mass=mass, cda=cda, cruise=0, kv=0, tw=0, name=name,
                              contact=contact))
        self.wps.append([pos])
        return len(self.rows) - 1

    def add_guided(self, waypoints, mass=1.2, cda=0.03, cruise=12.0, kv=1.5, max_tw=2.5, name="drone", contact=None):
        self.rows.append(dict(kind=GUIDED, x=waypoints[0], v=(0, 0, 0), mass=mass, cda=cda, cruise=cruise, kv=kv,
                              tw=max_tw, name=name, contact=contact))
        self.wps.append(list(waypoints[1:]) + [waypoints[0]])
        return len(self.rows) - 1

    def add_tracers(self, n, box_lo, box_hi, seed=1):
        rng = np.random.default_rng(seed)
        lo, hi = np.asarray(box_lo, float), np.asarray(box_hi, float)
        self.box = (lo, hi)
        for p in rng.uniform(lo, hi, size=(n, 3)):
            self.rows.append(dict(kind=TRACER, x=tuple(p), v=(0, 0, 0), mass=1, cda=0, cruise=0, kv=0, tw=0,
                                  name="tracer"))
            self.wps.append([tuple(p)])

    def upload(self, device, wp_radius=3.0, brake_m=15.0):
        R = self.rows
        f32 = lambda key: wp.array(np.array([r[key] for r in R], np.float32), dtype=float, device=device)
        self.n = len(R)
        self.kind_np = np.array([r["kind"] for r in R], np.int32)
        self.names = [r["name"] for r in R]
        self.kind = wp.array(self.kind_np, dtype=int, device=device)
        self.mass, self.cda, self.cruise, self.kv, self.max_tw = f32("mass"), f32("cda"), f32("cruise"), f32("kv"), f32("tw")
        starts, counts, pts = [], [], []
        for w in self.wps:
            starts.append(len(pts)); counts.append(len(w)); pts.extend(w)
        self.wp_pts = wp.array(np.array(pts, np.float32), dtype=wp.vec3, device=device)
        self.wp_start = wp.array(np.array(starts, np.int32), dtype=int, device=device)
        self.wp_count = wp.array(np.array(counts, np.int32), dtype=int, device=device)
        self.wp_idx = wp.zeros(self.n, dtype=int, device=device)
        self.x = wp.array(np.array([r["x"] for r in R], np.float32), dtype=wp.vec3, device=device)
        self.v = wp.array(np.array([r["v"] for r in R], np.float32), dtype=wp.vec3, device=device)
        lo, hi = getattr(self, "box", (np.zeros(3), np.ones(3)))
        self.box_lo, self.box_hi = wp.vec3(*lo), wp.vec3(*hi)
        self.wp_radius, self.brake_m = float(wp_radius), float(brake_m)
        self.device = device
        n1 = max(self.n, 1)
        self.imp_t = wp.full(n1, -1.0, dtype=float, device=device)
        self.imp_x, self.imp_v, self.imp_n = (wp.zeros(n1, dtype=wp.vec3, device=device) for _ in range(3))
        self.imp_cnt = wp.zeros(n1, dtype=int, device=device)
        self.imp_vmax = wp.zeros(n1, dtype=float, device=device)
        self.gmode_np = np.ones(self.n, np.int32)
        if self.terrain is not None:
            self.tp = self.terrain.params(device)
            self.gmode = wp.array(self.gmode_np, dtype=int, device=device)

    def contact_bind(self, cm):
        """Cross-domain contact (core/contact.py): every non-tracer body with contact enabled is a sphere."""
        idx, prox, mass, names = [], [], [], []
        for i, r in enumerate(self.rows):
            c = r.get("contact") or {}
            if r["kind"] == TRACER or not c.get("enabled", True):
                continue
            prox.append(dict(body=len(idx), shape="sphere", size=[c.get("radius", 0.3 if r["kind"] == GUIDED else 0.1)],
                             material=c.get("material", "default"), terrain=c.get("terrain", True)))
            idx.append(i)
            mass.append(r["mass"])
            names.append(r["name"])
            if self.terrain is not None and c.get("terrain", True) and cm.cfg["terrain"]:
                self.gmode_np[i] = 2               # the contact manager owns this body's ground contact
        if not idx:
            return
        if self.terrain is not None:
            self.gmode.assign(self.gmode_np)
        water = any((self.rows[i].get("contact") or {}).get("water", True) for i in idx)
        cm.bind_points("bodies", self.x, self.v, idx, mass, prox, names=names, water=water)

    def step(self, t, dt, atm, wind, device):
        if self.n == 0:
            return
        if self.terrain is not None:
            wp.launch(step_bodies_ground, self.n,
                      [float(t), float(dt), atm] + wind.args() +
                      [self.kind, self.mass, self.cda, self.cruise, self.kv, self.max_tw,
                       self.wp_pts, self.wp_start, self.wp_count, self.wp_idx, self.wp_radius, self.brake_m,
                       self.box_lo, self.box_hi, self.tp, self.gmode, self.x, self.v] + self._imp_args(), device=device)
            return
        wp.launch(step_bodies, self.n,
                  [float(t), float(dt), atm] + wind.args() +
                  [self.kind, self.mass, self.cda, self.cruise, self.kv, self.max_tw,
                   self.wp_pts, self.wp_start, self.wp_count, self.wp_idx, self.wp_radius, self.brake_m,
                   self.box_lo, self.box_hi, self.x, self.v] + self._imp_args(), device=device)

    def _imp_args(self):
        return [self.impact_speed, self.imp_t, self.imp_x, self.imp_v, self.imp_n, self.imp_cnt, self.imp_vmax]

    def ground_impacts(self):
        """Recorded ground impacts (non-tracer bodies that hit their floor at >= impact_speed along the normal):
        list of dict(body index, name, t [s], x [m], v [m/s] (velocity at impact), speed, v_normal (into the
        ground), normal (terrain normal; flat ground +z), slope_deg, n_impacts, v_normal_max)."""
        if not getattr(self, "n", 0):
            return []
        cnt = self.imp_cnt.numpy()[:self.n]
        hit = np.where((cnt > 0) & (self.kind_np != TRACER))[0]
        if not len(hit):
            return []
        T_, X, V, N, VM = (a.numpy() for a in (self.imp_t, self.imp_x, self.imp_v, self.imp_n, self.imp_vmax))
        out = []
        for i in hit:
            n = N[i].astype(float)
            v = V[i].astype(float)
            out.append(dict(body=int(i), name=self.names[i], t=round(float(T_[i]), 6), x=X[i].astype(float).tolist(),
                            v=v.tolist(), speed=float(np.linalg.norm(v)), v_normal=float(-v @ n),
                            normal=n.tolist(), slope_deg=float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0)))),
                            n_impacts=int(cnt[i]), v_normal_max=float(VM[i])))
        return out

    def summary(self):
        gi = self.ground_impacts()
        return dict(ground_impacts=gi) if gi else {}

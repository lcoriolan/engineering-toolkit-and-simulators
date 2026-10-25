#!/usr/bin/env python3
# DESCRIPTION: TerrainDomain logs the terrain as a static true-scale mesh body (index 0 of its slice) and tells the
#              renderer (meta.render_hooks = ["land"]) to drop its checker ground and colour by surface.
#              LandVehicles / ContactBodies get the World logging protocol (log_setup / log_frame) here: chassis as
#              true-scale visual meshes, wheels as cylinders spinning/steering with the logged attitude.
#              LandScene(atm, wind, terrain, domains) = World(atm, wind, [TerrainDomain(terrain)] + domains): kept as
#              the land-only convenience constructor (same run() arguments as before).
import math
import os
import time

import numpy as np
import warp as wp

from engineering_sim.core.world import World

from .contact import ContactBodies
from .vehicles import LandVehicles


def terrain_mesh(terrain, step=1, z_off=-0.02):
    """Heightfield -> (V, F, face surface id) world-frame mesh (every `step`-th node); z_off keeps tyres from
    z-fighting the ground."""
    h = terrain.h[::step, ::step].astype(float)
    ny, nx = h.shape
    X, Y = np.meshgrid(terrain.x0 + terrain.dx * step * np.arange(nx), terrain.y0 + terrain.dx * step * np.arange(ny))
    V = np.c_[X.ravel(), Y.ravel(), h.ravel() + z_off]
    idx = np.arange(nx * ny).reshape(ny, nx)
    F = np.c_[idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()]
    Fm = terrain.mat[::step, ::step][:-1, :-1].ravel()
    return V.astype(np.float32), F.astype(np.int32), Fm.astype(np.int32)


def vehicle_visual_mesh(spec):
    """Readable true-scale body in the BODY frame about the CG (z = -h_cg is the ground plane at rest): a lower
    tub between the wheels, an upper body over them, and (if tall enough) a cabin. Visual only — contact uses
    the chassis box."""
    L, W, H = spec["shape"]
    h = spec["h_cg"]
    ws = spec["wheels"]
    R = max(w["radius"] for w in ws)
    wheel_w = spec["wheel_size"][1]
    inner = max(min(abs(w["pos"][1]) for w in ws if w["wtype"] != 2) * 2 - wheel_w - 0.02, 0.4 * W)
    z_ground = -h
    parts = [((L * 0.92, inner, 0.55 * R), z_ground + 0.25 * R + 0.275 * R + 0.15 * R)]
    z_top_low = z_ground + 1.1 * (2 * R) if H > 3 * R else z_ground + 1.6 * R
    upper_h = max(0.18 * H, 0.04)
    parts.append(((L, W, upper_h), z_top_low + upper_h / 2))
    if H > 1.0:   # cabin
        parts.append(((0.55 * L, 0.85 * W, H - (z_top_low - z_ground) - upper_h), None))
    V, F = [], []
    zc = z_top_low + upper_h
    for (lx, ly, lz), zc_part in parts:
        if zc_part is None:
            zc_part = zc + lz / 2
        bv, bf = box_mesh(lx, ly, lz)
        k = len(V)
        V.extend((bv + [(-0.05 * L if zc_part > zc else 0.0), 0, zc_part]).tolist())
        F.extend([tuple(int(i) + k for i in f) for f in bf])
    return np.array(V), F


def box_mesh(lx, ly, lz):
    V = np.array([[x, y, z] for x in (-lx / 2, lx / 2) for y in (-ly / 2, ly / 2) for z in (-lz / 2, lz / 2)], float)
    F = np.array([(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    return V, F


def wheel_mesh(radius, width, n=20, spokes=True):
    """Cylinder along body +X (the renderer's cylinder convention; vehicles.py logs wheel attitude so +X = axle).
    Two contrasting spoke blocks make the spin visible."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    V = [[-width / 2, radius * math.cos(t), radius * math.sin(t)] for t in a] + \
        [[width / 2, radius * math.cos(t), radius * math.sin(t)] for t in a] + [[-width / 2, 0, 0], [width / 2, 0, 0]]
    F = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    F += [((i + 1) % n, i, 2 * n) for i in range(n)] + [(n + i, n + (i + 1) % n, 2 * n + 1) for i in range(n)]
    V = np.array(V, float)
    if spokes:   # a radial bar proud of each side face
        b = len(V)
        w, r = 0.02 * radius + 0.004, 0.85 * radius
        for sx in (-1, 1):
            x0 = sx * (width / 2 + 0.01 * radius + 0.002)
            k = len(V)
            V = np.vstack([V, [[x0, -w, -r], [x0, w, -r], [x0, w, r], [x0, -w, r]]])
            F.append((k, k + 1, k + 2, k + 3))
    return V, np.array(F, dtype=object)


def _save_mesh(path, V, F):
    F = [tuple(int(i) for i in f) for f in F]
    if len({len(f) for f in F}) > 1:          # mixed polygons -> fan triangles (renderer wants a regular F array)
        F = [(f[0], f[k], f[k + 1]) for f in F for k in range(1, len(f) - 1)]
    np.savez_compressed(path, V=np.asarray(V, np.float32), F=np.array(F, np.int32))


class TerrainDomain:
    """Static domain: the terrain itself (no state; logged once as a mesh body at the origin)."""
    step_every = 1 << 30

    def __init__(self, terrain, mesh_step=1):
        self.terrain, self.mesh_step = terrain, mesh_step

    def upload(self, device):
        pass

    def step(self, t, dt, atm, wind, device):
        pass

    def log_setup(self, log, run_dir):
        os.makedirs(os.path.join(run_dir, "mesh"), exist_ok=True)
        Vt, Ft, Fm = terrain_mesh(self.terrain, self.mesh_step)
        fn = os.path.join(run_dir, "mesh", "terrain.npz")
        _save_mesh(fn, Vt, Ft)
        z = dict(np.load(fn))
        from engineering_sim.core.terrain import SURFACE_NAMES
        np.savez_compressed(fn, face_surface=Fm, surfaces=np.array(SURFACE_NAMES), **z)
        log.add_body("terrain", kind="passive", shape="box", size=[1, 1, 1], mesh="mesh/terrain.npz", domain="land")

    def log_frame(self):
        return dict(body_x=np.zeros((1, 3), np.float32), body_v=np.zeros((1, 3), np.float32),
                    body_q=np.array([[1, 0, 0, 0]], np.float32))

    def log_meta(self):
        return dict(terrain=self.terrain.describe(), render_hooks=["land"])


def _vehicles_log_setup(self, log, run_dir):
    os.makedirs(os.path.join(run_dir, "mesh"), exist_ok=True)
    for j, row in enumerate(self.log_rows()):
        fn = "mesh/%s.npz" % row["name"]
        if row["shape"] == "wheel":
            _save_mesh(os.path.join(run_dir, fn), *wheel_mesh(*row["size"]))
        else:
            _save_mesh(os.path.join(run_dir, fn), *vehicle_visual_mesh(self.rows[j]))
        log.add_body(row["name"], kind=row["kind"], shape="box" if row["shape"] != "wheel" else "cylinder",
                     size=row["size"], mesh=fn, domain="land")


def _contact_log_setup(self, log, run_dir):
    os.makedirs(os.path.join(run_dir, "mesh"), exist_ok=True)
    for row in self.log_rows():
        fn = "mesh/%s.npz" % row["name"]
        _save_mesh(os.path.join(run_dir, fn), *box_mesh(*row["size"]))
        log.add_body(row["name"], kind="vehicle", shape="box", size=row["size"], mesh=fn, domain="land")


def _xvq_frame(self):
    x, v, q = self.frame()
    return dict(body_x=x, body_v=v, body_q=q)


def _vehicles_contact_bind(self, cm):
    """Cross-domain contact (core/contact.py, 2026-10-15): the chassis box (body-frame half extents + offset from the
    CG) of every vehicle collides with other domains' bodies; the ground stays with the wheels + chassis corners of
    the land model itself (terrain=False). Material per entity `contact.material` (default steel, ASSUMED)."""
    if not self.n:
        return
    prox, keep = [], []
    for k, r in enumerate(self.rows):
        c = r.get("contact") or {}
        if not c.get("enabled", True):
            continue
        prox.append(dict(body=len(keep), shape="box", size=list(r["half"]), offset=list(r["box_off"]),
                         material=c.get("material", "steel")))
        keep.append(k)
    if keep:
        cm.bind_rigid("land_vehicles", self.S.x, self.S.q, self.S.v, self.S.wb, keep,
                      [self.rows[k]["mass"] for k in keep], [self.rows[k]["inertia"] for k in keep], prox,
                      names=[self.rows[k]["name"] for k in keep], terrain=False, water=False)


def _contact_bodies_contact_bind(self, cm):
    """Landers / crates: their box (logged size) collides with other domains' bodies; their terrain contact stays
    with their own contact points (terrain=False)."""
    if not self.n:
        return
    prox, keep = [], []
    for k, r in enumerate(self.rows):
        c = r.get("contact") or {}
        if not c.get("enabled", True):
            continue
        prox.append(dict(body=len(keep), shape="box", size=[0.5 * s for s in r["size"]],
                         material=c.get("material", "composite")))
        keep.append(k)
    if keep:
        cm.bind_rigid("contact_bodies", self.x, self.q, self.v, self.wb, keep, [self.rows[k]["mass"] for k in keep],
                      [self.rows[k]["I"] for k in keep], prox, names=[self.rows[k]["name"] for k in keep],
                      terrain=False, water=False)


LandVehicles.log_setup, LandVehicles.log_frame = _vehicles_log_setup, _xvq_frame
ContactBodies.log_setup, ContactBodies.log_frame = _contact_log_setup, _xvq_frame
LandVehicles.contact_bind, ContactBodies.contact_bind = _vehicles_contact_bind, _contact_bodies_contact_bind


class LandScene(World):
    """World with the terrain as its first domain (land-only convenience; any World domain may be added)."""

    def __init__(self, atm, wind, terrain, domains, dt=1e-3, device=None, env_describe=None, terrain_step=1):
        self.terrain, self.env_describe = terrain, env_describe
        self.terrain_domain = TerrainDomain(terrain, terrain_step)
        super().__init__(atm, wind, [self.terrain_domain] + list(domains), dt=dt, device=device)

    def run(self, duration_s, run_dir, log_hz=30.0, meta=None, chunk_frames=30, series=None, terrain_step=None,
            progress=True, realtime=False, use_graph=True):
        if terrain_step is not None:
            self.terrain_domain.mesh_step = terrain_step
        meta = dict(meta or {}, physics_dt=self.dt)
        return super().run(duration_s, run_dir, log_hz=log_hz, chunk_frames=chunk_frames, meta=meta,
                           realtime=realtime, progress=progress, series=series, use_graph=use_graph)

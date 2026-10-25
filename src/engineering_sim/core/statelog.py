#!/usr/bin/env python3
# DESCRIPTION: Writes schema "unified_sim_statelog/1" so every sim renders through the same Blender renderer and
#              shows up on the same dashboards. core/world.World uses the same layout.
#
#              manifest.json keys: schema, created, log_hz, frames, chunks[], done, atmos, wind, meta,
#                bodies: [{name, kind, shape, size}]   kind: 0 passive, 1 guided, 2 tracer, 3 kinematic, 4 vehicle
#                        shape: "dart" | "octa" | "box" | "cylinder" | "boat" | "sphere"; size = metres
#                        (box: [lx,ly,lz]; cylinder: [radius, length] along body +X; others scalar)
#                        optional mesh: run-relative npz path with V[n,3] (metres, body frame about the CG) and F
#                        (true-scale geometry; the renderer does not inflate it)            (added 2026-10-15)
#                        optional color [r,g,b,a]; optional domain (owning domain tag, e.g. "land", used by render
#                        hooks); optional proxy (text: the DISPLAY position is not the true position, e.g. a
#                        satellite drawn along its true line of sight at a display range; stamped on every frame)
#                                                                                     (added 2026-10-15 UNIFY)
#                beams:  [{name, base, L, D, ...}]  (as written by core/beam.py)
#                series: [{name, unit}]             optional scalar channels shown in the render stamp
#                water:  {nx, dx, follow, sea{...}} optional free surface: a square nx*nx grid of spacing dx whose
#                        origin is logged per frame (follows a body)                         (added 2026-10-15)
#                        optional plane [x0,y0,x1,y1]: static still-water plane (z = 0) around the grid (a lake)
#                meta.render_hooks: [names] -> render/plugins/<name>_render.py post_build/post_frame hooks (UNIFY)
#              chunk arrays: t[F] (+ any of) body_x[F,Nb,3], body_v[F,Nb,3], body_q[F,Nb,4] (w,x,y,z body->world;
#                an all-zero quaternion = no attitude logged for that body -> oriented along its velocity),
#                beam_nodes[F,Nbeam,NS,3], beam_root_M[F,Nbeam], probe_wind[F,Np,3], series[F,Ns],
#                water_eta[F,nx*nx] (surface elevation, row-major y then x), water_origin[F,2] (grid corner x0,y0)
#              Readers of the original schema ignore the optional keys, so the schema id stays /1.
#
#   USAGE:     log = StateLog(run_dir, log_hz=30, env=env, meta={...})
#              log.add_body("rocket", kind=4, shape="cylinder", size=[0.04, 1.2])
#              log.frame(t, body_x=..., body_v=..., body_q=..., series=[alt, thrust])   # per frame
#              log.close()
import json
import os
import time

import numpy as np

SCHEMA = "unified_sim_statelog/1"
KINDS = dict(passive=0, guided=1, tracer=2, kinematic=3, vehicle=4)


class StateLog:
    def __init__(self, run_dir, log_hz, env=None, meta=None, chunk_frames=30):
        self.dir = run_dir
        os.makedirs(os.path.join(run_dir, "chunks"), exist_ok=True)
        self.m = dict(schema=SCHEMA, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), log_hz=float(log_hz), frames=0,
                      chunks=[], done=False, meta=meta or {}, bodies=[], series=[])
        if env is not None:
            self.m.update(env.describe())
        else:   # renderer stamp expects these keys
            self.m.update(atmos=dict(site_elev_m=0.0, dT_K=0.0, p0_pa=101325.0, rh=0.0),
                          wind=dict(u10=0.0, wind_dir_deg=0.0, sigma_uvw=[0, 0, 0], uniform=False, u_uniform=0.0))
        self.chunk_frames, self.buf = chunk_frames, []

    def add_body(self, name, kind="vehicle", shape="octa", size=1.0, mesh=None, V=None, F=None, color=None, **extra):
        """mesh: run-relative npz name; or pass V (body frame, metres, about the CG) and F to have it written.
        color: optional RGBA (renderer colour for this body); extra: optional row keys (e.g. domain="land",
        proxy="<how the display position differs from the true one>")."""
        row = dict(name=name, kind=KINDS.get(kind, kind), shape=shape, size=size)
        if color is not None:
            row["color"] = [float(c) for c in color]
        row.update(extra)
        if V is not None:
            mesh = mesh or "mesh_%s.npz" % name
            np.savez(os.path.join(self.dir, mesh), V=np.asarray(V, np.float32), F=np.asarray(F, np.int32))
        if mesh:
            row["mesh"] = mesh
        self.m["bodies"].append(row)
        return len(self.m["bodies"]) - 1

    def add_water(self, nx, dx, follow=None, sea=None, plane=None):
        self.m["water"] = dict(nx=int(nx), dx=float(dx), follow=follow, sea=sea or {})
        if plane is not None:
            self.m["water"]["plane"] = [float(c) for c in plane]

    def add_beams(self, rows):
        self.m["beams"] = rows

    def add_series(self, name, unit=""):
        self.m["series"].append(dict(name=name, unit=unit))

    def frame(self, t, **arrays):
        self.buf.append(dict(t=float(t), **{k: np.asarray(v, np.float32) for k, v in arrays.items()}))
        if len(self.buf) >= self.chunk_frames:
            self._flush()

    def _flush(self):
        if not self.buf:
            return
        name = "c%05d.npz" % len(self.m["chunks"])
        keys = [k for k in self.buf[0] if k != "t"]
        tmp = os.path.join(self.dir, "chunks", name + ".tmp.npz")
        np.savez(tmp, t=np.array([b["t"] for b in self.buf]), **{k: np.stack([b[k] for b in self.buf]) for k in keys})
        os.replace(tmp, os.path.join(self.dir, "chunks", name))
        self.m["chunks"].append(dict(file=name, frames=len(self.buf), t0=self.buf[0]["t"], t1=self.buf[-1]["t"]))
        self.m["frames"] += len(self.buf)
        self.buf = []
        self._write_manifest()

    def _write_manifest(self):
        tmp = os.path.join(self.dir, "manifest.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(self.m, fh, indent=1, default=float)
        os.replace(tmp, os.path.join(self.dir, "manifest.json"))

    def close(self, **summary):
        self._flush()
        self.m["done"] = True
        self.m["summary"] = summary
        self._write_manifest()
        return self.m


def quat_from_euler(roll, pitch, yaw):
    """ZYX (yaw-pitch-roll) Euler [rad] -> quaternion (w,x,y,z), body->world."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])

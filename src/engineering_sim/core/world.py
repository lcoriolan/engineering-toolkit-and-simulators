#!/usr/bin/env python3
#              CONTACT: optional cross-domain contact pass after every step)
# DESCRIPTION: Owns the shared environment (atmos + wind) and every domain of a scenario. Each step advances ALL
#              domains on the same clock on the physics GPU (env SIMGPU_PHYSICS_DEVICE). Without a contact manager,
#              domains are coupled ONLY through the shared environment (air, wind, terrain, sea), so the stepping order
#              between domains does not change any result (and the CUDA-graph group below is used).
#              With World(..., contact=ContactManager) (core/contact.py; scenario `contact:` block), every domain that
#              implements contact_bind(cm) — plus core Bodies and Beams, adapted here — is bound once after upload,
#              and ONE contact pass runs after every world step (all domains advanced to t+dt, then contacts resolved
#              and written back; plain stepping — the graph group is not used because contact interleaves every step).
#              Impact/splash events go to manifest["events"]; the pass is deterministic (sorted contact order).
#
#              DOMAIN PROTOCOL (duck-typed; core Bodies / Beams are adapted here, so they need nothing extra):
#                upload(device)                          required: allocate on the physics device
#                step(t, dt, atm, wind, device)          required: advance [t, t + dt)
#                step_every  (int, optional)             multirate: stepped once every k world steps with dt*k
#                                                        (CPU flight software at its loop rate, slow propagators)
#                step_rel(t_off, dt, atm, wind, device)  optional: device-clock step (reads self.tarr[0] + t_off) —
#                + attribute tarr (set by the World)     every such domain is advanced inside ONE CUDA graph per
#                                                        log frame (identical results, no per-launch Python cost)
#                log_setup(log, run_dir)                 optional: register bodies (log.add_body, meshes), beams,
#                                                        water and its series (log.add_series) with the StateLog
#                log_frame() -> dict                     optional: body_x/body_v[/body_q] for ITS bodies (in the order
#                                                        it registered them) + any other per-frame arrays
#                                                        (beam_nodes, water_eta, ...) + 'series' list for its series
#                log_meta() -> dict                      optional: merged into manifest.meta (render_hooks are unioned)
#                summary() -> dict                       optional: end-of-run results (manifest.summary.domains)
#                contact_bind(cm)                        optional: register bodies + collision proxies with the
#                                                        contact manager (protocol: core/contact.py header)
#
#              STATE LOG: core/statelog.py (schema "unified_sim_statelog/1") — the ONLY interface to renderers and
#              dashboards; manifest rewritten atomically per chunk so a near-live renderer can follow a running sim.
#              body_q all-zero for a body = "no attitude logged" (renderer orients it along its velocity).
import json
import os
import time

import numpy as np
import warp as wp

from . import atmos as A
from . import wind as W
from . import beam as B
from . import physics_device
from .statelog import StateLog, SCHEMA

__all__ = ["World", "load_run", "SCHEMA"]


from ..execution import operation as _simulation_operation

@wp.kernel
def _probe(p: wp.array(dtype=wp.vec3), t: float, w: W.WindParams, mk: wp.array(dtype=wp.vec3),
           ma: wp.array(dtype=float), mph: wp.array(dtype=float), ms: wp.array(dtype=wp.vec3),
           out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = W.wind_at(p[i], t, w, mk, ma, mph, ms)


def _is_bodies(d):
    return hasattr(d, "kind_np") and hasattr(d, "x") and not hasattr(d, "log_frame")


def _is_beams(d):
    return isinstance(d, B.Beams) and not hasattr(d, "log_frame")


class World:
    def __init__(self, atmos_params, wind, domains, dt=1e-3, device=None, probes=(), contact=None):
        self.device = physics_device(device)
        self.contact = contact
        self.atm, self.wind, self.domains, self.dt = atmos_params, wind, list(domains), float(dt)
        self.t, self.k = 0.0, 0
        self._tp = 0.0                    # plain-stepping clock (dt accumulated per step, exactly as v0.1 World.step)
        self.tarr = wp.zeros(1, dtype=float, device=self.device)     # ONE device clock for graph-captured domains
        self.probes_np = np.asarray(probes, np.float32).reshape(-1, 3)
        for d in self.domains:
            if hasattr(d, "step_rel"):
                d.tarr = self.tarr
            try:
                d._world = self               # read-only back-reference (clock) for loggers such as a sea surface
            except AttributeError:
                pass
            d.upload(self.device)
        self.every = [max(1, int(getattr(d, "step_every", 1) or 1)) for d in self.domains]
        self.probe_p = wp.array(self.probes_np if len(self.probes_np) else np.zeros((1, 3), np.float32),
                                dtype=wp.vec3, device=self.device)
        self.probe_out = wp.zeros(max(1, len(self.probes_np)), dtype=wp.vec3, device=self.device)
        self._graph, self._graph_n = None, 0
        self._is_cuda = wp.get_device(self.device).is_cuda
        if contact is not None:
            for d in self.domains:
                if hasattr(d, "contact_bind"):
                    d.contact_bind(contact)
                elif _is_beams(d) and d.n:
                    _bind_beams(contact, d)
            contact.finalize(self)

    # ------------------------------------------------------------------------------------------------ stepping
    def _step_domains(self, idx, t, k):
        for i in idx:
            e = self.every[i]
            if k % e == 0:
                self.domains[i].step(t, self.dt * e, self.atm, self.wind, self.device)

    def step(self):
        self._step_domains(range(len(self.domains)), self._tp, self.k)
        self._tp += self.dt
        self.t = self._tp
        self.k += 1
        if self.contact is not None:
            self.contact.step(self.t, self.dt)

    def advance(self, n_steps, use_graph=True):
        """n_steps of dt. Graph-capable domains (step_rel, step_every == 1) run as ONE CUDA graph launch per block on
        CUDA (their clock: self.t, block-wise as the v0.1 LandScene); every other domain is stepped plainly over the
        same interval on the plain clock self._tp (dt accumulated per step, as the v0.1 World). Domains are
        decoupled, so a domain's results do not depend on which other domains share the World."""
        G = [i for i, d in enumerate(self.domains) if hasattr(d, "step_rel") and self.every[i] == 1] \
            if (use_graph and self._is_cuda and self.contact is None) else []
        O = [i for i in range(len(self.domains)) if i not in G]
        for d in self.domains:            # sub-rate schedules (e.g. wind sampling) restart every block in both modes
            if hasattr(d, "_sub"):
                d._sub = 0
        if not G:
            for _ in range(n_steps):
                self.step()
            return
        k0 = self.k
        if self._graph is None or self._graph_n != n_steps:
            # first block: one plain step (loads modules), capture the n-step graph (capture does not execute), then
            # finish this block with plain steps -> exactly n_steps either way
            self._step_domains(G, self.t, self.k)
            self.t += self.dt
            saved = [getattr(self.domains[i], "_sub", None) for i in G]
            for i in G:
                if hasattr(self.domains[i], "_sub"):
                    self.domains[i]._sub = 0
            with wp.ScopedCapture(device=self.device) as cap:
                for s in range(n_steps):
                    for i in G:
                        self.domains[i].step_rel(s * self.dt, self.dt, self.atm, self.wind, self.device)
            for i, sv in zip(G, saved):
                if sv is not None:
                    self.domains[i]._sub = sv
            self._graph, self._graph_n = cap.graph, n_steps
            for _ in range(n_steps - 1):
                self._step_domains(G, self.t, self.k)
                self.t += self.dt
        else:
            self.tarr.fill_(self.t)
            wp.capture_launch(self._graph)
            self.t += n_steps * self.dt
        for j in range(n_steps):          # everything not in the graph: same interval on the plain clock, so a
            self._step_domains(O, self._tp, k0 + j)   # domain sees identical times whether or not a graph group exists
            self._tp += self.dt
        self.k = k0 + n_steps

    # ------------------------------------------------------------------------------------------------ logging
    def _setup_log(self, log, run_dir):
        self.slices = []
        for d in self.domains:
            n0 = len(log.m["bodies"])
            if hasattr(d, "log_setup"):
                d.log_setup(log, run_dir)
            elif _is_bodies(d):           # v0.1 row format kept (renderer picks dart/octa glyphs by kind)
                log.m["bodies"] += [dict(name=n, kind=int(k)) for n, k in zip(d.names, d.kind_np)]
            elif _is_beams(d):
                rows = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()
                         if k not in ("M", "K", "C")} for r in d.rows]
                log.m["beams"] = log.m.get("beams", []) + rows
            self.slices.append((n0, len(log.m["bodies"])))
            if hasattr(d, "log_meta"):
                for k, v in (d.log_meta() or {}).items():
                    if k == "render_hooks":
                        log.m["meta"]["render_hooks"] = sorted(set(log.m["meta"].get("render_hooks", [])) | set(v))
                    else:
                        log.m["meta"][k] = v

    def snapshot(self):
        snap = dict(t=self.t)
        X, Vv, Q, anyq, ser = [], [], [], False, []
        for d, (n0, n1) in zip(self.domains, getattr(self, "slices", [(0, 0)] * len(self.domains))):
            fr = {}
            if hasattr(d, "log_frame"):
                fr = dict(d.log_frame() or {})
            elif _is_bodies(d):
                fr = dict(body_x=d.x.numpy().copy(), body_v=d.v.numpy().copy())
            elif _is_beams(d) and d.n:
                fr = dict(beam_nodes=d.nodes.numpy().copy(), beam_root_M=d.root_moment())
            if "body_x" in fr:
                X.append(np.asarray(fr.pop("body_x"), np.float32))
                Vv.append(np.asarray(fr.pop("body_v", np.zeros_like(X[-1])), np.float32))
                q = fr.pop("body_q", None)
                anyq |= q is not None
                Q.append(np.asarray(q, np.float32) if q is not None else np.zeros((len(X[-1]), 4), np.float32))
            if "series" in fr:
                ser += list(np.atleast_1d(fr.pop("series")))
            for k, v in fr.items():
                if k in snap:
                    raise ValueError("two domains log the same array %r" % k)
                snap[k] = v
        if X:
            snap["body_x"], snap["body_v"] = np.concatenate(X), np.concatenate(Vv)
            if anyq:
                snap["body_q"] = np.concatenate(Q)
        for s in getattr(self, "_series_fns", []):
            ser.append(s[2](self))
        if ser:
            snap["series"] = np.asarray(ser, np.float32)
        if len(self.probes_np):
            wp.launch(_probe, len(self.probes_np), [self.probe_p, float(self.t)] + self.wind.args() + [self.probe_out],
                      device=self.device)
            snap["probe_wind"] = self.probe_out.numpy().copy()
        return snap

    @_simulation_operation('world.run', kind='transient')
    def run(self, duration_s, run_dir, log_hz=30.0, chunk_frames=30, meta=None, realtime=False, progress=True,
            series=None, use_graph=True, env_describe=None):
        """series: [(name, unit, fn(world) -> float)] world-level channels (after the domains' own series)."""
        os.makedirs(os.path.join(run_dir, "chunks"), exist_ok=True)
        spf = max(1, int(round(1.0 / (log_hz * self.dt))))
        frame_dt = spf * self.dt
        n_frames = int(round(duration_s / frame_dt))
        log = StateLog(run_dir, 1.0 / frame_dt, meta=dict(meta or {}), chunk_frames=chunk_frames)
        log.m.update(atmos=_atm_dict(self.atm), wind=self.wind.describe())
        if env_describe:
            log.m.update(env_describe)
        log.m.update(physics_dt=self.dt, steps_per_frame=spf, device=str(wp.get_device(self.device)),
                     probes=self.probes_np.tolist(), log_hz_requested=log_hz)
        self._setup_log(log, run_dir)
        self._series_fns = list(series or [])
        for s in self._series_fns:
            log.add_series(s[0], s[1])
        t_wall0 = time.time()

        if self.contact is not None:
            log.m["contact"] = self.contact.describe()
            log.m["events"] = self.contact.events          # same list object: grows as the run goes (near-live)
            log.m["meta"]["render_hooks"] = sorted(set(log.m["meta"].get("render_hooks", [])) | {"contact"})

        def emit():
            s = self.snapshot()
            log.frame(s.pop("t"), **s)

        emit()
        for f in range(1, n_frames + 1):
            self.advance(spf, use_graph=use_graph)
            emit()
            if progress and (f % int(max(1, 5 * log_hz)) == 0 or f == n_frames):
                print("t=%.2fs frames=%d wall=%.1fs" % (self.t, f + 1, time.time() - t_wall0), flush=True)
            if realtime:
                lag = self.t - (time.time() - t_wall0)
                if lag > 0:
                    time.sleep(lag)
        wp.synchronize_device(self.device)
        wall = time.time() - t_wall0
        summ = dict(wall_s=round(wall, 3), realtime_factor=round(self.t / max(wall, 1e-9), 2))
        doms = {}
        for i, d in enumerate(self.domains):
            if hasattr(d, "summary"):
                doms["%d_%s" % (i, type(d).__name__)] = d.summary()
        if doms:
            summ["domains"] = doms
        if self.contact is not None:
            summ["contact"] = self.contact.summary()
        log.m.update(wall_s=summ["wall_s"], realtime_factor=summ["realtime_factor"])
        return log.close(**summ)


def _bind_beams(cm, beams):
    """Masts as kinematic (infinite-mass) capsules base -> tip (undeflected; tip deflection <= a few % of L is
    ignored for contact — ASSUMED). Things hit masts; masts do not respond."""
    rows = beams.rows
    ctr = np.array([np.asarray(r["base"], float) + [0.0, 0.0, 0.5 * r["L"]] for r in rows])
    s2 = 0.5 ** 0.5
    prox = [dict(body=k, shape="capsule", size=[0.5 * r["D"], 0.5 * r["L"]], quat_wxyz=(s2, 0.0, -s2, 0.0),
                 material="aluminium", terrain=False) for k, r in enumerate(rows)]
    cm.bind_host("beams", len(rows), lambda: dict(x=ctr, v=np.zeros_like(ctr)), lambda *a: None, None, prox,
                 names=[r.get("name", "mast%d" % k) for k, r in enumerate(rows)], terrain=False, kinematic=True)


def _atm_dict(a):
    return dict(site_elev_m=a.site_elev_m, dT_K=a.dT_K, p0_pa=a.p0_pa, rh=a.rh)


def load_run(run_dir):
    """Concatenate all chunks -> (manifest, dict of arrays)."""
    m = json.load(open(os.path.join(run_dir, "manifest.json"), encoding="utf-8"))
    parts = [np.load(os.path.join(run_dir, "chunks", c["file"])) for c in m["chunks"]]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files} if parts else {}
    return m, out

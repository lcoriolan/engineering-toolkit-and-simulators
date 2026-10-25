#!/usr/bin/env python3
# DESCRIPTION: RocketDesignDomain flies the entity's design (inline `rocket:` spec or `spec_file:` YAML of schema
#              rocket_design/1) in 6-DOF ONCE at build time, in the scenario's own core air (site atmos + wind of the
#              World's Env), from `pad` (ground height added), and then replays the logged trajectory on the World
#              clock (sustainer + every spent stage as its own body). Honest limit: it is not coupled to other
#              domains while flying (no contact with other entities; the 6-DOF and its events are exact, the replay is
#              interpolated at the log rate, 20 Hz). Entity keys: rocket | spec_file, pad [x, y], t_ignite, rail, fc.
import math

import numpy as np


class RocketDesignDomain:
    own_dt = 0.05

    def __init__(self, e, ctx):
        from . import rules as RU
        from .airframe import Rocket
        from .flight import fly, AirTable
        from .motors import load
        self.e = e
        self.step_every = max(1, int(round(self.own_dt / ctx.dt)))
        spec = e.get("rocket")
        if spec is None:
            import yaml
            doc = yaml.safe_load(open(e["spec_file"]))
            spec = doc["rocket"]
            e.setdefault("rail", doc.get("rail"))
        env = dict(atmos=dict(ctx.env.atmos_cfg), wind={k: v for k, v in ctx.env.wind_cfg.items()})
        db = load()
        air = AirTable(env)
        self.rocket = Rocket(RU.derive(spec, db, float(air.tab[0, 0])), db)
        pad = e.get("pad", [0.0, 0.0])
        self.pad = np.array([pad[0], pad[1], ctx.ground(pad[0], pad[1])], float)
        self.t0 = float(e.get("t_ignite", 0.0))
        self.res = fly(self.rocket, env=env, rail=e.get("rail"), fc=e.get("fc"))
        self.tr = self.res["traj"]
        self.ix = {c: i for i, c in enumerate(self.res["cols"])}
        self.t = 0.0

    def upload(self, device):
        pass

    def step(self, t, dt, atm, wind, device):
        self.t = t

    def _interp(self, arr, tt, cols):
        if len(arr) == 0:
            return np.zeros(len(cols))
        return np.array([np.interp(tt, arr[:, 0], arr[:, c]) for c in cols])

    def log_setup(self, log, run_dir):
        st = self.rocket.stages
        log.add_body(self.e["name"], kind="vehicle", shape="cylinder",
                     size=[0.5 * st[0].d * 3, st[0].x_aft], color=self.e.get("color"))
        for s in st[1:]:
            log.add_body("%s %s" % (self.e["name"], s.name), kind="vehicle", shape="cylinder",
                         size=[0.5 * s.d * 3, s.x_aft - s.x_top])
        for s, u in (("alt", "m"), ("mach", "")):
            log.add_series("%s %s" % (self.e["name"], s), u)

    def log_frame(self):
        tt = self.t - self.t0
        ix = self.ix
        x = self._interp(self.tr, tt, [ix["x"], ix["y"], ix["z"]]) + self.pad
        v = self._interp(self.tr, tt, [ix["vx"], ix["vy"], ix["vz"]])
        q = self._interp(self.tr, tt, [ix["qw"], ix["qx"], ix["qy"], ix["qz"]])
        q = q / max(np.linalg.norm(q), 1e-9)
        xs, vs, qs = [x], [v], [q]
        for s in self.rocket.stages[1:]:
            trk = self.res["spent_tracks"].get(s.name)
            if trk is None or len(trk) == 0 or tt < trk[0, 0]:
                xs.append(x)          # still attached: ride with the stack
            else:
                xs.append(self._interp(trk, tt, [1, 2, 3]) + self.pad)
            vs.append(v)
            qs.append(q)
        M = float(np.interp(tt, self.tr[:, 0], self.tr[:, ix["mach"]])) if len(self.tr) else 0.0
        return dict(body_x=xs, body_v=vs, body_q=qs, series=[float(x[2] - self.pad[2]), M])

    def summary(self):
        s = self.res["summary"]
        return dict(apogee_agl_m=s["apogee_agl_m"], max_mach=s["max_mach"]["M"], events=s["events"],
                    landing_xy=[s["landing_xy_m"][0] + self.pad[0], s["landing_xy_m"][1] + self.pad[1]],
                    spent=s["spent"], note="flown in 6-DOF at build time in the scenario air; replayed on the clock")

#!/usr/bin/env python3
# DESCRIPTION: No dynamics: each logged frame evaluates the SAME SeaState the vessels feel on an nx*nx grid of spacing
#              dx (GPU) and logs water_eta / water_origin. The grid either follows a vessel (origin placed so the
#              vessel sits `lead` of the way along x and centred in y — the PWC mission D-phase layout) or stays at a
#              fixed origin. plane = [x0, y0, x1, y1] adds a static still-water plane (a lake) around the grid.
#              Still water level is world z = 0 (vessels and the renderer assume it).
import numpy as np
import warp as wp


class SeaSurface:
    step_every = 1 << 30            # nothing to integrate

    def __init__(self, sea, nx=140, dx=0.6, follow=None, origin=(0.0, 0.0), lead=0.35, plane=None):
        """follow: (Vessels, vessel index) or None (fixed origin)."""
        self.sea, self.nx, self.dx, self.follow, self.origin = sea, int(nx), float(dx), follow, origin
        self.lead, self.plane = lead, plane

    def upload(self, device):
        self.grid = wp.zeros(self.nx * self.nx, dtype=float, device=self.sea.device)

    def step(self, t, dt, atm, wind, device):
        pass

    def log_setup(self, log, run_dir):
        fol = None
        if self.follow is not None:
            v, vi = self.follow
            fol = v.specs[vi]["name"]
        log.add_water(self.nx, self.dx, follow=fol, sea=self.sea.describe(), plane=self.plane)

    def log_frame(self):
        t = self._world._tp              # the vessels' clock (plain stepping)
        if self.follow is not None:
            v, vi = self.follow
            p = v.rb.x.numpy()[v.specs[vi]["bi"]]
            x0, y0 = p[0] - self.nx * self.dx * self.lead, p[1] - self.nx * self.dx * 0.5
        else:
            x0, y0 = self.origin
        self.sea.surface(x0, y0, self.dx, self.nx, t, out=self.grid)
        return dict(water_eta=self.grid.numpy(), water_origin=np.array([x0, y0]))

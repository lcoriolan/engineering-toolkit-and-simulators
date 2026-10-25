#!/usr/bin/env python3
#              to model everything from it"
# DESCRIPTION: Broadband ENERGY room acoustics for a rectangular (shoebox) room, sound speed from the unified core's
#              site air. Two models, both standard:
#                diffuse field (Sabine):  T60 = 24 ln10 V / (c A),  A = sum S_i alpha_i  (Kuttruff, Room Acoustics, 5th
#                                         ed., eq. 5.33); critical distance r_c = sqrt(Q A / (16 pi)) (Kuttruff 5.43);
#                                         DRR(d) = (r_c / d)^2 — the law legacy sensor's pdrr.drr_from_range uses.
#                image sources (Allen & Berkley 1979, energy form): every image of the source in the 6 walls carries
#                                         energy prod(1 - alpha_wall)^(hits) / r^2; the direct path is 1/d^2.
#                                         DRR = E_direct / E_reflected, images summed out to c * T60 (60 dB decay).
#              Walls: alpha per wall (x0, x1, y0, y1, z0, z1), random-incidence energy absorption, frequency-flat
#              (ASSUMED broadband). Air absorption inside the room is neglected (rooms << 1 km). ALL RESULTS MODELED.
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "env_bridge"))
from engineering_sim.core import air_np as AIR


class ShoeboxRoom:
    def __init__(self, Lx, Ly, Lz, alpha, c=None):
        self.L = np.array([Lx, Ly, Lz], float)
        a = np.broadcast_to(np.asarray(alpha, float), (6,)).copy()
        assert np.all((a > 0) & (a <= 1)), "wall absorption must be in (0, 1]"
        self.alpha = a                                     # x0, x1, y0, y1, z0, z1
        self.c = float(AIR.ref_state(0.0)["c"]) if c is None else float(c)

    @property
    def volume(self):
        return float(np.prod(self.L))

    def absorption_area(self):
        Lx, Ly, Lz = self.L
        S = np.array([Ly * Lz, Ly * Lz, Lx * Lz, Lx * Lz, Lx * Ly, Lx * Ly])
        return float(np.sum(S * self.alpha))

    def t60_sabine(self):
        return 24.0 * math.log(10.0) * self.volume / (self.c * self.absorption_area())

    def critical_distance(self, Q=1.0):
        return math.sqrt(Q * self.absorption_area() / (16.0 * math.pi))

    def drr_diffuse(self, d, Q=1.0):
        return (self.critical_distance(Q) / max(float(d), 1e-9)) ** 2

    def drr_image(self, src, rcv, t_max=None):
        """Direct-to-reverberant energy ratio from the image-source energy sum (images to c * t_max)."""
        src, rcv = np.asarray(src, float), np.asarray(rcv, float)
        R = self.c * (t_max if t_max is not None else self.t60_sabine())
        e_dir = 1.0 / max(float(np.sum((src - rcv) ** 2)), 1e-12)
        e_ref = 0.0
        n = [int(math.ceil(R / (2.0 * self.L[k]))) + 1 for k in range(3)]
        grids = []
        for k in range(3):
            # Allen & Berkley 1979: image coordinate 2 m L + (1 - 2q) s, q in {0, 1}; it has |m - q| hits on the
            # wall at 0 and |m| hits on the wall at L.
            m = np.arange(-n[k], n[k] + 1)
            pos = np.concatenate([2.0 * m * self.L[k] + src[k], 2.0 * m * self.L[k] - src[k]])
            h0 = np.concatenate([np.abs(m), np.abs(m - 1)]).astype(float)
            h1 = np.concatenate([np.abs(m), np.abs(m)]).astype(float)
            grids.append((pos, h0, h1))
        (px, ax0, ax1), (py, ay0, ay1), (pz, az0, az1) = grids
        ga = np.log(1.0 - self.alpha)                      # log energy factor per hit, per wall
        lx = ax0 * ga[0] + ax1 * ga[1]
        ly = ay0 * ga[2] + ay1 * ga[3]
        lz = az0 * ga[4] + az1 * ga[5]
        dx2 = (px - rcv[0]) ** 2
        dy2 = (py - rcv[1]) ** 2
        for iz in range(len(pz)):
            d2 = dx2[:, None] + dy2[None, :] + (pz[iz] - rcv[2]) ** 2
            w = np.exp(lx[:, None] + ly[None, :] + lz[iz])
            ok = (d2 <= R * R) & (d2 > 0)
            e_ref += float(np.sum(w[ok] / d2[ok]))
        # the direct path is the even image with m = 0 on every axis (zero hits): remove it from the reflected sum
        e_ref -= e_dir
        return e_dir / max(e_ref, 1e-30)


def selftest():
    ok, msgs = True, []
    r = ShoeboxRoom(20.0, 20.0, 4.0, 0.20, c=343.0)
    t60 = r.t60_sabine()
    t60_hand = 0.161 * r.volume / r.absorption_area()        # the classic 0.161 V/A at c = 343 m/s
    ok &= abs(t60 - t60_hand) / t60_hand < 0.002
    msgs.append("Sabine T60 %.3f s vs 0.161 V/A %.3f s" % (t60, t60_hand))
    rc = r.critical_distance()
    ok &= abs(r.drr_diffuse(rc) - 1.0) < 1e-12
    msgs.append("DRR(r_c = %.2f m) = 1 (diffuse law)" % rc)
    # image-source: in a large, fairly absorptive room the ISM DRR at mid distances follows the diffuse law within
    # a factor ~2 (Kuttruff: the diffuse assumption is approximate in flat rooms); and DRR falls monotonically with d
    src = np.array([10.0, 10.0, 1.5])
    ds = [1.0, 2.0, 4.0, 8.0]
    drr = [r.drr_image(src, src + np.array([d, 0.0, 0.0])) for d in ds]
    ok &= all(a > b for a, b in zip(drr, drr[1:]))
    ratio = drr[1] / r.drr_diffuse(2.0)
    ok &= 0.3 < ratio < 3.0
    msgs.append("ISM DRR at 1/2/4/8 m = %s (monotone); ISM/diffuse at 2 m = %.2f" % ([round(x, 3) for x in drr], ratio))
    # hard walls -> reflected energy explodes -> DRR tiny; anechoic-ish walls -> DRR large
    hard = ShoeboxRoom(20.0, 20.0, 4.0, 0.02, c=343.0).drr_image(src, src + [2.0, 0, 0], t_max=0.5)
    soft = ShoeboxRoom(20.0, 20.0, 4.0, 0.95, c=343.0).drr_image(src, src + [2.0, 0, 0])
    ok &= soft > drr[1] > hard
    msgs.append("absorption ordering: DRR soft %.2f > alpha 0.2 %.3f > hard %.4f" % (soft, drr[1], hard))
    print("ROOM SELFTEST " + "; ".join(msgs) + " -> " + ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)

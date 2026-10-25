# DESCRIPTION: One GPU thread per vehicle, fixed-step RK4, the same force terms as forces.py (host reference).
#
#              Why float64: |r| ~ 7e6 m, so float32 resolves only ~0.5 m and its velocity increments (~1e-3 m/s per
#              10 s step at 7.6 km/s) lose ~4 significant digits per step; a float32 RK4 drifts km/day. float64 keeps
#              integration error at the RK4 truncation level. The P4's float64 rate is 1/32 of float32, which still
#              propagates ~1e4 vehicles x 1 day @ 10 s in seconds (benchmark in proof).
#              Sun/Moon positions are evaluated on the host (float64, analytic) at every RK4 stage time and uploaded
#              once per chunk; the density is the ONE air model (density.AirColumn) sampled as a log table.
#
#              Thermosphere (thermosphere.Thermosphere, SPACE2 2026-10-15): when `air` has table3d(), drag above the
#              table base uses a 3-D log(rho) table over (geodetic alt, local solar time, geodetic lat) rebuilt every
#              air.refresh_s of sim time (chunks are cut at refresh boundaries); below the base the 1-D column table
#              is used as before. With a plain AirColumn the kernel path is unchanged (flag bit 128 off).
#
#              USAGE:  prop = OrbitBatch(r0[N,3], v0[N,3], jd0, cd_am[N], cr_am[N], air=AirColumn(env))
#                      t, r_hist, v_hist = prop.run(h=10.0, n_steps=8640, every=6)
import math

import numpy as np
import warp as wp

from engineering_sim.core import physics_device
from .constants import MU_EARTH, R_GRAV, J2, J3, J4, MU_SUN, MU_MOON, AU, P_SRP_1AU, OMEGA_EARTH, WGS84_A, WGS84_F
from .ephem import sun_eci, moon_eci

F64 = wp.float64
_MU = wp.constant(F64(MU_EARTH))
_R = wp.constant(F64(R_GRAV))
_J2 = wp.constant(F64(J2))
_J3 = wp.constant(F64(J3))
_J4 = wp.constant(F64(J4))
_MUS = wp.constant(F64(MU_SUN))
_MUM = wp.constant(F64(MU_MOON))
_AU = wp.constant(F64(AU))
_PSRP = wp.constant(F64(P_SRP_1AU))
_WE = wp.constant(F64(OMEGA_EARTH))
_A = wp.constant(F64(WGS84_A))
_F = wp.constant(F64(WGS84_F))

FLAG = dict(j2=1, j3=2, j4=4, sun=8, moon=16, srp=32, drag=64)
FLAG_RHO3 = 128             # drag density from the 3-D thermosphere table above r3.h0
H_DECAY_M = 80e3            # vehicle flagged decayed (state frozen) below this altitude


@wp.func
def _third(r: wp.vec3d, s: wp.vec3d, mu: F64):
    d = s - r
    dn = wp.length(d)
    sn = wp.length(s)
    return mu * (d / (dn * dn * dn) - s / (sn * sn * sn))


@wp.func
def _alt(r: wp.vec3d):
    rn = wp.length(r)
    s = r[2] / rn
    return rn - _A * (F64(1.0) - _F * s * s)


@wp.func
def _rho(h: F64, logrho: wp.array(dtype=F64), dh: F64):
    x = wp.max(h, F64(0.0)) / dh
    n = logrho.shape[0]
    i = int(x)
    if i >= n - 1:
        # beyond the table: continue the last band's log slope
        g = (logrho[n - 1] - logrho[n - 2])
        return wp.exp(logrho[n - 1] + g * (x - F64(n - 1)))
    w = x - F64(i)
    return wp.exp(logrho[i] * (F64(1.0) - w) + logrho[i + 1] * w)


@wp.struct
class Rho3:
    tab: wp.array3d(dtype=F64)   # log(rho) [nh, n_lst, n_lat]
    h0: F64
    dh: F64
    dlst: F64
    lat0: F64
    dlat: F64


@wp.func
def _rho3(h: F64, lst: F64, lat: F64, r3: Rho3):
    """Trilinear in log(rho); same formula and order as thermosphere.Thermosphere.interp (host)."""
    nh = r3.tab.shape[0]
    nl = r3.tab.shape[1]
    na = r3.tab.shape[2]
    x = wp.max((h - r3.h0) / r3.dh, F64(0.0))
    i = wp.min(int(wp.floor(x)), nh - 2)
    wx = x - F64(i)
    y = (lst - F64(24.0) * wp.floor(lst / F64(24.0))) / r3.dlst
    jf = wp.floor(y)
    wy = y - jf
    j = int(jf) % nl
    j1 = (j + 1) % nl
    z = wp.min(wp.max((lat - r3.lat0) / r3.dlat, F64(0.0)), F64(na - 1))
    k = wp.min(int(wp.floor(z)), na - 2)
    wz = z - F64(k)
    one = F64(1.0)
    a0 = (one - wz) * r3.tab[i, j, k] + wz * r3.tab[i, j, k + 1]
    a1 = (one - wz) * r3.tab[i, j1, k] + wz * r3.tab[i, j1, k + 1]
    b0 = (one - wz) * r3.tab[i + 1, j, k] + wz * r3.tab[i + 1, j, k + 1]
    b1 = (one - wz) * r3.tab[i + 1, j1, k] + wz * r3.tab[i + 1, j1, k + 1]
    lo = (one - wy) * a0 + wy * a1
    hi = (one - wy) * b0 + wy * b1
    return wp.exp((one - wx) * lo + wx * hi)


@wp.func
def _lat_gd(r: wp.vec3d):
    e2 = _F * (F64(2.0) - _F)
    p = wp.sqrt(r[0] * r[0] + r[1] * r[1])
    return wp.atan2(r[2], p * (F64(1.0) - e2)) * F64(57.29577951308232)


@wp.func
def accel(r: wp.vec3d, v: wp.vec3d, sun: wp.vec3d, moon: wp.vec3d, cd_am: F64, cr_am: F64,
          logrho: wp.array(dtype=F64), dh: F64, flags: int, r3: Rho3, lst_off: F64):
    x = r[0]
    y = r[1]
    z = r[2]
    r2 = wp.dot(r, r)
    rn = wp.sqrt(r2)
    a = -_MU * r / (rn * r2)
    zr2 = z * z / r2
    one = F64(1.0)
    if (flags & 1) != 0:
        k = F64(-1.5) * _J2 * _MU * _R * _R / (r2 * r2 * rn)
        a = a + wp.vec3d(k * x * (one - F64(5.0) * zr2), k * y * (one - F64(5.0) * zr2), k * z * (F64(3.0) - F64(5.0) * zr2))
    if (flags & 2) != 0:
        k = F64(-2.5) * _J3 * _MU * _R * _R * _R / (r2 * r2 * r2 * rn)
        f = F64(3.0) * z - F64(7.0) * z * z * z / r2
        a = a + wp.vec3d(k * x * f, k * y * f, k * (F64(6.0) * z * z - F64(7.0) * z * z * z * z / r2 - F64(0.6) * r2))
    if (flags & 4) != 0:
        k = F64(1.875) * _J4 * _MU * _R * _R * _R * _R / (r2 * r2 * r2 * rn)
        f = one - F64(14.0) * zr2 + F64(21.0) * zr2 * zr2
        a = a + wp.vec3d(k * x * f, k * y * f, k * z * (F64(5.0) - F64(70.0) / F64(3.0) * zr2 + F64(21.0) * zr2 * zr2))
    if (flags & 8) != 0:
        a = a + _third(r, sun, _MUS)
    if (flags & 16) != 0:
        a = a + _third(r, moon, _MUM)
    if (flags & 32) != 0:
        u = wp.normalize(sun)
        proj = wp.dot(r, u)
        perp = wp.length(r - proj * u)
        lit = one
        if proj < F64(0.0) and perp < _A:
            lit = F64(0.0)
        d = r - sun
        dn = wp.length(d)
        a = a + lit * _PSRP * cr_am * (_AU / dn) * (_AU / dn) * d / dn
    if (flags & 64) != 0:
        vr = v - wp.cross(wp.vec3d(F64(0.0), F64(0.0), _WE), r)
        hh = _alt(r)
        rho = F64(0.0)
        if (flags & 128) != 0 and hh >= r3.h0:
            lst = lst_off + wp.atan2(r[1], r[0]) * F64(3.819718634205488)     # rad -> h (12/pi)
            rho = _rho3(hh, lst, _lat_gd(r), r3)
        else:
            rho = _rho(hh, logrho, dh)
        a = a - F64(0.5) * rho * cd_am * wp.length(vr) * vr
    return a


@wp.kernel
def _rk4_chunk(r: wp.array(dtype=wp.vec3d), v: wp.array(dtype=wp.vec3d), alive: wp.array(dtype=int),
               cd_am: wp.array(dtype=F64), cr_am: wp.array(dtype=F64),
               sun: wp.array(dtype=wp.vec3d), moon: wp.array(dtype=wp.vec3d),
               logrho: wp.array(dtype=F64), dh: F64, h: F64, n_steps: int, every: int, flags: int,
               r3: Rho3, lsto: wp.array(dtype=F64),
               r_out: wp.array2d(dtype=wp.vec3d), v_out: wp.array2d(dtype=wp.vec3d)):
    i = wp.tid()
    ri = r[i]
    vi = v[i]
    al = alive[i]
    hh = F64(0.5) * h
    for k in range(n_steps):
        if al != 0:
            s0 = sun[2 * k]
            s1 = sun[2 * k + 1]
            s2 = sun[2 * k + 2]
            m0 = moon[2 * k]
            m1 = moon[2 * k + 1]
            m2 = moon[2 * k + 2]
            l0 = lsto[2 * k]
            l1 = lsto[2 * k + 1]
            l2 = lsto[2 * k + 2]
            k1v = accel(ri, vi, s0, m0, cd_am[i], cr_am[i], logrho, dh, flags, r3, l0)
            k1r = vi
            k2v = accel(ri + hh * k1r, vi + hh * k1v, s1, m1, cd_am[i], cr_am[i], logrho, dh, flags, r3, l1)
            k2r = vi + hh * k1v
            k3v = accel(ri + hh * k2r, vi + hh * k2v, s1, m1, cd_am[i], cr_am[i], logrho, dh, flags, r3, l1)
            k3r = vi + hh * k2v
            k4v = accel(ri + h * k3r, vi + h * k3v, s2, m2, cd_am[i], cr_am[i], logrho, dh, flags, r3, l2)
            k4r = vi + h * k3v
            ri = ri + h / F64(6.0) * (k1r + F64(2.0) * k2r + F64(2.0) * k3r + k4r)
            vi = vi + h / F64(6.0) * (k1v + F64(2.0) * k2v + F64(2.0) * k3v + k4v)
            if _alt(ri) < F64(80000.0):
                al = 0
        if (k + 1) % every == 0:
            r_out[(k + 1) // every - 1, i] = ri
            v_out[(k + 1) // every - 1, i] = vi
    r[i] = ri
    v[i] = vi
    alive[i] = al


class OrbitBatch:
    def __init__(self, r0, v0, jd0, cd_am=0.0, cr_am=0.0, air=None, terms=("j2", "j3", "j4", "sun", "moon", "srp", "drag"),
                 device=None):
        self.device = physics_device(device)
        r0, v0 = np.array(r0, float, ndmin=2), np.array(v0, float, ndmin=2)
        n = len(r0)
        self.n, self.jd0, self.t = n, float(jd0), 0.0
        self.flags = sum(FLAG[t] for t in terms if not (t == "drag" and air is None))
        self.r = wp.array(r0, dtype=wp.vec3d, device=self.device)
        self.v = wp.array(v0, dtype=wp.vec3d, device=self.device)
        self.alive = wp.array(np.ones(n, np.int32), dtype=int, device=self.device)
        self.cd = wp.array(np.broadcast_to(np.asarray(cd_am, float), (n,)).copy(), dtype=F64, device=self.device)
        self.cr = wp.array(np.broadcast_to(np.asarray(cr_am, float), (n,)).copy(), dtype=F64, device=self.device)
        self.air = air
        self.dyn = air is not None and hasattr(air, "table3d") and "drag" in terms
        if self.dyn:
            self.flags |= FLAG_RHO3
        self._slot = None
        self.r3 = self._rho3_struct(None)
        if air is not None:
            tab = air.table()
            self.logrho = wp.array(tab["log_rho"], dtype=F64, device=self.device)
            self.dh = tab["dh"]
        else:
            self.logrho = wp.zeros(2, dtype=F64, device=self.device)
            self.dh = 1.0

    def _rho3_struct(self, tab):
        r3 = Rho3()
        if tab is None:
            r3.tab = wp.zeros((2, 2, 2), dtype=F64, device=self.device)
            r3.h0, r3.dh, r3.dlst, r3.lat0, r3.dlat = F64(1e30), F64(1.0), F64(1.0), F64(0.0), F64(1.0)
        else:
            r3.tab = wp.array(tab["log_rho"], dtype=F64, device=self.device)
            r3.h0, r3.dh, r3.dlst = F64(tab["h0"]), F64(tab["dh"]), F64(tab["dlst"])
            r3.lat0, r3.dlat = F64(tab["lat0"]), F64(tab["dlat"])
        return r3

    def run(self, h, n_steps, every=1, chunk=None):
        """Advance n_steps of h seconds; returns (t [n_rec], r [n_rec,N,3], v [n_rec,N,3]) every `every` steps."""
        chunk = chunk or every * max(1, 2048 // every)
        assert chunk % every == 0 and n_steps % every == 0, "n_steps and chunk must be multiples of every"
        ts, rs, vs = [], [], []
        done = 0
        while done < n_steps:
            m = min(chunk, n_steps - done)
            lst_o = np.zeros(2 * m + 1)
            if self.dyn:
                from .thermosphere import lst_offset_hours
                jd_now = self.jd0 + self.t / 86400.0
                k = self.air.slot(jd_now + 1e-9 / 86400.0)
                if k != self._slot:
                    self.r3 = self._rho3_struct(self.air.table3d(self.air.slot_jd(k)))
                    self._slot = k
                # cut the chunk at the next refresh boundary (whole steps; the boundary stage uses this table)
                t_next = (self.air.slot_jd(k + 1) - self.jd0) * 86400.0
                m_ref = max(every, int(math.floor((t_next - self.t) / h + 1e-9)) // every * every)
                m = min(m, m_ref)
            stage_t = self.t + 0.5 * h * np.arange(2 * m + 1)
            jd = self.jd0 + stage_t / 86400.0
            if self.dyn:
                lst_o = lst_offset_hours(jd)
            else:
                lst_o = np.zeros(2 * m + 1)
            lsto = wp.array(np.asarray(lst_o, float), dtype=F64, device=self.device)
            sun = wp.array(sun_eci(jd), dtype=wp.vec3d, device=self.device)
            moon = wp.array(moon_eci(jd), dtype=wp.vec3d, device=self.device)
            nrec = m // every
            ro = wp.zeros((max(nrec, 1), self.n), dtype=wp.vec3d, device=self.device)
            vo = wp.zeros((max(nrec, 1), self.n), dtype=wp.vec3d, device=self.device)
            wp.launch(_rk4_chunk, dim=self.n, device=self.device,
                      inputs=[self.r, self.v, self.alive, self.cd, self.cr, sun, moon, self.logrho, F64(self.dh), F64(h),
                              m, every, self.flags, self.r3, lsto, ro, vo])
            if nrec:
                ts.append(self.t + h * every * np.arange(1, nrec + 1))
                rs.append(ro.numpy())
                vs.append(vo.numpy())
            self.t += m * h
            done += m
        return (np.concatenate(ts) if ts else np.zeros(0), np.concatenate(rs) if rs else np.zeros((0, self.n, 3)),
                np.concatenate(vs) if vs else np.zeros((0, self.n, 3)))

    def state(self):
        return self.r.numpy().astype(float), self.v.numpy().astype(float), self.alive.numpy()

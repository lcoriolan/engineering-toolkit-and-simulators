#!/usr/bin/env python3
# DESCRIPTION: One GPU thread per (source -> receiver plane p, launch angle i). Rays in the vertical range-height
#              plane of a horizontally stratified MOVING medium (sound speed c(z), along-path wind u(z)); exact ray
#              equations with the wave-normal angle psi as state (Ostashev, "Acoustics in Moving Inhomogeneous Media",
#              1997, ch. 3; the Snell invariant cos(psi) / (c + u cos(psi)) is conserved):
#                  dr/dt = c cos(psi) + u        dz/dt = c sin(psi)        dpsi/dt = -cos(psi) (c' + u' cos(psi))
#              integrated with midpoint RK2 at a fixed time step (ds / c_ref), all in float64 (phase accuracy). Ground = the terrain profile under the
#              plane (piecewise linear in range); a ray that reaches it is reflected specularly about the local slope
#              (psi -> 2 phi - psi) and the grazing angle + surface index of every bounce are recorded (up to NB).
#              For each receiver range r_j of the plane the FIRST forward crossing is recorded: height, travel time,
#              wave-normal and group angles, path length, bounce count. Eigenrays / amplitudes / coherent sums are
#              formed on the host (propagate.py). Rays that turn back in range (steep terrain), leave the top of the
#              table or exceed NB bounces are ended there (documented limitation).
import os
import sys

import numpy as np
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from engineering_sim.core import physics_device  # noqa: E402

wp.config.quiet = True
NB_MAX = 48


F = wp.float64


@wp.func
def _row(a: wp.array2d(dtype=wp.float64), p: int, x: wp.float64, x0: wp.float64, dx: wp.float64, n: int):
    f = wp.clamp((x - x0) / dx, F(0.0), F(n - 1) - F(1.0e-6))
    i = int(f)
    w = f - F(i)
    return a[p, i] * (F(1.0) - w) + a[p, i + 1] * w


@wp.func
def _rowi(a: wp.array2d(dtype=int), p: int, x: wp.float64, x0: wp.float64, dx: wp.float64, n: int):
    f = wp.clamp((x - x0) / dx + F(0.5), F(0.0), F(n - 1))
    return a[p, int(f)]


@wp.kernel
def trace_kernel(psi0: wp.array(dtype=wp.float64), src_z: wp.array(dtype=wp.float64),
                 pc: wp.array2d(dtype=wp.float64), pdc: wp.array2d(dtype=wp.float64),
                 pu: wp.array2d(dtype=wp.float64), pdu: wp.array2d(dtype=wp.float64), z0: wp.float64,
                 dz: wp.float64, nz: int,
                 gh: wp.array2d(dtype=wp.float64), gm: wp.array2d(dtype=int), drg: wp.float64, ng: int,
                 rec_r: wp.array2d(dtype=wp.float64), nj: wp.array(dtype=int),
                 dt: wp.float64, max_steps: int, z_top: wp.float64, nb_max: int,
                 oz: wp.array3d(dtype=wp.float64), ot: wp.array3d(dtype=wp.float64),
                 opsi: wp.array3d(dtype=wp.float64), oth: wp.array3d(dtype=wp.float64),
                 oL: wp.array3d(dtype=wp.float64), onb: wp.array3d(dtype=int),
                 bpsi: wp.array3d(dtype=wp.float64), bmat: wp.array3d(dtype=int), bL: wp.array3d(dtype=wp.float64),
                 nbounce: wp.array2d(dtype=int)):
    p, i = wp.tid()
    half = F(0.5)
    r = F(0.0)
    z = src_z[p]
    psi = psi0[i]
    L = F(0.0)
    nb = int(0)
    j = int(0)
    njp = nj[p]
    r_end = rec_r[p, njp - 1]
    step = int(0)
    alive = int(1)
    while alive == 1 and step < max_steps and j < njp:
        # midpoint RK2 in time (time = step * dt exactly: no accumulated rounding in the phase)
        c1 = _row(pc, p, z, z0, dz, nz)
        g1 = _row(pdc, p, z, z0, dz, nz)
        q1 = _row(pdu, p, z, z0, dz, nz)
        cp = wp.cos(psi)
        sp = wp.sin(psi)
        zm = z + half * dt * c1 * sp
        psim = psi - half * dt * cp * (g1 + q1 * cp)
        c2 = _row(pc, p, zm, z0, dz, nz)
        u2 = _row(pu, p, zm, z0, dz, nz)
        g2 = _row(pdc, p, zm, z0, dz, nz)
        q2 = _row(pdu, p, zm, z0, dz, nz)
        cpm = wp.cos(psim)
        spm = wp.sin(psim)
        vr = c2 * cpm + u2
        vz = c2 * spm
        rn = r + dt * vr
        zn = z + dt * vz
        psin = psi - dt * cpm * (g2 + q2 * cpm)
        Ln = L + dt * wp.sqrt(vr * vr + vz * vz)
        if vr <= F(1.0e-3):
            alive = 0
        # ground: specular reflection of the wave normal about the local slope
        gz = _row(gh, p, rn, F(0.0), drg, ng)
        if alive == 1 and zn < gz:
            ga = _row(gh, p, rn - half * drg, F(0.0), drg, ng)
            gb = _row(gh, p, rn + half * drg, F(0.0), drg, ng)
            phi = wp.atan((gb - ga) / drg)
            graze = phi - psin
            if nb < nb_max:
                bpsi[p, i, nb] = graze
                bmat[p, i, nb] = _rowi(gm, p, rn, F(0.0), drg, ng)
                bL[p, i, nb] = Ln
            nb = nb + 1
            psin = F(2.0) * phi - psin
            zn = gz + (gz - zn)
            if zn < gz:
                zn = gz + F(1.0e-6)
            if nb > nb_max:
                alive = 0
        # receiver-range crossings (first forward crossing of each r_j)
        while j < njp and rn >= rec_r[p, j] and rn > r:
            rr = rec_r[p, j]
            w = (rr - r) / (rn - r)
            zc = z + w * (zn - z)
            pe = psi + w * (psin - psi)
            oz[p, i, j] = zc
            ot[p, i, j] = (F(step) + w) * dt
            opsi[p, i, j] = pe
            ce = _row(pc, p, zc, z0, dz, nz)
            ue = _row(pu, p, zc, z0, dz, nz)
            oth[p, i, j] = wp.atan2(ce * wp.sin(pe), ce * wp.cos(pe) + ue)
            oL[p, i, j] = L + w * (Ln - L)
            onb[p, i, j] = nb
            j = j + 1
        r = rn
        z = zn
        psi = psin
        L = Ln
        step = step + 1
        if z > z_top or r > r_end:
            alive = 0
        if psi > F(1.5707) or psi < F(-1.5707):
            alive = 0
    nbounce[p, i] = nb


def trace(psi0, src_z, profiles, ground_h, ground_mat, drg, rec_r, ds=0.5, z_top=None, device=None,
          nb_max=NB_MAX, c_ref=None):
    """psi0 [NR] launch wave-normal angles [rad]; src_z [P] world source heights; profiles: list of P Profile objects
    on ONE common z grid; ground_h/ground_mat [P, NG] terrain height / surface index at r = k drg; rec_r [P, NJ]
    ascending receiver ranges (NaN-padded; nj = finite count). Returns dict of numpy arrays."""
    dev = physics_device(device)
    P, NR = len(src_z), len(psi0)
    rec_r = np.asarray(rec_r, float)
    nj = np.array([int(np.sum(np.isfinite(rr))) for rr in rec_r], np.int32)
    NJ = rec_r.shape[1]
    rr = np.where(np.isfinite(rec_r), rec_r, 1e30)
    pr0 = profiles[0]
    nz = len(pr0.z)
    stack = lambda key: np.stack([getattr(pf, key) for pf in profiles]).astype(np.float64)  # noqa: E731
    c_ref = c_ref or float(np.median(pr0.c))
    dt = ds / c_ref
    rmax = float(np.nanmax(rec_r))
    zt = z_top if z_top is not None else float(pr0.z[-1] - 1.0)
    # steps: enough for the longest relevant path (to r_max via the top of the table and back)
    max_steps = int(2.5 * (rmax + 2.0 * max(zt - float(np.min(ground_h)), 1.0)) / ds) + 100
    A = lambda a, dt_=wp.float64: wp.array(np.ascontiguousarray(a), dtype=dt_, device=dev)  # noqa: E731
    nan3 = np.full((P, NR, NJ), np.nan, np.float64)
    out = dict(oz=A(nan3), ot=A(nan3), opsi=A(nan3), oth=A(nan3), oL=A(nan3),
               onb=A(np.full((P, NR, NJ), -1, np.int32), wp.int32),
               bpsi=A(np.zeros((P, NR, nb_max), np.float64)), bmat=A(np.zeros((P, NR, nb_max), np.int32), wp.int32),
               bL=A(np.zeros((P, NR, nb_max), np.float64)), nbounce=A(np.zeros((P, NR), np.int32), wp.int32))
    wp.launch(trace_kernel, dim=(P, NR),
              inputs=[A(np.asarray(psi0, np.float64)), A(np.asarray(src_z, np.float64)),
                      A(stack("c")), A(stack("dc")), A(stack("u")), A(stack("du")), wp.float64(pr0.z0), wp.float64(pr0.dz),
                      int(nz), A(np.asarray(ground_h, np.float64)), A(np.asarray(ground_mat, np.int32), wp.int32),
                      wp.float64(drg), int(np.asarray(ground_h).shape[1]), A(rr.astype(np.float64)), A(nj, wp.int32),
                      wp.float64(dt), int(max_steps), wp.float64(zt), int(nb_max)],
              outputs=[out[k] for k in ("oz", "ot", "opsi", "oth", "oL", "onb", "bpsi", "bmat", "bL", "nbounce")],
              device=dev)
    res = {k: v.numpy() for k, v in out.items()}
    res.update(psi0=np.asarray(psi0, float), nj=nj, dt=dt, max_steps=max_steps, device=str(dev))
    return res

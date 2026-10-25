#!/usr/bin/env python3
# DESCRIPTION: The wave-based reference for LOW-ANGLE propagation over flat impedance ground in a refracting moving
#              atmosphere (effective sound speed c_eff(z) = c(z) + u_par(z)). It resolves what geometric rays cannot:
#              diffraction into refraction shadow zones, caustics, and refraction inside sub-metre near-ground layers
#              (the log wind profile), which decide the level at a node 1.5 m above the ground.
#              Method (Salomons, "Computational Atmospheric Acoustics", 2001, app. G; West, Gilbert & Sack 1992):
#                p(r, z) = psi(r, z) exp(i ka r) / sqrt(r),  wide-angle Pade(1,1):
#                (1 + q/4) dpsi/dr = (i ka / 2) q psi,  q = eps(z) + ka^-2 d2/dz2,  eps = (k(z)/ka)^2 - 1
#                k(z) = w / c_eff(z) + i alpha(f) (ISO 9613-1 via atmospheric absorption, Np/m)
#                Crank-Nicolson in range: [1 + (1/4 - i ka dr/4) q] psi(r+dr) = [1 + (1/4 + i ka dr/4) q] psi(r)
#                ground z = 0: dpsi/dz + i k0 beta psi = 0, beta = 1/Z (ground.py impedance) via a ghost point
#                top: absorbing layer, eps += i A (z - z_t)^2 / (z_max - z_t)^2 over 50 wavelengths, psi = 0 at z_max
#                starter: Salomons wide-angle Gaussian + image weighted by the normal-incidence R = (Z-1)/(Z+1)
#                grid: dz = lambda / 10, dr = lambda / 8 (unconditionally stable; accuracy checked against the
#                closed form in tests/test_acoustic_prop.py)
#              GPU, method="pcr" (default): parallel cyclic reduction across z (see _march_pcr). method="thomas":
#              reference, one Warp thread per (case, frequency) marches its own contiguous column (float64 complex as vec2d; the
#              tridiagonal LHS factorisation is computed once per thread, the medium being range independent).
#              precision=32 (default; float32 complex, ~8x faster on the Tesla P4) or 64 (reference); both are checked.
#              Limits: flat ground (terrain -> ray model), elevation angles <~ 30 deg (Pade(1,1)); 2-D (cylindrical
#              spreading of a point source). ALL OUTPUTS MODELED.
import math
import os
import sys

import numpy as np
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from engineering_sim.core import physics_device  # noqa: E402
from engineering_sim.domains.acoustic_prop import ground as G  # noqa: E402

wp.config.quiet = True
F = wp.float64


@wp.func
def cmul(a: wp.vec2d, b: wp.vec2d):
    return wp.vec2d(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@wp.func
def cinv(b: wp.vec2d):
    d = b[0] * b[0] + b[1] * b[1]
    return wp.vec2d(b[0] / d, -b[1] / d)


@wp.kernel
def pe_kernel(bd: wp.array2d(dtype=wp.vec2d), ad: wp.array2d(dtype=wp.vec2d), nz: wp.array(dtype=int),
              s: wp.array(dtype=wp.float64), kdr4: wp.array(dtype=wp.float64), dr: wp.array(dtype=wp.float64),
              nsteps: wp.array(dtype=int), rec_r: wp.array2d(dtype=wp.float64), nj: wp.array(dtype=int),
              jr: wp.array(dtype=int), wr: wp.array(dtype=wp.float64),
              psi: wp.array2d(dtype=wp.vec2d), cp: wp.array2d(dtype=wp.vec2d), im: wp.array2d(dtype=wp.vec2d),
              dp: wp.array2d(dtype=wp.vec2d), out: wp.array2d(dtype=wp.vec2d), out_r: wp.array2d(dtype=wp.float64)):
    # arrays are [thread, z]: each thread walks its own contiguous column (cache-line reuse; few threads)
    # bd[t, j] = 1 + b q_jj (RHS diagonal), ad[t, j] = 1 + a q_jj (LHS diagonal); row 0 includes the ground BC
    t = wp.tid()
    n = nz[t]
    st = s[t]
    a = wp.vec2d(F(0.25), -kdr4[t])        # LHS coefficient of q
    b = wp.vec2d(F(0.25), kdr4[t])         # RHS coefficient of q
    a_off = a * st                          # a * s  (off-diagonals)
    a_off0 = a * (F(2.0) * st)              # row 0 upper (ghost point)
    b_off = b * st
    b_off0 = b * (F(2.0) * st)
    # ---- factorise A once (Thomas: c'_j and 1/m_j) -----------------------------------------------------------
    i0 = cinv(ad[t, 0])
    im[t, 0] = i0
    c = cmul(a_off0, i0)
    cp[t, 0] = c
    for j in range(1, n):
        ij = cinv(ad[t, j] - cmul(a_off, c))
        im[t, j] = ij
        c = cmul(a_off, ij)
        cp[t, j] = c
    jrec = int(0)
    njt = nj[t]
    zero = wp.vec2d(F(0.0), F(0.0))
    for m in range(nsteps[t]):
        # forward sweep with the RHS B psi formed on the fly (psi still holds the old field)
        pc = psi[t, 0]
        pn = psi[t, 1]
        d = cmul(cmul(bd[t, 0], pc) + cmul(b_off0, pn), im[t, 0])
        dp[t, 0] = d
        pm = pc
        pc = pn
        for j in range(1, n):
            pn = zero
            if j < n - 1:
                pn = psi[t, j + 1]
            rhs = cmul(bd[t, j], pc) + cmul(b_off, pm + pn)
            d = cmul(rhs - cmul(a_off, d), im[t, j])
            dp[t, j] = d
            pm = pc
            pc = pn
        # back substitution
        x = dp[t, n - 1]
        psi[t, n - 1] = x
        for k in range(1, n):
            j = n - 1 - k
            x = dp[t, j] - cmul(cp[t, j], x)
            psi[t, j] = x
        rr = F(m + 1) * dr[t]
        while jrec < njt and rr >= rec_r[t, jrec]:
            jj = jr[t]
            w = wr[t]
            out[t, jrec] = psi[t, jj] * (F(1.0) - w) + psi[t, jj + 1] * w
            out_r[t, jrec] = rr
            jrec = jrec + 1


F32 = wp.float32


@wp.func
def cmul32(a: wp.vec2f, b: wp.vec2f):
    return wp.vec2f(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@wp.func
def cinv32(b: wp.vec2f):
    d = b[0] * b[0] + b[1] * b[1]
    return wp.vec2f(b[0] / d, -b[1] / d)


@wp.kernel
def pe_kernel32(bd: wp.array2d(dtype=wp.vec2f), ad: wp.array2d(dtype=wp.vec2f), nz: wp.array(dtype=int),
              s: wp.array(dtype=wp.float32), kdr4: wp.array(dtype=wp.float32), dr: wp.array(dtype=wp.float32),
              nsteps: wp.array(dtype=int), rec_r: wp.array2d(dtype=wp.float32), nj: wp.array(dtype=int),
              jr: wp.array(dtype=int), wr: wp.array(dtype=wp.float32),
              psi: wp.array2d(dtype=wp.vec2f), cp: wp.array2d(dtype=wp.vec2f), im: wp.array2d(dtype=wp.vec2f),
              dp: wp.array2d(dtype=wp.vec2f), out: wp.array2d(dtype=wp.vec2f), out_r: wp.array2d(dtype=wp.float32)):
    # arrays are [thread, z]: each thread walks its own contiguous column (cache-line reuse; few threads)
    # bd[t, j] = 1 + b q_jj (RHS diagonal), ad[t, j] = 1 + a q_jj (LHS diagonal); row 0 includes the ground BC
    t = wp.tid()
    n = nz[t]
    st = s[t]
    a = wp.vec2f(F32(0.25), -kdr4[t])        # LHS coefficient of q
    b = wp.vec2f(F32(0.25), kdr4[t])         # RHS coefficient of q
    a_off = a * st                          # a * s  (off-diagonals)
    a_off0 = a * (F32(2.0) * st)              # row 0 upper (ghost point)
    b_off = b * st
    b_off0 = b * (F32(2.0) * st)
    # ---- factorise A once (Thomas: c'_j and 1/m_j) -----------------------------------------------------------
    i0 = cinv32(ad[t, 0])
    im[t, 0] = i0
    c = cmul32(a_off0, i0)
    cp[t, 0] = c
    for j in range(1, n):
        ij = cinv32(ad[t, j] - cmul32(a_off, c))
        im[t, j] = ij
        c = cmul32(a_off, ij)
        cp[t, j] = c
    jrec = int(0)
    njt = nj[t]
    zero = wp.vec2f(F32(0.0), F32(0.0))
    for m in range(nsteps[t]):
        # forward sweep with the RHS B psi formed on the fly (psi still holds the old field)
        pc = psi[t, 0]
        pn = psi[t, 1]
        d = cmul32(cmul32(bd[t, 0], pc) + cmul32(b_off0, pn), im[t, 0])
        dp[t, 0] = d
        pm = pc
        pc = pn
        for j in range(1, n):
            pn = zero
            if j < n - 1:
                pn = psi[t, j + 1]
            rhs = cmul32(bd[t, j], pc) + cmul32(b_off, pm + pn)
            d = cmul32(rhs - cmul32(a_off, d), im[t, j])
            dp[t, j] = d
            pm = pc
            pc = pn
        # back substitution
        x = dp[t, n - 1]
        psi[t, n - 1] = x
        for k in range(1, n):
            j = n - 1 - k
            x = dp[t, j] - cmul32(cp[t, j], x)
            psi[t, j] = x
        rr = F32(m + 1) * dr[t]
        while jrec < njt and rr >= rec_r[t, jrec]:
            jj = jr[t]
            w = wr[t]
            out[t, jrec] = psi[t, jj] * (F32(1.0) - w) + psi[t, jj + 1] * w
            out_r[t, jrec] = rr
            jrec = jrec + 1


# ------------------------------------------------------------------------------------------------------------------
# PCR engine (default): parallel cyclic reduction across z, one GPU thread per (frequency, z node); the reduction
# multipliers depend only on the (range-independent) LHS matrix, so they are computed once; every range step is then
# rhs -> ceil(log2 nz) reduction sweeps -> divide, captured as ONE CUDA graph replayed per step.
# ------------------------------------------------------------------------------------------------------------------
V2 = wp.vec2f
F32 = wp.float32


@wp.func
def cm(a: wp.vec2f, b: wp.vec2f):
    return wp.vec2f(a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


@wp.func
def cdv(a: wp.vec2f, b: wp.vec2f):
    d = b[0] * b[0] + b[1] * b[1]
    return wp.vec2f((a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d)


@wp.kernel
def pcr_init(ad: wp.array2d(dtype=wp.vec2f), aoff: wp.array(dtype=wp.vec2f), aoff0: wp.array(dtype=wp.vec2f),
             nz: wp.array(dtype=int), A: wp.array2d(dtype=wp.vec2f), B: wp.array2d(dtype=wp.vec2f),
             C: wp.array2d(dtype=wp.vec2f)):
    t, i = wp.tid()
    n = nz[t]
    z = wp.vec2f(F32(0.0), F32(0.0))
    if i >= n:
        A[t, i] = z
        B[t, i] = wp.vec2f(F32(1.0), F32(0.0))
        C[t, i] = z
        return
    B[t, i] = ad[t, i]
    if i == 0:
        A[t, i] = z
        C[t, i] = aoff0[t]
    else:
        A[t, i] = aoff[t]
        if i == n - 1:
            C[t, i] = z
        else:
            C[t, i] = aoff[t]


@wp.kernel
def pcr_coef(A: wp.array2d(dtype=wp.vec2f), B: wp.array2d(dtype=wp.vec2f), C: wp.array2d(dtype=wp.vec2f),
             nz: wp.array(dtype=int), st: int, AL: wp.array2d(dtype=wp.vec2f), GA: wp.array2d(dtype=wp.vec2f),
             A2: wp.array2d(dtype=wp.vec2f), B2: wp.array2d(dtype=wp.vec2f), C2: wp.array2d(dtype=wp.vec2f)):
    t, i = wp.tid()
    n = nz[t]
    z = wp.vec2f(F32(0.0), F32(0.0))
    if i >= n:
        AL[t, i] = z
        GA[t, i] = z
        A2[t, i] = z
        B2[t, i] = wp.vec2f(F32(1.0), F32(0.0))
        C2[t, i] = z
        return
    al = z
    ga = z
    b = B[t, i]
    an = z
    cn = z
    if i - st >= 0:
        al = -cdv(A[t, i], B[t, i - st])
        an = cm(al, A[t, i - st])
        b = b + cm(al, C[t, i - st])
    if i + st < n:
        ga = -cdv(C[t, i], B[t, i + st])
        cn = cm(ga, C[t, i + st])
        b = b + cm(ga, A[t, i + st])
    AL[t, i] = al
    GA[t, i] = ga
    A2[t, i] = an
    B2[t, i] = b
    C2[t, i] = cn


@wp.kernel
def pcr_rhs(psi: wp.array2d(dtype=wp.vec2f), bd: wp.array2d(dtype=wp.vec2f), boff: wp.array(dtype=wp.vec2f),
            boff0: wp.array(dtype=wp.vec2f), nz: wp.array(dtype=int), step: wp.array(dtype=int),
            nsteps: wp.array(dtype=int), d: wp.array2d(dtype=wp.vec2f)):
    t, i = wp.tid()
    n = nz[t]
    if i >= n or step[t] >= nsteps[t]:
        return
    v = cm(bd[t, i], psi[t, i])
    if i == 0:
        v = v + cm(boff0[t], psi[t, 1])
    else:
        nb = psi[t, i - 1]
        if i < n - 1:
            nb = nb + psi[t, i + 1]
        v = v + cm(boff[t], nb)
    d[t, i] = v


@wp.kernel
def pcr_sweep(d: wp.array2d(dtype=wp.vec2f), AL: wp.array2d(dtype=wp.vec2f), GA: wp.array2d(dtype=wp.vec2f),
              nz: wp.array(dtype=int), st: int, step: wp.array(dtype=int), nsteps: wp.array(dtype=int),
              d2: wp.array2d(dtype=wp.vec2f)):
    t, i = wp.tid()
    n = nz[t]
    if i >= n or step[t] >= nsteps[t]:
        return
    v = d[t, i]
    if i - st >= 0:
        v = v + cm(AL[t, i], d[t, i - st])
    if i + st < n:
        v = v + cm(GA[t, i], d[t, i + st])
    d2[t, i] = v


@wp.kernel
def pcr_fin(d: wp.array2d(dtype=wp.vec2f), Bf: wp.array2d(dtype=wp.vec2f), nz: wp.array(dtype=int),
            step: wp.array(dtype=int), nsteps: wp.array(dtype=int), psi: wp.array2d(dtype=wp.vec2f)):
    t, i = wp.tid()
    if i >= nz[t] or step[t] >= nsteps[t]:
        return
    psi[t, i] = cdv(d[t, i], Bf[t, i])


@wp.kernel
def pcr_rec(psi: wp.array2d(dtype=wp.vec2f), dr: wp.array(dtype=wp.float32), step: wp.array(dtype=int),
            nsteps: wp.array(dtype=int), rec_r: wp.array2d(dtype=wp.float32), nj: wp.array(dtype=int),
            jrec: wp.array(dtype=int), jr: wp.array(dtype=int), wr: wp.array(dtype=wp.float32),
            out: wp.array2d(dtype=wp.vec2f), out_r: wp.array2d(dtype=wp.float32)):
    t = wp.tid()
    m = step[t]
    if m >= nsteps[t]:
        return
    rr = F32(m + 1) * dr[t]
    j = jrec[t]
    while j < nj[t] and rr >= rec_r[t, j]:
        w = wr[t]
        out[t, j] = psi[t, jr[t]] * (F32(1.0) - w) + psi[t, jr[t] + 1] * w
        out_r[t, j] = rr
        j = j + 1
    jrec[t] = j
    step[t] = m + 1


@wp.kernel
def pcr_rec_grid(psi: wp.array2d(dtype=wp.vec2f), dr: wp.array(dtype=wp.float32), step: wp.array(dtype=int),
                 nsteps: wp.array(dtype=int), rec_r: wp.array2d(dtype=wp.float32), nj: wp.array(dtype=int),
                 jrec: wp.array(dtype=int), jrs: wp.array2d(dtype=int), wrs: wp.array2d(dtype=wp.float32), nh: int,
                 out: wp.array2d(dtype=wp.vec2f), out_r: wp.array2d(dtype=wp.float32)):
    t = wp.tid()
    m = step[t]
    if m >= nsteps[t]:
        return
    rr = F32(m + 1) * dr[t]
    j = jrec[t]
    while j < nj[t] and rr >= rec_r[t, j]:
        for h in range(nh):
            w = wrs[t, h]
            out[t, j * nh + h] = psi[t, jrs[t, h]] * (F32(1.0) - w) + psi[t, jrs[t, h] + 1] * w
        out_r[t, j] = rr
        j = j + 1
    jrec[t] = j
    step[t] = m + 1


def _march_pcr(rows, rec_r, dev, heights=None):
    """rows: per-thread PE setup; rec_r [T, NJ] receiver ranges per thread (NaN padded). heights [NH] (AGL): record
    the field at every height of the grid (pcr_rec_grid) instead of the single receiver height of each row."""
    T = len(rows)
    NZ = max(len(r["ad"]) for r in rows)
    K = int(math.ceil(math.log2(max(NZ, 2))))
    v2 = lambda a: wp.array(np.ascontiguousarray(a, np.float32), dtype=V2, device=dev)  # noqa: E731
    f32 = lambda a: wp.array(np.ascontiguousarray(a, np.float32), dtype=wp.float32, device=dev)  # noqa: E731
    i32 = lambda a: wp.array(np.ascontiguousarray(a, np.int32), dtype=wp.int32, device=dev)  # noqa: E731
    cplx = lambda z: np.stack([np.real(z), np.imag(z)], -1)  # noqa: E731
    AD, BD, PS = (np.zeros((T, NZ, 2), np.float32) for _ in range(3))
    for t, r in enumerate(rows):
        n = len(r["ad"])
        AD[t, :n], BD[t, :n], PS[t, :n] = cplx(r["ad"]), cplx(r["bd"]), cplx(r["psi0"])
    nz = i32([len(r["ad"]) for r in rows])
    aoff = v2(cplx(np.array([r["a"] * r["s"] for r in rows])))
    aoff0 = v2(cplx(np.array([2.0 * r["a"] * r["s"] for r in rows])))
    boff = v2(cplx(np.array([r["b"] * r["s"] for r in rows])))
    boff0 = v2(cplx(np.array([2.0 * r["b"] * r["s"] for r in rows])))
    ad, bd, psi = v2(AD), v2(BD), v2(PS)
    sh = (T, NZ)
    Am, Bm, Cm, A2, B2, C2 = (wp.zeros(sh, dtype=V2, device=dev) for _ in range(6))
    AL = [wp.zeros(sh, dtype=V2, device=dev) for _ in range(K)]
    GA = [wp.zeros(sh, dtype=V2, device=dev) for _ in range(K)]
    wp.launch(pcr_init, dim=sh, inputs=[ad, aoff, aoff0, nz, Am, Bm, Cm], device=dev)
    for k in range(K):
        wp.launch(pcr_coef, dim=sh, inputs=[Am, Bm, Cm, nz, 1 << k, AL[k], GA[k], A2, B2, C2], device=dev)
        Am, A2, Bm, B2, Cm, C2 = A2, Am, B2, Bm, C2, Cm
    Bf = Bm
    rec_r = np.atleast_2d(np.asarray(rec_r, float))
    NJ = rec_r.shape[1]
    out = wp.zeros((T, NJ), dtype=V2, device=dev)
    out_r = wp.zeros((T, NJ), dtype=wp.float32, device=dev)
    step = wp.zeros(T, dtype=wp.int32, device=dev)
    jrec = wp.zeros(T, dtype=wp.int32, device=dev)
    nsteps = i32([r["nsteps"] for r in rows])
    dr = f32([r["dr"] for r in rows])
    rec = f32(np.where(np.isfinite(rec_r), rec_r, 1e30))
    njv = i32(np.sum(np.isfinite(rec_r), axis=1))
    jr = i32([r["jr"] for r in rows])
    wr = f32([r["wr"] for r in rows])
    d0, d1 = wp.zeros(sh, dtype=V2, device=dev), wp.zeros(sh, dtype=V2, device=dev)

    if heights is not None:
        H = np.asarray(heights, float)
        NH = len(H)
        JR = np.zeros((T, NH), np.int32)
        WR = np.zeros((T, NH), np.float32)
        for t, r in enumerate(rows):
            q = H / r["dz"]
            JR[t] = np.minimum(q.astype(int), len(r["ad"]) - 2)
            WR[t] = q - JR[t]
        jrs, wrs = i32(JR), f32(WR)
        out = wp.zeros((T, NJ * NH), dtype=V2, device=dev)

    def one_step():
        wp.launch(pcr_rhs, dim=sh, inputs=[psi, bd, boff, boff0, nz, step, nsteps, d0], device=dev)
        a, b = d0, d1
        for k in range(K):
            wp.launch(pcr_sweep, dim=sh, inputs=[a, AL[k], GA[k], nz, 1 << k, step, nsteps, b], device=dev)
            a, b = b, a
        wp.launch(pcr_fin, dim=sh, inputs=[a, Bf, nz, step, nsteps, psi], device=dev)
        if heights is None:
            wp.launch(pcr_rec, dim=T, inputs=[psi, dr, step, nsteps, rec, njv, jrec, jr, wr, out, out_r], device=dev)
        else:
            wp.launch(pcr_rec_grid, dim=T, inputs=[psi, dr, step, nsteps, rec, njv, jrec, jrs, wrs, NH, out, out_r],
                      device=dev)

    one_step()                                   # loads the module (and does step 1)
    with wp.ScopedCapture(device=dev) as cap:
        one_step()
    for _ in range(max(r["nsteps"] for r in rows) - 2):
        wp.capture_launch(cap.graph)
    o = out.numpy().astype(float)
    if heights is not None:
        o = o.reshape(T, NJ, NH, 2)
    return o[..., 0] + 1j * o[..., 1], out_r.numpy().astype(float)


def _setup_row(c_eff, z_grid, h_src, h_rcv, r_max, f, sigma_kpa, alpha_np, impedance_model, c_a, margin_m,
               dz_per_lambda, dr_per_lambda, abs_layer_lambda, A_abs):
    c0g = float(np.interp(0.0, z_grid, c_eff))
    c_a = c0g if c_a is None else c_a
    lam = c_a / f
    dz = lam / dz_per_lambda
    dr = lam / dr_per_lambda
    z_t = max(h_src, h_rcv) + max(margin_m, 20.0 * lam)
    z_m = z_t + abs_layer_lambda * lam
    n = int(math.ceil(z_m / dz))
    z = dz * np.arange(n)               # node 0 = the ground
    w = 2.0 * math.pi * f
    ka = w / c_a
    ce = np.interp(z, z_grid, c_eff)
    k = w / ce + 1j * alpha_np
    eps = (k / ka) ** 2 - 1.0
    lay = z > z_t
    eps[lay] += 1j * A_abs * ((z[lay] - z_t) / (z_m - z_t)) ** 2
    Zg = G.impedance(f, sigma_kpa, impedance_model)
    beta = 1.0 / Zg
    k0 = w / c0g
    s = 1.0 / (ka * dz) ** 2
    q00 = eps[0] + s * (-2.0 + 2j * dz * k0 * beta)
    # Salomons wide-angle Gaussian starter + image (normal-incidence reflection)
    R0 = (Zg - 1.0) / (Zg + 1.0)
    g = lambda zz: (1.3717 - 0.3701 * (ka * zz) ** 2) * np.exp(-(ka * zz) ** 2 / 3.0)  # noqa: E731
    psi0 = np.sqrt(1j * ka) * (g(z - h_src) + R0 * g(z + h_src))
    jr = min(int(h_rcv / dz), n - 2)
    wr = h_rcv / dz - jr
    nst = int(math.ceil(r_max / dr)) + 1
    qd = eps - 2.0 * s
    qd[0] = q00
    aq, bq = 0.25 - 1j * ka * dr / 4.0, 0.25 + 1j * ka * dr / 4.0
    return dict(ad=1.0 + aq * qd, bd=1.0 + bq * qd, s=s, kdr4=ka * dr / 4.0, dr=dr, nsteps=nst, psi0=psi0,
                jr=jr, wr=wr, ka=ka, a=aq, b=bq, n=n, dz=dz)


DEFAULTS = dict(impedance_model="miki", c_a=None, margin_m=50.0, dz_per_lambda=10.0, dr_per_lambda=8.0,
                abs_layer_lambda=50.0, A_abs=1.0)


def pe_cases(cases, freqs, device=None, **kw):
    """Many flat-ground cases x many frequencies in ONE GPU march (PCR). cases: list of dict(c_eff [on z_grid],
    z_grid (height above the local ground), h_src, h_rcv (AGL), ranges [NJ] ascending, sigma_kpa, alpha_np [F]
    (Np/m per frequency)). Returns a list (per case) of (p [NJ, F] complex, free field at 1 m = 1; r_actual [NJ, F])."""
    opt = dict(DEFAULTS, **kw)
    dev = physics_device(device)
    freqs = np.asarray(freqs, float)
    rows, recs, idx = [], [], []
    NJ = max(len(c["ranges"]) for c in cases)
    for ci, c in enumerate(cases):
        rg = np.asarray(c["ranges"], float)
        for fi, f in enumerate(freqs):
            rows.append(_setup_row(c["c_eff"], c["z_grid"], c["h_src"], c["h_rcv"], rg[-1], f, c["sigma_kpa"],
                                   c["alpha_np"][fi], opt["impedance_model"], opt["c_a"], opt["margin_m"],
                                   opt["dz_per_lambda"], opt["dr_per_lambda"], opt["abs_layer_lambda"], opt["A_abs"]))
            rr = np.full(NJ, np.nan)
            rr[:len(rg)] = rg
            recs.append(rr)
            idx.append((ci, fi))
    psi_c, rr = _march_pcr(rows, np.array(recs), dev)
    ka = np.array([r["ka"] for r in rows])[:, None]
    p = psi_c * np.exp(1j * ka * rr) / np.sqrt(np.maximum(rr, 1e-9))
    out = []
    for ci, c in enumerate(cases):
        nj = len(c["ranges"])
        sel = [k for k, (a, _) in enumerate(idx) if a == ci]
        out.append((p[sel, :nj].T, rr[sel, :nj].T))
    return out


def pe_grid(case, freqs, heights, device=None, **kw):
    """One flat-ground case, the field on a (range x height) grid: case = dict(c_eff, z_grid, h_src, ranges [NJ],
    sigma_kpa, alpha_np [F]) (h_rcv ignored); heights [NH] AGL. Returns p [NJ, NH, F] (free field at 1 m = 1)."""
    opt = dict(DEFAULTS, **kw)
    dev = physics_device(device)
    freqs = np.asarray(freqs, float)
    heights = np.asarray(heights, float)
    rg = np.asarray(case["ranges"], float)
    rows = [_setup_row(case["c_eff"], case["z_grid"], case["h_src"], float(np.max(heights)), rg[-1], f,
                       case["sigma_kpa"], case["alpha_np"][fi], opt["impedance_model"], opt["c_a"], opt["margin_m"],
                       opt["dz_per_lambda"], opt["dr_per_lambda"], opt["abs_layer_lambda"], opt["A_abs"])
            for fi, f in enumerate(freqs)]
    psi_c, rr = _march_pcr(rows, np.tile(rg, (len(freqs), 1)), dev, heights=heights)   # [T, NJ, NH]
    ka = np.array([r["ka"] for r in rows])[:, None, None]
    p = psi_c * np.exp(1j * ka * rr[:, :, None]) / np.sqrt(np.maximum(rr[:, :, None], 1e-9))
    return np.transpose(p, (1, 2, 0))


def pe_levels(profiles_c_eff, z_grid, h_src, h_rcv, ranges, freqs, sigma_kpa, alpha_np_per_m, impedance_model="miki",
              c_a=None, margin_m=50.0, dz_per_lambda=10.0, dr_per_lambda=8.0, abs_layer_lambda=50.0, A_abs=1.0,
              precision=32, method="pcr", device=None):
    """One flat-ground case, many frequencies (method "pcr" -> pe_cases; "thomas" = the sequential reference
    marcher, precision 32 | 64). Returns complex p [NJ, F] (free field at 1 m = 1) and the recorded ranges [NJ, F]."""
    kw = dict(impedance_model=impedance_model, c_a=c_a, margin_m=margin_m, dz_per_lambda=dz_per_lambda,
              dr_per_lambda=dr_per_lambda, abs_layer_lambda=abs_layer_lambda, A_abs=A_abs)
    freqs = np.asarray(freqs, float)
    ranges = np.asarray(ranges, float)
    if method == "pcr":
        return pe_cases([dict(c_eff=profiles_c_eff, z_grid=z_grid, h_src=h_src, h_rcv=h_rcv, ranges=ranges,
                              sigma_kpa=sigma_kpa, alpha_np=np.asarray(alpha_np_per_m, float))], freqs, device,
                        **kw)[0]
    dev = physics_device(device)
    T = len(freqs)
    rows = [_setup_row(profiles_c_eff, z_grid, h_src, h_rcv, ranges[-1], f, sigma_kpa, alpha_np_per_m[fi],
                       impedance_model, c_a, margin_m, dz_per_lambda, dr_per_lambda, abs_layer_lambda, A_abs)
            for fi, f in enumerate(freqs)]
    nzs = [r["n"] for r in rows]
    drs = [r["dr"] for r in rows]
    NZ = max(nzs)
    AD, BD, Psi = (np.zeros((T, NZ, 2)) for _ in range(3))
    for t, rw in enumerate(rows):
        n = len(rw["ad"])
        AD[t, :n, 0], AD[t, :n, 1] = rw["ad"].real, rw["ad"].imag
        BD[t, :n, 0], BD[t, :n, 1] = rw["bd"].real, rw["bd"].imag
        Psi[t, :n, 0], Psi[t, :n, 1] = rw["psi0"].real, rw["psi0"].imag
    fl, vt, kern = (wp.float64, wp.vec2d, pe_kernel) if precision == 64 else (wp.float32, wp.vec2f, pe_kernel32)
    A = lambda a, d=fl: wp.array(np.ascontiguousarray(a), dtype=d, device=dev)  # noqa: E731
    v2 = lambda a: wp.array(np.ascontiguousarray(a), dtype=vt, device=dev)  # noqa: E731
    NJ = len(ranges)
    out = wp.zeros((T, NJ), dtype=vt, device=dev)
    out_r = wp.zeros((T, NJ), dtype=fl, device=dev)
    psi = v2(Psi)
    scratch = [wp.zeros((T, NZ), dtype=vt, device=dev) for _ in range(3)]
    wp.launch(kern, dim=T,
              inputs=[v2(BD), v2(AD), A(np.array(nzs, np.int32), wp.int32), A(np.array([r["s"] for r in rows])),
                      A(np.array([r["kdr4"] for r in rows])), A(np.array(drs)),
                      A(np.array([r["nsteps"] for r in rows], np.int32), wp.int32),
                      A(np.tile(ranges, (T, 1))), A(np.full(T, NJ, np.int32), wp.int32),
                      A(np.array([r["jr"] for r in rows], np.int32), wp.int32), A(np.array([r["wr"] for r in rows])),
                      psi, scratch[0], scratch[1], scratch[2], out, out_r], device=dev)
    o = out.numpy().astype(float)
    rr = out_r.numpy().astype(float)
    psi_c = o[..., 0] + 1j * o[..., 1]                    # [T, NJ]
    ka = np.array([r["ka"] for r in rows])[:, None]
    p = psi_c * np.exp(1j * ka * rr) / np.sqrt(np.maximum(rr, 1e-9))
    return p.T, rr.T

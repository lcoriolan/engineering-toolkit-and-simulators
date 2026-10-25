#!/usr/bin/env python3
# DESCRIPTION: Righting arm GZ(heel) from 0 to 180 deg for any closed hull mesh and load case, in still water:
#                1. hydrostatics, two independent models of the same hull
#                   mesh     EXACT polyhedral clip of the closed triangle mesh by the water plane: every triangle is
#                            cut at z = 0 and the submerged part summed as tetrahedra from an apex ON the plane, so the
#                            waterplane cap contributes nothing (divergence theorem) -> volume + centre of buoyancy
#                            are exact for the mesh at any heel/trim (no small-angle or metacentric formula)
#                   columns  the GPU water domain's own model (hull.columns_from_mesh -> vessel._columns): prisms
#                            along body +Z, submerged fraction along each prism axis — evaluated here in numpy so the
#                            dynamic sim's hydrostatic error at large heel is visible next to the exact one
#                2. at every heel the craft is free to sink and trim: Newton on (CG height, trim) until buoyancy =
#                   weight and the centre of buoyancy is vertically above/below the CG longitudinally
#                   (rho V = M to 1e-6, x_B - x_G to 0.1 mm); continuation from the previous heel
#                3. GZ = y_B - y_G in the world frame (heel to PORT positive = port side down; GZ > 0 rights)
#              Occupant models (persons are point masses inside the load case's mass/CG):
#                rigid    people move with the deck (worst case: they cannot or do not balance)
#                upright  PHYSICAL: a person keeps their CG over their support (feet / seat): the CG of person i stays
#                         h_i above the support point along WORLD up instead of hull up, so G shifts by
#                         (m_i h_i / M)(z_world - z_hull) -> extra righting arm (m_i h_i / M) sin(tilt). It holds only
#                         while the support friction holds (tilt < atan(mu_support), mu ASSUMED); beyond that the
#                         person is treated as rigid (holding on; sliding to the low side is worse and not modelled);
#                         the couple fades out over the last 2 deg below the slip angle (same smoothstep as the GPU
#                         vessel kernel, so static GZ and the dynamic sim use one occupant model).
#                         h_i = person CG (load case) - support height (deck surface under the person from the hull
#                         mesh, or the seat part) — geometry, not a tuned value.
#              Outputs per case: GM (from the GZ slope at the equilibrium), equilibrium heel, angle and value of max
#              GZ, angle of vanishing stability (GZ -> 0 with negative slope past the max), area under GZ to the
#              vanishing angle (and to 30 / 40 deg), and the dynamic-heel energy check from a release at rest.
#              Labels: MODELED (results), ASSUMED (support friction, persons upright, seat heights), MEASURED = run.
import math

import numpy as np

G0 = 9.80665


# ---------------------------------------------------------------------------------------------- rotations
def rot(heel_deg, trim_deg):
    """Body (hull frame) -> world. Heel to PORT positive (port side down) = rotation -heel about world X; trim BOW UP
    positive = rotation -trim about world Y, applied after the heel (same order as the FreeCAD reference)."""
    a, b = math.radians(-heel_deg), math.radians(-trim_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    return Ry @ Rx


# ---------------------------------------------------------------------------------------------- hydrostatics
class MeshHydro:
    """Exact submerged volume / centroid of a closed, outward-oriented triangle mesh (hull frame, metres)."""

    def __init__(self, V, F):
        self.V = np.asarray(V, float)
        self.F = np.asarray(F, np.int64)
        a, b, c = (self.V[self.F[:, k]] for k in range(3))
        vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
        if vol < 0:                                        # make outward
            self.F = self.F[:, ::-1].copy()
        self.volume = abs(vol)

    def submerged(self, R, G, zc):
        """World vertices W = R (V - G) + (0, 0, zc); water plane z = 0. -> (volume, centroid world)."""
        W = (self.V - G) @ R.T
        W[:, 2] += zc
        T = W[self.F]                                      # [nf, 3, 3]
        z = T[:, :, 2]
        below = z < 0.0
        nb = below.sum(1)
        tris = [T[nb == 3]]
        # one vertex below: rotate it to slot 0 (cyclic -> orientation kept), keep (a, ab, ac)
        t1 = T[nb == 1]
        if len(t1):
            k = np.argmax(t1[:, :, 2] < 0.0, axis=1)
            t1 = _cycle(t1, k)
            a, b, c = t1[:, 0], t1[:, 1], t1[:, 2]
            pab = a + (b - a) * (a[:, 2] / (a[:, 2] - b[:, 2]))[:, None]
            pac = a + (c - a) * (a[:, 2] / (a[:, 2] - c[:, 2]))[:, None]
            tris.append(np.stack([a, pab, pac], 1))
        # two below: rotate the ABOVE vertex to slot 0: (u, a, b) -> polygon (pua, a, b, pub)
        t2 = T[nb == 2]
        if len(t2):
            k = np.argmax(t2[:, :, 2] >= 0.0, axis=1)
            t2 = _cycle(t2, k)
            u, a, b = t2[:, 0], t2[:, 1], t2[:, 2]
            pua = u + (a - u) * (u[:, 2] / (u[:, 2] - a[:, 2]))[:, None]
            pub = u + (b - u) * (u[:, 2] / (u[:, 2] - b[:, 2]))[:, None]
            tris.append(np.stack([pua, a, b], 1))
            tris.append(np.stack([pua, b, pub], 1))
        S = np.concatenate(tris, 0)
        if not len(S):
            return 0.0, np.zeros(3)
        # tetra (O, a, b, c), O = origin, which lies ON the water plane -> the cap adds nothing
        vol6 = np.einsum("ij,ij->i", S[:, 0], np.cross(S[:, 1], S[:, 2]))
        V = vol6.sum() / 6
        if V <= 1e-12:
            return 0.0, np.zeros(3)
        cen = (vol6[:, None] * S.sum(1)).sum(0) / 24 / V
        return float(V), cen


class MeshHydroGPU(MeshHydro):
    """MeshHydro on the physics GPU (warp): one thread per triangle clips it at the water plane and adds its
    tetrahedra (apex on the plane) to float64 accumulators. Same mathematics as MeshHydro.submerged (checked against
    it in run_stability V0); float64 atomics -> summation order varies at the 1e-15 level only."""

    def __init__(self, V, F, device=None):
        super().__init__(V, F)
        import warp as wp
        from engineering_sim.core import physics_device
        self.wp, self.dev = wp, physics_device(device)
        self.Vd = wp.array(self.V.astype(np.float64), dtype=wp.vec3d, device=self.dev)
        self.Fd = wp.array(self.F.astype(np.int32), dtype=wp.vec3i, device=self.dev)
        self.acc = wp.zeros(4, dtype=wp.float64, device=self.dev)
        self.k = _clip_k

    def submerged(self, R, G, zc):
        wp = self.wp
        self.acc.zero_()
        Rm = wp.mat33d(*[float(c) for c in np.asarray(R, float).ravel()])
        wp.launch(self.k, len(self.F), [self.Vd, self.Fd, Rm, wp.vec3d(*[float(c) for c in G]), float(zc), self.acc],
                  device=self.dev)
        a = self.acc.numpy()
        V = a[0] / 6.0
        if V <= 1e-12:
            return 0.0, np.zeros(3)
        return float(V), a[1:] / 24.0 / V


def _cycle(T, k):
    idx = (np.arange(3)[None, :] + k[:, None]) % 3
    return np.take_along_axis(T, idx[:, :, None].repeat(3, 2), 1)


class ColumnHydro:
    """The GPU water domain's column hydrostatics (domains/water/vessel._columns) in numpy, still water."""

    def __init__(self, cols):
        self.x, self.y = np.asarray(cols["x"], float), np.asarray(cols["y"], float)
        self.iv = np.asarray(cols["intervals"], float)                  # [nc, K, 2] hull frame z
        self.dA = float(cols["dA"])
        L = self.iv[:, :, 1] - self.iv[:, :, 0]
        self.volume = float((np.clip(L, 0, None)).sum() * self.dA)

    def submerged(self, R, G, zc):
        nc, K = self.iv.shape[:2]
        r = np.stack([np.repeat(self.x[:, None], K, 1), np.repeat(self.y[:, None], K, 1)], -1)
        pl = np.concatenate([r, self.iv[:, :, :1]], -1) - G
        pu = np.concatenate([r, self.iv[:, :, 1:]], -1) - G
        pl, pu = pl @ R.T, pu @ R.T
        pl[..., 2] += zc
        pu[..., 2] += zc
        ln = self.iv[:, :, 1] - self.iv[:, :, 0]
        ok = ln > 0
        swap = pu[..., 2] < pl[..., 2]
        lo = np.where(swap[..., None], pu, pl)
        hi = np.where(swap[..., None], pl, pu)
        dz = hi[..., 2] - lo[..., 2]
        frac = np.where(dz > 1e-6, np.clip((0.0 - lo[..., 2]) / np.maximum(dz, 1e-12), 0, 1),
                        (lo[..., 2] < 0.0).astype(float))
        frac = np.where(ok, frac, 0.0)
        vol = self.dA * ln * frac
        cen = lo + (hi - lo) * (0.5 * frac)[..., None]
        V = float(vol.sum())
        if V <= 1e-12:
            return 0.0, np.zeros(3)
        return V, (vol[..., None] * cen).sum((0, 1)) / V


# ---------------------------------------------------------------------------------------------- occupants
def deck_height(V, F, x, y):
    """Highest mesh surface z at (x, y) (hull frame): the deck / gunwale a person stands on."""
    P = np.asarray(V, float)[np.asarray(F)]
    a, b, c = P[:, 0], P[:, 1], P[:, 2]
    v0, v1 = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    v2 = np.array([x, y]) - a[:, :2]
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    ok = np.abs(den) > 1e-14
    dd = np.where(ok, den, 1.0)
    u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / dd
    v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / dd
    m = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    if not m.any():
        return None
    z = a[m, 2] + u[m] * (b[m, 2] - a[m, 2]) + v[m] * (c[m, 2] - a[m, 2])
    return float(z.max())


class Occupants:
    """Persons in a load case: rows of dict(name, m [kg], p (CG, hull frame) [m], h [m] CG above support, posture)."""

    def __init__(self, rows, mu_support=0.6, model="upright"):
        self.rows, self.model = list(rows), model
        self.mu = float(mu_support)                       # ASSUMED wet non-skid deck / seat cushion
        self.slip_deg = math.degrees(math.atan(self.mu))

    def K(self):
        """Upright-occupant couple coefficient sum(m g h) [N m]: moment = K sin(tilt) (restoring)."""
        return float(sum(r["m"] * G0 * r["h"] for r in self.rows))

    def g_shift(self, R, M):
        """World shift of the craft CG from the occupants' balance (zero for 'rigid'). Faded out with the SAME
        smoothstep as the GPU vessel kernel (vessel._vessel up_K): factor s((slip - tilt) / 2 deg), so footing is
        fully lost at the slip angle and the static and dynamic occupant models are identical."""
        if self.model != "upright" or not self.rows:
            return np.zeros(3), False
        zb = R @ np.array([0.0, 0.0, 1.0])
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, zb[2]))))
        c = min(max((self.slip_deg - tilt) / 2.0, 0.0), 1.0)
        fade = c * c * (3.0 - 2.0 * c)
        mh = sum(r["m"] * r["h"] for r in self.rows)
        return fade * (mh / M) * (np.array([0.0, 0.0, 1.0]) - zb), fade < 1.0


# ---------------------------------------------------------------------------------------------- equilibrium + GZ
def equilibrium(hydro, M, G, rho, heel, x0, occ=None, tol_m=1e-6, tol_x=1e-4, it_max=40, J0=None):
    """Free sinkage + trim at a fixed heel. x0 = (zc, trim_deg) start; J0 = a Jacobian to start from (continuation:
    the previous heel's) — recomputed by finite differences only when a step fails to reduce the residual.
    -> dict (+ 'J' for the next heel)."""
    x = np.array(x0, float)
    Vt = M / rho

    def res(x):
        R = rot(heel, x[1])
        V, B = hydro.submerged(R, G, x[0])
        dG, slipped = occ.g_shift(R, M) if occ is not None else (np.zeros(3), False)
        return np.array([V / Vt - 1.0, B[0] - dG[0]]), V, B, dG, slipped

    def jac(x, f):
        J = np.zeros((2, 2))
        for j, h in enumerate((1e-3, 0.05)):
            xp = x.copy()
            xp[j] += h
            J[:, j] = (res(xp)[0] - f) / h
        return J

    f, V, B, dG, sl = res(x)
    J = np.array(J0, float) if J0 is not None else None
    fresh = J is None
    if J is None:
        J = jac(x, f)
    ok, n_eval = False, 1
    for _ in range(it_max):
        if abs(f[0]) < tol_m and abs(f[1]) < tol_x:
            ok = True
            break
        try:
            dx = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            if fresh:
                break
            J, fresh = jac(x, f), True
            continue
        s_ = min(1.0, 0.05 / max(abs(dx[0]), 1e-12), 3.0 / max(abs(dx[1]), 1e-12))
        xn = x + s_ * dx
        fn = res(xn)
        n_eval += 1
        if np.abs(fn[0]).sum() < 0.7 * np.abs(f).sum() or (fresh and np.abs(fn[0]).sum() < np.abs(f).sum()):
            x = xn
            f, V, B, dG, sl = fn
            fresh = False                                  # keep J (Broyden-free chord iteration)
            continue
        if not fresh:                                      # stale Jacobian: rebuild at the current point
            J, fresh = jac(x, f), True
            n_eval += 2
            continue
        lam, moved = 0.5, False                            # fresh J and still no decrease: backtrack
        for _ls in range(6):
            xn = x + s_ * lam * dx
            fn = res(xn)
            n_eval += 1
            if np.abs(fn[0]).sum() < np.abs(f).sum():
                x = xn
                f, V, B, dG, sl = fn
                moved = True
                break
            lam *= 0.5
        if not moved:
            break
        J, fresh = jac(x, f), True
        n_eval += 2
    return dict(converged=ok, zc=float(x[0]), trim_deg=float(x[1]), volume=V, B=B.tolist(), G=dG.tolist(),
                GZ=float(B[1] - dG[1]), occupants_slipped=bool(sl), residual=f.tolist(), J=J.tolist(), n_eval=n_eval)


def gz_curve(hydro, M, G, rho, heels, occ=None, zc0=None):
    """GZ over heels (deg, ascending from 0). Continuation in (zc, trim)."""
    G = np.asarray(G, float)
    zc = zc0 if zc0 is not None else 0.3
    x0 = (zc, 0.0)
    rows, J = [], None
    for h in heels:
        r = equilibrium(hydro, M, G, rho, float(h), x0, occ, J0=J)
        if not r["converged"]:                             # retry from a few starts (trim branch changes)
            for tr in (0.0, -5.0, 5.0, -15.0, 15.0):
                for z in (x0[0], x0[0] + 0.1, x0[0] - 0.1):
                    r2 = equilibrium(hydro, M, G, rho, float(h), (z, tr), occ)
                    if r2["converged"]:
                        r = r2
                        break
                if r["converged"]:
                    break
        r["heel_deg"] = float(h)
        J = r.pop("J") if r["converged"] else None
        rows.append(r)
        if r["converged"]:
            x0 = (r["zc"], r["trim_deg"])
    return rows


def heel_grid(step=2.5, fine_to=60.0, coarse=5.0):
    """0, step, ... fine_to, then coarse steps to 180 (deg)."""
    a = np.arange(0.0, fine_to + 1e-9, step)
    b = np.arange(fine_to + coarse, 180.0 + 1e-9, coarse)
    return np.concatenate([a, b])


def full_curve(hydro, M, G, rho, step=2.5, occ=None, zc0=None, coarse=5.0):
    """GZ from -180 to +180 deg (heel to port +): two continuations out of the upright state (step to 60 deg,
    `coarse` beyond)."""
    grid = heel_grid(step, 60.0, max(coarse, step))
    pos = gz_curve(hydro, M, G, rho, grid, occ, zc0)
    neg = gz_curve(hydro, M, G, rho, -grid[1:], occ, zc0)
    return sorted(neg + pos, key=lambda r: r["heel_deg"])


def _trap(y, x):
    y, x = np.asarray(y, float), np.asarray(x, float)
    return float(((y[1:] + y[:-1]) * np.diff(x)).sum() / 2) if len(x) > 1 else 0.0


def _zeros(h, gz, sign):
    """Zero crossings of gz(h) with slope sign (+1 rising, -1 falling) -> list of angles."""
    out = []
    for k in range(len(h) - 1):
        a, b = gz[k], gz[k + 1]
        if (sign > 0 and a < 0 <= b) or (sign < 0 and a > 0 >= b):
            out.append(float(h[k] - a * (h[k + 1] - h[k]) / (b - a)))
    return out


def characterise(rows, releases=(0.0, 5.0, 10.0, 15.0, 20.0, 30.0, -5.0, -10.0, -20.0)):
    """Stability numbers from a signed GZ curve (heel to port +; GZ > 0 rights a port heel). Converged rows only.
    equilibrium = rising zero crossing nearest 0; vanishing angles = the falling crossings bracketing it."""
    R = [r for r in rows if r["converged"]]
    h = np.array([r["heel_deg"] for r in R])
    gz = np.array([r["GZ"] for r in R])
    out = dict(n=len(R), n_failed=len(rows) - len(R),
               failed_heels=[r["heel_deg"] for r in rows if not r["converged"]])
    if len(R) < 5:
        return out
    rising = _zeros(h, gz, +1)
    up = [a for a in rising if abs(a) < 90.0]           # an UPRIGHT equilibrium (deck normal above horizontal)
    if not up:
        side = (h >= 0) & (h <= 90)
        j = np.where(side)[0][int(np.argmax(gz[side]))]
        out.update(equilibrium_heel_deg=None, capsizes_statically=True,
                   stable_states_deg=rising,
                   note="no upright equilibrium: GZ < 0 (heeling to port) from upright to past 90 deg; the only "
                        "stable floating state is the capsized one",
                   closest_to_righting_heel_deg=float(h[j]), min_heeling_arm_m=float(-gz[j]),
                   heeling_arm_upright_m=float(-np.interp(0.0, h, gz)),
                   falling_crossings_deg=_zeros(h, gz, -1))
        return out
    eq = min(up, key=abs)
    dn = _zeros(h, gz, -1)
    vp = min([a for a in dn if a > eq], default=None)
    vm = max([a for a in dn if a < eq], default=None)
    out.update(equilibrium_heel_deg=eq, capsizes_statically=False, vanishing_stability_port_deg=vp,
               vanishing_stability_stbd_deg=vm,
               range_of_stability_deg=(vp - vm) if (vp is not None and vm is not None) else None)
    k = int(np.argmin(np.abs(h - eq)))
    k = min(max(k, 1), len(h) - 2)
    out["GM_m"] = float((gz[k + 1] - gz[k - 1]) / math.radians(h[k + 1] - h[k - 1]))
    mp = (h >= eq) & (h <= (vp if vp is not None else h[-1]))
    if mp.any():
        j = np.where(mp)[0][int(np.argmax(gz[mp]))]
        out["gz_max_port_m"], out["heel_at_gz_max_port_deg"] = float(gz[j]), float(h[j])
        out["area_eq_to_vanishing_port_m_rad"] = float(_trap(gz[mp], np.radians(h[mp])))
        for lim in (30.0, 40.0):
            m2 = (h >= eq) & (h <= lim)
            out["area_eq_to_%d_m_rad" % lim] = float(_trap(gz[m2], np.radians(h[m2]))) if m2.sum() > 1 else None
    mm = (h <= eq) & (h >= (vm if vm is not None else h[0]))
    if mm.any():
        j = np.where(mm)[0][int(np.argmin(gz[mm]))]
        out["gz_max_stbd_m"], out["heel_at_gz_max_stbd_deg"] = float(-gz[j]), float(h[j])
    out["dynamic_release"] = [release_check(h, gz, r0, vm, vp) for r0 in releases]
    return out


def release_check(h, gz, r0, vm, vp):
    """Undamped energy bound: released at rest at r0, the roll stays inside the connected set where
    U(phi) = M g * integral_0^phi GZ <= U(r0). Capsize if that set reaches past a vanishing angle."""
    th = np.radians(h)
    U = np.concatenate([[0.0], np.cumsum(0.5 * (gz[1:] + gz[:-1]) * np.diff(th))])
    U0 = float(np.interp(math.radians(r0), th, U))
    k0 = int(np.argmin(np.abs(h - r0)))
    lo = hi = k0
    while hi + 1 < len(h) and U[hi + 1] <= U0 + 1e-12:
        hi += 1
    while lo - 1 >= 0 and U[lo - 1] <= U0 + 1e-12:
        lo -= 1
    a, b = float(h[lo]), float(h[hi])
    cap = (vp is not None and b >= vp) or (vm is not None and a <= vm) or hi == len(h) - 1 or lo == 0
    return dict(release_deg=r0, roll_range_deg=[a, b], capsizes=bool(cap))

# ---------------------------------------------------------------------------------------------- GPU mesh clip
try:                                   # warp is optional: the numpy MeshHydro needs only numpy (CPU containers)
    import warp as wp

    @wp.func
    def _tet(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, acc: wp.array(dtype=wp.float64)):
        v6 = wp.dot(a, wp.cross(b, c))
        sm = a + b + c
        wp.atomic_add(acc, 0, v6)
        wp.atomic_add(acc, 1, v6 * sm[0])
        wp.atomic_add(acc, 2, v6 * sm[1])
        wp.atomic_add(acc, 3, v6 * sm[2])

    @wp.func
    def _cut(a: wp.vec3d, b: wp.vec3d):
        return a + (b - a) * (a[2] / (a[2] - b[2]))

    @wp.kernel
    def _clip_k(V: wp.array(dtype=wp.vec3d), F: wp.array(dtype=wp.vec3i), R: wp.mat33d, G: wp.vec3d,
                zc: wp.float64, acc: wp.array(dtype=wp.float64)):
        i = wp.tid()
        f = F[i]
        up = wp.vec3d(wp.float64(0.0), wp.float64(0.0), zc)
        p0 = R * (V[f[0]] - G) + up
        p1 = R * (V[f[1]] - G) + up
        p2 = R * (V[f[2]] - G) + up
        zero = wp.float64(0.0)
        b0 = p0[2] < zero
        b1 = p1[2] < zero
        b2 = p2[2] < zero
        nb = int(0)
        if b0:
            nb = nb + 1
        if b1:
            nb = nb + 1
        if b2:
            nb = nb + 1
        if nb == 0:
            return
        if nb == 3:
            _tet(p0, p1, p2, acc)
            return
        # cyclic rotation (keeps orientation): nb == 1 -> the below vertex first; nb == 2 -> the above vertex first
        a = p0
        b = p1
        c = p2
        if nb == 1:
            if b1:
                a = p1
                b = p2
                c = p0
            elif b2:
                a = p2
                b = p0
                c = p1
            _tet(a, _cut(a, b), _cut(a, c), acc)
        else:
            if not b1:
                a = p1
                b = p2
                c = p0
            elif not b2:
                a = p2
                b = p0
                c = p1
            pua = _cut(a, b)
            pub = _cut(a, c)
            _tet(pua, b, c, acc)
            _tet(pua, c, pub, acc)
except Exception:                      # pragma: no cover
    _clip_k = None

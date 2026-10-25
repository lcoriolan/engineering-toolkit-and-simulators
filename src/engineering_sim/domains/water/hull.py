#!/usr/bin/env python3
# DESCRIPTION: Any closed (watertight) hull mesh is converted ON THE GPU into a grid of vertical columns: for each
#              (x, y) cell, rays cast upward collect every entry/exit, giving up to K solid intervals [z_lo, z_hi]
#              per cell (multi-lobe hulls, tunnels and recessed wells are handled). In the vessel kernel each
#              interval is a prism along body +Z: its submerged part under the LOCAL wave elevation gives the
#              Froude-Krylov/hydrostatic force (equivalent-water-level form; Smith effect neglected, ASSUMED) at the
#              submerged centroid. Exact for any heel/trim as cell size -> 0.
#              Sources of meshes: (1) the PROJECT CAD — e.g. modular-pwc sealed_hydrostatic_envelope.brep
#              tessellated by tessellate_brep.py (FreeCAD) -> npz(V [m], F); (2) parametric hulls here (box,
#              prismatic deadrise planing hull) used for validation against closed-form results.
#              Units: metres. Hull (CAD) frame: +X forward from the transom, +Y port, +Z up from the keel.
import math

import numpy as np
import warp as wp

from engineering_sim.core import physics_device

K_MAX = 4   # max solid intervals per column


@wp.kernel
def _cast_up(mesh: wp.uint64, starts: wp.array(dtype=wp.vec3), max_t: float, n_max: int,
             out_t: wp.array2d(dtype=float), out_n: wp.array2d(dtype=wp.vec3), nhit: wp.array(dtype=int)):
    i = wp.tid()
    d = wp.vec3(0.0, 0.0, 1.0)
    off = float(0.0)
    cnt = int(0)
    for h in range(n_max):
        if cnt == h:
            q = wp.mesh_query_ray(mesh, starts[i] + d * off, d, max_t - off)
            if q.result:
                tt = off + q.t
                out_t[i, cnt] = tt
                out_n[i, cnt] = wp.normalize(q.normal)
                cnt += 1
                off = tt + 1.0e-5
    nhit[i] = cnt


def box_mesh(L, B, D):
    V = np.array([[x, y, z] for x in (0, L) for y in (-B / 2, B / 2) for z in (0, D)], float)
    F = [(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4), (2, 6, 7), (2, 7, 3), (0, 4, 6),
         (0, 6, 2), (1, 3, 7), (1, 7, 5)]
    return V, np.array(F, np.int32)


def prism_mesh(L, B, D, deadrise_deg):
    """Constant-deadrise prismatic planing hull: keel z=0 at y=0, hard chines at (+/-B/2, B/2 tan(beta)),
    vertical sides to the deck z=D, flat transom x=0 and flat bow x=L (validation geometry, Savitsky prism)."""
    zc = B / 2 * math.tan(math.radians(deadrise_deg))
    sec = np.array([[0, 0], [B / 2, zc], [B / 2, D], [-B / 2, D], [-B / 2, zc]], float)   # (y, z) CCW
    n = len(sec)
    V = np.array([[x, y, z] for x in (0.0, L) for (y, z) in sec], float)
    F = []
    for j in range(n):
        a, b = j, (j + 1) % n
        F += [(a, n + a, n + b), (a, n + b, b)]
    for j in range(1, n - 1):
        F += [(0, j + 1, j), (n, n + j, n + j + 1)]
    return V, np.array(F, np.int32)


def columns_from_mesh(V, F, nx=96, ny=40, device=None, jitter=1e-4):
    """Cast nx*ny upward rays through the closed mesh (hull frame, metres) -> column dict (hull frame)."""
    dev = physics_device(device)
    V = np.asarray(V, np.float64)
    lo, hi = V.min(0), V.max(0)
    dx, dy = (hi[0] - lo[0]) / nx, (hi[1] - lo[1]) / ny
    xs = lo[0] + (np.arange(nx) + 0.5) * dx
    ys = lo[1] + (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    X, Y = X.ravel() + jitter * 0.37, Y.ravel() + jitter * 0.61      # avoid exact edge/vertex hits
    z0 = lo[2] - 0.5
    starts = np.c_[X, Y, np.full_like(X, z0)].astype(np.float32)
    mesh = wp.Mesh(points=wp.array(V.astype(np.float32), dtype=wp.vec3, device=dev),
                   indices=wp.array(np.asarray(F, np.int32).ravel(), dtype=int, device=dev))
    n_max = 2 * K_MAX + 2
    out_t = wp.zeros((len(X), n_max), dtype=float, device=dev)
    out_n = wp.zeros((len(X), n_max), dtype=wp.vec3, device=dev)
    nhit = wp.zeros(len(X), dtype=int, device=dev)
    wp.launch(_cast_up, len(X), [mesh.id, wp.array(starts, dtype=wp.vec3, device=dev), float(hi[2] - z0 + 1.0),
                                 n_max, out_t, out_n, nhit], device=dev)
    T, N, H = out_t.numpy(), out_n.numpy(), nhit.numpy()
    keep, ints, bot_n, odd = [], [], [], 0
    for i in range(len(X)):
        h = H[i]
        if h < 2:
            continue
        if h % 2:
            odd += 1
            h -= 1
        zs = z0 + T[i, :h]
        iv = [(zs[j], zs[j + 1]) for j in range(0, h, 2) if zs[j + 1] - zs[j] > 1e-6][:K_MAX]
        if not iv:
            continue
        keep.append(i)
        ints.append(iv + [(0.0, 0.0)] * (K_MAX - len(iv)))
        bot_n.append(N[i, 0])
    keep = np.array(keep)
    cols = dict(x=X[keep], y=Y[keep], dx=dx, dy=dy, dA=dx * dy, intervals=np.array(ints, np.float64),
                bottom_normal=np.array(bot_n, np.float64), grid=(nx, ny), odd_ray_columns=odd,
                bounds=[lo.tolist(), hi.tolist()])
    cols["keel"] = keel_flags(cols)
    return cols


def keel_flags(cols, tol=1e-3):
    """Mark the lowest column of every x-station (the keel line used for Savitsky's wetted keel length)."""
    x = np.round(cols["x"] / cols["dx"]).astype(int)
    zb = cols["intervals"][:, 0, 0]
    flag = np.zeros(len(x), np.int32)
    for s in np.unique(x):
        idx = np.where(x == s)[0]
        flag[idx[np.argmin(zb[idx])]] = 1
    return flag


def volume_curve(cols, waterlines):
    """Even-keel hydrostatics from the columns (hull frame): volume, LCB/VCB, waterplane area, I_T, I_L."""
    out = []
    iv = cols["intervals"]
    for wl in np.atleast_1d(waterlines):
        sub = np.clip(wl - iv[:, :, 0], 0, iv[:, :, 1] - iv[:, :, 0])          # [Nc, K]
        vol_c = sub.sum(1) * cols["dA"]
        zc = (np.where(sub > 0, iv[:, :, 0] + sub / 2, 0) * sub).sum(1) / np.maximum(sub.sum(1), 1e-12)
        V = vol_c.sum()
        cut = ((iv[:, :, 0] < wl) & (iv[:, :, 1] > wl)).any(1)
        A = cut.sum() * cols["dA"]
        xf = (cols["x"][cut]).mean() if cut.any() else 0.0
        out.append(dict(waterline_m=float(wl), volume_m3=float(V),
                        lcb_m=float((vol_c * cols["x"]).sum() / max(V, 1e-12)),
                        vcb_m=float((vol_c * zc).sum() / max(V, 1e-12)), waterplane_m2=float(A), lcf_m=float(xf),
                        I_T_m4=float((cols["y"][cut] ** 2).sum() * cols["dA"]),
                        I_L_m4=float(((cols["x"][cut] - xf) ** 2).sum() * cols["dA"])))
    return out


def deadrise_stats(cols, beam_window=None):
    """Area-weighted local bottom deadrise [deg] from the first-hit normals (diagnostic for Savitsky's beta)."""
    n = cols["bottom_normal"]
    beta = np.degrees(np.arctan2(np.abs(n[:, 1]), np.abs(n[:, 2]) + 1e-12))
    m = np.ones(len(beta), bool) if beam_window is None else (np.abs(cols["y"]) <= beam_window / 2)
    return dict(mean_deg=float(beta[m].mean()), median_deg=float(np.median(beta[m])),
                p10_deg=float(np.percentile(beta[m], 10)), p90_deg=float(np.percentile(beta[m], 90)))

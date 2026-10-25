#!/usr/bin/env python3
# DESCRIPTION: After every World step the ContactManager gathers the bodies of ALL bound domains (GPU arrays copied
#              on the device; CPU flight-software domains through small host bindings), finds contacts between their
#              collision proxies and against the static terrain heightfield (incl. a carved lake bed), resolves them
#              with sequential impulses (restitution + Coulomb friction, parametric per material pair), pushes
#              penetrating bodies apart, applies a simple water-entry load (buoyancy + quadratic drag) to bodies that
#              enter the sea, logs impact/splash events and writes the velocity/position changes back to each domain.
#
#              PIPELINE (per contact step, physics GPU):
#                gather      domain state -> global body arrays X, Q, V, W(world)        (one kernel per GPU binding)
#                proxies     world centre / rotation / bounding radius per proxy
#                broadphase  wp.HashGrid (uniform spatial hash) over proxy centres; pair (i, j>i) kept if bounding
#                            spheres overlap, bodies differ, not both kinematic, not the same collision group
#                narrowphase sphere / capsule / box(OBB) proxies: sphere-sphere, sphere-capsule, capsule-capsule
#                            exact (closest points); sphere-box exact (closest point / min-face inside); capsule-box
#                            = 3 spheres on the axis; box-box = separating-axis test over the 15 axes (6 face normals +
#                            9 edge crosses) -> minimum-overlap axis: FACE -> the incident face of the other box clipped
#                            against the reference face's side planes (Sutherland-Hodgman), up to 8 contact points with
#                            their own depths; EDGE -> closest points of the two support edges, 1 point (2026-10-16 GAPS:
#                            was corners-inside-the-other only, which missed face-on contacts of equal boxes and put an
#                            offset face impact on 2 corners -> wrong lever arm, impulse 5 % low)
#                terrain     sphere/capsule ends vs the local tangent plane of the bilinear heightfield; box corners as
#                            points (r = 0). mu = material mu x surface peak friction D (core/terrain SURFACES), as the
#                            land domain's contact; e = min(material e, terrain restitution)
#                order       contacts are written in arbitrary GPU order, then RADIX-SORTED by a key built from the
#                            proxy pair (+ sub-contact index) -> the solve order and hence every result is
#                            deterministic (bitwise run-to-run) regardless of hash-grid / atomic ordering
#                solve       ONE GPU thread, sequential impulses over the sorted list, `iterations` sweeps:
#                            normal impulse to the target separating speed -e*vn0 (vn0 = approach speed before the
#                            step; e = 0 below restitution_threshold so resting contact does not jitter), accumulated
#                            and clamped >= 0; friction impulse along the tangential slip, accumulated and clamped to
#                            the Coulomb cone |jt| <= mu*jn. Effective mass uses the full rigid inverse inertia
#                            (world frame) at the contact point; point bodies have no rotational response.
#                            Then position projection: penetration beyond `slop` x `beta`, split by inverse mass and
#                            divided by the body's contact count (stops 4 box corners over-correcting)
#                water       bodies with the water flag, where the sea surface is above the terrain (lake) or no terrain
#                            is given (open sea): submerged fraction f from the centre depth over the body's equivalent
#                            diameter; buoyancy rho_w V f g, drag 0.5 rho_w Cd A f |v_rel| v_rel integrated implicitly
#                            (v_rel / (1 + k |v_rel| dt), unconditionally stable); v_rel against the wave orbital velocity.
#                            Vessels are NOT water-flagged: their buoyancy/hydrodynamics are the water domain's job
#                scatter     dV, dW, dX applied to the domains' own state (only bodies that were touched -> untouched
#                            bodies are bit-identical to a run without contact). The domains already advanced x, q with
#                            the PRE-contact velocity, so x also gets dV dt and q the rotation dW dt: the result equals
#                            a semi-implicit Euler step with the contact solved before the position update (no static-
#                            friction creep on slopes). Host (CPU) domains stepping at k x dt get the same dV dt
#                            (ASSUMED adequate: their contact correction is per world step).
#
#              DOMAIN CONTACT PROTOCOL (duck-typed, optional; core.world calls it once after upload):
#                domain.contact_bind(cm)   register bodies + proxies with ONE of
#                  cm.bind_rigid(name, x, q, v, w_body, idx, mass, inertia, proxies, ...)   GPU 6-DOF state layout of
#                      core/rigid6 (x world CG, q body->world xyzw, v world, w BODY rates) — Rigid6, land vehicles,
#                      land contact bodies, water vessels
#                  cm.bind_points(name, x, v, idx, mass, proxies, ...)                      GPU point masses (core bodies)
#                  cm.bind_host(name, n, get, put, mass, inertia, proxies, kinematic, ...)  CPU domains / prescribed
#                      motion: get() -> dict(x[n,3], v[n,3], q[n,4] xyzw | None, w[n,3] body | None);
#                      put(dv[n,3], dw_body[n,3], dx[n,3], touched[n]) applies the contact result
#                proxies: list of dict(body=k, shape="sphere"|"capsule"|"box", size=[r] | [r, half_len] | [hx,hy,hz],
#                         offset=[x,y,z] body frame, quat_wxyz (proxy axes in body frame; capsule axis = proxy +X),
#                         material="default", group=0)
#                per-binding flags: terrain (collide with the heightfield), water (buoyancy/drag), kinematic
#                         (infinite mass: prescribed motion, masts, nodes)
#
#              EVENTS: impact = a body pair (or body-terrain) coming into contact with approach speed >= event_speed;
#              splash = a water-flagged body entering water. Each event: t, kind, a, b, x, y, z, v_rel (m/s),
#              impulse (N s, summed over that pair's contacts in the step). World.run writes them to manifest["events"].
#              Every number above is a parameter (scenario `contact:` block); defaults are ASSUMED (see DEFAULTS).
import math

import numpy as np
import warp as wp

from . import physics_device
from . import terrain as T
from .constants import G0

wp.set_module_options({"enable_backward": False})

SPHERE, CAPSULE, BOX = 0, 1, 2
SHAPES = dict(sphere=SPHERE, capsule=CAPSULE, box=BOX)
F_BODY, F_TERRAIN, F_WATER = 1, 2, 4
INT_MAX = 2147483647

# material: (mu, e). ASSUMED representative values (dry contact; e at impact speeds of a few m/s), parameters.
MATERIALS = {
    "default":   (0.5, 0.3),
    "rubber":    (0.9, 0.6),
    "steel":     (0.4, 0.5),
    "aluminium": (0.4, 0.4),
    "composite": (0.4, 0.3),   # CF / GRP structures, drone frames, dart bodies
    "hull":      (0.3, 0.2),   # PWC / boat hull (polyethylene / GRP)
    "plastic":   (0.35, 0.4),
    "soft":      (0.6, 0.05),  # sandbags, crates with padding
}
DEFAULTS = dict(iterations=8, every=1, terrain=True, water=True, restitution_threshold=0.1, slop=1e-3, beta=0.8,
                terrain_restitution=0.2, event_speed=0.5, water_rho=1000.0, water_cd=1.0, max_contacts=None,
                grid_dim=32)


@wp.struct
class Contact:
    a: int
    b: int            # -1 = static terrain
    pi: int
    pj: int           # -1 = terrain
    p: wp.vec3        # world contact point
    n: wp.vec3        # unit normal, pushes body a away from b
    d: float          # penetration depth [m]
    mu: float
    e: float
    vn0: float        # normal relative speed before the solve (< 0 = approaching)
    jn: float         # accumulated normal impulse [N s]
    jt: wp.vec3       # accumulated friction impulse [N s]


# ------------------------------------------------------------------------------------------------ gather / scatter
@wp.kernel
def _gather_rigid(off: int, idx: wp.array(dtype=int), x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.quat),
                  v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), X: wp.array(dtype=wp.vec3),
                  Q: wp.array(dtype=wp.quat), V: wp.array(dtype=wp.vec3), W: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    i = idx[k]
    qq = q[i]
    X[off + k] = x[i]
    Q[off + k] = qq
    V[off + k] = v[i]
    W[off + k] = wp.quat_rotate(qq, w[i])


@wp.kernel
def _gather_points(off: int, idx: wp.array(dtype=int), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                   X: wp.array(dtype=wp.vec3), Q: wp.array(dtype=wp.quat), V: wp.array(dtype=wp.vec3),
                   W: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    i = idx[k]
    X[off + k] = x[i]
    Q[off + k] = wp.quat_identity()
    V[off + k] = v[i]
    W[off + k] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def _finish(dt: float, touched: wp.array(dtype=int), dV: wp.array(dtype=wp.vec3), dX: wp.array(dtype=wp.vec3)):
    """The domains already advanced x with the pre-contact velocity (x += v dt); the semi-implicit update with the
    post-contact velocity needs x += (v + dV) dt -> add dV dt (= solving contact before the position update; without
    it a body held by static friction on a slope creeps by g dt^2 sin(th) per step)."""
    i = wp.tid()
    if touched[i] == 1:
        dX[i] = dX[i] + dV[i] * dt


@wp.kernel
def _scatter_rigid(off: int, dt: float, idx: wp.array(dtype=int), touched: wp.array(dtype=int),
                   dV: wp.array(dtype=wp.vec3), dW: wp.array(dtype=wp.vec3), dX: wp.array(dtype=wp.vec3),
                   x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.quat), v: wp.array(dtype=wp.vec3),
                   w: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    if touched[off + k] == 0:
        return
    i = idx[k]
    v[i] = v[i] + dV[off + k]
    x[i] = x[i] + dX[off + k]
    qi = q[i]
    dwb = wp.quat_rotate_inv(qi, dW[off + k])
    w[i] = w[i] + dwb
    om = wp.length(dwb)
    if om > 1.0e-12:                      # same semi-implicit correction for the attitude: rotate by dW dt
        q[i] = wp.normalize(qi * wp.quat_from_axis_angle(dwb / om, om * dt))


@wp.kernel
def _scatter_points(off: int, idx: wp.array(dtype=int), touched: wp.array(dtype=int), dV: wp.array(dtype=wp.vec3),
                    dX: wp.array(dtype=wp.vec3), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    if touched[off + k] == 0:
        return
    i = idx[k]
    v[i] = v[i] + dV[off + k]
    x[i] = x[i] + dX[off + k]


@wp.kernel
def _merge_host(mask: wp.array(dtype=int), sX: wp.array(dtype=wp.vec3), sQ: wp.array(dtype=wp.quat),
                sV: wp.array(dtype=wp.vec3), sW: wp.array(dtype=wp.vec3), X: wp.array(dtype=wp.vec3),
                Q: wp.array(dtype=wp.quat), V: wp.array(dtype=wp.vec3), W: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if mask[i] == 1:
        X[i] = sX[i]
        Q[i] = sQ[i]
        V[i] = sV[i]
        W[i] = sW[i]


@wp.kernel
def _clear(dV: wp.array(dtype=wp.vec3), dW: wp.array(dtype=wp.vec3), dX: wp.array(dtype=wp.vec3),
           touched: wp.array(dtype=int), ncnt: wp.array(dtype=int)):
    i = wp.tid()
    dV[i] = wp.vec3(0.0, 0.0, 0.0)
    dW[i] = wp.vec3(0.0, 0.0, 0.0)
    dX[i] = wp.vec3(0.0, 0.0, 0.0)
    touched[i] = 0
    ncnt[i] = 0


@wp.kernel
def _clear_keys(keys: wp.array(dtype=int), vals: wp.array(dtype=int)):
    i = wp.tid()
    keys[i] = 2147483647
    vals[i] = i


# ------------------------------------------------------------------------------------------------ proxies
@wp.kernel
def _proxy_world(pb: wp.array(dtype=int), poff: wp.array(dtype=wp.vec3), prot: wp.array(dtype=wp.quat),
                 ptype: wp.array(dtype=int), psize: wp.array(dtype=wp.vec3), X: wp.array(dtype=wp.vec3),
                 Q: wp.array(dtype=wp.quat), pc: wp.array(dtype=wp.vec3), pq: wp.array(dtype=wp.quat),
                 prad: wp.array(dtype=float)):
    i = wp.tid()
    b = pb[i]
    qb = Q[b]
    pc[i] = X[b] + wp.quat_rotate(qb, poff[i])
    pq[i] = qb * prot[i]
    s = psize[i]
    t = ptype[i]
    r = s[0]
    if t == 1:
        r = s[0] + s[1]
    if t == 2:
        r = wp.length(s)
    prad[i] = r


@wp.func
def _emit(c: Contact, key: int, cont: wp.array(dtype=Contact), keys: wp.array(dtype=int), cnt: wp.array(dtype=int),
          cap: int):
    slot = wp.atomic_add(cnt, 0, 1)
    if slot < cap:
        cont[slot] = c
        keys[slot] = key


@wp.func
def _mk(a: int, b: int, pi: int, pj: int, p: wp.vec3, n: wp.vec3, d: float, mu: float, e: float):
    c = Contact()
    c.a = a
    c.b = b
    c.pi = pi
    c.pj = pj
    c.p = p
    c.n = n
    c.d = d
    c.mu = mu
    c.e = e
    c.vn0 = 0.0
    c.jn = 0.0
    c.jt = wp.vec3(0.0, 0.0, 0.0)
    return c


@wp.func
def _seg_point(p: wp.vec3, a: wp.vec3, b: wp.vec3):
    ab = b - a
    t = wp.clamp(wp.dot(p - a, ab) / wp.max(wp.dot(ab, ab), 1.0e-12), 0.0, 1.0)
    return a + ab * t


@wp.func
def _seg_seg(p1: wp.vec3, q1: wp.vec3, p2: wp.vec3, q2: wp.vec3):
    """Closest points between segments p1q1 and p2q2 (Ericson, Real-Time Collision Detection 5.1.9)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)
    s = float(0.0)
    t = float(0.0)
    if a <= 1.0e-12 and e <= 1.0e-12:
        s = 0.0
        t = 0.0
    elif a <= 1.0e-12:
        s = 0.0
        t = wp.clamp(f / e, 0.0, 1.0)
    else:
        c = wp.dot(d1, r)
        if e <= 1.0e-12:
            t = 0.0
            s = wp.clamp(-c / a, 0.0, 1.0)
        else:
            b = wp.dot(d1, d2)
            den = a * e - b * b
            if den > 1.0e-12:
                s = wp.clamp((b * f - c * e) / den, 0.0, 1.0)
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = wp.clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = wp.clamp((b - c) / a, 0.0, 1.0)
    return p1 + d1 * s, p2 + d2 * t


@wp.func
def _sph_sph(ca: wp.vec3, ra: float, cb: wp.vec3, rb: float):
    """-> (depth, normal b->a, point). depth <= 0: no contact."""
    dlt = ca - cb
    dist = wp.length(dlt)
    n = wp.vec3(0.0, 0.0, 1.0)
    if dist > 1.0e-9:
        n = dlt / dist
    depth = ra + rb - dist
    return depth, n, cb + n * (rb - 0.5 * depth)


@wp.func
def _sph_box(cs: wp.vec3, r: float, cb: wp.vec3, qb: wp.quat, h: wp.vec3):
    """Sphere vs OBB -> (depth, normal box->sphere, point on the box surface)."""
    l = wp.quat_rotate_inv(qb, cs - cb)
    cl = wp.vec3(wp.clamp(l[0], -h[0], h[0]), wp.clamp(l[1], -h[1], h[1]), wp.clamp(l[2], -h[2], h[2]))
    dl = l - cl
    dist = wp.length(dl)
    depth = float(-1.0)
    nl = wp.vec3(0.0, 0.0, 1.0)
    pl = cl
    if dist > 1.0e-9:
        nl = dl / dist
        depth = r - dist
    else:                                     # centre inside the box: push out through the nearest face
        ax = int(0)
        m = h[0] - wp.abs(l[0])
        if h[1] - wp.abs(l[1]) < m:
            ax = 1
            m = h[1] - wp.abs(l[1])
        if h[2] - wp.abs(l[2]) < m:
            ax = 2
            m = h[2] - wp.abs(l[2])
        sg = float(1.0)
        if l[ax] < 0.0:
            sg = -1.0
        nl = wp.vec3(0.0, 0.0, 0.0)
        nl[ax] = sg
        pl = l
        pl[ax] = sg * h[ax]
        depth = r + m
    return depth, wp.quat_rotate(qb, nl), cb + wp.quat_rotate(qb, pl)


@wp.func
def _corner(h: wp.vec3, k: int):
    sx = float(1.0)
    sy = float(1.0)
    sz = float(1.0)
    if (k & 1) != 0:
        sx = -1.0
    if (k & 2) != 0:
        sy = -1.0
    if (k & 4) != 0:
        sz = -1.0
    return wp.vec3(sx * h[0], sy * h[1], sz * h[2])


@wp.func
def _pt_in_box(p: wp.vec3, cb: wp.vec3, qb: wp.quat, h: wp.vec3):
    """-> (depth, outward normal of B at the nearest face) ; depth <= 0: point outside."""
    l = wp.quat_rotate_inv(qb, p - cb)
    dx = h[0] - wp.abs(l[0])
    dy = h[1] - wp.abs(l[1])
    dz = h[2] - wp.abs(l[2])
    m = wp.min(dx, wp.min(dy, dz))
    nl = wp.vec3(0.0, 0.0, 0.0)
    ax = int(2)
    if m == dx:
        ax = 0
    elif m == dy:
        ax = 1
    if l[ax] < 0.0:
        nl[ax] = -1.0
    else:
        nl[ax] = 1.0
    return m, wp.quat_rotate(qb, nl)


mat83 = wp.types.matrix(shape=(8, 3), dtype=float)
vec8 = wp.types.vector(length=8, dtype=float)
CLIP_TOL = 1.0e-5          # [m] a vertex ON a side plane (equal boxes, face-on) counts as inside


@wp.func
def _bax(q: wp.quat, k: int):
    """Box axis k (0, 1, 2) in the world frame."""
    if k == 0:
        return wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    if k == 1:
        return wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    return wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))


@wp.func
def _proj(q: wp.quat, h: wp.vec3, L: wp.vec3):
    """Half-extent of an OBB projected on the unit axis L."""
    return (h[0] * wp.abs(wp.dot(_bax(q, 0), L)) + h[1] * wp.abs(wp.dot(_bax(q, 1), L)) +
            h[2] * wp.abs(wp.dot(_bax(q, 2), L)))


@wp.func
def _row(P: mat83, i: int):
    return wp.vec3(P[i, 0], P[i, 1], P[i, 2])


@wp.func
def _clip(P: mat83, n: int, u: wp.vec3, off: float):
    """Sutherland-Hodgman: keep the part of polygon P[0:n] with dot(u, p) <= off (+ CLIP_TOL)."""
    Q = mat83()
    m = int(0)
    for i in range(n):
        j = i + 1
        if j == n:
            j = 0
        a = _row(P, i)
        b = _row(P, j)
        da = wp.dot(u, a) - off - CLIP_TOL
        db = wp.dot(u, b) - off - CLIP_TOL
        if da <= 0.0 and m < 8:
            Q[m, 0] = a[0]
            Q[m, 1] = a[1]
            Q[m, 2] = a[2]
            m = m + 1
        if ((da < 0.0 and db > 0.0) or (da > 0.0 and db < 0.0)) and m < 8:
            x = a + (b - a) * (da / (da - db))
            Q[m, 0] = x[0]
            Q[m, 1] = x[1]
            Q[m, 2] = x[2]
            m = m + 1
    return Q, m


@wp.func
def _face_manifold(cr: wp.vec3, qr: wp.quat, hr: wp.vec3, ir: int, nrm: wp.vec3, ci: wp.vec3, qi: wp.quat,
                   hi: wp.vec3):
    """Reference box R (face ir, outward normal nrm toward box I) vs incident box I: the face of I most anti-parallel
    to nrm, clipped to R's face rectangle; returns (count, points on I's face, depth of each below R's face)."""
    jb = int(0)
    best = float(-1.0)
    for k in range(3):
        c = wp.abs(wp.dot(_bax(qi, k), nrm))
        if c > best:
            best = c
            jb = k
    uj = _bax(qi, jb)
    sg = float(1.0)
    if wp.dot(uj, nrm) > 0.0:
        sg = -1.0
    j1 = (jb + 1) % 3
    j2 = (jb + 2) % 3
    fc = ci + uj * (hi[jb] * sg)
    e1 = _bax(qi, j1) * hi[j1]
    e2 = _bax(qi, j2) * hi[j2]
    P = mat83()
    for k in range(4):
        s1 = float(1.0)
        s2 = float(1.0)
        if k == 1 or k == 2:
            s1 = -1.0
        if k >= 2:
            s2 = -1.0
        v = fc + e1 * s1 + e2 * s2
        P[k, 0] = v[0]
        P[k, 1] = v[1]
        P[k, 2] = v[2]
    n = int(4)
    i1 = (ir + 1) % 3
    i2 = (ir + 2) % 3
    u1 = _bax(qr, i1)
    u2 = _bax(qr, i2)
    P, n = _clip(P, n, u1, hr[i1] + wp.dot(u1, cr))
    P, n = _clip(P, n, -u1, hr[i1] - wp.dot(u1, cr))
    P, n = _clip(P, n, u2, hr[i2] + wp.dot(u2, cr))
    P, n = _clip(P, n, -u2, hr[i2] - wp.dot(u2, cr))
    D = vec8()
    for k in range(n):
        D[k] = hr[ir] - wp.dot(nrm, _row(P, k) - cr)
    return n, P, D


@wp.func
def _box_box(ca: wp.vec3, qa: wp.quat, ha: wp.vec3, cb: wp.vec3, qb: wp.quat, hb: wp.vec3):
    """OBB-OBB separating-axis test + contact manifold. Returns (count, points, depths, normal pushing A away from B,
    kind 0 none / 1 face of A / 2 face of B / 3 edge-edge). Contact points are midway between the two surfaces."""
    dl = cb - ca
    sep = int(0)
    fa = float(1.0e30)
    ia = int(0)
    fb = float(1.0e30)
    ib = int(0)
    fe = float(1.0e30)
    ea = int(0)
    eb = int(0)
    for k in range(3):
        L = _bax(qa, k)
        o = ha[k] + _proj(qb, hb, L) - wp.abs(wp.dot(dl, L))
        if o < 0.0:
            sep = 1
        if o < fa:
            fa = o
            ia = k
        L2 = _bax(qb, k)
        o2 = _proj(qa, ha, L2) + hb[k] - wp.abs(wp.dot(dl, L2))
        if o2 < 0.0:
            sep = 1
        if o2 < fb:
            fb = o2
            ib = k
    for i in range(3):
        for j in range(3):
            L = wp.cross(_bax(qa, i), _bax(qb, j))
            ln = wp.length(L)
            if ln > 1.0e-5:
                L = L / ln
                o = _proj(qa, ha, L) + _proj(qb, hb, L) - wp.abs(wp.dot(dl, L))
                if o < 0.0:
                    sep = 1
                if o < fe:
                    fe = o
                    ea = i
                    eb = j
    P = mat83()
    D = vec8()
    n = int(0)
    nrm = wp.vec3(0.0, 0.0, 1.0)
    kind = int(0)
    if sep == 0:
        ff = wp.min(fa, fb)
        if fe < 0.95 * ff - 1.0e-4:                        # edge-edge (face axes preferred within 5 %: stable manifolds)
            L = wp.normalize(wp.cross(_bax(qa, ea), _bax(qb, eb)))
            if wp.dot(dl, L) < 0.0:
                L = -L
            pa = ca
            pb = cb
            for k in range(3):
                if k != ea:
                    u = _bax(qa, k)
                    s = float(1.0)
                    if wp.dot(u, L) < 0.0:
                        s = -1.0
                    pa = pa + u * (ha[k] * s)
                if k != eb:
                    u2 = _bax(qb, k)
                    s2 = float(1.0)
                    if wp.dot(u2, L) > 0.0:
                        s2 = -1.0
                    pb = pb + u2 * (hb[k] * s2)
            da = _bax(qa, ea) * ha[ea]
            db = _bax(qb, eb) * hb[eb]
            c1, c2 = _seg_seg(pa - da, pa + da, pb - db, pb + db)
            v = 0.5 * (c1 + c2)
            P[0, 0] = v[0]
            P[0, 1] = v[1]
            P[0, 2] = v[2]
            D[0] = fe
            n = 1
            nrm = -L
            kind = 3
        elif fa <= fb * 1.0001 + 1.0e-7:                     # reference face on A (normal A -> B)
            L = _bax(qa, ia)
            if wp.dot(dl, L) < 0.0:
                L = -L
            n, P, D = _face_manifold(ca, qa, ha, ia, L, cb, qb, hb)
            for k in range(n):                               # B's face points -> midway into A's face
                v = _row(P, k) + L * (0.5 * D[k])
                P[k, 0] = v[0]
                P[k, 1] = v[1]
                P[k, 2] = v[2]
            nrm = -L
            kind = 1
        else:                                                # reference face on B (normal B -> A)
            L = _bax(qb, ib)
            if wp.dot(dl, L) > 0.0:
                L = -L
            n, P, D = _face_manifold(cb, qb, hb, ib, L, ca, qa, ha)
            for k in range(n):
                v = _row(P, k) + L * (0.5 * D[k])
                P[k, 0] = v[0]
                P[k, 1] = v[1]
                P[k, 2] = v[2]
            nrm = L
            kind = 2
    return n, P, D, nrm, kind


@wp.func
def _cap_ends(c: wp.vec3, q: wp.quat, hl: float, k: int):
    """k = 0, 1, 2 -> end, centre, other end of a capsule axis (proxy +X)."""
    ax = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    return c + ax * (hl * float(k - 1))


@wp.kernel
def _pairs(grid: wp.uint64, rmax: float, P: int, pb: wp.array(dtype=int), ptype: wp.array(dtype=int),
           psize: wp.array(dtype=wp.vec3), pflag: wp.array(dtype=int), pgrp: wp.array(dtype=int),
           pmat: wp.array(dtype=int), pc: wp.array(dtype=wp.vec3), pq: wp.array(dtype=wp.quat),
           prad: wp.array(dtype=float), kin: wp.array(dtype=int), mmu: wp.array2d(dtype=float),
           me: wp.array2d(dtype=float), cont: wp.array(dtype=Contact), keys: wp.array(dtype=int),
           cnt: wp.array(dtype=int), cap: int):
    i = wp.tid()
    if (pflag[i] & 1) == 0:
        return
    q = wp.hash_grid_query(grid, pc[i], prad[i] + rmax)
    j = int(0)
    while wp.hash_grid_query_next(q, j):
        if j <= i:
            continue
        if (pflag[j] & 1) == 0:
            continue
        bi = pb[i]
        bj = pb[j]
        if bi == bj:
            continue
        if pgrp[i] != 0 and pgrp[i] == pgrp[j]:
            continue
        if kin[bi] == 1 and kin[bj] == 1:
            continue
        if wp.length(pc[i] - pc[j]) > prad[i] + prad[j]:
            continue
        # canonical order: A = lower shape type (sphere < capsule < box); normal always pushes A away from B
        ia = i
        ib = j
        if ptype[j] < ptype[i]:
            ia = j
            ib = i
        ta = ptype[ia]
        tb = ptype[ib]
        A = pb[ia]
        B = pb[ib]
        sa = psize[ia]
        sb = psize[ib]
        ca = pc[ia]
        cb = pc[ib]
        mu = mmu[pmat[ia], pmat[ib]]
        e = me[pmat[ia], pmat[ib]]
        kbase = (i * (P + 1) + j) * 16
        if ta == 0 and tb == 0:
            d, n, p = _sph_sph(ca, sa[0], cb, sb[0])
            if d > 0.0:
                _emit(_mk(A, B, ia, ib, p, n, d, mu, e), kbase, cont, keys, cnt, cap)
        elif ta == 0 and tb == 1:
            cpt = _seg_point(ca, _cap_ends(cb, pq[ib], sb[1], 0), _cap_ends(cb, pq[ib], sb[1], 2))
            d, n, p = _sph_sph(ca, sa[0], cpt, sb[0])
            if d > 0.0:
                _emit(_mk(A, B, ia, ib, p, n, d, mu, e), kbase, cont, keys, cnt, cap)
        elif ta == 1 and tb == 1:
            c1, c2 = _seg_seg(_cap_ends(ca, pq[ia], sa[1], 0), _cap_ends(ca, pq[ia], sa[1], 2),
                              _cap_ends(cb, pq[ib], sb[1], 0), _cap_ends(cb, pq[ib], sb[1], 2))
            d, n, p = _sph_sph(c1, sa[0], c2, sb[0])
            if d > 0.0:
                _emit(_mk(A, B, ia, ib, p, n, d, mu, e), kbase, cont, keys, cnt, cap)
        elif ta == 0 and tb == 2:
            d, n, p = _sph_box(ca, sa[0], cb, pq[ib], sb)
            if d > 0.0:
                _emit(_mk(A, B, ia, ib, p, n, d, mu, e), kbase, cont, keys, cnt, cap)
        elif ta == 1 and tb == 2:
            for k in range(3):
                d, n, p = _sph_box(_cap_ends(ca, pq[ia], sa[1], k), sa[0], cb, pq[ib], sb)
                if d > 0.0:
                    _emit(_mk(A, B, ia, ib, p, n, d, mu, e), kbase + k, cont, keys, cnt, cap)
        else:                                   # box-box: SAT + clipped face manifold (or one edge-edge point)
            nb, Pm, Dm, nm, kd = _box_box(ca, pq[ia], sa, cb, pq[ib], sb)
            for k in range(nb):
                if Dm[k] > 0.0:
                    _emit(_mk(A, B, ia, ib, _row(Pm, k), nm, Dm[k], mu, e), kbase + k, cont, keys, cnt, cap)


@wp.kernel
def _terrain(P: int, tp: T.TerrainParams, e_ter: float, pb: wp.array(dtype=int), ptype: wp.array(dtype=int),
             psize: wp.array(dtype=wp.vec3), pflag: wp.array(dtype=int), pmat: wp.array(dtype=int),
             pc: wp.array(dtype=wp.vec3), pq: wp.array(dtype=wp.quat), kin: wp.array(dtype=int),
             mat_mu: wp.array(dtype=float), mat_e: wp.array(dtype=float), cont: wp.array(dtype=Contact),
             keys: wp.array(dtype=int), cnt: wp.array(dtype=int), cap: int):
    i = wp.tid()
    if (pflag[i] & 2) == 0:
        return
    b = pb[i]
    if kin[b] == 1:
        return
    t = ptype[i]
    s = psize[i]
    c = pc[i]
    e = wp.min(mat_e[pmat[i]], e_ter)
    kbase = (i * (P + 1) + P) * 16
    n_pts = 1
    if t == 1:
        n_pts = 3
    if t == 2:
        n_pts = 8
    for k in range(n_pts):
        p = c
        r = s[0]
        if t == 1:
            p = _cap_ends(c, pq[i], s[1], k)
        if t == 2:
            p = c + wp.quat_rotate(pq[i], _corner(s, k))
            r = 0.0
        hn = T.height_normal(p[0], p[1], tp)
        n = wp.vec3(hn[0], hn[1], hn[2])
        dist = (p[2] - hn[3]) * n[2]
        depth = r - dist
        if depth > 0.0:
            mu = mat_mu[pmat[i]] * tp.mf[T.material(p[0], p[1], tp)][2]
            _emit(_mk(b, -1, i, -1, p - n * r, n, depth, mu, e), kbase + k, cont, keys, cnt, cap)


# ------------------------------------------------------------------------------------------------ solve
@wp.func
def _iinv(q: wp.quat, invI: wp.vec3, x: wp.vec3):
    return wp.quat_rotate(q, wp.cw_mul(invI, wp.quat_rotate_inv(q, x)))


@wp.func
def _vel_at(k: int, p: wp.vec3, X: wp.array(dtype=wp.vec3), V: wp.array(dtype=wp.vec3), W: wp.array(dtype=wp.vec3),
            dV: wp.array(dtype=wp.vec3), dW: wp.array(dtype=wp.vec3)):
    if k < 0:
        return wp.vec3(0.0, 0.0, 0.0)
    return V[k] + dV[k] + wp.cross(W[k] + dW[k], p - X[k])


@wp.func
def _k_dir(k: int, p: wp.vec3, d: wp.vec3, X: wp.array(dtype=wp.vec3), Q: wp.array(dtype=wp.quat),
           invm: wp.array(dtype=float), invI: wp.array(dtype=wp.vec3)):
    if k < 0:
        return 0.0
    r = p - X[k]
    return invm[k] + wp.dot(wp.cross(_iinv(Q[k], invI[k], wp.cross(r, d)), r), d)


@wp.func
def _apply(k: int, p: wp.vec3, J: wp.vec3, X: wp.array(dtype=wp.vec3), Q: wp.array(dtype=wp.quat),
           invm: wp.array(dtype=float), invI: wp.array(dtype=wp.vec3), dV: wp.array(dtype=wp.vec3),
           dW: wp.array(dtype=wp.vec3), touched: wp.array(dtype=int)):
    if k < 0:
        return
    if invm[k] == 0.0:
        return
    dV[k] = dV[k] + J * invm[k]
    dW[k] = dW[k] + _iinv(Q[k], invI[k], wp.cross(p - X[k], J))
    touched[k] = 1


@wp.kernel
def _solve(cnt: wp.array(dtype=int), cap: int, order: wp.array(dtype=int), cont: wp.array(dtype=Contact),
           iters: int, v_rest: float, slop: float, beta: float, X: wp.array(dtype=wp.vec3), Q: wp.array(dtype=wp.quat),
           V: wp.array(dtype=wp.vec3), W: wp.array(dtype=wp.vec3), invm: wp.array(dtype=float),
           invI: wp.array(dtype=wp.vec3), dV: wp.array(dtype=wp.vec3), dW: wp.array(dtype=wp.vec3),
           dX: wp.array(dtype=wp.vec3), touched: wp.array(dtype=int), ncnt: wp.array(dtype=int)):
    n = wp.min(cnt[0], cap)
    for k in range(n):                                   # approach speeds before any impulse of this step
        ci = order[k]
        c = cont[ci]
        vr = _vel_at(c.a, c.p, X, V, W, dV, dW) - _vel_at(c.b, c.p, X, V, W, dV, dW)
        c.vn0 = wp.dot(vr, c.n)
        cont[ci] = c
        ncnt[c.a] = ncnt[c.a] + 1
        if c.b >= 0:
            ncnt[c.b] = ncnt[c.b] + 1
    for it in range(iters):
        for k in range(n):
            ci = order[k]
            c = cont[ci]
            a = c.a
            b = c.b
            nrm = c.n
            p = c.p
            vr = _vel_at(a, p, X, V, W, dV, dW) - _vel_at(b, p, X, V, W, dV, dW)
            vn = wp.dot(vr, nrm)
            kn = _k_dir(a, p, nrm, X, Q, invm, invI) + _k_dir(b, p, nrm, X, Q, invm, invI)
            if kn <= 0.0:
                continue
            tgt = float(0.0)
            if c.vn0 < -v_rest:
                tgt = -c.e * c.vn0
            jn_new = wp.max(c.jn + (tgt - vn) / kn, 0.0)
            dj = jn_new - c.jn
            c.jn = jn_new
            _apply(a, p, nrm * dj, X, Q, invm, invI, dV, dW, touched)
            _apply(b, p, nrm * (-dj), X, Q, invm, invI, dV, dW, touched)
            # Coulomb friction (2-D, along the current slip direction), accumulated in the cone |jt| <= mu jn
            vr = _vel_at(a, p, X, V, W, dV, dW) - _vel_at(b, p, X, V, W, dV, dW)
            vt = vr - nrm * wp.dot(vr, nrm)
            vts = wp.length(vt)
            if vts > 1.0e-9 and c.mu > 0.0:
                tdir = vt / vts
                kt = _k_dir(a, p, tdir, X, Q, invm, invI) + _k_dir(b, p, tdir, X, Q, invm, invI)
                if kt > 0.0:
                    jt_new = c.jt - tdir * (vts / kt)
                    lim = c.mu * c.jn
                    m = wp.length(jt_new)
                    if m > lim:
                        jt_new = jt_new * (lim / m)
                    dJ = jt_new - c.jt
                    c.jt = jt_new
                    _apply(a, p, dJ, X, Q, invm, invI, dV, dW, touched)
                    _apply(b, p, -dJ, X, Q, invm, invI, dV, dW, touched)
            cont[ci] = c
    for k in range(n):                                   # position projection (linear), split by inverse mass
        c = cont[order[k]]
        corr = wp.max(c.d - slop, 0.0) * beta
        if corr <= 0.0:
            continue
        wa = invm[c.a]
        wb = float(0.0)
        if c.b >= 0:
            wb = invm[c.b]
        if wa + wb <= 0.0:
            continue
        if wa > 0.0:
            dX[c.a] = dX[c.a] + c.n * (corr * wa / (wa + wb) / float(ncnt[c.a]))
            touched[c.a] = 1
        if wb > 0.0:
            dX[c.b] = dX[c.b] - c.n * (corr * wb / (wa + wb) / float(ncnt[c.b]))
            touched[c.b] = 1


try:                                   # sea-state sampling for the water-entry hook (domains/water/waves)
    from engineering_sim.domains.water import waves as _WV
except Exception:                      # pragma: no cover  (core used without the domains package on the path)
    _WV = None

if _WV is not None:
    @wp.kernel
    def _water_k(t: float, dt: float, has_ter: int, tp: T.TerrainParams, wv: _WV.WaveParams,
                 kv: wp.array(dtype=wp.vec3), om: wp.array(dtype=float), amp: wp.array(dtype=float),
                 ph: wp.array(dtype=float), wflag: wp.array(dtype=int), wr: wp.array(dtype=float),
                 wvol: wp.array(dtype=float), warea: wp.array(dtype=float), rho_w: float, cd: float,
                 X: wp.array(dtype=wp.vec3), V: wp.array(dtype=wp.vec3), invm: wp.array(dtype=float),
                 dV: wp.array(dtype=wp.vec3), touched: wp.array(dtype=int), wet: wp.array(dtype=int)):
        i = wp.tid()
        if wflag[i] == 0 or invm[i] == 0.0:
            wet[i] = 0
            return
        p = X[i]
        s = _WV.wave_sample(p[0], p[1], p[2], t, wv, kv, om, amp, ph)
        eta = s[0]
        if has_ter == 1:
            if T.height(p[0], p[1], tp) >= eta:
                wet[i] = 0
                return
        r = wr[i]
        f = wp.clamp((eta - (p[2] - r)) / (2.0 * r), 0.0, 1.0)
        if f <= 0.0:
            wet[i] = 0
            return
        wet[i] = 1
        u = wp.vec3(s[3], s[4], s[5])
        v = V[i] + dV[i]
        vr = v - u
        k = 0.5 * rho_w * cd * warea[i] * f * invm[i]
        vrn = vr / (1.0 + k * wp.length(vr) * dt)
        buoy = wp.vec3(0.0, 0.0, rho_w * wvol[i] * f * G0 * invm[i] * dt)
        dV[i] = dV[i] + (vrn - vr) + buoy
        touched[i] = 1


# ------------------------------------------------------------------------------------------------ manager
class ContactManager:
    """Cross-domain contact. cfg = the scenario `contact:` block (all parameters; DEFAULTS are ASSUMED)."""

    def __init__(self, cfg=None, terrain=None, sea=None, device=None):
        cfg = dict(cfg or {})
        self.cfg = dict(DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS})
        self.terrain, self.sea = terrain, sea
        self.device = physics_device(device)
        mats = dict(MATERIALS)
        for k, v in (cfg.get("materials") or {}).items():
            mats[k] = (float(v.get("mu", mats.get(k, MATERIALS["default"])[0])),
                       float(v.get("e", mats.get(k, MATERIALS["default"])[1])))
        self.mat_names = list(mats)
        self.mats = mats
        n = len(mats)
        mu = np.zeros((n, n), np.float32)
        e = np.zeros((n, n), np.float32)
        for i, a in enumerate(self.mat_names):
            for j, b in enumerate(self.mat_names):
                mu[i, j] = math.sqrt(mats[a][0] * mats[b][0])     # ASSUMED combine rules (pair table overrides)
                e[i, j] = 0.5 * (mats[a][1] + mats[b][1])
        for pr in cfg.get("pairs") or []:
            i, j = self.mat_names.index(pr["a"]), self.mat_names.index(pr["b"])
            for (u, w) in ((i, j), (j, i)):
                if "mu" in pr:
                    mu[u, w] = pr["mu"]
                if "e" in pr:
                    e[u, w] = pr["e"]
        self.mu_tab, self.e_tab = mu, e
        self.bindings, self.proxies, self.names = [], [], []
        self.mass, self.inertia, self.kin, self.water = [], [], [], []
        self.events, self.active, self.wet_prev = [], set(), None
        self.n_steps, self.n_contacts_max, self.overflow, self.k = 0, 0, 0, 0
        self.t = 0.0

    # ---------------------------------------------------------------- binding API (domain contact protocol)
    def _add_bodies(self, names, mass, inertia, kinematic, water):
        b0 = len(self.names)
        for k, nm in enumerate(names):
            self.names.append(nm)
            m = float(mass[k]) if mass is not None else 0.0
            self.mass.append(m)
            I = inertia[k] if inertia is not None else None
            self.inertia.append(None if I is None else [float(c) for c in I])
            self.kin.append(1 if kinematic or m <= 0 else 0)
            self.water.append(1 if water else 0)
        return b0

    def _add_proxies(self, b0, proxies, terrain, flags_extra=0):
        for p in proxies:
            shape = SHAPES[p.get("shape", "sphere")]
            size = list(map(float, p.get("size", [0.1])))
            size = (size + [0.0, 0.0, 0.0])[:3]
            mat = p.get("material", "default")
            if mat not in self.mat_names:
                raise ValueError("contact material %r (known: %s; add it under contact.materials)" % (
                    mat, ", ".join(self.mat_names)))
            qw = p.get("quat_wxyz", (1, 0, 0, 0))
            fl = F_BODY | (F_TERRAIN if (terrain and p.get("terrain", True)) else 0) | flags_extra
            self.proxies.append(dict(body=b0 + int(p.get("body", 0)), shape=shape, size=size,
                                     off=list(map(float, p.get("offset", (0, 0, 0)))),
                                     q=[qw[1], qw[2], qw[3], qw[0]], mat=self.mat_names.index(mat),
                                     grp=int(p.get("group", 0)), flags=fl))

    def bind_rigid(self, name, x, q, v, w, idx, mass, inertia, proxies, names=None, terrain=True, water=False,
                   kinematic=False):
        idx = list(map(int, idx))
        nm = names or ["%s[%d]" % (name, i) for i in idx]
        b0 = self._add_bodies(nm, mass, inertia, kinematic, water)
        self._add_proxies(b0, proxies, terrain and self.cfg["terrain"])
        self.bindings.append(dict(kind="rigid", name=name, off=b0, n=len(idx), arrays=(x, q, v, w), idx=idx))
        return b0

    def bind_points(self, name, x, v, idx, mass, proxies, names=None, terrain=True, water=True, kinematic=False):
        idx = list(map(int, idx))
        nm = names or ["%s[%d]" % (name, i) for i in idx]
        b0 = self._add_bodies(nm, mass, None, kinematic, water)
        self._add_proxies(b0, proxies, terrain and self.cfg["terrain"])
        self.bindings.append(dict(kind="points", name=name, off=b0, n=len(idx), arrays=(x, v), idx=idx))
        return b0

    def bind_host(self, name, n, get, put, mass, proxies, inertia=None, names=None, terrain=True, water=False,
                  kinematic=False):
        nm = names or ["%s[%d]" % (name, i) for i in range(n)]
        b0 = self._add_bodies(nm, mass if mass is not None else [0.0] * n, inertia, kinematic, water)
        self._add_proxies(b0, proxies, terrain and self.cfg["terrain"])
        self.bindings.append(dict(kind="host", name=name, off=b0, n=n, get=get, put=put))
        return b0

    def add_event(self, **ev):
        """Domain-reported event (e.g. a CPU dart's own ground impact)."""
        self.events.append({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in ev.items()})

    # ---------------------------------------------------------------- upload
    def finalize(self, world):
        dev = self.device
        N, P = len(self.names), len(self.proxies)
        self.N, self.P = N, P
        if P > 11000:
            raise ValueError("contact: %d proxies > 11000 (int32 sort key)" % P)
        if N == 0 or P == 0:
            return
        invm = np.array([0.0 if k else 1.0 / m for m, k in zip(self.mass, self.kin)], np.float32)
        invI = np.array([[0.0, 0.0, 0.0] if (k or I is None) else [1.0 / c if c > 0 else 0.0 for c in I]
                         for I, k in zip(self.inertia, self.kin)], np.float32)
        z3 = lambda: wp.zeros(N, dtype=wp.vec3, device=dev)
        self.X, self.V, self.W, self.dV, self.dW, self.dX = (z3() for _ in range(6))
        self.Q = wp.array(np.tile([0, 0, 0, 1], (N, 1)).astype(np.float32), dtype=wp.quat, device=dev)
        self.invm = wp.array(invm, dtype=float, device=dev)
        self.invI = wp.array(invI, dtype=wp.vec3, device=dev)
        self.kin_d = wp.array(np.array(self.kin, np.int32), dtype=int, device=dev)
        self.touched = wp.zeros(N, dtype=int, device=dev)
        self.ncnt = wp.zeros(N, dtype=int, device=dev)
        Pr = self.proxies
        self.pb = wp.array(np.array([p["body"] for p in Pr], np.int32), dtype=int, device=dev)
        self.ptype = wp.array(np.array([p["shape"] for p in Pr], np.int32), dtype=int, device=dev)
        self.psize = wp.array(np.array([p["size"] for p in Pr], np.float32), dtype=wp.vec3, device=dev)
        self.poff = wp.array(np.array([p["off"] for p in Pr], np.float32), dtype=wp.vec3, device=dev)
        self.prot = wp.array(np.array([p["q"] for p in Pr], np.float32), dtype=wp.quat, device=dev)
        self.pmat = wp.array(np.array([p["mat"] for p in Pr], np.int32), dtype=int, device=dev)
        self.pgrp = wp.array(np.array([p["grp"] for p in Pr], np.int32), dtype=int, device=dev)
        self.pflag = wp.array(np.array([p["flags"] for p in Pr], np.int32), dtype=int, device=dev)
        self.pc, self.pq = wp.zeros(P, dtype=wp.vec3, device=dev), wp.zeros(P, dtype=wp.quat, device=dev)
        self.prad = wp.zeros(P, dtype=float, device=dev)
        self.mu_d = wp.array(self.mu_tab, dtype=float, device=dev)
        self.e_d = wp.array(self.e_tab, dtype=float, device=dev)
        self.mat_mu = wp.array(np.array([self.mats[k][0] for k in self.mat_names], np.float32), dtype=float, device=dev)
        self.mat_e = wp.array(np.array([self.mats[k][1] for k in self.mat_names], np.float32), dtype=float, device=dev)
        cap = int(self.cfg["max_contacts"] or max(4096, 64 * P))
        self.cap = cap
        self.cont = wp.zeros(cap, dtype=Contact, device=dev)
        self.keys = wp.zeros(2 * cap, dtype=int, device=dev)
        self.vals = wp.zeros(2 * cap, dtype=int, device=dev)
        self.cnt = wp.zeros(1, dtype=int, device=dev)
        g = int(self.cfg["grid_dim"])
        self.grid = wp.HashGrid(g, g, g, device=dev)
        self.size_max = float(max(_bound(p) for p in Pr))
        for bd in self.bindings:
            if bd["kind"] in ("rigid", "points"):
                bd["idx_d"] = wp.array(np.array(bd["idx"], np.int32), dtype=int, device=dev)
        # water: per body equivalent radius / volume / frontal area from its proxies (largest proxy, ASSUMED)
        self.has_sea = self.sea is not None and _WV is not None and self.cfg["water"] and any(self.water)
        if self.has_sea:
            r_eq, vol, area = np.zeros(N, np.float32), np.zeros(N, np.float32), np.zeros(N, np.float32)
            for p in Pr:
                b, s = p["body"], p["size"]
                if p["shape"] == SPHERE:
                    v_, a_, r_ = 4 / 3 * math.pi * s[0] ** 3, math.pi * s[0] ** 2, s[0]
                elif p["shape"] == CAPSULE:
                    v_ = math.pi * s[0] ** 2 * 2 * s[1] + 4 / 3 * math.pi * s[0] ** 3
                    a_, r_ = math.pi * s[0] ** 2, s[0]
                else:
                    v_, a_ = 8 * s[0] * s[1] * s[2], 4 * s[1] * s[2]
                    r_ = (v_ * 3 / (4 * math.pi)) ** (1 / 3)
                if v_ > vol[b]:
                    vol[b], area[b], r_eq[b] = v_, a_, r_
            self.wflag = wp.array(np.array(self.water, np.int32), dtype=int, device=dev)
            self.wr = wp.array(np.maximum(r_eq, 1e-3), dtype=float, device=dev)
            self.wvol = wp.array(vol, dtype=float, device=dev)
            self.warea = wp.array(area, dtype=float, device=dev)
            self.wet = wp.zeros(N, dtype=int, device=dev)
            self.wet_prev = np.zeros(N, np.int32)
        self.tp = self.terrain.params(dev) if (self.terrain is not None) else None
        if self.tp is None:   # kernels take a TerrainParams argument; give them a 2x2 flat dummy (never sampled)
            self._dummy_ter = T.Terrain.flat(size=2.0, dx=2.0)
            self.tp_dummy = self._dummy_ter.params(dev)

    # ---------------------------------------------------------------- per step
    def _build_cmds(self, dt_c):
        """Record every fixed-argument launch once (per-step Python cost of wp.launch argument packing removed);
        host bindings share ONE pinned full-size staging set (4 H2D + 4 D2H copies per step in total)."""
        dev, N, P = self.device, self.N, self.P
        rc = dict(device=dev, record_cmd=True)
        C = self._cmds = {}
        C["clear"] = wp.launch(_clear, N, [self.dV, self.dW, self.dX, self.touched, self.ncnt], **rc)
        C["clear_keys"] = wp.launch(_clear_keys, 2 * self.cap, [self.keys, self.vals], **rc)
        C["gather"], C["scatter"] = [], []
        for bd in self.bindings:
            if bd["kind"] == "rigid":
                x, q, v, w = bd["arrays"]
                C["gather"].append(wp.launch(_gather_rigid, bd["n"], [bd["off"], bd["idx_d"], x, q, v, w, self.X, self.Q,
                                                                      self.V, self.W], **rc))
                C["scatter"].append(wp.launch(_scatter_rigid, bd["n"], [bd["off"], float(dt_c), bd["idx_d"], self.touched,
                                                                        self.dV, self.dW, self.dX, x, q, v, w], **rc))
            elif bd["kind"] == "points":
                x, v = bd["arrays"]
                C["gather"].append(wp.launch(_gather_points, bd["n"], [bd["off"], bd["idx_d"], x, v, self.X, self.Q,
                                                                       self.V, self.W], **rc))
                C["scatter"].append(wp.launch(_scatter_points, bd["n"], [bd["off"], bd["idx_d"], self.touched, self.dV,
                                                                         self.dX, x, v], **rc))
        self.host = [bd for bd in self.bindings if bd["kind"] == "host"]
        if self.host:
            mask = np.zeros(N, np.int32)
            for bd in self.host:
                mask[bd["off"]:bd["off"] + bd["n"]] = 1
            self.hmask = wp.array(mask, dtype=int, device=dev)
            pin = lambda dt_: wp.zeros(N, dtype=dt_, device="cpu", pinned=True)
            self.hX, self.hV, self.hW, self.hdV, self.hdW, self.hdX = (pin(wp.vec3) for _ in range(6))
            self.hQ, self.htc = pin(wp.quat), pin(int)
            self.hQ.numpy()[:] = [0, 0, 0, 1]
            self.sX, self.sV, self.sW = (wp.zeros(N, dtype=wp.vec3, device=dev) for _ in range(3))
            self.sQ = wp.zeros(N, dtype=wp.quat, device=dev)
            C["merge"] = wp.launch(_merge_host, N, [self.hmask, self.sX, self.sQ, self.sV, self.sW, self.X, self.Q,
                                                    self.V, self.W], **rc)
        C["proxy"] = wp.launch(_proxy_world, P, [self.pb, self.poff, self.prot, self.ptype, self.psize, self.X, self.Q,
                                                 self.pc, self.pq, self.prad], **rc)
        self.grid.build(self.pc, max(self.size_max, 0.05))
        C["pairs"] = wp.launch(_pairs, P, [self.grid.id, self.size_max, P, self.pb, self.ptype, self.psize, self.pflag,
                                           self.pgrp, self.pmat, self.pc, self.pq, self.prad, self.kin_d, self.mu_d,
                                           self.e_d, self.cont, self.keys, self.cnt, self.cap], **rc)
        if self.tp is not None and self.cfg["terrain"]:
            C["terrain"] = wp.launch(_terrain, P, [P, self.tp, float(self.cfg["terrain_restitution"]), self.pb,
                                                   self.ptype, self.psize, self.pflag, self.pmat, self.pc, self.pq,
                                                   self.kin_d, self.mat_mu, self.mat_e, self.cont, self.keys, self.cnt,
                                                   self.cap], **rc)
        C["solve"] = wp.launch(_solve, 1, [self.cnt, self.cap, self.vals, self.cont, int(self.cfg["iterations"]),
                                           float(self.cfg["restitution_threshold"]), float(self.cfg["slop"]),
                                           float(self.cfg["beta"]), self.X, self.Q, self.V, self.W, self.invm, self.invI,
                                           self.dV, self.dW, self.dX, self.touched, self.ncnt], **rc)
        if self.has_sea:
            tp = self.tp if self.tp is not None else self.tp_dummy
            C["water"] = wp.launch(_water_k, N, [0.0, float(dt_c), int(self.tp is not None), tp] + self.sea.args() +
                                   [self.wflag, self.wr, self.wvol, self.warea, float(self.cfg["water_rho"]),
                                    float(self.cfg["water_cd"]), self.X, self.V, self.invm, self.dV, self.touched,
                                    self.wet], **rc)
        C["finish"] = wp.launch(_finish, N, [float(dt_c), self.touched, self.dV, self.dX], **rc)
        self._cmd_dt = dt_c

    def step(self, t, dt):
        """Run one contact pass at time t (after all domains advanced to t) over interval dt."""
        if not getattr(self, "P", 0) or not getattr(self, "N", 0):
            return
        self.k += 1
        if self.k % max(1, int(self.cfg["every"])) != 0:
            return
        dt_c = dt * max(1, int(self.cfg["every"]))
        if getattr(self, "_cmd_dt", None) != dt_c:
            self._build_cmds(dt_c)
        C, dev = self._cmds, self.device
        self.t = t
        C["clear"].launch()
        C["clear_keys"].launch()
        self.cnt.zero_()
        for c in C["gather"]:
            c.launch()
        if self.host:
            hX, hV, hQ, hW = self.hX.numpy(), self.hV.numpy(), self.hQ.numpy(), self.hW.numpy()
            for bd in self.host:
                s, o, n = bd["get"](), bd["off"], bd["n"]
                sl = slice(o, o + n)
                hX[sl] = np.asarray(s["x"], np.float32).reshape(n, 3)
                hV[sl] = np.asarray(s["v"], np.float32).reshape(n, 3)
                q = s.get("q")
                hQ[sl] = [0, 0, 0, 1] if q is None else np.asarray(q, np.float32).reshape(n, 4)
                w = s.get("w")
                hW[sl] = 0.0 if w is None else _rot(hQ[sl].astype(float), np.asarray(w, float)).astype(np.float32)
            for src, dst in ((self.hX, self.sX), (self.hV, self.sV), (self.hQ, self.sQ), (self.hW, self.sW)):
                wp.copy(dst, src)
            C["merge"].launch()
        C["proxy"].launch()
        self.grid.build(self.pc, max(self.size_max, 0.05))
        C["pairs"].launch()
        if "terrain" in C:
            C["terrain"].launch()
        wp.utils.radix_sort_pairs(self.keys, self.vals, self.cap)
        C["solve"].launch()
        if "water" in C:
            C["water"].set_param_at_index(0, float(t))
            C["water"].launch()
        C["finish"].launch()
        for c in C["scatter"]:
            c.launch()
        if self.host:
            for src, dst in ((self.dV, self.hdV), (self.dW, self.hdW), (self.dX, self.hdX), (self.touched, self.htc)):
                wp.copy(dst, src)
            wp.synchronize_device(dev)
        self._events(t)                           # synchronises (contact count + list)
        if self.host:
            tc = self.htc.numpy()
            for bd in self.host:
                sl = slice(bd["off"], bd["off"] + bd["n"])
                if tc[sl].any():
                    q = self.hQ.numpy()[sl].astype(float)
                    bd["put"](self.hdV.numpy()[sl].astype(float), _rot_inv(q, self.hdW.numpy()[sl].astype(float)),
                              self.hdX.numpy()[sl].astype(float), tc[sl].astype(bool))
        self.n_steps += 1

    def _events(self, t):
        n = int(self.cnt.numpy()[0])
        self.n_contacts_max = max(self.n_contacts_max, n)
        if n > self.cap:
            self.overflow += n - self.cap
            n = self.cap
        cur = {}
        if n:
            order = self.vals[0:n].numpy()
            C = self.cont[0:n].numpy()[order]
            for c in C:
                a, b = int(c["a"]), int(c["b"])
                key = (a, b) if b < 0 else (min(a, b), max(a, b))
                r = cur.setdefault(key, dict(vn=0.0, J=0.0, p=np.zeros(3), k=0))
                r["vn"] = min(r["vn"], float(c["vn0"]))
                r["J"] += float(c["jn"])
                r["p"] += np.asarray(c["p"], float)
                r["k"] += 1
        for key, r in cur.items():
            if key not in self.active and -r["vn"] >= self.cfg["event_speed"]:
                a, b = key
                p = r["p"] / r["k"]
                self.events.append(dict(t=round(float(t), 4), kind="impact", a=self.names[a],
                                        b="terrain" if b < 0 else self.names[b], x=float(p[0]), y=float(p[1]),
                                        z=float(p[2]), v_rel=round(-r["vn"], 4), impulse=round(r["J"], 5)))
        self.active = set(cur)
        if self.has_sea:
            wet = self.wet.numpy()
            new = np.where((wet == 1) & (self.wet_prev == 0))[0]
            if len(new):
                X, V = self.X.numpy(), self.V.numpy()
                for i in new:
                    self.events.append(dict(t=round(float(t), 4), kind="splash", a=self.names[i], b="water",
                                            x=float(X[i][0]), y=float(X[i][1]), z=float(X[i][2]),
                                            v_rel=round(float(np.linalg.norm(V[i])), 4), impulse=0.0))
            self.wet_prev = wet.copy()

    # ---------------------------------------------------------------- reporting
    def describe(self):
        return dict(cfg=self.cfg, bodies=len(self.names), proxies=len(self.proxies),
                    materials={k: dict(mu=v[0], e=v[1]) for k, v in self.mats.items()},
                    bindings=[dict(name=b["name"], kind=b["kind"], n=b["n"]) for b in self.bindings])

    def summary(self):
        kinds = {}
        for e in self.events:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        return dict(steps=self.n_steps, max_contacts_in_a_step=self.n_contacts_max, overflow=self.overflow,
                    events=kinds, pairs_hit=sorted({"%s|%s" % (e["a"], e["b"]) for e in self.events
                                                    if e["kind"] == "impact"}))


def _bound(p):
    s = p["size"]
    r = s[0] if p["shape"] == SPHERE else (s[0] + s[1] if p["shape"] == CAPSULE else math.sqrt(sum(c * c for c in s)))
    return r + float(np.linalg.norm(p["off"]))


def _rot(q_xyzw, v):
    u, w = q_xyzw[:, :3], q_xyzw[:, 3:4]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _rot_inv(q_xyzw, v):
    qc = q_xyzw.copy()
    qc[:, :3] *= -1
    return _rot(qc, v)

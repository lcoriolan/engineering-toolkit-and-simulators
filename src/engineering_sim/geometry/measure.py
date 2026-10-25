import math
import numpy as np
from . import stl as stl_io

def outer_and_holes(vertices, faces, axis, value):
    """Slice at coord[axis]==value; return (outer_loop, holes).

    ``outer_loop`` is the closed loop with the largest enclosed area (the wing
    section); ``holes`` are the remaining closed loops (spar tubes and slots).
    Both are (N, 2) arrays in the two non-axis source coordinates (Y_src, Z_src
    for axis=0). Open chains are ignored (they only occur at joint faces, which
    the station lists avoid).
    """
    closed, _open = stl_io.slice_plane(vertices, faces, axis, value)
    if not closed:
        return (None, [])
    loops = sorted(closed, key=lambda p: -abs(stl_io.polygon_area(p)))
    return (loops[0], loops[1:])

def le_te(poly):
    """Return ((z_le, y_le), (z_te, y_te)) for a wing-section polygon in (Y_src, Z_src).

    LE is the most-forward point (min Z_src), TE the most-aft (max Z_src). The
    chord line height uses the Y_src of those two points.
    """
    z = poly[:, 1]
    i_le = int(np.argmin(z))
    i_te = int(np.argmax(z))
    return ((poly[i_le, 1], poly[i_le, 0]), (poly[i_te, 1], poly[i_te, 0]))

def section_stats(poly):
    """Basic station metrics from a wing-section polygon (Y_src, Z_src)."""
    (z_le, y_le), (z_te, y_te) = le_te(poly)
    chord = z_te - z_le
    z_chord = 0.5 * (y_le + y_te)
    twist = np.degrees(np.arctan2(y_te - y_le, z_te - z_le))
    return {'le_x': z_le, 'te_x': z_te, 'chord': chord, 'z_chord': z_chord, 'twist_deg': twist}

def normalize_airfoil(poly, n_points=60):
    """Normalise a wing-section polygon to (x/c, z/c) ordered TE->upper->LE->lower->TE.

    Principle: x/c = (Z_src - Z_le)/chord, z/c = (Y_src - z_chord)/chord where
    z_chord is the mean Y_src of LE and TE. The upper and lower surfaces are
    re-interpolated onto a half-cosine x/c grid (clustered at LE and TE) so the
    section has ~2*n_points+2 points. Returns (coords, t_c, camber_max).
    """
    (z_le, y_le), (z_te, y_te) = le_te(poly)
    chord = z_te - z_le
    if chord <= 0:
        return (None, 0.0, 0.0)
    z_chord = 0.5 * (y_le + y_te)
    xc = (poly[:, 1] - z_le) / chord
    zc = (poly[:, 0] - z_chord) / chord
    upper = zc > 1e-06
    lower = zc < -1e-06
    xs = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_points + 1) / n_points))
    if upper.sum() >= 2 and lower.sum() >= 2:
        xu = xc[upper]
        zu = zc[upper]
        xl = xc[lower]
        zl = zc[lower]
        order_u = np.argsort(xu)
        order_l = np.argsort(xl)
        zu_i = np.interp(xs, xu[order_u], zu[order_u])
        zl_i = np.interp(xs, xl[order_l], zl[order_l])
    else:
        xa = xc[zc >= 0] if upper.sum() else xc
        za = zc[zc >= 0] if upper.sum() else -zc
        order = np.argsort(xa)
        zu_i = np.interp(xs, xa[order], za[order])
        zl_i = -zu_i
    coords = []
    coords.append((1.0, 0.0))
    for i in range(n_points - 1, 0, -1):
        coords.append((float(xs[i]), float(zu_i[i])))
    coords.append((0.0, 0.0))
    for i in range(1, n_points):
        coords.append((float(xs[i]), float(zl_i[i])))
    coords.append((1.0, 0.0))
    coords = np.asarray(coords)
    t_c = float(np.max(zc) - np.min(zc))
    camber_max = float(np.max(0.5 * (zu_i + zl_i)))
    return (coords, t_c, camber_max)

def fit_circle(poly):
    """Least-squares circle fit (algebraic, Kasa) of a 2D point loop.

    Solves x^2+y^2 = D x + E y + F for (D,E,F); centre (D/2, E/2),
    r = sqrt(F + cx^2 + cy^2). Returns (cx, cy, r, rms) where rms is the RMS
    residual |dist - r|.
    """
    x = poly[:, 0]
    y = poly[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    d, e, f = sol
    cx, cy = (d / 2.0, e / 2.0)
    r = float(np.sqrt(max(f + cx * cx + cy * cy, 0.0)))
    dist = np.hypot(x - cx, y - cy)
    rms = float(np.sqrt(np.mean((dist - r) ** 2)))
    return (cx, cy, r, rms)

def classify_hole(poly):
    """Classify a closed hole loop as round (spar tube) or rect (slot).

    ``poly`` is in (Y_src, Z_src). A hole is round when the least-squares circle
    residual is < 8% of its radius; otherwise it is a rectangle (reported by its
    chordwise w = Z_src extent and thickness h = Y_src extent). Returns a dict
    with shape, size and centre: cy = Y_src centre, cz = Z_src centre.
    """
    cx, cy, r, rms = fit_circle(poly)
    if r > 1e-06 and rms / r < 0.08:
        return {'shape': 'round', 'd': 2.0 * r, 'cy': cx, 'cz': cy, 'r': r}
    ymin = poly[:, 0].min()
    ymax = poly[:, 0].max()
    zmin = poly[:, 1].min()
    zmax = poly[:, 1].max()
    return {'shape': 'rect', 'w': float(zmax - zmin), 'h': float(ymax - ymin), 'cy': 0.5 * (ymin + ymax), 'cz': 0.5 * (zmin + zmax)}

def combine_mass(items):
    """Combine parts/point-masses into total mass, CG and inertia about the CG.

    ``items`` is a list of dicts each with mass_g, cg (3,), and either I (3x3
    about its own CG) or None (point mass). Returns
    (total_g, cg(3,), inertia(3x3) g*mm^2 about the CG). Parallel-axis theorem:
    I_CG += I_self + m (|d|^2 I - d d^T), d = cg - CG.
    """
    total = 0.0
    moment = np.zeros(3)
    for it in items:
        total += it['mass_g']
        moment += it['mass_g'] * np.asarray(it['cg'], dtype=np.float64)
    if total <= 0:
        return (0.0, np.zeros(3), np.zeros((3, 3)))
    cg = moment / total
    I = np.zeros((3, 3))
    for it in items:
        d = np.asarray(it['cg'], dtype=np.float64) - cg
        m = it['mass_g']
        I += m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        if it.get('I') is not None:
            I += it['I']
    return (total, cg, I)

def point_item(mass_g, cg):
    return {'mass_g': mass_g, 'cg': np.asarray(cg, dtype=np.float64), 'I': None}

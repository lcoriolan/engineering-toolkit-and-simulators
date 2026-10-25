# DESCRIPTION: Section properties section: area, first moment and centroid, second moments (Ix, Iy, Ixy,
#              polar J), parallel-axis and principal values, section modulus S = I/c, plastic modulus Z and
#              shape factor, radius of gyration, shear centre and torsion constant for standard shapes
#              (rectangle, hollow rectangle, circle, tube, I/W, channel, tee, angle) plus a general
#              polygon and a rectangle-composite procedure.
"""Section properties calculators.

Coordinate convention for the standard shapes: x is horizontal (to the right), y vertical (up), with the
origin stated in each variable description. Second moments Ix, Iy are about centroidal axes; Ixy is the
product of inertia about the centroid (signed). ``J`` is the polar second moment J = Ix + Iy; ``J_t`` is
the Saint-Venant torsion constant (a different quantity for non-circular sections — for open thin-walled
shapes it is approximated by Σ b·t³/3). The plastic modulus Z is computed by bisecting the equal-area
(plastic) neutral axis; the shape factor is Z/S. Principal moments are ordered I1 ≥ I2 with theta_p the
angle of the I1 axis from +x, theta_p = ½·atan2(−2·Ixy, Ix − Iy) with Ixy = ∫xy dA.

Reference relations are the standard ones in mechanics-of-materials texts (e.g. Beer & Johnston; Hibbeler;
Young & Budynas, *Roark's Formulas for Stress and Strain*); they are cited by relation name, recalled
rather than re-read, so no page numbers are given.
"""
import math

import numpy as np

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('section_props', 'Section properties', order=30,
                  summary='Area, centroid, second moments, section modulus, plastic modulus, principal '
                          'axes, radius of gyration, shear centre and torsion constant for standard and '
                          'general sections.')


# ---------------------------------------------------------------------------------------------------
# Shared geometry engines
# ---------------------------------------------------------------------------------------------------
def _finite(x, name):
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        raise MechError(f'{name} is not a finite number')
    return float(x)


def _geom_outputs(g):
    """Common output keys for a section (excludes the torsion constant J_t)."""
    return {k: g[k] for k in ('A', 'xbar', 'ybar', 'Ix', 'Iy', 'Ixy', 'J', 'Sx', 'Sy', 'Zx', 'Zy',
                              'rx', 'ry', 'shape_x', 'shape_y', 'I1', 'I2', 'theta_p')}


def _rect_geom(rects):
    """Full geometry of a composite of axis-aligned rectangles.

    Each row is [b, h, xc, yc]: width b, height h, centroid at (xc, yc). Rectangles must not overlap.
    Returns area, centroid, centroidal second moments Ix/Iy/Ixy, polar J, elastic section moduli
    Sx/Sy (= I/c to the extreme fibre), plastic moduli Zx/Zy (equal-area axis by bisection), radii of
    gyration rx/ry, shape factors, and principal moments/angle.
    """
    rects = [[_finite(x, 'rectangle dim') for x in r] for r in rects]
    if not rects:
        raise MechError('rects: give at least one [b, h, xc, yc] row')
    A = sum(b * h for b, h, xc, yc in rects)
    if A <= 0:
        raise MechError('rects: total area must be positive')
    xbar = sum(b * h * xc for b, h, xc, yc in rects) / A
    ybar = sum(b * h * yc for b, h, xc, yc in rects) / A
    Ix = sum(b * h ** 3 / 12 + b * h * (yc - ybar) ** 2 for b, h, xc, yc in rects)
    Iy = sum(h * b ** 3 / 12 + b * h * (xc - xbar) ** 2 for b, h, xc, yc in rects)
    Ixy = sum(b * h * (xc - xbar) * (yc - ybar) for b, h, xc, yc in rects)
    xmin = min(xc - b / 2 for b, h, xc, yc in rects)
    xmax = max(xc + b / 2 for b, h, xc, yc in rects)
    ymin = min(yc - h / 2 for b, h, xc, yc in rects)
    ymax = max(yc + h / 2 for b, h, xc, yc in rects)
    cx = max(abs(xmax - xbar), abs(xbar - xmin))
    cy = max(abs(ymax - ybar), abs(ybar - ymin))
    Zx = _plastic_rects(rects, 'x', A, ybar)
    Zy = _plastic_rects(rects, 'y', A, xbar)
    Iavg, R = (Ix + Iy) / 2, math.hypot((Ix - Iy) / 2, Ixy)
    return {'A': A, 'xbar': xbar, 'ybar': ybar, 'Ix': Ix, 'Iy': Iy, 'Ixy': Ixy, 'J': Ix + Iy,
            'Sx': Ix / cy, 'Sy': Iy / cx, 'Zx': Zx, 'Zy': Zy,
            'rx': math.sqrt(Ix / A), 'ry': math.sqrt(Iy / A),
            'shape_x': Zx / (Ix / cy), 'shape_y': Zy / (Iy / cx),
            'I1': Iavg + R, 'I2': Iavg - R, 'theta_p': math.atan2(-2 * Ixy, Ix - Iy) / 2}


def _plastic_rects(rects, axis, A, bar):
    """Plastic modulus about a horizontal ('x') or vertical ('y') axis by bisecting the equal-area axis.

    The equal-area axis is found by bisection on the coordinate perpendicular to ``axis``; the plastic
    modulus is then Z = Q_above + Q_below = 2·Q_above − A·(bar − c), where Q_above is the first moment of
    the material on the positive side of the axis about that axis and ``bar`` the centroid coordinate.
    """
    if axis == 'x':
        lo = min(yc - h / 2 for b, h, xc, yc in rects)
        hi = max(yc + h / 2 for b, h, xc, yc in rects)

        def half(c):
            area = q = 0.0
            for b, h, xc, yc in rects:
                top, bot = yc + h / 2, yc - h / 2
                if bot >= c:
                    area += b * h
                    q += b * h * (yc - c)
                elif top > c:
                    hh = top - c
                    area += b * hh
                    q += b * hh * hh / 2
            return area, q
    else:
        lo = min(xc - b / 2 for b, h, xc, yc in rects)
        hi = max(xc + b / 2 for b, h, xc, yc in rects)

        def half(c):
            area = q = 0.0
            for b, h, xc, yc in rects:
                right, left = xc + b / 2, xc - b / 2
                if left >= c:
                    area += b * h
                    q += b * h * (xc - c)
                elif right > c:
                    ww = right - c
                    area += ww * h
                    q += h * ww * ww / 2
            return area, q

    # bisect for area_above(c) == A/2 (area_above is monotone decreasing in c)
    for _ in range(200):
        c = (lo + hi) / 2
        a, _ = half(c)
        if a > A / 2:
            lo = c
        else:
            hi = c
    c = (lo + hi) / 2
    _, q = half(c)
    return 2 * q - A * (bar - c)


def _poly_geom(vertices):
    """Full geometry of a simple polygon given as [[x, y], ...] vertices (m), in any winding order.

    Area, centroid and second moments use Green's-theorem edge sums (exact for any simple polygon); the
    plastic modulus Zx is obtained by bisecting the equal-area horizontal axis with an edge-integral
    half-plane clip (exact, and correct for concave polygons). Concave sections are also served by the
    rectangle-composite procedure.
    """
    vs = [[_finite(p[0], 'x'), _finite(p[1], 'y')] for p in vertices]
    n = len(vs)
    if n < 3:
        raise MechError('vertices: give at least three [x, y] rows')
    A2 = sum(vs[i][0] * vs[(i + 1) % n][1] - vs[(i + 1) % n][0] * vs[i][1] for i in range(n))
    if abs(A2) < 1e-300:
        raise MechError('vertices: degenerate polygon (zero area)')
    if A2 < 0:
        vs = list(reversed(vs))
        A2 = -A2
    A = A2 / 2
    qx = qy = 0.0
    ix0 = iy0 = ixy0 = 0.0
    for i in range(n):
        x1, y1 = vs[i]
        x2, y2 = vs[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        qx += (y1 + y2) * cross
        qy += (x1 + x2) * cross
        ix0 += cross * (y1 * y1 + y1 * y2 + y2 * y2)
        iy0 += cross * (x1 * x1 + x1 * x2 + x2 * x2)
        ixy0 += cross * (x1 * y2 + 2 * x1 * y1 + 2 * x2 * y2 + x2 * y1)
    xbar = qy / (6 * A)
    ybar = qx / (6 * A)
    Ix = ix0 / 12 - A * ybar * ybar
    Iy = iy0 / 12 - A * xbar * xbar
    Ixy = ixy0 / 24 - A * xbar * ybar
    cx = max(abs(x - xbar) for x, y in vs)
    cy = max(abs(y - ybar) for x, y in vs)
    Zx = _plastic_poly(vs, 1, A, ybar)
    Zy = _plastic_poly(vs, 0, A, xbar)
    Iavg, R = (Ix + Iy) / 2, math.hypot((Ix - Iy) / 2, Ixy)
    return {'A': A, 'xbar': xbar, 'ybar': ybar, 'Ix': Ix, 'Iy': Iy, 'Ixy': Ixy, 'J': Ix + Iy,
            'Sx': Ix / cy, 'Sy': Iy / cx, 'Zx': Zx, 'Zy': Zy,
            'rx': math.sqrt(Ix / A), 'ry': math.sqrt(Iy / A),
            'shape_x': Zx / (Ix / cy), 'shape_y': Zy / (Iy / cx),
            'I1': Iavg + R, 'I2': Iavg - R, 'theta_p': math.atan2(-2 * Ixy, Ix - Iy) / 2}


def _q_edge(pa, ta, pb, tb):
    """∫ p(t)·t dt along a straight segment from (pa, ta) to (pb, tb); t is the distance coordinate."""
    if tb == ta:
        return 0.0
    s = (pb - pa) / (tb - ta)
    return pa * (tb * tb - ta * ta) / 2 + s * (tb ** 3 - ta ** 3) / 3 - s * ta * (tb * tb - ta * ta) / 2


def _poly_half(vs, c, axis):
    """(area, first-moment-about-the-line) of the part of a CCW polygon on the positive side of the line
    y = c (axis == 1) or x = c (axis == 0). Exact for any simple polygon.

    The loop below is the horizontal-axis form: it integrates ∮ x dy (area) and ∮ x(y−c) dy (first moment)
    along the polygon edges on the y > c side, where the clip line y = c contributes nothing because dy = 0
    there. For a vertical axis the analogous edge sum would be ∮ y dx, which (a) is the negative of the
    area for a CCW polygon and (b) also omits the non-zero contribution of the vertical clip line, so it
    breaks for concave polygons. We therefore swap x↔y and reverse the vertex order (a reflection across
    y = x, which preserves CCW orientation) and reuse the horizontal-axis logic verbatim.
    """
    if axis == 0:
        vs = [(y, x) for x, y in reversed(vs)]
    n = len(vs)
    area = q = 0.0
    for i in range(n):
        x1, y1 = vs[i]
        x2, y2 = vs[(i + 1) % n]
        t1, t2, p1, p2 = y1, y2, x1, x2
        if t1 >= c and t2 >= c:
            area += (p1 + p2) * (t2 - t1) / 2
            q += _q_edge(p1, t1 - c, p2, t2 - c)
        elif t1 < c < t2:
            pc = p1 + (p2 - p1) * (c - t1) / (t2 - t1)
            area += (pc + p2) * (t2 - c) / 2
            q += _q_edge(pc, 0.0, p2, t2 - c)
        elif t2 < c < t1:
            pc = p1 + (p2 - p1) * (c - t1) / (t2 - t1)
            # Downward edge crossing the cut: its dy (c − t1) is negative, like the fully-above branch's (t2 − t1).

            area += (p1 + pc) * (c - t1) / 2
            q += _q_edge(p1, t1 - c, pc, 0.0)
    return area, q


def _plastic_poly(vs, axis, A, bar):
    """Plastic modulus of a polygon about a horizontal (axis=1) or vertical (axis=0) axis by bisection."""
    if axis == 1:
        lo = min(y for x, y in vs)
        hi = max(y for x, y in vs)
    else:
        lo = min(x for x, y in vs)
        hi = max(x for x, y in vs)
    for _ in range(200):
        c = (lo + hi) / 2
        a, _ = _poly_half(vs, c, axis)
        if a > A / 2:
            lo = c
        else:
            hi = c
    c = (lo + hi) / 2
    _, q = _poly_half(vs, c, axis)
    return 2 * q - A * (bar - c)


# ---------------------------------------------------------------------------------------------------
# Standard-shape procedures (rectilinear shapes go through the shared rectangle engine)
# ---------------------------------------------------------------------------------------------------
def _shape(rects, jt=None):
    g = _geom_outputs(_rect_geom(rects))
    if jt is not None:
        g['J_t'] = jt
    return g


def _rectangle(v):
    b, h = _finite(v['b'], 'b'), _finite(v['h'], 'h')
    w, t = (b, h) if b >= h else (h, b)
    jt = w * t ** 3 * (1 / 3 - 0.21 * (t / w) * (1 - (t / w) ** 4 / 12))
    return _shape([[b, h, b / 2, h / 2]], jt)


def _hollow_rectangle(v):
    b, h, bi, hi = (_finite(v[k], k) for k in ('b', 'h', 'bi', 'hi'))
    if bi <= 0 or hi <= 0 or bi >= b or hi >= h:
        raise MechError('hollow_rectangle: need 0 < bi < b and 0 < hi < h')
    tw, th = (b - bi) / 2, (h - hi) / 2  # wall thicknesses (symmetric walls)
    rects = [[b, th, b / 2, th / 2], [b, th, b / 2, h - th / 2],
             [tw, hi, tw / 2, h / 2], [tw, hi, b - tw / 2, h / 2]]
    return _shape(rects)


def _circle(v):
    d = _finite(v['d'], 'd')
    A = math.pi * d ** 2 / 4
    I = math.pi * d ** 4 / 64
    J = math.pi * d ** 4 / 32
    S = math.pi * d ** 3 / 32
    Z = d ** 3 / 6
    return {'A': A, 'xbar': d / 2, 'ybar': d / 2, 'Ix': I, 'Iy': I, 'Ixy': 0.0, 'J': J,
            'Sx': S, 'Sy': S, 'Zx': Z, 'Zy': Z, 'rx': d / 4, 'ry': d / 4,
            'shape_x': Z / S, 'shape_y': Z / S, 'I1': I, 'I2': I, 'theta_p': 0.0, 'J_t': J}


def _tube(v):
    do, di = _finite(v['do'], 'do'), _finite(v['di'], 'di')
    if di <= 0 or di >= do:
        raise MechError('tube: need 0 < di < do')
    A = math.pi * (do ** 2 - di ** 2) / 4
    I = math.pi * (do ** 4 - di ** 4) / 64
    J = math.pi * (do ** 4 - di ** 4) / 32
    S = math.pi * (do ** 4 - di ** 4) / (32 * do)
    Z = (do ** 3 - di ** 3) / 6
    r = math.sqrt(I / A)
    return {'A': A, 'xbar': do / 2, 'ybar': do / 2, 'Ix': I, 'Iy': I, 'Ixy': 0.0, 'J': J,
            'Sx': S, 'Sy': S, 'Zx': Z, 'Zy': Z, 'rx': r, 'ry': r,
            'shape_x': Z / S, 'shape_y': Z / S, 'I1': I, 'I2': I, 'theta_p': 0.0, 'J_t': J}


def _i_section(v):
    bf, tf, d, tw = (_finite(v[k], k) for k in ('bf', 'tf', 'd', 'tw'))
    if tf <= 0 or tw <= 0 or d <= 2 * tf or bf <= tw:
        raise MechError('i_section: need tf, tw > 0, d > 2·tf and bf > tw')
    rects = [[bf, tf, bf / 2, d - tf / 2], [bf, tf, bf / 2, tf / 2], [tw, d - 2 * tf, bf / 2, d / 2]]
    jt = (2 * bf * tf ** 3 + (d - 2 * tf) * tw ** 3) / 3  # thin-walled open-section approximation
    return _shape(rects, jt)


def _channel(v):
    bf, tf, d, tw = (_finite(v[k], k) for k in ('bf', 'tf', 'd', 'tw'))
    if tf <= 0 or tw <= 0 or d <= 2 * tf or bf <= tw:
        raise MechError('channel: need tf, tw > 0, d > 2·tf and bf > tw')
    rects = [[bf, tf, bf / 2, d - tf / 2], [bf, tf, bf / 2, tf / 2], [tw, d - 2 * tf, tw / 2, d / 2]]
    jt = (2 * bf * tf ** 3 + (d - 2 * tf) * tw ** 3) / 3
    return _shape(rects, jt)


def _tee(v):
    bf, tf, d, tw = (_finite(v[k], k) for k in ('bf', 'tf', 'd', 'tw'))
    if tf <= 0 or tw <= 0 or d <= tf or bf <= tw:
        raise MechError('tee: need tf, tw > 0, d > tf and bf > tw')
    rects = [[bf, tf, bf / 2, d - tf / 2], [tw, d - tf, bf / 2, (d - tf) / 2]]
    jt = (bf * tf ** 3 + (d - tf) * tw ** 3) / 3
    return _shape(rects, jt)


def _angle(v):
    L, t = _finite(v['L'], 'L'), _finite(v['t'], 't')
    if t <= 0 or t >= L:
        raise MechError('angle: need 0 < t < L')
    rects = [[t, L - t, t / 2, (L + t) / 2], [L - t, t, (L + t) / 2, t / 2], [t, t, t / 2, t / 2]]
    jt = 2 * L * t ** 3 / 3  # thin-walled open-section approximation (two legs)
    return _shape(rects, jt)


def _principal_axes(v):
    Ix, Iy, Ixy = _finite(v['Ix'], 'Ix'), _finite(v['Iy'], 'Iy'), _finite(v['Ixy'], 'Ixy')
    Iavg = (Ix + Iy) / 2
    R = math.hypot((Ix - Iy) / 2, Ixy)
    return {'I1': Iavg + R, 'I2': Iavg - R, 'theta_p': math.atan2(-2 * Ixy, Ix - Iy) / 2}


def _channel_shear_center(v):
    """Shear centre of a thin-walled channel (uniform thickness t): distance e from the web centreline,
    on the side opposite the flange toes, e = 3·b²/(6·b + h). b = flange width measured from the web
    centreline to the flange tip; h = web depth measured between flange centrelines."""
    b, h, t = _finite(v['b'], 'b'), _finite(v['h'], 'h'), _finite(v['t'], 't')
    if b <= 0 or h <= 0 or t <= 0:
        raise MechError('channel_shear_center: b, h, t must be positive')
    Ix = t * h ** 2 * (h + 6 * b) / 12
    return {'Ix': Ix, 'e': 3 * b ** 2 / (6 * b + h), 'A': t * (2 * b + h)}


def _polygon(v):
    return _geom_outputs(_poly_geom(v['vertices']))


def _composite(v):
    return _geom_outputs(_rect_geom(v['rects']))


# ---------------------------------------------------------------------------------------------------
# Calculators
# ---------------------------------------------------------------------------------------------------
_GEO_VARS = [
    V('A', 'm^2', 'cross-sectional area'),
    V('xbar', 'm', 'centroid x̄', label='x̄'),
    V('ybar', 'm', 'centroid ȳ', label='ȳ'),
    V('Ix', 'm^4', 'second moment about the centroidal x-axis'),
    V('Iy', 'm^4', 'second moment about the centroidal y-axis'),
    V('Ixy', 'm^4', 'product of inertia about the centroid, Ixy = ∫xy·dA (signed)', signed=True),
    V('J', 'm^4', 'polar second moment J = Ix + Iy'),
    V('Sx', 'm^3', 'elastic section modulus about x, Ix/cx'),
    V('Sy', 'm^3', 'elastic section modulus about y, Iy/cy'),
    V('Zx', 'm^3', 'plastic section modulus about x'),
    V('Zy', 'm^3', 'plastic section modulus about y'),
    V('rx', 'm', 'radius of gyration about x, sqrt(Ix/A)'),
    V('ry', 'm', 'radius of gyration about y, sqrt(Iy/A)'),
    V('shape_x', '', 'shape factor Zx/Sx'),
    V('shape_y', '', 'shape factor Zy/Sy'),
    V('I1', 'm^4', 'major principal moment'),
    V('I2', 'm^4', 'minor principal moment'),
    V('theta_p', 'rad', 'angle of the I1 axis from +x', signed=True, label='θp'),
]

CALCS = [
    Calc('first_moment', 'First moment of area about the x-axis (Q = A·ȳ)',
         vars=[V('Q', 'm^3', 'first moment about the x-axis'), V('A', 'm^2', 'area'),
               V('ybar', 'm', 'centroid distance from the x-axis', label='ȳ')],
         equations=['Q = A*ybar'], outputs=['Q'],
         example={'A': 0.02, 'ybar': 0.1},
         refs=['First moment of area Q = A·ȳ (recalled)']),

    Calc('parallel_axis', 'Parallel-axis theorem (I′ = I + A·d²)',
         vars=[V('I_prime', 'm^4', 'moment about the offset axis', label="I′"),
               V('I', 'm^4', 'moment about the centroidal axis'), V('A', 'm^2', 'area'),
               V('d', 'm', 'distance between the parallel axes')],
         equations=['I_prime = I + A*d**2'], outputs=['I_prime'],
         example={'I': 8e-6, 'A': 0.02, 'd': 0.1},
         refs=['Parallel-axis (Steiner) theorem (recalled)']),

    Calc('section_modulus', 'Elastic section modulus (S = I/c)',
         vars=[V('S', 'm^3', 'section modulus'), V('I', 'm^4', 'second moment about the axis'),
               V('c', 'm', 'distance from the neutral axis to the extreme fibre')],
         equations=['S = I/c'], outputs=['S'],
         example={'I': 8e-6, 'c': 0.1},
         refs=['Elastic section modulus S = I/c (recalled)']),

    Calc('radius_gyration', 'Radius of gyration (r = √(I/A))',
         vars=[V('r', 'm', 'radius of gyration'), V('I', 'm^4', 'second moment about the axis'),
               V('A', 'm^2', 'area')],
         equations=['r = sqrt(I/A)'], outputs=['r'],
         example={'I': 8e-6, 'A': 0.02},
         refs=['Radius of gyration r = sqrt(I/A) (recalled)']),

    Calc('principal_axes', 'Principal moments and orientation from Ix, Iy, Ixy',
         vars=[V('Ix', 'm^4', 'second moment about the centroidal x-axis'),
               V('Iy', 'm^4', 'second moment about the centroidal y-axis'),
               V('Ixy', 'm^4', 'product of inertia about the centroid, Ixy = ∫xy·dA (signed)', signed=True),
               V('I1', 'm^4', 'major principal moment'), V('I2', 'm^4', 'minor principal moment'),
               V('theta_p', 'rad', 'angle of the I1 axis from +x', signed=True, label='θp')],
         procedure=_principal_axes, outputs=['I1', 'I2', 'theta_p'],
         example={'Ix': 10e-6, 'Iy': 4e-6, 'Ixy': 3e-6},
         refs=['Principal moments of a second-moment tensor; θp = ½·atan2(−2·Ixy, Ix − Iy) (recalled)']),

    Calc('rectangle', 'Rectangle section properties',
         vars=[V('b', 'm', 'width (horizontal)'), V('h', 'm', 'height (vertical)')] + _GEO_VARS +
              [V('J_t', 'm^4', 'Saint-Venant torsion constant (long side b_t ≥ t_t)')],
         procedure=_rectangle, outputs=['A', 'xbar', 'ybar', 'Ix', 'Iy', 'Ixy', 'J', 'Sx', 'Sy',
                                        'Zx', 'Zy', 'rx', 'ry', 'shape_x', 'shape_y', 'I1', 'I2',
                                        'theta_p', 'J_t'],
         example={'b': 0.2, 'h': 0.1},
         assumptions=['Shape factor = 1.5 for a rectangle (landmark). J_t uses the standard '
                      'Saint-Venant approximation b·t³(1/3 − 0.21(t/b)(1 − (t/b)⁴/12)).'],
         refs=['Rectangle: A = bh, Ix = bh³/12, Zx = bh²/4 (recalled)']),

    Calc('hollow_rectangle', 'Hollow (box) rectangle section properties',
         vars=[V('b', 'm', 'outer width (horizontal)'), V('h', 'm', 'outer height (vertical)'),
               V('bi', 'm', 'inner width (hole)'), V('hi', 'm', 'inner height (hole)')] + _GEO_VARS,
         procedure=_hollow_rectangle, outputs=['A', 'xbar', 'ybar', 'Ix', 'Iy', 'Ixy', 'J', 'Sx', 'Sy',
                                               'Zx', 'Zy', 'rx', 'ry', 'shape_x', 'shape_y', 'I1', 'I2',
                                               'theta_p'],
         example={'b': 0.2, 'h': 0.1, 'bi': 0.16, 'hi': 0.06},
         assumptions=['Symmetric walls; Ix = (b·h³ − bi·hi³)/12. Torsion constant omitted '
                      '(thin-wall closed-section formula not recalled confidently).'],
         refs=['Hollow rectangle: subtract the hole moments of area (recalled)']),

    Calc('circle', 'Solid circular section properties',
         vars=[V('d', 'm', 'diameter')] + _GEO_VARS + [V('J_t', 'm^4', 'Saint-Venant torsion constant')],
         procedure=_circle, outputs=['A', 'xbar', 'ybar', 'Ix', 'Iy', 'Ixy', 'J', 'Sx', 'Sy',
                                     'Zx', 'Zy', 'rx', 'ry', 'shape_x', 'shape_y', 'I1', 'I2',
                                     'theta_p', 'J_t'],
         example={'d': 0.2},
         assumptions=['Shape factor = 16/(3π) ≈ 1.698 for a solid circle.'],
         refs=['Circle: I = πd⁴/64, J = πd⁴/32, Z = d³/6 (recalled)']),

    Calc('tube', 'Circular tube section properties',
         vars=[V('do', 'm', 'outer diameter'), V('di', 'm', 'inner diameter (hole)')] + _GEO_VARS +
              [V('J_t', 'm^4', 'Saint-Venant torsion constant')],
         procedure=_tube, outputs=['A', 'xbar', 'ybar', 'Ix', 'Iy', 'Ixy', 'J', 'Sx', 'Sy',
                                   'Zx', 'Zy', 'rx', 'ry', 'shape_x', 'shape_y', 'I1', 'I2',
                                   'theta_p', 'J_t'],
         example={'do': 0.2, 'di': 0.16},
         assumptions=['J_t = π(do⁴ − di⁴)/32 exactly for a circular tube.'],
         refs=['Hollow circle: I = π(do⁴ − di⁴)/64, Z = (do³ − di³)/6 (recalled)']),

    Calc('i_section', 'I/W-section properties (symmetric, equal flanges)',
         vars=[V('bf', 'm', 'flange width'), V('tf', 'm', 'flange thickness'),
               V('d', 'm', 'overall depth'), V('tw', 'm', 'web thickness')] + _GEO_VARS +
              [V('J_t', 'm^4', 'Saint-Venant torsion constant (thin-walled open approximation)')],
         procedure=_i_section, outputs=['A', 'xbar', 'ybar', 'Ix', 'Iy', 'Ixy', 'J', 'Sx', 'Sy',
                                        'Zx', 'Zy', 'rx', 'ry', 'shape_x', 'shape_y', 'I1', 'I2',
                                        'theta_p', 'J_t'],
         example={'bf': 0.2, 'tf': 0.02, 'd': 0.4, 'tw': 0.01},
         assumptions=['Equal flanges, symmetric. Ix = [bf·d³ − (bf − tw)(d − 2tf)³]/12. J_t is the '
                      'open-section approximation Σ b·t³/3.'],
         refs=['I-section: A = 2·bf·tf + (d − 2tf)·tw; plastic Zx = bf·tf(d − tf) + tw(d − 2tf)²/4 '
               '(recalled)']),

    Calc('channel', 'Channel section properties',
         vars=[V('bf', 'm', 'flange width (from back of web to toe)'), V('tf', 'm', 'flange thickness'),
               V('d', 'm', 'overall depth'), V('tw', 'm', 'web thickness')] + _GEO_VARS +
              [V('J_t', 'm^4', 'Saint-Venant torsion constant (thin-walled open approximation)')],
         procedure=_channel, outputs=['A', 'xbar', 'ybar', 'Ix', 'Iy', 'Ixy', 'J', 'Sx', 'Sy',
                                      'Zx', 'Zy', 'rx', 'ry', 'shape_x', 'shape_y', 'I1', 'I2',
                                      'theta_p', 'J_t'],
         example={'bf': 0.1, 'tf': 0.012, 'd': 0.2, 'tw': 0.008},
         assumptions=['x̄ is measured from the back of the web; the shear centre lies outside the web '
                      '(see channel_shear_center). J_t is the open-section approximation Σ b·t³/3.'],
         refs=['Channel centroid and second moments by the parallel-axis theorem (recalled)']),

    Calc('tee', 'Tee section properties',
         vars=[V('bf', 'm', 'flange width'), V('tf', 'm', 'flange thickness'),
               V('d', 'm', 'overall depth'), V('tw', 'm', 'stem thickness')] + _GEO_VARS +
              [V('J_t', 'm^4', 'Saint-Venant torsion constant (thin-walled open approximation)')],
         procedure=_tee, outputs=['A', 'xbar', 'ybar', 'Ix', 'Iy', 'Ixy', 'J', 'Sx', 'Sy',
                                  'Zx', 'Zy', 'rx', 'ry', 'shape_x', 'shape_y', 'I1', 'I2',
                                  'theta_p', 'J_t'],
         example={'bf': 0.15, 'tf': 0.02, 'd': 0.2, 'tw': 0.01},
         assumptions=['ȳ is measured from the bottom of the stem; the shear centre is at the '
                      'flange/stem junction on the symmetry axis.'],
         refs=['Tee centroid and second moments by the parallel-axis theorem (recalled)']),

    Calc('angle', 'Equal-leg angle section properties',
         vars=[V('L', 'm', 'leg length'), V('t', 'm', 'leg thickness')] + _GEO_VARS +
              [V('J_t', 'm^4', 'Saint-Venant torsion constant (thin-walled open approximation)')],
         procedure=_angle, outputs=['A', 'xbar', 'ybar', 'Ix', 'Iy', 'Ixy', 'J', 'Sx', 'Sy',
                                    'Zx', 'Zy', 'rx', 'ry', 'shape_x', 'shape_y', 'I1', 'I2',
                                    'theta_p', 'J_t'],
         example={'L': 0.1, 't': 0.01},
         assumptions=['x̄ = ȳ measured from the heel (back of the legs); principal axes at ±45° to the '
                      'legs; A = t(2L − t). J_t is the open-section approximation (two legs).'],
         refs=['Equal-leg angle: A = t(2L − t), ȳ = (L² + Lt − t²)/(2(2L − t)) (recalled)']),

    Calc('channel_shear_center', 'Shear centre of a thin-walled channel',
         vars=[V('b', 'm', 'flange width (web centreline to flange tip)'),
               V('h', 'm', 'web depth (between flange centrelines)'),
               V('t', 'm', 'uniform wall thickness'),
               V('Ix', 'm^4', 'second moment about the centroidal x-axis'),
               V('e', 'm', 'shear-centre distance from the web centreline (opposite the toes)'),
               V('A', 'm^2', 'area')],
         procedure=_channel_shear_center, outputs=['Ix', 'e', 'A'],
         example={'b': 0.05, 'h': 0.1, 't': 0.005},
         assumptions=['Thin uniform wall (t ≪ b, h). Shear centre lies on the symmetry axis, outside the '
                      'web.'],
         refs=['Channel shear-centre distance e = 3b²/(6b + h) (recalled)']),

    Calc('polygon_properties', 'General polygon section properties',
         vars=[V('vertices', '', '[[x, y], ...] vertices (m), any winding order', kind='json')] + _GEO_VARS,
         procedure=_polygon, outputs=['A', 'xbar', 'ybar', 'Ix', 'Iy', 'Ixy', 'J', 'Sx', 'Sy',
                                      'Zx', 'Zy', 'rx', 'ry', 'shape_x', 'shape_y', 'I1', 'I2', 'theta_p'],
         example={'vertices': [[0, 0], [0.2, 0], [0.2, 0.1], [0, 0.1]]},
         assumptions=['Simple polygon; Zx/Zy by bisection of the equal-area axis. Concave sections are '
                      'also served by the rectangle-composite procedure.'],
         refs=['Area, centroid and second moments of a polygon by Green’s theorem (recalled)']),

    Calc('composite', 'Composite of rectangles: full section properties',
         vars=[V('rects', '', '[[b, h, xc, yc], ...] rows (m); non-overlapping rectangles', kind='json')] +
              _GEO_VARS,
         procedure=_composite, outputs=['A', 'xbar', 'ybar', 'Ix', 'Iy', 'Ixy', 'J', 'Sx', 'Sy',
                                        'Zx', 'Zy', 'rx', 'ry', 'shape_x', 'shape_y', 'I1', 'I2', 'theta_p'],
         example={'rects': [[0.2, 0.02, 0.1, 0.19], [0.2, 0.02, 0.1, 0.01], [0.01, 0.16, 0.1, 0.1]]},
         assumptions=['Parallel-axis combination; Zx/Zy by bisection of the equal-area axis.'],
         refs=['Composite section by the parallel-axis theorem (recalled)']),
]

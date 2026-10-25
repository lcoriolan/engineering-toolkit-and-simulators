# DESCRIPTION: Engineering mathematics section: plane geometry, vectors, matrices, calculus (derivative,
#              integral, Laplace and Fourier transforms), numerical methods (Newton–Raphson, Simpson,
#              Runge–Kutta), statistics (descriptives, regression, confidence intervals), unit
#              conversion, percent error and interpolation.
"""Engineering mathematics calculators.

Symbolic-math procedures (derivative, integral, Laplace, Fourier, Newton, Simpson, Runge–Kutta) take an
expression as a **one-element JSON list** holding the sympy string, e.g. ``["x**2 + 3*x"]``. The string is
parsed by ``core.safe_expr`` (an AST allow-list — no ``eval``, ``sympify`` or ``parse_expr``), so a payload
such as ``().__class__.__mro__`` or ``Symbol.__new__.__globals__[...]`` cannot execute; any expression whose
free symbols are not the declared variable(s) is rejected. Variables are fixed per calculator: ``x`` for
derivative/integral/Fourier/Newton/Simpson, ``t`` for the Laplace transform (transform variable ``s``),
``x`` and ``y`` for the ODE right-hand side in Runge–Kutta. Symbolic differentiation/integration and the
Laplace/Fourier transforms run under ``core.time_limit`` and raise MechError on timeout. Angles are radians
internally; the output of a statistical or geometric procedure is a bare dimensionless number (enter data
in one consistent unit system).
"""
import math

import numpy as np
import sympy as sp
from scipy import stats

from engineering_sim.science.mech.core import Calc, MechError, Q_, V, _Timeout, safe_expr, section, time_limit

SECTION = section('math', 'Engineering mathematics', order=10,
                  summary='Geometry, vectors, matrices, calculus and transforms, numerical methods, '
                          'statistics, unit conversion, percent error and interpolation.')

PI = math.pi

# ---------------------------------------------------------------------------------------------------
# Safe expression parsing (AST allow-list, no eval)
# ---------------------------------------------------------------------------------------------------
# User expressions are parsed ONLY by core.safe_expr (see core.py): an ast.parse allow-list that accepts
# numbers, the declared variable names, + − × ÷ ** % ^ and allow-listed functions, and rejects attribute
# access, subscripts, calls with keywords, strings, lambdas and every other node, so a payload such as
# ``().__class__.__mro__[-1].__subclasses__()`` or ``Symbol.__new__.__globals__[...]`` cannot execute.
# sympify / parse_expr / eval are never used on user text here.
_MAX_EXPR_LEN = 200


def _expr_text(raw, name='expr'):
    """Pull the expression string out of its one-element JSON-list wrapper."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 1 or not isinstance(raw[0], str):
        raise MechError(f'{name}: pass a one-element JSON list holding the expression, '
                        f'e.g. {name}=["x**2 + 3*x"]')
    return raw[0]


def _parse(expr, var, extra_syms=()):
    """Parse ``expr`` with core.safe_expr; only ``var`` (and ``extra_syms``) may appear free."""
    if not isinstance(expr, str) or not expr.strip():
        raise MechError('expression must be a non-empty string')
    if len(expr) > _MAX_EXPR_LEN:
        raise MechError(f'expression is longer than {_MAX_EXPR_LEN} characters')
    parsed, syms = safe_expr(expr.strip(), (var,) + tuple(extra_syms))
    # safe_expr's allow-list cannot produce these, but guard anyway so a future widening of the list
    # cannot smuggle an unevaluated operator through.
    if parsed.has(sp.Derivative, sp.Integral, sp.Sum, sp.Limit):
        raise MechError('nested calculus operators (Derivative/Integral/Sum/Limit) are not allowed')
    return parsed, syms


def _lambdified(parsed, syms, order):
    """numpy lambdify of ``parsed`` over ``order`` (a tuple of symbols), or raise a user-facing error."""
    try:
        return sp.lambdify(order, parsed, modules=['numpy'])
    except Exception as exc:
        raise MechError(f'expression cannot be compiled for numeric evaluation: {exc}')


def _finite(x, name='value'):
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        raise MechError(f'{name} is not a finite number')
    return float(x)


def _sym_op(fn, what):
    """Run a symbolic operation (diff/integrate/transform) under core.time_limit.

    A timeout (``_Timeout``) or a sympy failure becomes a user-facing MechError, so a hard-to-handle
    expression fails fast instead of hanging the worker thread.
    """
    try:
        with time_limit(5.0):
            return fn()
    except MechError:
        raise
    except _Timeout:
        raise MechError(f'{what} timed out (expression too hard); try a simpler expression')
    except Exception as exc:
        raise MechError(f'{what} failed: {exc}')


def _real_value(val, what):
    """Real float from a sympy/numpy numeric result; MechError if non-numeric or meaningfully complex."""
    try:
        c = complex(val)
    except (TypeError, ValueError):
        raise MechError(f'{what}: result is not a finite number ({val})')
    if abs(c.imag) > 1e-9 * max(1.0, abs(c.real)):
        raise MechError(f'{what}: result is complex-valued ({val}); check the limits/integrand')
    return _finite(c.real)


# ---------------------------------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------------------------------
def _derivative(v):
    """Symbolic and numeric derivative of f(x): derivative_expr (string) and its value at x = at."""
    expr, syms = _parse(_expr_text(v['expr']), 'x')
    x = syms['x']
    d = _sym_op(lambda: sp.diff(expr, x), 'symbolic differentiation')
    fn = _lambdified(d, syms, (x,))
    return {'derivative_expr': sp.sstr(d), 'value': _finite(fn(_finite(v['at'], 'at')))}


def _integral(v):
    """Indefinite integral (string) and the definite integral of f(x) over [a, b]."""
    expr, syms = _parse(_expr_text(v['expr']), 'x')
    x = syms['x']
    a, b = _finite(v['a'], 'a'), _finite(v['b'], 'b')
    if b <= a:
        raise MechError('integral: need b > a')
    indef = _sym_op(lambda: sp.integrate(expr, x), 'indefinite integration')
    val = _sym_op(lambda: sp.integrate(expr, (x, a, b)), 'definite integration')
    # Refuse a complex result outright: silently taking .real would hide a sign/branch error.
    return {'integral_expr': sp.sstr(indef), 'definite': _real_value(val, 'integral: definite value')}


def _laplace(v):
    """Laplace transform L{f(t)}(s) (string) and its value at s = s_val."""
    expr, syms = _parse(_expr_text(v['expr']), 't')
    t = syms['t']
    s = sp.Symbol('s', real=True)
    F = _sym_op(lambda: sp.laplace_transform(expr, t, s, noconds=True), 'Laplace transform')
    sval = _finite(v['s_val'], 's_val')
    try:
        fn = _lambdified(F, syms, (s,))
        val = fn(sval)
    except MechError:
        raise
    except Exception:
        val = F.subs(s, sval)
    return {'transform': sp.sstr(F), 'value': _real_value(val, 'laplace_transform: value')}


def _fourier(v):
    """Fourier transform F{f(x)}(k) (sympy convention F(k)=∫f(x)e^(-2πikx)dx); value at k = k_val."""
    expr, syms = _parse(_expr_text(v['expr']), 'x')
    x = syms['x']
    k = sp.Symbol('k', real=True)
    F = _sym_op(lambda: sp.fourier_transform(expr, x, k), 'Fourier transform')
    kval = _finite(v['k_val'], 'k_val')
    try:
        fn = _lambdified(F, syms, (k,))
        val = fn(kval)
    except MechError:
        raise
    except Exception:
        val = F.subs(k, kval)
    return {'transform': sp.sstr(F), 'value': _real_value(val, 'fourier_transform: value')}


def _newton(v):
    """Newton–Raphson root of f(x)=0 from a starting guess x0 (iterated to 1e-12 relative tolerance)."""
    expr, syms = _parse(_expr_text(v['expr']), 'x')
    x = syms['x']
    f = _lambdified(expr, syms, (x,))
    df = _lambdified(_sym_op(lambda: sp.diff(expr, x), 'symbolic differentiation'), syms, (x,))
    xn = _finite(v['x0'], 'x0')
    for i in range(100):
        fx, d = f(xn), df(xn)
        if not (math.isfinite(fx) and math.isfinite(d)) or d == 0:
            raise MechError(f'Newton–Raphson broke down at x = {xn:.6g} (derivative vanished); '
                            f'pick another x0')
        xn1 = xn - fx / d
        if abs(xn1 - xn) <= 1e-12 * max(1.0, abs(xn1)):
            xn = xn1
            break
        xn = xn1
    else:
        raise MechError('Newton–Raphson did not converge in 100 iterations')
    if not math.isfinite(xn):
        raise MechError('Newton–Raphson diverged; pick a starting guess nearer a root')
    return {'root': xn, 'iterations': i + 1}


def _simpson(v):
    """Composite Simpson's rule for ∫_a^b f(x) dx with n equal panels (n even). Exact for cubics."""
    expr, syms = _parse(_expr_text(v['expr']), 'x')
    x = syms['x']
    f = _lambdified(expr, syms, (x,))
    a, b = _finite(v['a'], 'a'), _finite(v['b'], 'b')
    n = int(v['n'])
    if b <= a:
        raise MechError('simpson: need b > a')
    if n < 2 or n % 2 != 0:
        raise MechError('simpson: n must be an even integer >= 2')
    h = (b - a) / n
    ys = [f(a + i * h) for i in range(n + 1)]
    if any(not math.isfinite(y) for y in ys):
        raise MechError('simpson: integrand is not finite somewhere on [a, b]')
    total = ys[0] + ys[-1] + 4.0 * sum(ys[1:-1:2]) + 2.0 * sum(ys[2:-1:2])
    return {'integral': total * h / 3.0}


def _runge_kutta(v):
    """Classic 4th-order Runge–Kutta for y' = f(x, y) over n steps of size h from (x0, y0)."""
    expr, syms = _parse(_expr_text(v['expr']), 'x', extra_syms=('y',))
    x, y = syms['x'], syms['y']
    f = _lambdified(expr, syms, (x, y))
    xn, yn = _finite(v['x0'], 'x0'), _finite(v['y0'], 'y0')
    h = _finite(v['h'], 'h')
    n = int(v['n'])
    if n < 1:
        raise MechError('runge_kutta: n must be >= 1')
    for _ in range(n):
        k1 = f(xn, yn)
        k2 = f(xn + h / 2.0, yn + h * k1 / 2.0)
        k3 = f(xn + h / 2.0, yn + h * k2 / 2.0)
        k4 = f(xn + h, yn + h * k3)
        if not all(math.isfinite(k) for k in (k1, k2, k3, k4)):
            raise MechError('runge_kutta: right-hand side is not finite on this step')
        yn = yn + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        xn = xn + h
    return {'y_final': yn}


def _stats(v):
    """Mean, sample/population variance and standard deviation of a data list (sample std uses n-1)."""
    data = v['data']
    if not isinstance(data, list) or len(data) < 2:
        raise MechError('data: give a JSON list of at least 2 numbers')
    xs = [_finite(x, 'data entry') for x in data]
    n = len(xs)
    mean = sum(xs) / n
    var_s = sum((x - mean) ** 2 for x in xs) / (n - 1)
    var_p = sum((x - mean) ** 2 for x in xs) / n
    return {'mean': mean, 'sample_std': math.sqrt(var_s), 'sample_var': var_s, 'pop_var': var_p,
            'count': n}


def _regression(v):
    """Ordinary least-squares fit y = slope*x + intercept with Pearson r and r²."""
    pts = v['points']
    if not isinstance(pts, list) or len(pts) < 2:
        raise MechError('points: give a JSON list of at least 2 [x, y] rows')
    xs, ys = [], []
    for p in pts:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise MechError('points: each row must be [x, y]')
        xs.append(_finite(p[0], 'x')); ys.append(_finite(p[1], 'y'))
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0:
        raise MechError('points: x values are identical (vertical line); fit is undefined')
    slope = sxy / sxx
    intercept = my - slope * mx
    r = sxy / math.sqrt(sxx * syy) if syy > 0 else 0.0
    return {'slope': slope, 'intercept': intercept, 'r': r, 'r2': r * r}


def _confidence(v):
    """Two-sided confidence interval for the mean of a data list using a Student-t quantile."""
    data = v['data']
    conf = _finite(v['conf'], 'conf')
    if not (0 < conf < 1):
        raise MechError('conf must be strictly between 0 and 1')
    if not isinstance(data, list) or len(data) < 2:
        raise MechError('data: give a JSON list of at least 2 numbers')
    xs = [_finite(x, 'data entry') for x in data]
    n = len(xs)
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    t = float(stats.t.ppf(1.0 - (1.0 - conf) / 2.0, n - 1))
    margin = t * sd / math.sqrt(n)
    return {'margin': margin, 'mean': mean, 'lower': mean - margin, 'upper': mean + margin}


def _unit_convert(v):
    """Convert a value between any two pint units via the shared core registry (offset units supported)."""
    value = _finite(v['value'], 'value')
    frm = v['unit_from']
    to = v['unit_to']
    if not isinstance(frm, (list, tuple)) or len(frm) != 1 or not isinstance(frm[0], str):
        raise MechError('unit_from: pass a one-element JSON list, e.g. unit_from=["in"]')
    if not isinstance(to, (list, tuple)) or len(to) != 1 or not isinstance(to[0], str):
        raise MechError('unit_to: pass a one-element JSON list, e.g. unit_to=["mm"]')
    frm, to = frm[0], to[0]
    frm = {'%': 'percent'}.get(frm, frm)
    to = {'%': 'percent'}.get(to, to)
    try:
        q = Q_(value, frm or 'dimensionless')
        out = q.to(to or 'dimensionless').magnitude
    except Exception as exc:
        raise MechError(f'cannot convert {frm!r} to {to!r}: {exc}')
    return {'value_out': _finite(out, 'value_out')}


def _vector_ops(v):
    """3-D vector algebra: magnitudes, dot and cross products, resultant and included angle."""
    a, b = v['a'], v['b']
    if not isinstance(a, (list, tuple)) or len(a) != 3 or not isinstance(b, (list, tuple)) or len(b) != 3:
        raise MechError('a and b: give two 3-component JSON lists')
    a = [_finite(x, 'a component') for x in a]
    b = [_finite(x, 'b component') for x in b]
    dot = sum(ai * bi for ai, bi in zip(a, b))
    cross = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    ma, mb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if ma == 0 or mb == 0:
        raise MechError('a and b must be non-zero vectors')
    ang = math.acos(max(-1.0, min(1.0, dot / (ma * mb))))
    return {'dot': dot, 'cross': cross, 'mag_a': ma, 'mag_b': mb, 'angle': ang,
            'sum': [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
            'sum_mag': math.sqrt(sum((a[i] + b[i]) ** 2 for i in range(3)))}


def _matrix_ops(v):
    """Determinant, trace and inverse of a square 2x2 or 3x3 matrix."""
    m = v['matrix']
    if not isinstance(m, (list, tuple)) or len(m) not in (2, 3) or not all(
            isinstance(row, (list, tuple)) and len(row) == len(m) for row in m):
        raise MechError('matrix: give a square 2x2 or 3x3 JSON list of rows')
    a = np.array([[float(x) for x in row] for row in m])
    det = float(np.linalg.det(a))
    if abs(det) < 1e-15:
        raise MechError('matrix is singular (zero determinant); inverse is undefined')
    inv = np.linalg.inv(a).tolist()
    return {'det': det, 'trace': float(np.trace(a)), 'inverse': inv}


def _eigen(v):
    """Eigenvalues (descending) and unit eigenvectors of a real symmetric 2x2 or 3x3 matrix."""
    m = v['matrix']
    if not isinstance(m, (list, tuple)) or len(m) not in (2, 3) or not all(
            isinstance(row, (list, tuple)) and len(row) == len(m) for row in m):
        raise MechError('matrix: give a square 2x2 or 3x3 JSON list of rows')
    a = np.array([[float(x) for x in row] for row in m])
    if not np.allclose(a, a.T, atol=1e-12):
        raise MechError('eigen: eigenvalues/eigenvectors require a symmetric matrix '
                        '(off-diagonal symmetry violated)')
    w, vec = np.linalg.eigh(a)
    order = np.argsort(w)[::-1]
    return {'eigenvalues': [float(w[i]) for i in order],
            'eigenvectors': [vec[:, i].tolist() for i in order]}


def _triangle_centroid(v):
    """Centroid (average of vertices) of a triangle given by three [x, y] vertices."""
    verts = v['vertices']
    if not isinstance(verts, (list, tuple)) or len(verts) != 3 or not all(
            isinstance(p, (list, tuple)) and len(p) == 2 for p in verts):
        raise MechError('vertices: give three [x, y] JSON rows')
    pts = [[_finite(p[0], 'x'), _finite(p[1], 'y')] for p in verts]
    return {'x_bar': sum(p[0] for p in pts) / 3.0, 'y_bar': sum(p[1] for p in pts) / 3.0}


# ---------------------------------------------------------------------------------------------------
# Calculators
# ---------------------------------------------------------------------------------------------------
CALCS = [
    Calc('circle_area', 'Circle: area and circumference',
         vars=[V('r', 'm', 'radius'), V('A', 'm^2', 'area'), V('C', 'm', 'circumference')],
         equations=['A = pi*r**2', 'C = 2*pi*r'], outputs=['A', 'C'],
         example={'r': 1.0}, keywords=['disk', 'perimeter'],
         refs=['Area and circumference of a circle (Euclidean geometry; recalled)']),

    Calc('sphere_volume', 'Sphere: volume and surface area',
         vars=[V('r', 'm', 'radius'), V('V', 'm^3', 'volume'), V('A', 'm^2', 'surface area')],
         equations=['V = 4*pi*r**3/3', 'A = 4*pi*r**2'], outputs=['V', 'A'],
         example={'r': 1.0}, keywords=['ball'],
         refs=['Volume and surface area of a sphere (Euclidean geometry; recalled)']),

    Calc('arc_length', 'Circular arc length (s = r·θ)',
         vars=[V('r', 'm', 'radius'), V('theta', 'rad', 'central angle', hi=2 * PI, label='θ'),
               V('s', 'm', 'arc length')],
         equations=['s = r*theta'], outputs=['s'],
         example={'r': 0.5, 'theta': math.radians(60)},
         keywords=['segment', 'sector'],
         refs=['Circular arc length s = rθ (θ in radians; recalled)']),

    Calc('curvature', 'Curvature of a circle (κ = 1/R)',
         vars=[V('R', 'm', 'radius of curvature'), V('kappa', '1/m', 'curvature', label='κ')],
         equations=['kappa = 1/R'], outputs=['kappa'],
         example={'R': 0.25}, keywords=['radius of curvature'],
         refs=['Curvature of a circular arc, κ = 1/R (recalled)']),

    Calc('triangle_centroid', 'Centroid of a triangle (average of vertices)',
         vars=[V('vertices', '', 'three [x, y] vertex rows (m)', kind='json'),
               V('x_bar', 'm', 'x̄ of the centroid', label='x̄'),
               V('y_bar', 'm', 'ȳ of the centroid', label='ȳ')],
         procedure=_triangle_centroid, outputs=['x_bar', 'y_bar'],
         example={'vertices': [[0, 0], [4, 0], [1, 3]]},
         refs=['Centroid of a triangle = mean of its vertices (recalled)']),

    Calc('vector_components', 'Vector components from magnitude and direction',
         vars=[V('r', '', 'magnitude'), V('theta', 'rad', 'direction from +x, CCW +', lo=-2 * PI,
                                            hi=2 * PI, label='θ'),
               V('x', '', 'x component', signed=True), V('y', '', 'y component', signed=True)],
         equations=['x = r*cos(theta)', 'y = r*sin(theta)'], outputs=['x', 'y'],
         example={'r': 5.0, 'theta': math.radians(30)},
         assumptions=['Dimensionless vector; inverse solves for θ return the root closest to the example '
                      '(check the quadrant).'],
         refs=['Rectangular components of a vector']),

    Calc('vector_ops', '3-D vector algebra: magnitudes, dot, cross, resultant, angle',
         vars=[V('a', '', 'first vector [ax, ay, az]', kind='json'),
               V('b', '', 'second vector [bx, by, bz]', kind='json'),
               V('dot', '', 'dot product a·b', signed=True),
               V('cross', '', 'cross product a×b as [cx, cy, cz]', kind='json'),
               V('mag_a', '', '|a|'), V('mag_b', '', '|b|'),
               V('angle', 'rad', 'angle between a and b', hi=PI, label='θ'),
               V('sum', '', 'resultant a + b as [sx, sy, sz]', kind='json'),
               V('sum_mag', '', '|a + b|')],
         procedure=_vector_ops, outputs=['dot', 'cross', 'mag_a', 'mag_b', 'angle', 'sum', 'sum_mag'],
         example={'a': [1, 2, 3], 'b': [4, 5, 6]},
         refs=['Dot product, cross product, vector magnitude and included angle (recalled)']),

    Calc('det2', 'Determinant of a 2×2 matrix (ad − bc)',
         vars=[V('a', '', 'row 1, col 1'), V('b', '', 'row 1, col 2'),
               V('c', '', 'row 2, col 1'), V('d', '', 'row 2, col 2'),
               V('det', '', 'determinant ad − bc', signed=True)],
         equations=['det = a*d - b*c'], outputs=['det'],
         example={'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0},
         refs=['Determinant of a 2×2 matrix (recalled)']),

    Calc('matrix_ops', 'Matrix determinant, trace and inverse (2×2 or 3×3)',
         vars=[V('matrix', '', 'square matrix as a JSON list of rows', kind='json'),
               V('det', '', 'determinant', signed=True), V('trace', '', 'trace (sum of diagonal)', signed=True),
               V('inverse', '', 'inverse matrix as a JSON list of rows', kind='json')],
         procedure=_matrix_ops, outputs=['det', 'trace', 'inverse'],
         example={'matrix': [[1, 2], [3, 4]]},
         assumptions=['Raises an error for a singular matrix.'],
         refs=['Determinant, trace and inverse of a square matrix (recalled)']),

    Calc('eigen', 'Eigenvalues and eigenvectors (real symmetric 2×2 or 3×3)',
         vars=[V('matrix', '', 'symmetric matrix as a JSON list of rows', kind='json'),
               V('eigenvalues', '', 'eigenvalues, descending', kind='json'),
               V('eigenvectors', '', 'unit eigenvectors (columns, same order as eigenvalues)', kind='json')],
         procedure=_eigen, outputs=['eigenvalues', 'eigenvectors'],
         example={'matrix': [[2, 1], [1, 2]]},
         assumptions=['Restricted to real symmetric matrices so the eigen-decomposition is real and '
                      'the eigenvectors are orthogonal.'],
         refs=['Eigendecomposition of a real symmetric matrix (recalled)']),

    Calc('derivative', 'Derivative of f(x): symbolic and numeric',
         vars=[V('expr', '', 'expression in x, as a one-element JSON list ["f(x)"]', kind='json'),
               V('at', '', 'x value at which to evaluate the derivative'),
               V('derivative_expr', '', 'symbolic derivative (expression string)', kind='json'),
               V('value', '', "f'(x) evaluated at x = at", signed=True)],
         procedure=_derivative, outputs=['derivative_expr', 'value'],
         example={'expr': ['x**2 + 3*x'], 'at': 3.0},
         keywords=['differentiation', 'slope'],
         refs=['Symbolic differentiation (sympy); derivative by the power rule (recalled)']),

    Calc('integral', 'Definite and indefinite integral of f(x)',
         vars=[V('expr', '', 'expression in x, as a one-element JSON list ["f(x)"]', kind='json'),
               V('a', '', 'lower limit'), V('b', '', 'upper limit (b > a)'),
               V('integral_expr', '', 'indefinite integral (expression string)', kind='json'),
               V('definite', '', 'definite integral over [a, b]', signed=True)],
         procedure=_integral, outputs=['integral_expr', 'definite'],
         example={'expr': ['x'], 'a': 0.0, 'b': 2.0},
         keywords=['antiderivative', 'quadrature'],
         refs=['Fundamental theorem of calculus; symbolic integration (recalled)']),

    Calc('laplace_transform', 'Laplace transform L{f(t)}(s): symbolic and numeric',
         vars=[V('expr', '', 'expression in t, as a one-element JSON list ["f(t)"]', kind='json'),
               V('s_val', '', 's value at which to evaluate the transform'),
               V('transform', '', 'Laplace transform (expression string in s)', kind='json'),
               V('value', '', 'transform evaluated at s = s_val', signed=True)],
         procedure=_laplace, outputs=['transform', 'value'],
         example={'expr': ['exp(-3*t)'], 's_val': 1.0},
         keywords=['transfer function', 's-domain'],
         refs=['Unilateral Laplace transform L{f(t)} = ∫₀^∞ f(t)e^(−st) dt (recalled)']),

    Calc('fourier_transform', 'Fourier transform F{f(x)}(k): symbolic and numeric',
         vars=[V('expr', '', 'expression in x, as a one-element JSON list ["f(x)"]', kind='json'),
               V('k_val', '', 'k value at which to evaluate the transform'),
               V('transform', '', 'Fourier transform (expression string in k)', kind='json'),
               V('value', '', 'transform evaluated at k = k_val', signed=True)],
         procedure=_fourier, outputs=['transform', 'value'],
         example={'expr': ['exp(-x**2)'], 'k_val': 1.0},
         keywords=['spectrum', 'frequency domain'],
         refs=['Fourier transform F(k) = ∫ f(x)e^(−2πikx) dx (sympy convention; recalled)']),

    Calc('newton_raphson', 'Newton–Raphson root of f(x) = 0',
         vars=[V('expr', '', 'expression in x, as a one-element JSON list ["f(x)"]', kind='json'),
               V('x0', '', 'starting guess'),
               V('root', '', 'converged root', signed=True),
               V('iterations', '', 'iterations used', integer=True)],
         procedure=_newton, outputs=['root', 'iterations'],
         no_invert={'x0': 'the converged root is independent of the starting guess within its basin '
                           'of attraction, so x0 cannot be recovered from the result'},
         example={'expr': ['x**2 - 2'], 'x0': 1.0},
         keywords=['tangent method', 'fixed point'],
         refs=['Newton–Raphson iteration x_{n+1} = x_n − f(x_n)/f′(x_n) (recalled)']),

    Calc('simpson_rule', "Composite Simpson's rule (n even panels)",
         vars=[V('expr', '', 'integrand in x, as a one-element JSON list ["f(x)"]', kind='json'),
               V('a', '', 'lower limit'), V('b', '', 'upper limit (b > a)'),
               V('n', '', 'number of panels (even)', lo=2, hi=1000, integer=True),
               V('integral', '', 'approximate integral', signed=True)],
         procedure=_simpson, outputs=['integral'],
         example={'expr': ['x**3'], 'a': 0.0, 'b': 1.0, 'n': 4},
         keywords=['quadrature', 'parabolic'],
         assumptions=['Exact for polynomials of degree ≤ 3.'],
         refs=["Composite Simpson's rule (recalled)"]),

    Calc('runge_kutta', "Classic 4th-order Runge–Kutta for y' = f(x, y)",
         vars=[V('expr', '', "right-hand side f(x, y), as a one-element JSON list ['f(x, y)']", kind='json'),
               V('x0', '', 'initial x', signed=True), V('y0', '', 'initial y', signed=True),
               V('h', '', 'step size'), V('n', '', 'number of steps', lo=1, hi=100000, integer=True),
               V('y_final', '', 'y at x = x0 + n·h', signed=True)],
         procedure=_runge_kutta, outputs=['y_final'],
         example={'expr': ['x'], 'x0': 0.0, 'y0': 0.0, 'h': 0.25, 'n': 4},
         keywords=['ode', 'ivp', 'integration'],
         assumptions=['Fixed step size h; global error O(h⁴). Exact when f depends only on x.'],
         refs=['Classic fourth-order Runge–Kutta method (recalled)']),

    Calc('stats_basic', 'Mean, variance and standard deviation of a data set',
         vars=[V('data', '', 'JSON list of numbers (one consistent unit)', kind='json'),
               V('mean', '', 'arithmetic mean', signed=True),
               V('sample_std', '', 'sample standard deviation (n−1)'),
               V('sample_var', '', 'sample variance (n−1)'),
               V('pop_var', '', 'population variance (n)'),
               V('count', '', 'number of data points', integer=True)],
         procedure=_stats, outputs=['mean', 'sample_std', 'sample_var', 'pop_var', 'count'],
         example={'data': [2, 4, 4, 4, 5, 5, 7, 9]},
         keywords=['average', 'dispersion', 'spread'],
         refs=['Sample and population variance/standard deviation (recalled)']),

    Calc('linear_regression', 'Least-squares line fit: slope, intercept, correlation',
         vars=[V('points', '', 'JSON list of [x, y] rows', kind='json'),
               V('slope', '', 'best-fit slope', signed=True), V('intercept', '', 'best-fit intercept', signed=True),
               V('r', '', 'Pearson correlation coefficient (−1..1)', signed=True),
               V('r2', '', 'coefficient of determination r²')],
         procedure=_regression, outputs=['slope', 'intercept', 'r', 'r2'],
         example={'points': [[1, 1], [2, 3], [3, 5]]},
         keywords=['ols', 'trend', 'line of best fit'],
         refs=['Ordinary least-squares regression; Pearson correlation (recalled)']),

    Calc('confidence_interval', 'Confidence interval for the mean (Student-t)',
         vars=[V('data', '', 'JSON list of numbers (one consistent unit)', kind='json'),
               V('conf', '', 'confidence level (0 < conf < 1)', lo=0.0, hi=1.0),
               V('margin', '', 'half-width t*·s/√n'),
               V('mean', '', 'sample mean', signed=True),
               V('lower', '', 'lower bound mean − margin', signed=True),
               V('upper', '', 'upper bound mean + margin', signed=True)],
         procedure=_confidence, outputs=['margin', 'mean', 'lower', 'upper'],
         example={'data': [2, 4, 4, 4, 5, 5, 7, 9], 'conf': 0.95},
         keywords=['t-test', 'uncertainty', 'interval estimate'],
         assumptions=['Assumes the data are independent samples from a normal population.'],
         refs=['Student-t confidence interval for a mean; t-quantile from scipy.stats (recalled)']),

    Calc('unit_convert', 'Unit conversion via the shared pint registry',
         vars=[V('value', '', 'numeric value to convert', signed=True),
               V('unit_from', '', 'source unit, as a one-element JSON list ["in"]', kind='json'),
               V('unit_to', '', 'target unit, as a one-element JSON list ["mm"]', kind='json'),
               V('value_out', '', 'converted value', signed=True)],
         procedure=_unit_convert, outputs=['value_out'],
         example={'value': 12.0, 'unit_from': ['in'], 'unit_to': ['mm']},
         keywords=['pint', 'si', 'imperial'],
         assumptions=['Uses the shared core.Q_ registry (no second unit table). Temperature offset units '
                      '(degC, degF) are handled.'],
         refs=['pint unit conversion (the one registry shared with the rest of mechcalc)']),

    Calc('percent_error', 'Percent error of a measurement vs a true value',
         vars=[V('measured', '', 'measured (or approximate) value', signed=True),
               V('true', '', 'true (or reference) value'),
               V('pct', '', 'percent error (%)', signed=True)],
         equations=['pct = (measured - true)/true*100'], outputs=['pct'],
         example={'measured': 10.5, 'true': 10.0},
         assumptions=['Enter measured and true in the same units (the ratio is dimensionless).'],
         refs=['Percent error = (measured − true)/true × 100% (recalled)']),

    Calc('linear_interp', 'Linear interpolation between two points',
         vars=[V('x0', 'm', 'first x'), V('y0', '', 'first y'),
               V('x1', 'm', 'second x (x1 ≠ x0)'), V('y1', '', 'second y'),
               V('x', 'm', 'query x'), V('y', '', 'interpolated y', signed=True)],
         equations=['y = y0 + (x - x0)*(y1 - y0)/(x1 - x0)'], outputs=['y'],
         example={'x0': 0.0, 'y0': 0.0, 'x1': 2.0, 'y1': 10.0, 'x': 1.0},
         keywords=['lerp', 'lookup'],
         refs=['Linear interpolation between two points (recalled)']),
]

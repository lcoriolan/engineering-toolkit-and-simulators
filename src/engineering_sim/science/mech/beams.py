# DESCRIPTION: Beams and deflection section: a general prismatic-beam solver (point loads, point
#              moments, uniform and linearly varying distributed loads; simple/cantilever/propped/
#              fixed-fixed/continuous supports; reactions including indeterminate cases; SFD/BMD/slope/
#              deflection arrays and charts) plus the standard closed-form table cases (cantilever and
#              simply supported point/UDL/off-centre loads), the energy methods (moment-area,
#              Castigliano, unit-load) and equivalent spring stiffness.
"""Beams and deflection calculators.

Conventions (general solver): x from the left end (0) to the right end (L); loads positive downward for
P and w, point moments positive counter-clockwise; vertical reactions positive upward; bending moment
positive sagging (concave up, top in compression); deflection positive downward; slope = d(deflection)/dx
(positive when the downward deflection increases with x). The general solver writes the load as
singularity (Macaulay) functions, integrates twice to slope and deflection with two integration
constants, and solves the linear system formed by overall equilibrium (ΣFy = 0, ΣM = 0) plus the support
conditions (v = 0 at simple/roller/pin supports; v = 0 and θ = 0 at fixed ends). Indeterminate beams
(propped, fixed-fixed, continuous over n supports) are therefore solved by superposition/compatibility in
the same step, and the method of double integration / moment-area / Castigliano / unit-load closed forms
must agree with it (they do; see the oracles).

Reference relations are the standard beam-table and energy-method results in mechanics-of-materials texts
(e.g. Beer & Johnston, Hibbeler); cited by relation name, recalled rather than re-read.
"""
import math

import numpy as np

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts

SECTION = section('beams', 'Beams and deflection', order=45,
                  summary='General prismatic-beam solver (reactions, shear, moment, slope, deflection) '
                          'and the standard beam-table and energy-method deflection formulas.')


def _finite(x, name):
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        raise MechError(f'{name} is not a finite number')
    return float(x)


def _mac(x, a, n):
    """Singularity (Macaulay) function <x-a>^n for n >= 0 (scalar)."""
    if n == 0:
        return 1.0 if x >= a else 0.0
    return (x - a) ** n if x >= a else 0.0


def _mac_v(xs, a, n):
    """Vectorised singularity function over a numpy array xs."""
    xs = np.asarray(xs, dtype=float)
    if n == 0:
        return (xs >= a).astype(float)
    return np.where(xs >= a, (xs - a) ** n, 0.0)


def _parse_loads(loads):
    """Normalise the loads input into (point_loads, point_moms, dist).

    point_loads: [(P, a)] with P downward positive; point_moms: [(M, a)] with M CCW positive;
    dist: [(wa, wb, a, b)] trapezoidal loads (wa at a, wb at b) downward positive, wa == wb for a UDL.
    """
    point_loads, point_moms, dist = [], [], []
    if not isinstance(loads, list):
        raise MechError('loads: give a JSON list of load rows ([] for none)')
    for row in loads:
        if not isinstance(row, (list, tuple)) or not row:
            raise MechError('loads: each row must be ["P"|"M"|"w"|"lin", ...]')
        kind = row[0]
        if kind == 'P':
            point_loads.append((_finite(row[1], 'load'), _finite(row[2], 'load position')))
        elif kind == 'M':
            point_moms.append((_finite(row[1], 'moment'), _finite(row[2], 'moment position')))
        elif kind == 'w':
            dist.append((_finite(row[1], 'w'), _finite(row[1], 'w'), _finite(row[2], 'a'),
                         _finite(row[3], 'b')))
        elif kind == 'lin':
            dist.append((_finite(row[1], 'wa'), _finite(row[2], 'wb'), _finite(row[3], 'a'),
                         _finite(row[4], 'b')))
        else:
            raise MechError(f'loads: unknown load kind {kind!r} (use "P", "M", "w" or "lin")')
    for wa, wb, a, b in dist:
        if b <= a:
            raise MechError('loads: distributed load needs b > a')
    return point_loads, point_moms, dist


def _parse_supports(supports):
    simple, fixed = [], []
    if not isinstance(supports, list) or not supports:
        raise MechError('supports: give a JSON list of [x, "simple"|"fixed"] rows')
    for s in supports:
        if not isinstance(s, (list, tuple)) or len(s) != 2 or s[1] not in ('simple', 'fixed'):
            raise MechError('supports: each row must be [x, "simple" or "fixed"]')
        x = _finite(s[0], 'support position')
        (simple if s[1] == 'simple' else fixed).append(x)
    return simple, fixed


def _solve_beam(E, I, L, point_loads, point_moms, dist, simple, fixed):
    """Solve reactions and integration constants for the prismatic beam (Euler–Bernoulli).

    Unknowns: one vertical reaction per support (simple then fixed order), one CCW moment reaction per
    fixed support, and the two integration constants C3 = EI·θ(0), C4 = EI·v(0). Equations: overall
    ΣFy = 0, ΣM = 0, v = 0 at simple supports, and v = 0 plus θ = 0 at fixed supports.
    """
    nR = len(simple) + len(fixed)
    nF = len(fixed)
    supp_pts = simple + fixed  # reaction points, in order
    nU = nR + nF + 2
    iC3, iC4 = nR + nF, nR + nF + 1

    Wtot = sum(P for P, _ in point_loads)
    Wmom = sum(P * a for P, a in point_loads)
    for wa, wb, a, b in dist:
        W = (wa + wb) / 2 * (b - a)
        cx = a + (b - a) * (2 * wb + wa) / (3 * (wa + wb)) if wa + wb != 0 else (a + b) / 2
        Wtot += W
        Wmom += W * cx
    M_applied = sum(M for M, _ in point_moms)

    A = np.zeros((nU, nU))
    b = np.zeros(nU)
    for j in range(nR):
        A[0, j] = 1.0
    b[0] = Wtot
    for j in range(nR):
        A[1, j] = supp_pts[j]
    for j in range(nF):
        A[1, nR + j] = 1.0
    b[1] = Wmom - M_applied

    def coeffs(x):
        """Coefficients (per unknown) and constants of EI·θ(x) and EI·v(x)."""
        ct = np.zeros(nU)
        cv = np.zeros(nU)
        cth = cvh = 0.0
        for j in range(nR):
            xj = supp_pts[j]
            ct[j] += -_mac(x, xj, 2) / 2.0
            cv[j] += -_mac(x, xj, 3) / 6.0
        for j in range(nF):
            xf = fixed[j]
            ct[nR + j] += _mac(x, xf, 1)
            cv[nR + j] += _mac(x, xf, 2) / 2.0
        ct[iC3] += 1.0
        cv[iC3] += x
        cv[iC4] += 1.0
        for P, a in point_loads:
            cth += P / 2 * _mac(x, a, 2)
            cvh += P / 6 * _mac(x, a, 3)
        for M, a in point_moms:
            cth += M * _mac(x, a, 1)
            cvh += M / 2 * _mac(x, a, 2)
        for wa, wb, a, b in dist:
            sl = (wb - wa) / (b - a)
            cth += wa / 6 * _mac(x, a, 3) + sl / 24 * _mac(x, a, 4) \
                - wb / 6 * _mac(x, b, 3) - sl / 24 * _mac(x, b, 4)
            cvh += wa / 24 * _mac(x, a, 4) + sl / 120 * _mac(x, a, 5) \
                - wb / 24 * _mac(x, b, 4) - sl / 120 * _mac(x, b, 5)
        return ct, cv, cth, cvh

    row = 2
    for xj in simple:
        ct, cv, cth, cvh = coeffs(xj)
        A[row] = cv
        b[row] = -cvh
        row += 1
    for xf in fixed:
        ct, cv, cth, cvh = coeffs(xf)
        A[row] = cv
        b[row] = -cvh
        row += 1
        A[row] = ct
        b[row] = -cth
        row += 1

    if np.linalg.matrix_rank(A) < nU:
        raise MechError('the support configuration is a mechanism or redundant (singular system)')
    sol = np.linalg.solve(A, b)
    return sol[:nR], sol[nR:nR + nF], sol[iC3], sol[iC4], supp_pts


def _fields(xs, R, supp_pts, Mf, fixed, C3, C4, EI, point_loads, point_moms, dist):
    """Vectorised V, M, θ, v on the grid xs (θ and v already divided by EI)."""
    xs = np.asarray(xs, dtype=float)
    V = np.zeros_like(xs)
    M = np.zeros_like(xs)
    th = np.zeros_like(xs)
    de = np.zeros_like(xs)
    for j, Rj in enumerate(R):
        xj = supp_pts[j]
        V += Rj * _mac_v(xs, xj, 0)
        M += Rj * _mac_v(xs, xj, 1)
        th -= Rj / 2 * _mac_v(xs, xj, 2)
        de -= Rj / 6 * _mac_v(xs, xj, 3)
    for P, a in point_loads:
        V -= P * _mac_v(xs, a, 0)
        M -= P * _mac_v(xs, a, 1)
        th += P / 2 * _mac_v(xs, a, 2)
        de += P / 6 * _mac_v(xs, a, 3)
    for M0, a in point_moms:
        M -= M0 * _mac_v(xs, a, 0)
        th += M0 * _mac_v(xs, a, 1)
        de += M0 / 2 * _mac_v(xs, a, 2)
    for j, Mfj in enumerate(Mf):
        xf = fixed[j]
        M -= Mfj * _mac_v(xs, xf, 0)
        th += Mfj * _mac_v(xs, xf, 1)
        de += Mfj / 2 * _mac_v(xs, xf, 2)
    for wa, wb, a, b in dist:
        sl = (wb - wa) / (b - a)
        V -= wa * _mac_v(xs, a, 1) + sl / 2 * _mac_v(xs, a, 2) \
            - wb * _mac_v(xs, b, 1) - sl / 2 * _mac_v(xs, b, 2)
        M -= wa / 2 * _mac_v(xs, a, 2) + sl / 6 * _mac_v(xs, a, 3) \
            - wb / 2 * _mac_v(xs, b, 2) - sl / 6 * _mac_v(xs, b, 3)
        th += wa / 6 * _mac_v(xs, a, 3) + sl / 24 * _mac_v(xs, a, 4) \
            - wb / 6 * _mac_v(xs, b, 3) - sl / 24 * _mac_v(xs, b, 4)
        de += wa / 24 * _mac_v(xs, a, 4) + sl / 120 * _mac_v(xs, a, 5) \
            - wb / 24 * _mac_v(xs, b, 4) - sl / 120 * _mac_v(xs, b, 5)
    th += C3
    de += C3 * xs + C4
    return V, M, th / EI, de / EI


def _max_abs(xs, fn, jumps=()):
    """Location and magnitude of the maximum |field| (a vectorised function of x).

    The field is piecewise-smooth with step jumps at ``jumps`` (concentrated moments and fixed-support
    reaction moments). A dense grid plus golden-section refinement finds interior smooth maxima; each
    jump is additionally probed on both sides (d ± eps) so a peak that sits exactly at a discontinuity —
    e.g. the fixed-end moment of a cantilever at x = L — is reported at its true magnitude instead of
    collapsing to the post-jump value (zero just outside a right-end support).
    """
    span = float(xs[-1]) - float(xs[0])
    eps = 1e-9 * max(1.0, span)
    cand = np.unique(np.concatenate([np.asarray(xs, float),
                                     np.array([d - eps for d in jumps], float),
                                     np.array([d + eps for d in jumps], float)]))
    ys = fn(cand)
    i = int(np.argmax(np.abs(ys)))
    xm, peak = cand[i], abs(ys[i])
    jump = next((d for d in jumps if abs(xm - d) <= 2 * eps), None)
    if jump is not None:
        # Peak is at a step: report the jump location with the larger-side magnitude (the probe that won).
        return float(jump), peak
    a = cand[max(0, i - 1)]
    b = cand[min(len(cand) - 1, i + 1)]
    g = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - g * (b - a)
    d = a + g * (b - a)
    fc, fd = abs(fn(np.array([c]))[0]), abs(fn(np.array([d]))[0])
    for _ in range(80):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = abs(fn(np.array([c]))[0])
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = abs(fn(np.array([d]))[0])
    xm = (a + b) / 2
    return xm, abs(fn(np.array([xm]))[0])


def _general_beam(v):
    """General prismatic Euler–Bernoulli beam: reactions, SFD/BMD/slope/deflection and their extrema."""
    E = _finite(v['E'], 'E')
    I = _finite(v['I'], 'I')
    L = _finite(v['L'], 'L')
    if E <= 0 or I <= 0 or L <= 0:
        raise MechError('general_beam: E, I and L must be positive')
    point_loads, point_moms, dist = _parse_loads(v.get('loads') or [])
    simple, fixed = _parse_supports(v['supports'])
    for x in simple + fixed:
        if not (0.0 <= x <= L):
            raise MechError(f'general_beam: support at x = {x} lies outside [0, L]')
    for P, a in point_loads + point_moms:
        if not (0.0 <= a <= L):
            raise MechError(f'general_beam: load at x = {a} lies outside [0, L]')
    for wa, wb, a, b in dist:
        if not (0.0 <= a <= L and 0.0 <= b <= L):
            raise MechError('general_beam: distributed load must lie inside [0, L]')
    R, Mf, C3, C4, supp_pts = _solve_beam(E, I, L, point_loads, point_moms, dist, simple, fixed)

    crit = sorted(set([0.0, L] + simple + fixed + [a for P, a in point_loads]
                      + [a for M, a in point_moms] + [c for wa, wb, a, b in dist for c in (a, b)]))
    EI = E * I

    # output arrays (a moderate grid plus every critical point, for crisp diagrams)
    grid = np.unique(np.concatenate([np.linspace(0.0, L, 201), np.array(crit)]))
    V, M, th, de = _fields(grid, R, supp_pts, Mf, fixed, C3, C4, EI, point_loads, point_moms, dist)
    # fine grid + critical points for the extrema
    fine = np.unique(np.concatenate([np.linspace(0.0, L, 4001), np.array(crit)]))

    def fV(xs):
        return _fields(xs, R, supp_pts, Mf, fixed, C3, C4, EI, point_loads, point_moms, dist)[0]

    def fM(xs):
        return _fields(xs, R, supp_pts, Mf, fixed, C3, C4, EI, point_loads, point_moms, dist)[1]

    def fd(xs):
        return _fields(xs, R, supp_pts, Mf, fixed, C3, C4, EI, point_loads, point_moms, dist)[3]

    xM, Mmax = _max_abs(fine, fM, jumps=fixed + [a for M, a in point_moms])
    xd, dmax = _max_abs(fine, fd)
    Vmax = float(np.max(np.abs(fV(fine))))

    return {'x': grid.tolist(), 'V': V.tolist(), 'M': M.tolist(), 'theta': th.tolist(),
            'delta': de.tolist(),
            'reactions': [[float(x), float(r)] for x, r in zip(supp_pts, R)],
            'M_fixed': [float(m) for m in Mf],
            'delta_max': dmax, 'x_delta_max': xd, 'M_max': Mmax, 'x_M_max': xM, 'V_max': Vmax}


def _sfd_bmd_chart(vals):
    """Chart specs for the shear-force and bending-moment diagrams."""
    x = vals.get('x') or []
    V = vals.get('V') or []
    M = vals.get('M') or []
    sfd = charts.chart('Shear force diagram', 'x [m]', 'V [N]',
                       [charts.series('V', x, V)], vlines=[0.0])
    bmd = charts.chart('Bending moment diagram', 'x [m]', 'M [N·m]',
                       [charts.series('M', x, M)], vlines=[0.0])
    return [sfd, bmd]


def _ss_point_offcenter(v):
    """Simply supported beam with an off-centre point load: reactions, max moment and load-point
    deflection, checked against the geometry relation L = a + b."""
    P = _finite(v['P'], 'P')
    a = _finite(v['a'], 'a')
    b = _finite(v['b'], 'b')
    L = _finite(v['L'], 'L')
    E = _finite(v['E'], 'E')
    I = _finite(v['I'], 'I')
    if L <= 0 or E <= 0 or I <= 0:
        raise MechError('ss_point_offcenter: need L, E and I > 0')
    if abs(a + b - L) > 1e-9 * L:
        raise MechError(f'ss_point_offcenter: span L must equal a + b '
                        f'(got a = {a}, b = {b}, L = {L})')
    return {'R_A': P * b / L, 'R_B': P * a / L, 'M_max': P * a * b / L,
            'delta_load': P * a ** 2 * b ** 2 / (3 * E * I * L)}


CALCS = [
    Calc('cantilever_end_load', 'Cantilever with an end point load',
         vars=[V('P', 'N', 'end load (downward +)'), V('L', 'm', 'length'),
               V('E', 'Pa', "Young's modulus"), V('I', 'm^4', 'second moment of area'),
               V('delta', 'm', 'tip deflection (downward +)'),
               V('theta', 'rad', 'tip slope', label='θ'),
               V('M_max', 'N*m', 'maximum bending moment (at the fixed end)')],
         equations=['delta = P*L**3/(3*E*I)', 'theta = P*L**2/(2*E*I)', 'M_max = P*L'],
         outputs=['delta', 'theta', 'M_max'],
         example={'P': 1000.0, 'L': 2.0, 'E': 200e9, 'I': 8e-6},
         assumptions=['Euler–Bernoulli, small deflection, prismatic, linear elastic.'],
         refs=['Cantilever end-load deflection by double integration (standard beam table; recalled)']),

    Calc('cantilever_udl', 'Cantilever with a uniform distributed load',
         vars=[V('w', 'N/m', 'load intensity (downward +)'), V('L', 'm', 'length'),
               V('E', 'Pa', "Young's modulus"), V('I', 'm^4', 'second moment of area'),
               V('delta', 'm', 'tip deflection (downward +)'), V('theta', 'rad', 'tip slope', label='θ'),
               V('M_max', 'N*m', 'maximum bending moment (at the fixed end)')],
         equations=['delta = w*L**4/(8*E*I)', 'theta = w*L**3/(6*E*I)', 'M_max = w*L**2/2'],
         outputs=['delta', 'theta', 'M_max'],
         example={'w': 2000.0, 'L': 2.0, 'E': 200e9, 'I': 8e-6},
         assumptions=['Euler–Bernoulli, small deflection, prismatic.'],
         refs=['Cantilever UDL deflection by double integration (standard beam table; recalled)']),

    Calc('ss_point_center', 'Simply supported beam with a central point load',
         vars=[V('P', 'N', 'central load (downward +)'), V('L', 'm', 'span'),
               V('E', 'Pa', "Young's modulus"), V('I', 'm^4', 'second moment of area'),
               V('delta', 'm', 'centre deflection (downward +)'),
               V('theta_A', 'rad', 'slope at the left support', label='θA'),
               V('M_max', 'N*m', 'maximum bending moment (at mid-span)'),
               V('R_A', 'N', 'left support reaction', signed=True),
               V('R_B', 'N', 'right support reaction', signed=True)],
         equations=['delta = P*L**3/(48*E*I)', 'theta_A = P*L**2/(16*E*I)', 'M_max = P*L/4',
                    'R_A = P/2', 'R_B = P/2'],
         outputs=['delta', 'theta_A', 'M_max', 'R_A', 'R_B'],
         example={'P': 10000.0, 'L': 4.0, 'E': 200e9, 'I': 8e-6},
         refs=['Simply supported central-load deflection (standard beam table; recalled)']),

    Calc('ss_udl', 'Simply supported beam with a uniform distributed load',
         vars=[V('w', 'N/m', 'load intensity (downward +)'), V('L', 'm', 'span'),
               V('E', 'Pa', "Young's modulus"), V('I', 'm^4', 'second moment of area'),
               V('delta', 'm', 'mid-span deflection (downward +)'),
               V('M_max', 'N*m', 'maximum bending moment (at mid-span)'),
               V('R_A', 'N', 'left support reaction', signed=True),
               V('R_B', 'N', 'right support reaction', signed=True)],
         equations=['delta = 5*w*L**4/(384*E*I)', 'M_max = w*L**2/8', 'R_A = w*L/2', 'R_B = w*L/2'],
         outputs=['delta', 'M_max', 'R_A', 'R_B'],
         example={'w': 2000.0, 'L': 4.0, 'E': 200e9, 'I': 8e-6},
         refs=['Simply supported UDL deflection (standard beam table; recalled)']),

    Calc('ss_point_offcenter', 'Simply supported beam with an off-centre point load',
         vars=[V('P', 'N', 'point load (downward +)'), V('a', 'm', 'load distance from the left support'),
               V('b', 'm', 'load distance from the right support'), V('L', 'm', 'span (L = a + b)'),
               V('E', 'Pa', "Young's modulus"), V('I', 'm^4', 'second moment of area'),
               V('R_A', 'N', 'left support reaction', signed=True),
               V('R_B', 'N', 'right support reaction', signed=True),
               V('M_max', 'N*m', 'maximum bending moment (at the load)'),
               V('delta_load', 'm', 'deflection under the load (downward +)')],
         procedure=_ss_point_offcenter,
         outputs=['R_A', 'R_B', 'M_max', 'delta_load'],
         example={'P': 8000.0, 'a': 1.5, 'b': 2.5, 'L': 4.0, 'E': 200e9, 'I': 8e-6},
         assumptions=['The span L is checked against a + b (a mismatch raises MechError).'],
         refs=['Simply supported off-centre load: R_A = P·b/L, R_B = P·a/L, M_max = P·a·b/L, '
               'δ_load = P·a²·b²/(3·E·I·L) (standard beam table; recalled)']),

    Calc('beam_stiffness', 'Equivalent spring stiffness of a beam (k = 3EI/L³, 48EI/L³)',
         vars=[V('E', 'Pa', "Young's modulus"), V('I', 'm^4', 'second moment of area'), V('L', 'm', 'span'),
               V('k_cant', 'N/m', 'stiffness of a cantilever with an end load (3EI/L³)'),
               V('k_ss', 'N/m', 'stiffness of a simply supported central load (48EI/L³)')],
         equations=['k_cant = 3*E*I/L**3', 'k_ss = 48*E*I/L**3'],
         outputs=['k_cant', 'k_ss'],
         example={'E': 200e9, 'I': 8e-6, 'L': 2.0},
         assumptions=['Stiffness k = P/δ for the stated load case; spring-equivalent only near that '
                      'deformation mode.'],
         refs=['Beam stiffness k = P/δ (recalled)']),

    Calc('moment_area', 'Cantilever end load by the moment-area method',
         vars=[V('P', 'N', 'end load (downward +)'), V('L', 'm', 'length'),
               V('E', 'Pa', "Young's modulus"), V('I', 'm^4', 'second moment of area'),
               V('delta', 'm', 'tip deflection'), V('theta', 'rad', 'tip slope', label='θ')],
         equations=['delta = P*L**3/(3*E*I)', 'theta = P*L**2/(2*E*I)'],
         outputs=['delta', 'theta'],
         example={'P': 1000.0, 'L': 2.0, 'E': 200e9, 'I': 8e-6},
         refs=['Moment-area theorems applied to a cantilever end load (agrees with the general solver '
               'and double integration; recalled)']),

    Calc('castigliano_energy', 'Cantilever end load by Castigliano’s theorem',
         vars=[V('P', 'N', 'end load (downward +)'), V('L', 'm', 'length'),
               V('E', 'Pa', "Young's modulus"), V('I', 'm^4', 'second moment of area'),
               V('delta', 'm', 'tip deflection')],
         equations=['delta = P*L**3/(3*E*I)'],
         outputs=['delta'],
         example={'P': 1000.0, 'L': 2.0, 'E': 200e9, 'I': 8e-6},
         refs=['Castigliano’s theorem: δ = ∂U/∂P with U = ∫M²/(2EI) dx (agrees with the general '
               'solver; recalled)']),

    Calc('unit_load', 'Simply supported central load by the unit-load method',
         vars=[V('P', 'N', 'central load (downward +)'), V('L', 'm', 'span'),
               V('E', 'Pa', "Young's modulus"), V('I', 'm^4', 'second moment of area'),
               V('delta', 'm', 'centre deflection')],
         equations=['delta = P*L**3/(48*E*I)'],
         outputs=['delta'],
         example={'P': 10000.0, 'L': 4.0, 'E': 200e9, 'I': 8e-6},
         refs=['Unit-load method δ = ∫m·M/(EI) dx for a central load (agrees with the general solver; '
               'recalled)']),

    Calc('deflection_superposition', 'Superpose two deflection contributions',
         vars=[V('delta1', 'm', 'deflection contribution 1', signed=True),
               V('delta2', 'm', 'deflection contribution 2', signed=True),
               V('delta', 'm', 'total deflection', signed=True)],
         equations=['delta = delta1 + delta2'],
         outputs=['delta'],
         example={'delta1': 0.004, 'delta2': 0.001},
         assumptions=['Valid for linear-elastic (small-deflection) beams where the contributions are '
                      'computed independently; add like-signed components.'],
         refs=['Principle of superposition for linear beams (recalled)']),

    Calc('general_beam', 'General prismatic beam: reactions, SFD/BMD, slope and deflection',
         vars=[V('E', 'Pa', "Young's modulus"), V('I', 'm^4', 'second moment of area'),
               V('L', 'm', 'beam length'),
               V('loads', '', 'rows ["P", P, x] point load (down), ["M", M, x] point moment (CCW +), '
                             '["w", w, a, b] uniform load (down), ["lin", wa, wb, a, b] linear load '
                             '(down); [] for none', kind='json'),
               V('supports', '', 'rows [x, "simple" or "fixed"]', kind='json'),
               V('x', '', 'sample points (m)', kind='json'),
               V('V', '', 'shear force (N), up + on the left face', kind='json'),
               V('M', '', 'bending moment (N·m), sagging +', kind='json'),
               V('theta', '', 'slope (rad)', kind='json'),
               V('delta', '', 'deflection (m), downward +', kind='json'),
               V('reactions', '', '[[x, R], ...] vertical reactions (N), up +', kind='json'),
               V('M_fixed', '', '[M, ...] moment reactions at fixed supports (N·m), CCW +', kind='json'),
               V('delta_max', 'm', 'maximum absolute deflection'),
               V('x_delta_max', 'm', 'location of maximum deflection'),
               V('M_max', 'N*m', 'maximum absolute bending moment'),
               V('x_M_max', 'm', 'location of maximum bending moment'),
               V('V_max', 'N', 'maximum absolute shear force')],
         procedure=_general_beam,
         outputs=['x', 'V', 'M', 'theta', 'delta', 'reactions', 'M_fixed', 'delta_max', 'x_delta_max',
                  'M_max', 'x_M_max', 'V_max'],
         example={'E': 200e9, 'I': 8e-6, 'L': 4.0, 'loads': [['P', 10000, 2]],
                  'supports': [[0, 'simple'], [4, 'simple']]},
         plots=[{'id': 'sfd_bmd', 'title': 'Shear and bending-moment diagrams', 'fn': _sfd_bmd_chart}],
         keywords=['indeterminate', 'propped', 'fixed-fixed', 'continuous', 'macaulay', 'singularity'],
         assumptions=['Euler–Bernoulli, small deflection, prismatic, linear elastic.',
                      'Indeterminate supports (propped, fixed-fixed, continuous) solved by '
                      'superposition of compatibility conditions in one linear system.',
                      'M_max/x_M_max are the first location achieving the maximum magnitude; on a '
                      'tie (e.g. fixed-fixed) the left-most is reported.'],
         refs=['Singularity (Macaulay) functions; double integration; superposition/compatibility '
               '(agrees with the closed-form table cases; recalled)']),
]

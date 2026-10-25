# DESCRIPTION: Shafts and connections: DE (distortion-energy) shaft diameter under combined bending and
#              torsion, stepped-shaft deflection/slope by numerical integration of M/EI, stepped-shaft
#              critical speed (Rayleigh), key shear/crushing and length, spline torque, flange-coupling
#              bolt stresses, and set-screw holding force.
"""Shafts and connections calculators.

Sign conventions: shaft axis is x; downward point loads are positive and produce downward (negative)
deflection y for a simply supported shaft (sagging positive moment). Reactions are positive upward.
The DE diameter forms follow the distortion-energy shaft design; Kf and Kfs are the user's fatigue
stress-concentration factors. The stepped-shaft deflection/slope solution here supplies the static
deflections that feed the Rayleigh/Dunkerley critical-speed relations, which live in the vibrations
section (this module's shaft_critical_speed is a convenience that applies vibrations' Rayleigh method
to those deflections).

Relations are standard shaft- and key-design forms (Shigley, *Mechanical Engineering Design*; Budynas &
Nisbett). Cited by relation name, recalled rather than re-read; no page or equation numbers are given.
"""
import math

import numpy as np

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('shafts', 'Shafts and connections', order=110,
                  summary='DE shaft sizing, stepped-shaft deflection/slope and critical speed, keys, '
                          'splines, flange couplings and set screws.')

PI = math.pi
G0 = 9.80665  # standard gravity m/s^2 (exact by definition)


# ---- shared stepped-shaft beam solution ----------------------------------------------------------
def _beam_solution(case, segments, loads, supports, E):
    """Static deflection/slope of a stepped round shaft by exact numerical integration of M/(EI).

    case      'simply_supported' (two supports at a, b) or 'cantilever' (fixed at x = 0)
    segments  [[length, diameter], ...] in m, in order from x = 0
    loads     [[x, F], ...] point loads, F positive downward
    supports  [a, b] for simply supported (positions in m); pass [] for a cantilever (not used)
    E         Young's modulus (Pa)

    The integration grid carries a node at every segment boundary, load position and support, so the
    trapezoid rule on M/EI (piecewise linear with those nodes) is exact — no step-size error from an
    EI jump or an off-node load. Returns {'x','y','slope','reactions','L'}: y and slope of the
    upward-positive y axis on a fine uniform grid (the kink nodes included), reactions [[x, R], ...]
    (upward positive), and total length. Sagging-positive moment gives downward (negative) y.
    """
    if case == 'cantilever':
        a = b = None
    elif case == 'simply_supported':
        if not isinstance(supports, (list, tuple)) or len(supports) != 2:
            raise MechError("shaft deflection: 'simply_supported' needs supports = [a, b]")
        a, b = float(supports[0]), float(supports[1])
        if not (0 <= a < b):
            raise MechError('shaft deflection: supports must satisfy 0 <= a < b')
    else:
        raise MechError("shaft deflection: case must be 'simply_supported' or 'cantilever'")
    if not isinstance(segments, list) or not segments:
        raise MechError('shaft deflection: segments = [[length, diameter], ...] in m')
    lengths = []
    for row in segments:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise MechError('shaft deflection: each segment must be [length_m, diameter_m]')
        try:
            Ls, ds = float(row[0]), float(row[1])
        except (TypeError, ValueError):
            raise MechError('shaft deflection: segment length and diameter must be numbers')
        if Ls <= 0 or ds <= 0:
            raise MechError('shaft deflection: segment length and diameter must be positive')
        lengths.append((Ls, ds))
    L = sum(Ls for Ls, _ in lengths)
    if not isinstance(loads, (list, tuple)):
        raise MechError('shaft deflection: loads = [[x_m, F_N], ...]')
    load_pts = []
    for row in loads:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise MechError('shaft deflection: each load must be [x_m, F_N]')
        try:
            x, F = float(row[0]), float(row[1])
        except (TypeError, ValueError):
            raise MechError('shaft deflection: load position and force must be numbers')
        if not (0.0 <= x <= L):
            raise MechError('shaft deflection: every load position must be within [0, L]')
        load_pts.append((x, F))
    if case == 'simply_supported' and not (a <= L and b <= L):
        raise MechError('shaft deflection: supports must lie within [0, L]')

    # Reactions (upward positive).
    if case == 'simply_supported':
        Ftot = sum(F for _, F in load_pts)
        sumFx = sum(F * x for x, F in load_pts)
        Rb = (sumFx - Ftot * a) / (b - a)
        Ra = Ftot - Rb
        reactions = [[a, Ra], [b, Rb]]
    else:
        reactions = [[0.0, sum(F for _, F in load_pts)]]

    # Segment boundary positions and per-segment EI.
    x0 = [0.0]
    for Ls, _ in lengths:
        x0.append(x0[-1] + Ls)
    EIseg = [E * PI * ds ** 4 / 64.0 for _, ds in lengths]

    # Integration grid: every kink (segment boundary, load, support) plus a fine uniform fill.
    N = 8000
    nodes = {0.0, L}
    nodes.update(x0)
    for x, _ in load_pts:
        nodes.add(x)
    if case == 'simply_supported':
        nodes.add(a)
        nodes.add(b)
    xg = np.array(sorted(nodes | {L * i / N for i in range(N + 1)}))

    # Bending moment M(x): sagging positive.
    M = np.zeros_like(xg)
    for x, F in load_pts:
        if case == 'cantilever':
            M -= F * np.maximum(x - xg, 0.0)   # moment of loads to the right (hogging negative)
        else:
            M -= F * np.maximum(xg - x, 0.0)   # load contribution
    if case == 'simply_supported':
        M += Ra * np.maximum(xg - a, 0.0) + Rb * np.maximum(xg - b, 0.0)

    # EI per interval (constant within an interval; midpoints pick the segment unambiguously), so the
    # M/EI jump at a diameter step is attributed to the correct side.
    xm = 0.5 * (xg[:-1] + xg[1:])
    seg = np.clip(np.searchsorted(x0, xm, side='right') - 1, 0, len(lengths) - 1)
    EIi = np.array(EIseg)[seg]
    fl = M[:-1] / EIi
    fr = M[1:] / EIi
    dx = np.diff(xg)

    # Exact integration of piecewise-linear M/EI: d(slope) = (f0 + f1)/2 dx ;
    # d(y) = slope0 dx + dx^2 (f0/3 + f1/6).
    slope_part = np.concatenate(([0.0], np.cumsum(0.5 * (fl + fr) * dx)))
    y_part = np.concatenate(([0.0], np.cumsum(slope_part[:-1] * dx + dx * dx * (fl / 3.0 + fr / 6.0))))

    if case == 'cantilever':
        y = y_part
        slope = slope_part
    else:
        # Linear correction c0 + c1·x to satisfy y(a) = y(b) = 0.
        ya = float(np.interp(a, xg, y_part))
        yb = float(np.interp(b, xg, y_part))
        c1 = (ya - yb) / (b - a)
        c0 = -ya - c1 * a
        y = y_part + c0 + c1 * xg
        slope = slope_part + c1

    return {'x': xg, 'y': y, 'slope': slope, 'reactions': reactions, 'L': L}


def _eval_points(case, loads, supports):
    """Sorted x positions at which deflection/slope are reported (loads + supports)."""
    pts = [float(x) for x, _ in loads]
    if case == 'simply_supported':
        pts += [float(s) for s in supports]
    else:
        pts.append(0.0)
    return sorted(set(pts))


def _shaft_deflection(values):
    """Deflection and slope at loads/supports plus maxima, for a stepped shaft."""
    case = values['case']
    loads = values['loads']
    supports = values.get('supports') or [0.0, 0.0]
    sol = _beam_solution(case, values['segments'], loads, supports, values['E'])
    xg, y, slope = sol['x'], sol['y'], sol['slope']
    pts = _eval_points(case, loads, supports)
    defls = [[x, float(np.interp(x, xg, y))] for x in pts]
    slps = [[x, float(np.interp(x, xg, slope))] for x in pts]
    iy = int(np.argmax(np.abs(y)))
    isl = int(np.argmax(np.abs(slope)))
    return {'deflections': defls, 'slopes': slps, 'reactions': sol['reactions'],
            'max_deflection': abs(float(y[iy])), 'max_deflection_x': float(xg[iy]),
            'max_slope': abs(float(slope[isl])), 'max_slope_x': float(xg[isl])}


def _shaft_critical_speed(values):
    """First bending critical speed of a stepped shaft by Rayleigh's method (static deflections under
    the mass weights). ω = sqrt(g·Σ m_i·δ_i / Σ m_i·δ_i²) with δ_i the deflections at the masses."""
    masses = values['masses']
    if not isinstance(masses, list) or not masses:
        raise MechError('critical_speed: masses = [[x_m, m_kg], ...]')
    weights = []
    for row in masses:
        if len(row) != 2:
            raise MechError('critical_speed: each row must be [x_m, m_kg]')
        weights.append([float(row[0]), float(row[1]) * G0])
    case = values['case']
    supports = values.get('supports') or [0.0, 0.0]
    sol = _beam_solution(case, values['segments'], weights, supports, values['E'])
    num = den = 0.0
    defls = []
    for row in masses:
        x, m = float(row[0]), float(row[1])
        y = float(np.interp(x, sol['x'], sol['y']))
        d_i = -y                      # downward-positive static deflection under the weights
        if d_i <= 0:
            raise MechError('critical_speed: a mass deflects upward; Rayleigh (weight-based) needs '
                            'downward deflections — check supports and mass positions')
        defls.append([x, y])
        num += m * d_i
        den += m * d_i * d_i
    if num <= 0 or den <= 0:
        raise MechError('critical_speed: Rayleigh numerator/denominator must be positive; check '
                        'supports and mass positions')
    return {'omega_cr': math.sqrt(G0 * num / den), 'deflections': defls}


CALCS = [
    Calc('shaft_diameter_de', 'Shaft diameter, combined bending and torsion (DE-Goodman/Gerber/ASME)',
         vars=[V('Ma', 'N*m', 'alternating bending moment (magnitude)'),
               V('Mm', 'N*m', 'midrange bending moment (magnitude)'),
               V('Ta', 'N*m', 'alternating torque (magnitude)'),
               V('Tm', 'N*m', 'midrange torque (magnitude)'),
               V('Kf', '', 'fatigue stress-concentration factor, bending', lo=1.0),
               V('Kfs', '', 'fatigue stress-concentration factor, torsion', lo=1.0),
               V('Se', 'Pa', 'fully corrected endurance limit'),
               V('Sut', 'Pa', 'ultimate tensile strength'),
               V('Sy', 'Pa', 'yield strength'),
               V('n', '', 'design factor of safety'),
               V('A', 'N*m', 'alternating von Mises numerator sqrt(4(Kf·Ma)² + 3(Kfs·Ta)²)'),
               V('B', 'N*m', 'midrange von Mises numerator sqrt(4(Kf·Mm)² + 3(Kfs·Tm)²)'),
               V('d_goodman', 'm', 'diameter from DE-Goodman'),
               V('d_gerber', 'm', 'diameter from DE-Gerber'),
               V('d_elliptic', 'm', 'diameter from DE-ASME elliptic')],
         equations=['A = sqrt(4*(Kf*Ma)**2 + 3*(Kfs*Ta)**2)',
                    'B = sqrt(4*(Kf*Mm)**2 + 3*(Kfs*Tm)**2)',
                    'd_goodman = (16*n/pi*(A/Se + B/Sut))**(1/3)',
                    'd_gerber = (8*n/(pi*Se)*(A + sqrt(A**2 + (2*B*Se/Sut)**2)))**(1/3)',
                    'd_elliptic = (16*n/pi*sqrt(A**2/Se**2 + B**2/Sy**2))**(1/3)'],
         outputs=['A', 'B', 'd_goodman', 'd_gerber', 'd_elliptic'],
         example={'Ma': 200.0, 'Mm': 100.0, 'Ta': 150.0, 'Tm': 100.0, 'Kf': 1.5, 'Kfs': 1.5,
                  'Se': 180e6, 'Sut': 600e6, 'Sy': 400e6, 'n': 2.0},
         keywords=['distortion energy', 'de-goodman', 'de-gerber', 'de-asme', 'fatigue', 'diameter'],
         assumptions=['Solid round shaft, fully reversed or steady loading as entered (magnitudes only). '
                      'Kf, Kfs are the fatigue notch factors from the geometry (user inputs, see the '
                      'fatigue section). For a rotating shaft with steady torque and reversed bending, '
                      'Ta = Tm = Mm = 0. The Gerber diameter is written without a 1/A factor so a '
                      'steady-only load (A = 0) gives the limit d = (16 n B/(π Sut))^(1/3) instead of '
                      'dividing by zero. d_elliptic needs Sy; d_goodman/d_gerber need Sut.'],
         refs=['Distortion-energy shaft sizing: DE-Goodman, DE-Gerber and DE-ASME-elliptic diameters '
               '(recalled)']),

    Calc('shaft_deflection_slope', 'Stepped-shaft deflection and slope (M/EI integration)',
         vars=[V('case', '', 'support condition', kind='text', choices=['simply_supported', 'cantilever']),
               V('segments', '', 'steps [[length (m), diameter (m)], ...] from x = 0', kind='json'),
               V('loads', '', 'point loads [[x (m), F (N, downward +)], ...]', kind='json'),
               V('supports', '', 'support positions [a, b] (m) — required only for simply_supported; '
                 'for a cantilever pass [] (the value is not used)', kind='json'),
               V('E', 'Pa', "Young's modulus"),
               V('reactions', '', 'support reactions [[x (m), R (N, upward +)], ...]', kind='json'),
               V('deflections', '', 'deflection [[x (m), y (m)], ...] at each load and support', kind='json'),
               V('slopes', '', 'slope [[x (m), dy/dx (rad)], ...] at each load and support', kind='json'),
               V('max_deflection', 'm', 'largest deflection magnitude'),
               V('max_deflection_x', 'm', 'position of the largest deflection'),
               V('max_slope', 'rad', 'largest slope magnitude'),
               V('max_slope_x', 'm', 'position of the largest slope')],
         procedure=_shaft_deflection,
         outputs=['reactions', 'deflections', 'slopes', 'max_deflection', 'max_deflection_x',
                  'max_slope', 'max_slope_x'],
         example={'case': 'simply_supported', 'segments': [[1.0, 0.04]], 'loads': [[0.5, 1000.0]],
                  'supports': [0.0, 1.0], 'E': 200e9},
         keywords=['deflection', 'slope', 'stepped shaft', 'macaulay', 'moment-area', 'beam'],
         assumptions=['Euler–Bernoulli beam theory, linear elastic, small deflections; point loads only. '
                      'Each step is a round section I = πd⁴/64. Integration nodes are placed at every '
                      'diameter step and load position, so M/(EI) is integrated exactly. Downward loads '
                      'give downward (negative) y; maxima are magnitudes. Bearings at the supports, gears '
                      'at the load positions. supports is required only for simply_supported (ignored for '
                      'a cantilever). This calculator supplies only the static deflections/slopes; the '
                      'Rayleigh/Dunkerley critical-speed relations live in the vibrations section.'],
         refs=['Double integration of M/(EI) with support boundary conditions']),

    Calc('shaft_critical_speed', 'Stepped-shaft first critical speed (Rayleigh)',
         vars=[V('case', '', 'support condition', kind='text', choices=['simply_supported', 'cantilever']),
               V('segments', '', 'steps [[length (m), diameter (m)], ...] from x = 0', kind='json'),
               V('masses', '', 'lumped masses [[x (m), m (kg)], ...]', kind='json'),
               V('supports', '', 'support positions [a, b] (m) — required only for simply_supported; '
                 'for a cantilever pass [] (the value is not used)', kind='json'),
               V('E', 'Pa', "Young's modulus"),
               V('omega_cr', 'rad/s', 'first bending critical speed', label='ωcr'),
               V('deflections', '', 'static deflection [[x (m), y (m)], ...] at each mass', kind='json')],
         procedure=_shaft_critical_speed, outputs=['omega_cr', 'deflections'],
         example={'case': 'simply_supported', 'segments': [[1.0, 0.04]], 'masses': [[0.5, 20.0]],
                  'supports': [0.0, 1.0], 'E': 200e9},
         keywords=['critical speed', 'rayleigh', 'whirl', 'dunkerley', 'natural frequency', 'rotor'],
         assumptions=['Rayleigh method applied to the static deflections computed by this module: '
                      'ω = sqrt(g·Σ m_i δ_i / Σ m_i δ_i²) with δ_i the deflections under the weights m_i·g. '
                      'Gives the first bending mode; upper bound on the exact frequency. Add '
                      'gyroscopic/bearing stiffness for an accurate rotor. The Rayleigh/Dunkerley relations '
                      'themselves are owned by the vibrations section; this calculator is the convenience '
                      'that feeds them the stepped-shaft deflections.'],
         refs=['Rayleigh method for the fundamental frequency of a shaft (recalled)']),

    Calc('key_shear_crush', 'Key shear and crushing (bearing) stress',
         vars=[V('tau', 'Pa', 'key shear stress', label='τ'),
               V('sigma_c', 'Pa', 'key crushing (bearing) stress', label='σc'),
               V('T', 'N*m', 'transmitted torque'),
               V('d', 'm', 'shaft diameter'), V('w', 'm', 'key width'), V('h', 'm', 'key height'),
               V('L', 'm', 'key length')],
         equations=['tau = 2*T/(d*w*L)', 'sigma_c = 4*T/(d*h*L)'],
         outputs=['tau', 'sigma_c'],
         example={'T': 200.0, 'd': 0.04, 'w': 0.012, 'h': 0.008, 'L': 0.05},
         keywords=['keyway', 'parallel key', 'shear', 'crushing', 'bearing'],
         assumptions=['Square parallel key; shear on the full width w over length L; crushing on half the '
                      'key height (h/2) against the shaft and hub. Force at the shaft surface F = 2T/d '
                      '(nominal radius; keyway depth neglected).'],
         refs=['Key shear τ = 2T/(d·w·L); crushing σc = 4T/(d·h·L) (standard forms, recalled)']),

    Calc('key_length', 'Key length required for shear and crushing',
         vars=[V('L_shear', 'm', 'length required to limit shear stress'),
               V('L_crush', 'm', 'length required to limit crushing stress'),
               V('T', 'N*m', 'transmitted torque'), V('d', 'm', 'shaft diameter'),
               V('w', 'm', 'key width'), V('h', 'm', 'key height'),
               V('tau_allow', 'Pa', 'allowable shear stress'),
               V('sigma_allow', 'Pa', 'allowable crushing (bearing) stress')],
         equations=['L_shear = 2*T/(d*w*tau_allow)', 'L_crush = 4*T/(d*h*sigma_allow)'],
         outputs=['L_shear', 'L_crush'],
         example={'T': 200.0, 'd': 0.04, 'w': 0.012, 'h': 0.008, 'tau_allow': 40e6,
                  'sigma_allow': 100e6},
         keywords=['keyway', 'key length', 'parallel key'],
         assumptions=['Design key length = max(L_shear, L_crush). Square parallel key, same areas as '
                      'key_shear_crush; allowables are the user\'s material/app factors.'],
         refs=['Key length from shear and crushing stress limits (recalled)']),

    Calc('spline_torque', 'Spline torque capacity (bearing-pressure model)',
         vars=[V('T', 'N*m', 'torque capacity'), V('p', 'Pa', 'allowable bearing pressure'),
               V('n', '', 'number of teeth', integer=True), V('L', 'm', 'engagement length'),
               V('D', 'm', 'major (outside) diameter'), V('d', 'm', 'minor (root) diameter'),
               V('h', 'm', 'tooth contact height (D−d)/2'),
               V('r_m', 'm', 'mean radius (D+d)/4')],
         equations=['T = p*n*L*h*r_m', 'h = (D - d)/2', 'r_m = (D + d)/4'],
         outputs=['T', 'h', 'r_m'],
         example={'p': 20e6, 'n': 6, 'L': 0.03, 'D': 0.05, 'd': 0.044},
         keywords=['spline', 'involute', 'straight-sided', 'torque capacity'],
         assumptions=['Approximate straight-sided/mean-radius model: torque = bearing pressure × total '
                      'contact area (n·L·h) × mean radius. Real involute splines derate for unequal tooth '
                      'load sharing (ANSI/AGMA); this is a first sizing estimate.'],
         refs=['Spline torque capacity T = p·A·r_m (standard approximation, recalled)']),

    Calc('flange_coupling', 'Flange coupling: bolt shear stress and torque capacity',
         vars=[V('T', 'N*m', 'coupling torque capacity'), V('F_b', 'N', 'shear force per bolt'),
               V('A_b', 'm^2', 'bolt cross-sectional area'),
               V('n_b', '', 'number of bolts', integer=True), V('R', 'm', 'bolt pitch-circle radius'),
               V('tau_b', 'Pa', 'allowable bolt shear stress'), V('d_b', 'm', 'bolt diameter')],
         equations=['T = n_b*F_b*R', 'F_b = tau_b*A_b', 'A_b = pi*d_b**2/4'],
         outputs=['T', 'F_b', 'A_b'],
         example={'n_b': 6, 'R': 0.06, 'tau_b': 60e6, 'd_b': 0.012},
         keywords=['rigid flange coupling', 'bolt shear', 'coupling'],
         assumptions=['Rigid flange coupling with the bolts in shear on the pitch circle; torque = n_b·F_b·R. '
                      'Preloaded friction couplings transmit torque by face friction instead (see bolted '
                      'section). Single shear; use the shear area accordingly for double-shear designs.'],
         refs=['Bolt-circle torque T = n·F·R']),

    Calc('set_screw', 'Set screw holding force and torque (friction model)',
         vars=[V('N', 'N', 'clamping (normal) force at the cup point'),
               V('F_hold', 'N', 'holding force against sliding'),
               V('T_hold', 'N*m', 'holding torque against rotation on the shaft'),
               V('T_set', 'N*m', 'set-screw tightening torque'), V('K', '', 'nut factor (≈0.2)'),
               V('d_set', 'm', 'set-screw nominal diameter'), V('mu', '', 'friction coefficient at the shaft'),
               V('d_shaft', 'm', 'shaft diameter')],
         equations=['N = T_set/(K*d_set)', 'F_hold = mu*N', 'T_hold = F_hold*d_shaft/2'],
         outputs=['N', 'F_hold', 'T_hold'],
         example={'T_set': 2.0, 'K': 0.2, 'd_set': 0.006, 'mu': 0.3, 'd_shaft': 0.02},
         keywords=['set screw', 'grub screw', 'cup point', 'holding power'],
         assumptions=['Friction model: the tightening torque develops a clamp N = T/(K·d) (nut factor K is a '
                      'user input), and the holding force/torque follow from the cup-point friction μ. '
                      'Manufacturer tabulated holding powers are more accurate for a specific screw/shaft; '
                      'no tabulated values are built in.'],
         refs=['Screw-clamp and friction holding model (recalled)']),
]

# DESCRIPTION: Bolted and riveted joints: tightening torque, tensile stress area, bolt/member stiffness
#              (frustum and Wileman), joint constant and external-load sharing, bolt fatigue, thread
#              stripping, bolt-group shear with eccentric loading, and rivet shear/bearing/tearing
#              efficiency.
"""Bolted and riveted joint calculators.

Conventions: preload F_i is the axial clamp force; a positive external load P adds to the bolt and
unloads the member. The joint constant is C = k_b/(k_b + k_m). Proof/yield/ultimate strengths are user
inputs (no material tables are embedded). Tensile stress area uses the ISO metric approximation
At = π/4 (d − 0.9382 p)². Member stiffness offers both the Shigley 30° frustum closed form and the
Wileman exponential (A, B are table inputs).

Relations are the standard bolted-joint forms (Shigley, *Mechanical Engineering Design*; Bickford,
*An Introduction to the Design and Behavior of Bolted Joints*). Cited by relation name, recalled rather
than re-read; no page or equation numbers are given.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('bolted', 'Bolted and riveted joints', order=120,
                  summary='Preload and torque, bolt/member stiffness, load sharing, bolt fatigue, thread '
                          'stripping, eccentric bolt groups, and rivet joints.')

PI = math.pi


# ---- procedures ---------------------------------------------------------------------------------
def _frustum(values):
    """Member stiffness of a joint with two equal clamped members (equal-member two-frustum form).

    One frustum grows from the head and one from the nut, each of length L/2, in series (hence the
    factor 2 in the denominator):

    k_m = π E d tanα / (2 ln[ ((L tanα + D − d)(D + d)) / ((L tanα + D + d)(D − d)) ])

    with D the washer/head face diameter, α the cone half-apex angle (≈30°), d the bolt diameter and
    L the total grip length.
    """
    E, d, D, alpha, L = values['E'], values['d'], values['D'], values['alpha'], values['L']
    if D <= d:
        raise MechError('member_stiffness_frustum: washer face diameter D must exceed the bolt diameter d')
    t = math.tan(alpha)
    if L * t + D - d <= 0:
        raise MechError('member_stiffness_frustum: L·tan(alpha) + D − d must be positive')
    arg = ((L * t + D - d) * (D + d)) / ((L * t + D + d) * (D - d))
    den = 2.0 * math.log(arg)
    if den <= 0:
        raise MechError('member_stiffness_frustum: frustum geometry gives a non-positive stiffness')
    return {'k_m': PI * E * d * t / den}


def _bolt_fatigue(values):
    """Bolt fatigue by the constant-preload Goodman line, plus the separation factor of safety.

    The bolt stress oscillates between σi = F_i/A_t and σi + C·Pmax/A_t, so the alternating stress is
    σa = C(Pmax − Pmin)/(2 A_t) and the midrange stress is σm = C(Pmax + Pmin)/(2 A_t) + σi. With the
    Goodman line drawn from the preload (the preload is always present), the factor of safety is
    n_f = Se(Sut − σi) / (Sut σa + Se(σm − σi)).
    """
    Fi, C, Pmax, Pmin, At, Se, Sut = (values[k] for k in ('F_i', 'C', 'Pmax', 'Pmin', 'A_t', 'Se', 'Sut'))
    if Pmax < Pmin:
        raise MechError('bolt_fatigue: Pmax must be >= Pmin')
    if not (0.0 < C < 1.0):
        raise MechError('bolt_fatigue: the joint constant C must lie between 0 and 1')
    if Pmax <= 0:
        raise MechError('bolt_fatigue: Pmax must be positive')
    if At <= 0 or Se <= 0 or Sut <= 0:
        raise MechError('bolt_fatigue: A_t, Se and Sut must be positive')
    sigma_i = Fi / At
    sigma_a = C * (Pmax - Pmin) / (2.0 * At)
    sigma_m = C * (Pmax + Pmin) / (2.0 * At) + sigma_i
    n_sep = Fi / ((1.0 - C) * Pmax)   # separation when F_m = F_i − (1−C)·Pmax reaches zero
    out = {'sigma_a': sigma_a, 'sigma_m': sigma_m, 'n_sep': n_sep}
    if Pmax == Pmin:
        # static (no alternating) load: the fatigue factor of safety is undefined
        out['n_fatigue'] = None
        out['_warnings'] = ['bolt_fatigue: Pmin == Pmax (static load, σa = 0) — the fatigue factor of '
                            'safety is undefined; use a static strength check instead']
    else:
        out['n_fatigue'] = Se * (Sut - sigma_i) / (Sut * sigma_a + Se * (sigma_m - sigma_i))
    return out


def _load_sharing(values):
    """External-load sharing, separation load and a separation warning for a preloaded joint.

    C = k_b/(k_b + k_m); the bolt carries F_i + C P and the member F_i − (1−C) P. When P exceeds
    P_sep = F_i/(1−C) the member load reaches zero — the joint separates and the bolt carries the
    full external load — which is surfaced as a warning.
    """
    kb, km, Fi, P = values['k_b'], values['k_m'], values['F_i'], values['P']
    if kb <= 0 or km <= 0:
        raise MechError('load_sharing: bolt and member stiffness must be positive')
    if Fi < 0 or P < 0:
        raise MechError('load_sharing: preload and external load must be non-negative')
    C = kb / (kb + km)
    F_b = Fi + C * P
    F_m = Fi - (1.0 - C) * P
    P_sep = Fi / (1.0 - C)
    out = {'F_b': F_b, 'F_m': F_m, 'C': C, 'P_sep': P_sep}
    if P > P_sep:
        out['_warnings'] = ['load_sharing: P exceeds P_sep — the joint has separated (F_m <= 0) and '
                            'the bolt carries the full external load P']
    return out


def _bolt_group(values):
    """Elastic (torsion) analysis of a bolt group under an eccentric shear load.

    Direct shear is shared equally; the moment M about the bolt centroid is resisted by shear
    proportional to each bolt's distance from the centroid (elastic method, no instantaneous-centre
    iteration). Returns per-bolt resultants and the largest one.
    """
    bolts = values['bolts']
    if not isinstance(bolts, list) or len(bolts) < 2:
        raise MechError('bolt_group_eccentric: need at least two bolt coordinates [[x, y], ...]')
    n = len(bolts)
    pts = []
    for row in bolts:
        if len(row) != 2:
            raise MechError('bolt_group_eccentric: each bolt must be [x_m, y_m]')
        pts.append((float(row[0]), float(row[1])))
    Fx, Fy = values['F_x'], values['F_y']
    xp, yp = values['x_p'], values['y_p']
    xc = sum(x for x, _ in pts) / n
    yc = sum(y for _, y in pts) / n
    M = (xp - xc) * Fy - (yp - yc) * Fx        # CCW positive, r × F
    sumr2 = sum((x - xc) ** 2 + (y - yc) ** 2 for x, y in pts)
    if sumr2 <= 0:
        raise MechError('bolt_group_eccentric: all bolts coincide with the centroid')
    dx, dy = Fx / n, Fy / n
    rows, max_shear = [], 0.0
    for x, y in pts:
        rx, ry = x - xc, y - yc
        ftx = -M * ry / sumr2
        fty = M * rx / sumr2
        bx, by = dx + ftx, dy + fty
        f = math.hypot(bx, by)
        max_shear = max(max_shear, f)
        rows.append([bx, by, f])
    return {'moment': M, 'direct_x': dx, 'direct_y': dy, 'max_shear': max_shear, 'bolt_forces': rows}


CALCS = [
    Calc('tightening_torque', 'Bolt preload from tightening torque (T = K·F·d)',
         vars=[V('T', 'N*m', 'tightening torque'), V('K', '', 'nut factor (0.2 typical dry steel, lubricated ~0.15)'),
               V('F_i', 'N', 'preload (clamping force)'), V('d', 'm', 'nominal bolt diameter')],
         equations=['T = K*F_i*d'], outputs=['T'],
         example={'K': 0.2, 'F_i': 20000.0, 'd': 0.012},
         keywords=['torque', 'preload', 'nut factor', 'tightening'],
         assumptions=['T = K·F_i·d relates torque to preload through the nut factor K (a user input that '
                      'depends on thread friction and finish); scatter of ±25 % is common. For a reliable '
                      'preload use torque-angle or direct tensioning.'],
         refs=['T = K F d (short-form torque–preload relation, recalled)']),

    Calc('tensile_stress_area', 'Bolt tensile stress area (ISO metric)',
         vars=[V('A_t', 'm^2', 'tensile stress area'),
               V('d', 'm', 'nominal (major) diameter'),
               V('p', 'm', 'thread pitch')],
         equations=['A_t = pi/4*(d - 0.9382*p)**2'], outputs=['A_t'],
         example={'d': 0.012, 'p': 0.00175},
         keywords=['tensile stress area', 'thread', 'metric', 'iso'],
         assumptions=['ISO metric (coarse and fine): At = π/4 (d − 0.9382 p)², the area of the mean of the '
                      'pitch and minor diameters. Unified (UN) threads use At = 0.7854 (d − 0.9743/n)² with d '
                      'in inches and n threads/in — not computed here.'],
         refs=['ISO metric tensile stress area At = π/4 (d − 0.9382 p)² (recalled)']),

    Calc('bolt_stiffness', 'Bolt stiffness (threaded + unthreaded portions)',
         vars=[V('k_b', 'N/m', 'bolt axial stiffness'),
               V('A_d', 'm^2', 'major-diameter (unthreaded) area'),
               V('d', 'm', 'nominal diameter'), V('A_t', 'm^2', 'tensile stress area'),
               V('E', 'Pa', "Young's modulus (bolt)"),
               V('L_t', 'm', 'threaded length in grip'), V('L_d', 'm', 'unthreaded length in grip')],
         equations=['A_d = pi*d**2/4', 'k_b = A_d*A_t*E/(A_d*L_t + A_t*L_d)'],
         outputs=['A_d', 'k_b'],
         example={'d': 0.012, 'A_t': 84.3e-6, 'E': 200e9, 'L_t': 0.03, 'L_d': 0.03},
         keywords=['bolt stiffness', 'spring', 'joint'],
         assumptions=['Two springs in series (threaded and unthreaded shank): 1/k_b = L_d/(A_d E) + '
                      'L_t/(A_t E). Head and nut flexibility are neglected.'],
         refs=['Bolt stiffness k_b = A_d A_t E / (A_d L_t + A_t L_d) (Shigley form, recalled)']),

    Calc('member_stiffness_frustum', 'Member stiffness (Shigley 30° cone frustum)',
         vars=[V('k_m', 'N/m', 'member (clamped material) stiffness'),
               V('E', 'Pa', "Young's modulus (member)"),
               V('d', 'm', 'bolt diameter'), V('D', 'm', 'washer/head face diameter (≈1.5d)'),
               V('alpha', 'rad', 'frustum half-apex angle (30° typical)', hi=PI / 2, label='α'),
               V('L', 'm', 'grip length')],
         procedure=_frustum, outputs=['k_m'],
         example={'E': 200e9, 'd': 0.012, 'D': 0.018, 'alpha': math.radians(30), 'L': 0.03},
         keywords=['member stiffness', 'frustum', 'cone', 'clamped', 'shigley'],
         assumptions=['Axisymmetric frustum, one from the head and one from the nut (two equal '
                      'clamped members, L/2 each, in series) with half-apex angle α (≈30° for steel). '
                      'Requires D > d. Valid while the frusta do not overlap the full grip.'],
         refs=['Shigley frustum member stiffness (closed form, recalled)']),

    Calc('member_stiffness_wileman', 'Member stiffness (Wileman exponential)',
         vars=[V('k_m', 'N/m', 'member stiffness'),
               V('E', 'Pa', "Young's modulus (member)"),
               V('d', 'm', 'bolt diameter'), V('L', 'm', 'grip length'),
               V('A', '', 'Wileman coefficient A (tabulated material/aspect input)'),
               V('B', '', 'Wileman coefficient B (tabulated material/aspect input)')],
         equations=['k_m = E*d*A*exp(B*d/L)'], outputs=['k_m'],
         example={'E': 200e9, 'd': 0.012, 'A': 0.787, 'B': 0.629, 'L': 0.03},
         keywords=['member stiffness', 'wileman', 'exponential', 'clamped'],
         assumptions=['Wileman–Choudury–Green empirical member stiffness k_m = E·d·A·exp(B·d/L); A and B '
                      'are tabulated (material and aspect ratio) user inputs.'],
         refs=['Wileman exponential member stiffness (recalled)']),

    Calc('joint_constant', 'Joint constant (C = kb/(kb + km))',
         vars=[V('C', '', 'joint constant (fraction of the external load carried by the bolt)'),
               V('k_b', 'N/m', 'bolt stiffness'), V('k_m', 'N/m', 'member stiffness')],
         equations=['C = k_b/(k_b + k_m)'], outputs=['C'],
         example={'k_b': 3.2e8, 'k_m': 2.4e9},
         keywords=['joint constant', 'stiffness ratio', 'load factor'],
         assumptions=['C is the fraction of an external tensile load that adds to the bolt; (1 − C) '
                      'unloads the member. Soft members (small k_m) give a large C.'],
         refs=['C = k_b/(k_b + k_m)']),

    Calc('preload', 'Preload from proof strength, and load/separation factors',
         vars=[V('F_i', 'N', 'preload'),
               V('k', '', 'preload as a fraction of proof load (≈0.75 reusable, ≈0.9 permanent)'),
               V('A_t', 'm^2', 'tensile stress area'),
               V('Sp', 'Pa', 'proof strength'),
               V('n_L', '', 'load factor (margin to proof strength under external load)'),
               V('n_sep', '', 'factor of safety against joint separation'),
               V('C', '', 'joint constant'),
               V('P', 'N', 'applied external (tensile) load')],
         equations=['F_i = k*A_t*Sp',
                    'n_L = (Sp*A_t - F_i)/(C*P)',
                    'n_sep = F_i/((1 - C)*P)'],
         outputs=['F_i', 'n_L', 'n_sep'],
         example={'k': 0.75, 'A_t': 84.3e-6, 'Sp': 600e6, 'C': 0.1176, 'P': 10000.0},
         keywords=['preload', 'proof strength', 'load factor', 'separation'],
         assumptions=['Preload as a fraction k of the proof load Sp·A_t (k ≈ 0.75 for reusable joints, '
                      '≈ 0.9 for permanent joints — guidance, not a built-in value). The load factor '
                      'n_L = (Sp A_t − F_i)/(C P) is the factor by which the external load could grow '
                      'before the bolt reaches proof strength; n_sep = F_i/((1−C) P) guards separation.'],
         refs=['Preload F_i = k A_t Sp and the load/separation factors (recalled)']),

    Calc('load_sharing', 'External-load sharing and separation load',
         vars=[V('C', '', 'joint constant'),
               V('F_b', 'N', 'bolt load under external load P'),
               V('F_m', 'N', 'member load under external load P', signed=True),
               V('P_sep', 'N', 'external load that just separates the joint'),
               V('k_b', 'N/m', 'bolt stiffness'), V('k_m', 'N/m', 'member stiffness'),
               V('F_i', 'N', 'preload'), V('P', 'N', 'applied external (tensile) load')],
         procedure=_load_sharing,
         outputs=['F_b', 'F_m', 'C', 'P_sep'],
         example={'k_b': 3.2e8, 'k_m': 2.4e9, 'F_i': 20000.0, 'P': 10000.0},
         keywords=['load sharing', 'separation', 'joint diagram', 'preload'],
         assumptions=['Linear elastic bolt and member. The joint separates (loses clamp) when the member '
                      'load reaches zero, at P = F_i/(1 − C); a warning is raised when P > P_sep (the '
                      'member load F_m goes negative and the bolt carries the full external load).'],
         refs=['Bolt/member load sharing: F_b = F_i + C P, F_m = F_i − (1−C) P']),

    Calc('bolt_fatigue', 'Bolt fatigue safety factor (Goodman) and separation',
         vars=[V('F_i', 'N', 'preload'),
               V('C', '', 'joint constant'),
               V('Pmax', 'N', 'maximum external load'), V('Pmin', 'N', 'minimum external load'),
               V('A_t', 'm^2', 'tensile stress area'),
               V('Se', 'Pa', 'bolt endurance limit (fully corrected)'),
               V('Sut', 'Pa', 'bolt ultimate tensile strength'),
               V('n_fatigue', '', 'Goodman factor of safety on the bolt'),
               V('n_sep', '', 'factor of safety against joint separation'),
               V('sigma_a', 'Pa', 'alternating bolt stress', label='σa'),
               V('sigma_m', 'Pa', 'midrange bolt stress', label='σm')],
         procedure=_bolt_fatigue, outputs=['n_fatigue', 'sigma_a', 'sigma_m', 'n_sep'],
         example={'F_i': 20000.0, 'C': 0.1176, 'Pmax': 10000.0, 'Pmin': 0.0, 'A_t': 84.3e-6,
                  'Se': 180e6, 'Sut': 600e6},
         keywords=['bolt fatigue', 'goodman', 'separation', 'preload', 'fluctuating load'],
         assumptions=['External load fluctuates between Pmin and Pmax; the bolt sees C·P of it. σi = '
                      'F_i/A_t; σa = C·(Pmax−Pmin)/(2 A_t); σm = C·(Pmax+Pmin)/(2 A_t) + σi. The constant-'
                      'preload Goodman form n_f = Se(Sut−σi)/(Sut σa + Se(σm−σi)) is used (the preload is '
                      'always present). Separation is guarded by n_sep = F_i/((1−C)·Pmax). Pmin = Pmax '
                      '(static) gives n_fatigue = null with a note.'],
         refs=['Bolt fatigue with constant preload and joint separation (recalled)']),

    Calc('thread_stripping', 'Thread stripping area and shear stress',
         vars=[V('A_strip', 'm^2', 'thread stripping (shear) area'),
               V('tau_strip', 'Pa', 'thread shear stress', label='τ'),
               V('F', 'N', 'axial load'),
               V('d_min', 'm', 'minor diameter (external thread root)'),
               V('L_e', 'm', 'length of engagement'),
               V('f', '', 'engaged-fraction of thread in shear (coarse user input, ≈0.5)')],
         equations=['A_strip = pi*d_min*L_e*f', 'tau_strip = F/A_strip'],
         outputs=['A_strip', 'tau_strip'],
         example={'F': 20000.0, 'd_min': 0.010, 'L_e': 0.012, 'f': 0.5},
         keywords=['thread stripping', 'shear area', 'engagement', 'nut height'],
         assumptions=['Simple cylinder model of the external-thread shear area: A = π·d_min·L_e·f with f '
                      'a coarse engaged-fraction user input (≈0.5). A refined treatment balances the '
                      'internal (nut) and external (bolt) thread shear areas; their diameter ratio '
                      'w_i/w_o ≈ 0.75–0.88 sets the equal-strength nut. Exact ASME B1.1 / ISO 898 '
                      'formulas give precise stripping areas.'],
         refs=['Thread shear area approximation (recalled)']),

    Calc('bolt_group_eccentric', 'Bolt group in shear with eccentric loading (elastic method)',
         vars=[V('bolts', '', 'bolt coordinates [[x (m), y (m)], ...]', kind='json'),
               V('F_x', 'N', 'applied load x-component', signed=True),
               V('F_y', 'N', 'applied load y-component', signed=True),
               V('x_p', 'm', 'x of the load application point', signed=True),
               V('y_p', 'm', 'y of the load application point', signed=True),
               V('moment', 'N*m', 'eccentric moment about the bolt centroid', signed=True),
               V('direct_x', 'N', 'direct shear per bolt, x', signed=True),
               V('direct_y', 'N', 'direct shear per bolt, y', signed=True),
               V('max_shear', 'N', 'largest bolt resultant shear'),
               V('bolt_forces', '', 'per-bolt [[Fx (N), Fy (N), F (N)], ...]', kind='json')],
         procedure=_bolt_group, outputs=['moment', 'direct_x', 'direct_y', 'max_shear', 'bolt_forces'],
         example={'bolts': [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]],
                  'F_x': 3000.0, 'F_y': 4000.0, 'x_p': 0.2, 'y_p': 0.15},
         keywords=['bolt group', 'eccentric load', 'torsion', 'elastic method', 'shear'],
         assumptions=['Elastic (torsion) method: equal direct shear F/n plus torsional shear proportional '
                      'to distance from the centroid (no instantaneous-centre plastic iteration). The '
                      'eccentric moment is M = (x_p−x̄)F_y − (y_p−ȳ)F_x.'],
         refs=['Elastic analysis of an eccentrically loaded bolt group (recalled)']),

    Calc('rivet_shear', 'Rivet shear and bearing stress',
         vars=[V('tau', 'Pa', 'rivet shear stress', label='τ'),
               V('sigma_b', 'Pa', 'rivet bearing stress', label='σb'),
               V('F', 'N', 'load transferred'),
               V('m', '', 'shear planes per rivet (1 or 2)', integer=True),
               V('n', '', 'number of rivets', integer=True),
               V('d', 'm', 'rivet diameter'), V('t', 'm', 'plate thickness')],
         equations=['tau = F/(m*n*pi*d**2/4)', 'sigma_b = F/(n*d*t)'],
         outputs=['tau', 'sigma_b'],
         example={'F': 20000.0, 'm': 1, 'n': 4, 'd': 0.008, 't': 0.006},
         keywords=['rivet', 'shear', 'bearing', 'lap joint', 'butt joint'],
         assumptions=['Average stresses: shear on m·n rivet cross-sections; bearing on the projected area '
                      'n·d·t of the plate. Single shear m = 1, double (butt with cover plates) m = 2.'],
         refs=['Rivet shear τ = F/(m·n·πd²/4); bearing σb = F/(n·d·t)']),

    Calc('rivet_efficiency', 'Riveted joint strength and efficiency (per pitch)',
         vars=[V('F_shear', 'N', 'joint strength in rivet shear (per pitch)'),
               V('F_bearing', 'N', 'joint strength in bearing (per pitch)'),
               V('F_tearing', 'N', 'joint strength in plate tearing (per pitch)'),
               V('F_solid', 'N', 'strength of the solid (unpierced) plate per pitch'),
               V('eta', '', 'joint efficiency (min strength / solid-plate strength)'),
               V('m', '', 'shear planes per rivet (1 or 2)', integer=True),
               V('n', '', 'rivets per pitch length', integer=True),
               V('d', 'm', 'rivet diameter'), V('t', 'm', 'plate thickness'), V('p', 'm', 'pitch (rivet spacing)'),
               V('tau_allow', 'Pa', 'allowable rivet shear stress'),
               V('sigma_b_allow', 'Pa', 'allowable bearing stress'),
               V('sigma_t_allow', 'Pa', 'allowable plate tensile stress')],
         equations=['F_shear = m*n*pi*d**2/4*tau_allow',
                    'F_bearing = n*d*t*sigma_b_allow',
                    'F_tearing = (p - d)*t*sigma_t_allow',
                    'F_solid = p*t*sigma_t_allow',
                    'eta = Min(F_shear, F_bearing, F_tearing)/F_solid'],
         outputs=['F_shear', 'F_bearing', 'F_tearing', 'F_solid', 'eta'],
         example={'m': 1, 'n': 1, 'd': 0.008, 't': 0.006, 'p': 0.024, 'tau_allow': 60e6,
                  'sigma_b_allow': 150e6, 'sigma_t_allow': 110e6},
         keywords=['rivet', 'efficiency', 'tearing', 'joint strength', 'pitch'],
         assumptions=['Per-pitch strengths with n rivets per pitch length p: shear m·n·πd²/4·τ_allow, '
                      'bearing n·d·t·σb, tearing across the outer-row net section (p − d)·t·σt (one hole per pitch in the outer row), solid plate '
                      'p·t·σt. The joint efficiency is η = min(F_shear, F_bearing, F_tearing)/F_solid. '
                      'Requires p > d for a positive net section.'],
         refs=['Riveted joint strengths and efficiency (standard forms, recalled)']),
]

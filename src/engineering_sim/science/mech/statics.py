# DESCRIPTION: Statics section (pilot / reference pattern for every other section): moments, resultants,
#              rigid-body and truss equilibrium, distributed loads, dry friction, belts, wedges, screw
#              jacks, cables, composite centroids and mechanical advantage.
"""Statics calculators.

Sign/axis convention for every 2-D procedure here: x to the right, y up, counter-clockwise moments and
angles positive, angles measured from +x. Truss member forces are positive in tension.

Reference relations are the standard ones found in any engineering-statics text (e.g. Hibbeler,
*Engineering Mechanics: Statics*; Beer & Johnston, *Vector Mechanics for Engineers*). They are cited by
relation name, recalled rather than re-read, so no page numbers are given.
"""
import math

import numpy as np

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts

SECTION = section('statics', 'Statics', order=20,
                  summary='Moments, resultants, equilibrium, trusses, friction, belts, wedges, screws, '
                          'cables, centroids and mechanical advantage.')

# Shared declarations -------------------------------------------------------------------------------
PI = math.pi


def _resultant(values):
    """Resultant of coplanar forces [[F (N), angle (deg from +x), x (m), y (m)], ...] about the origin."""
    forces = values['forces']
    if not isinstance(forces, list) or not forces:
        raise MechError('forces: give a list of [F_N, angle_deg, x_m, y_m] rows')
    Rx = Ry = Mo = 0.0
    for row in forces:
        if len(row) != 4:
            raise MechError('forces: each row must be [F_N, angle_deg, x_m, y_m]')
        F, ang, x, y = map(float, row)
        fx, fy = F * math.cos(math.radians(ang)), F * math.sin(math.radians(ang))
        Rx += fx
        Ry += fy
        Mo += x * fy - y * fx  # 2-D cross product r x F (CCW +)
    R = math.hypot(Rx, Ry)
    out = {'Rx': Rx, 'Ry': Ry, 'R': R, 'theta_R': math.atan2(Ry, Rx), 'M_O': Mo}
    # Where the single resultant force crosses the x axis (M_O = x_int * Ry); undefined for Ry = 0.
    out['x_int'] = Mo / Ry if abs(Ry) > 1e-12 * max(R, 1.0) else float('nan')
    return out


def _truss(values):
    """Pin-jointed plane truss by the method of joints (all joints at once as a linear system).

    nodes    [[x_m, y_m], ...]
    members  [[i, j], ...]            node indices (0-based)
    supports [[node, 'x'|'y'], ...]   one reaction component per entry (pin = x and y, roller = one)
    loads    [[node, Fx_N, Fy_N], ...]
    Statically determinate only: members + reactions must equal 2 x nodes and the matrix full rank.
    The method of sections gives identical forces for a determinate truss, so one solver serves both.
    """
    nodes = np.asarray(values['nodes'], dtype=float)
    members = [tuple(map(int, m)) for m in values['members']]
    supports = [(int(s[0]), str(s[1])) for s in values['supports']]
    loads = values.get('loads') or []
    n = len(nodes)
    m, r = len(members), len(supports)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise MechError('nodes: give [[x, y], ...] in m')
    if m + r != 2 * n:
        raise MechError(f'not statically determinate: members + reactions = {m + r}, 2 x joints = {2 * n}')
    A = np.zeros((2 * n, m + r))
    b = np.zeros(2 * n)
    for k, (i, j) in enumerate(members):
        d = nodes[j] - nodes[i]
        L = float(np.hypot(*d))
        if L == 0:
            raise MechError(f'member {k} has zero length')
        c, s = d / L
        # Tension pulls joint i toward j (+c,+s) and joint j toward i.
        A[2 * i, k] += c
        A[2 * i + 1, k] += s
        A[2 * j, k] -= c
        A[2 * j + 1, k] -= s
    for k, (node, comp) in enumerate(supports):
        if comp not in ('x', 'y'):
            raise MechError("supports: component must be 'x' or 'y'")
        A[2 * node + (0 if comp == 'x' else 1), m + k] = 1.0
    for node, fx, fy in loads:
        b[2 * int(node)] -= float(fx)
        b[2 * int(node) + 1] -= float(fy)
    if np.linalg.matrix_rank(A) < 2 * n:
        raise MechError('truss is a mechanism or has redundant supports (singular equilibrium matrix)')
    x = np.linalg.solve(A, b)
    forces = [float(v) for v in x[:m]]
    return {'member_forces': forces,
            'reactions': [[supports[k][0], supports[k][1], float(x[m + k])] for k in range(r)],
            'max_tension': max(max(forces), 0.0), 'max_compression': max(-min(forces), 0.0)}


def _rigid_body(values):
    """Three unknown reaction components on one planar rigid body (frame/machine member).

    loads     [[Fx_N, Fy_N, x_m, y_m], ...] and optional couples [[M_Nm], ...] in 'couples'
    unknowns  exactly 3 rows [[angle_deg, x_m, y_m], ...]: line of action direction and point
    Solves sum Fx = 0, sum Fy = 0, sum M_origin = 0 for the three magnitudes (negative = opposite sense).
    """
    loads = values.get('loads') or []
    couples = values.get('couples') or []
    unk = values['unknowns']
    if len(unk) != 3:
        raise MechError('unknowns: exactly three reaction lines are solvable by planar equilibrium')
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for k, (ang, x, y) in enumerate(unk):
        c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        A[:, k] = [c, s, x * s - y * c]
    for fx, fy, x, y in loads:
        b -= [fx, fy, x * fy - y * fx]
    for row in couples:
        b[2] -= float(row[0] if isinstance(row, list) else row)
    if abs(np.linalg.det(A)) < 1e-12:
        raise MechError('reaction lines are concurrent or parallel: improperly constrained body')
    sol = np.linalg.solve(A, b)
    return {'reactions': [float(v) for v in sol]}


def _composite(values):
    """Composite centroid / centre of gravity: parts [[w, x, y, z], ...], w = area, volume, mass or weight
    (consistent in one list; negative w for holes/cut-outs)."""
    parts = values['parts']
    W = sum(float(p[0]) for p in parts)
    if abs(W) < 1e-300:
        raise MechError('parts: total weight/area is zero')
    xs = [sum(float(p[0]) * float(p[k]) for p in parts) / W for k in (1, 2, 3)]
    return {'W_total': W, 'x_bar': xs[0], 'y_bar': xs[1], 'z_bar': xs[2]}


def _mohr_like_force_chart(values):
    """Force polygon for the resultant calculator (tip-to-tail)."""
    x, y = [0.0], [0.0]
    for F, ang, _, _ in values['forces']:
        x.append(x[-1] + F * math.cos(math.radians(ang)))
        y.append(y[-1] + F * math.sin(math.radians(ang)))
    return charts.chart('Force polygon', 'Fx [N]', 'Fy [N]',
                        [charts.series('forces tip-to-tail', x, y),
                         charts.series('resultant', [0, x[-1]], [0, y[-1]])], equal=True)


CALCS = [
    Calc('moment', 'Moment of a force (M = F·d)',
         vars=[V('M', 'N*m', 'moment about the point', signed=True),
               V('F', 'N', 'force magnitude', signed=True),
               V('d', 'm', 'perpendicular distance from point to line of action')],
         equations=['M = F*d'], outputs=['M'], example={'F': 250.0, 'd': 0.4},
         keywords=['torque', 'lever arm', 'couple'],
         assumptions=['d is the perpendicular (moment-arm) distance; for a couple, d is the spacing of '
                      'the two equal, opposite, parallel forces and M is the same about every point.'],
         refs=['Moment of a force, M = F d (standard statics relation)']),

    Calc('force_components', 'Force components (rectangular)',
         vars=[V('F', 'N', 'force magnitude'),
               V('theta', 'rad', 'direction from +x, CCW positive', lo=-2 * PI, hi=2 * PI, label='θ'),
               V('Fx', 'N', 'x component', signed=True), V('Fy', 'N', 'y component', signed=True)],
         equations=['Fx = F*cos(theta)', 'Fy = F*sin(theta)'], outputs=['Fx', 'Fy'],
         example={'F': 500.0, 'theta': math.radians(30)},
         assumptions=['Inverse solves for θ return the root closest to the example; check the quadrant.'],
         refs=['Rectangular components of a planar force']),

    Calc('resultant_two', 'Resultant of two concurrent forces (law of cosines)',
         vars=[V('R', 'N', 'resultant magnitude'), V('F1', 'N', 'first force'), V('F2', 'N', 'second force'),
               V('gamma', 'rad', 'angle between the two forces', hi=PI, label='γ')],
         equations=['R = sqrt(F1**2 + F2**2 + 2*F1*F2*cos(gamma))'], outputs=['R'],
         example={'F1': 300.0, 'F2': 400.0, 'gamma': math.radians(60)},
         refs=['Parallelogram law / law of cosines']),

    Calc('resultant_coplanar', 'Resultant of a coplanar force system',
         vars=[V('forces', '', 'rows [F (N), angle from +x (deg), x (m), y (m)] of each force application',
                 kind='json'),
               V('Rx', 'N', 'resultant x component', signed=True), V('Ry', 'N', 'resultant y component', signed=True),
               V('R', 'N', 'resultant magnitude'), V('theta_R', 'rad', 'resultant direction', signed=True, label='θR'),
               V('M_O', 'N*m', 'moment about the origin (CCW +)', signed=True),
               V('x_int', 'm', 'x-axis intercept of the resultant line of action', signed=True)],
         procedure=_resultant, outputs=['Rx', 'Ry', 'R', 'theta_R', 'M_O', 'x_int'],
         example={'forces': [[100, 0, 0, 1], [200, 90, 2, 0], [150, 225, 1, 1]]},
         plots=[{'id': 'polygon', 'title': 'Force polygon', 'fn': _mohr_like_force_chart}],
         keywords=['equivalent system', 'force couple'],
         refs=['Equivalent force-couple system; resultant line of action']),

    Calc('simple_beam_reactions', 'Simply supported beam reactions (single point load)',
         vars=[V('R_A', 'N', 'left support reaction', signed=True), V('R_B', 'N', 'right support reaction', signed=True),
               V('P', 'N', 'point load (downward +)', signed=True), V('a', 'm', 'load position from left support'),
               V('L', 'm', 'span')],
         equations=['R_A = P*(L - a)/L', 'R_B = P*a/L'], outputs=['R_A', 'R_B'],
         example={'P': 10e3, 'a': 1.5, 'L': 4.0},
         refs=['Sum of moments about each support']),

    Calc('rigid_body_3', 'Planar rigid-body equilibrium: three unknown reactions (frames and machines)',
         vars=[V('loads', '', 'known forces [[Fx (N), Fy (N), x (m), y (m)], ...]', kind='json'),
               V('couples', '', 'known couples [[M (N·m, CCW +)], ...]; [] if none', kind='json'),
               V('unknowns', '', 'three reaction lines [[angle (deg), x (m), y (m)], ...]', kind='json'),
               V('reactions', '', 'three reaction magnitudes in N (negative = opposite to stated angle)', kind='json')],
         procedure=_rigid_body, outputs=['reactions'],
         example={'loads': [[0, -1000, 2, 0]], 'couples': [], 'unknowns': [[0, 0, 0], [90, 0, 0], [90, 3, 0]]},
         keywords=['equilibrium', 'free body', 'frame', 'machine', 'pin'],
         assumptions=['Analyse a frame member by member: apply equal-and-opposite pin forces between members '
                      '(Newton III) and keep each free body to three unknowns.'],
         refs=['ΣFx = 0, ΣFy = 0, ΣM = 0 for a planar rigid body']),

    Calc('truss', 'Plane truss member forces (method of joints / sections)',
         vars=[V('nodes', '', 'joint coordinates [[x (m), y (m)], ...]', kind='json'),
               V('members', '', 'member end joints [[i, j], ...] (0-based)', kind='json'),
               V('supports', '', "reaction components [[joint, 'x' or 'y'], ...]", kind='json'),
               V('loads', '', 'joint loads [[joint, Fx (N), Fy (N)], ...]', kind='json'),
               V('member_forces', '', 'member axial forces in N, tension positive', kind='json'),
               V('reactions', '', '[[joint, component, force (N)], ...]', kind='json'),
               V('max_tension', 'N', 'largest tensile member force'),
               V('max_compression', 'N', 'largest compressive member force (magnitude)')],
         procedure=_truss, outputs=['member_forces', 'reactions', 'max_tension', 'max_compression'],
         example={'nodes': [[0, 0], [4, 0], [2, 3]], 'members': [[0, 1], [1, 2], [0, 2]],
                  'supports': [[0, 'x'], [0, 'y'], [1, 'y']], 'loads': [[2, 0, -12000]]},
         keywords=['joints', 'sections', 'pin-jointed', 'bridge'],
         assumptions=['Ideal truss: pinned joints, loads only at joints, straight two-force members.',
                      'Statically determinate (m + r = 2j) and stable; otherwise an error is raised.'],
         refs=['Method of joints; method of sections (identical results for determinate trusses)']),

    Calc('distributed_load', 'Resultant of a linearly varying distributed load',
         vars=[V('W', 'N', 'resultant force'), V('x_bar', 'm', 'resultant location from the w1 end', label='x̄'),
               V('w1', 'N/m', 'intensity at start'), V('w2', 'N/m', 'intensity at end'), V('L', 'm', 'loaded length')],
         equations=['W = (w1 + w2)*L/2', 'x_bar = L*(w1 + 2*w2)/(3*(w1 + w2))'], outputs=['W', 'x_bar'],
         example={'w1': 2000.0, 'w2': 5000.0, 'L': 3.0},
         keywords=['trapezoidal', 'triangular', 'uniform', 'udl'],
         assumptions=['w1 = w2 gives a uniform load (x̄ = L/2); w1 = 0 a triangle (x̄ = 2L/3).'],
         refs=['Resultant = area under the load curve, acting through its centroid']),

    Calc('friction', 'Dry (Coulomb) friction and angle of friction',
         vars=[V('F', 'N', 'friction force (limiting, at impending motion)'), V('mu', '', 'coefficient of friction', label='μ'),
               V('N', 'N', 'normal force'), V('phi', 'rad', 'angle of friction', hi=PI / 2, label='φ')],
         equations=['F = mu*N', 'phi = atan(mu)'], outputs=['F', 'phi'], example={'mu': 0.35, 'N': 800.0},
         keywords=['coulomb', 'static friction', 'kinetic friction', 'angle of repose'],
         assumptions=['F = μN is the limiting value; below impending motion the friction force is whatever '
                      'equilibrium requires, up to μsN. Use μk once sliding.'],
         refs=['Coulomb friction law; tan φ = μ']),

    Calc('incline_push', 'Force to push a block up / hold it on an incline (impending motion)',
         vars=[V('P_up', 'N', 'force parallel to the slope for impending motion up'),
               V('P_hold', 'N', 'force parallel to the slope to prevent sliding down (0 if self-holding)', signed=True),
               V('W', 'N', 'block weight'), V('alpha', 'rad', 'slope angle', hi=PI / 2, label='α'),
               V('mu', '', 'static friction coefficient', label='μ')],
         equations=['P_up = W*(sin(alpha) + mu*cos(alpha))', 'P_hold = W*(sin(alpha) - mu*cos(alpha))'],
         outputs=['P_up', 'P_hold'], example={'W': 500.0, 'alpha': math.radians(25), 'mu': 0.3},
         assumptions=['Force applied parallel to the incline. P_hold < 0 means the block does not slide '
                      '(tan α < μ).'],
         refs=['Equilibrium of a block on an inclined plane with Coulomb friction']),

    Calc('belt_friction', 'Flat belt / capstan friction (T₁/T₂ = e^{μθ})',
         vars=[V('T1', 'N', 'tight-side tension'), V('T2', 'N', 'slack-side tension'),
               V('mu', '', 'friction coefficient', label='μ'), V('theta', 'rad', 'wrap angle', label='θ')],
         equations=['T1 = T2*exp(mu*theta)'], outputs=['T1'], example={'T2': 100.0, 'mu': 0.3, 'theta': PI},
         keywords=['capstan', 'euler-eytelwein', 'rope', 'band'],
         assumptions=['Impending slip, massless flat belt, centrifugal tension neglected (see belts section).'],
         refs=['Capstan (Euler–Eytelwein) equation']),

    Calc('vbelt_friction', 'V-belt friction (T₁/T₂ = e^{μθ/sin(β/2)})',
         vars=[V('T1', 'N', 'tight-side tension'), V('T2', 'N', 'slack-side tension'),
               V('mu', '', 'friction coefficient', label='μ'), V('theta', 'rad', 'wrap angle', label='θ'),
               V('beta', 'rad', 'groove included angle', hi=PI, label='β')],
         equations=['T1 = T2*exp(mu*theta/sin(beta/2))'], outputs=['T1'],
         example={'T2': 100.0, 'mu': 0.3, 'theta': PI, 'beta': math.radians(36)},
         refs=['Capstan equation with wedge action of a V groove']),

    Calc('wedge', 'Wedge driven under a guided load',
         vars=[V('P', 'N', 'horizontal driving force for impending motion'), V('W', 'N', 'vertical load on the wedge'),
               V('alpha', 'rad', 'wedge angle (upper face)', hi=PI / 2, label='α'),
               V('phi', 'rad', 'friction angle on both wedge faces (φ = atan μ)', hi=PI / 2, label='φ')],
         equations=['P = W*(tan(phi) + tan(alpha + phi))'], outputs=['P'],
         example={'W': 5000.0, 'alpha': math.radians(10), 'phi': math.atan(0.25)},
         assumptions=['Wedge slides on a horizontal floor under a block that is guided vertically by a '
                      'frictionless wall; same friction angle on the floor and the inclined face.',
                      'Derivation: block ΣFy: R cos(α+φ) = W; wedge ΣFy: N1 = W; wedge ΣFx: '
                      'P = N1 tan φ + R sin(α+φ).'],
         refs=['Wedge equilibrium with friction on both faces']),

    Calc('screw_jack', 'Square-thread screw jack: raising and lowering torque',
         vars=[V('T_raise', 'N*m', 'torque to raise the load'), V('T_lower', 'N*m', 'torque to lower (negative = overhauls)', signed=True),
               V('lam', 'rad', 'lead angle', hi=PI / 2, label='λ'), V('phi', 'rad', 'friction angle', hi=PI / 2, label='φ'),
               V('W', 'N', 'axial load'), V('r_m', 'm', 'mean thread radius'), V('l', 'm', 'lead (advance per revolution)'),
               V('mu', '', 'thread friction coefficient', label='μ')],
         equations=['T_raise = W*r_m*tan(lam + phi)', 'T_lower = W*r_m*tan(phi - lam)',
                    'lam = atan(l/(2*pi*r_m))', 'phi = atan(mu)'],
         outputs=['T_raise', 'T_lower', 'lam', 'phi'], example={'W': 20e3, 'r_m': 0.015, 'l': 0.006, 'mu': 0.15},
         keywords=['power screw', 'self-locking'],
         assumptions=['Square thread, collar friction excluded (see power screws section for Acme/collar).',
                      'T_lower > 0 means self-locking (φ > λ).'],
         refs=['Screw as an inclined plane wrapped on a cylinder']),

    Calc('cable_parabolic', 'Cable under uniform load per horizontal length (parabolic)',
         vars=[V('T0', 'N', 'horizontal tension (at lowest point)'), V('T_max', 'N', 'maximum tension (at supports)'),
               V('s', 'm', 'cable length'), V('w', 'N/m', 'load per horizontal metre'),
               V('L', 'm', 'span between level supports'), V('h', 'm', 'sag at mid-span')],
         equations=['T0 = w*L**2/(8*h)', 'T_max = sqrt(T0**2 + (w*L/2)**2)',
                    's = L*(1 + (8/3)*(h/L)**2 - (32/5)*(h/L)**4)'],
         outputs=['T0', 'T_max', 's'], example={'w': 2000.0, 'L': 100.0, 'h': 8.0},
         keywords=['suspension', 'sag'],
         assumptions=['Level supports; load uniform per horizontal length (suspension bridge deck).',
                      'Length uses the series expansion, accurate for h/L ≲ 0.2.'],
         refs=['Parabolic cable; series expansion of arc length']),

    Calc('cable_catenary', 'Cable under its own weight (catenary)',
         vars=[V('c', 'm', 'catenary parameter T0/w'), V('T0', 'N', 'horizontal tension'), V('T_max', 'N', 'maximum tension'),
               V('s', 'm', 'cable length'), V('w', 'N/m', 'weight per unit cable length'),
               V('L', 'm', 'span between level supports'), V('h', 'm', 'sag')],
         equations=['h = c*(cosh(L/(2*c)) - 1)', 'T0 = w*c', 'T_max = w*(c + h)', 's = 2*c*sinh(L/(2*c))'],
         outputs=['c', 'T0', 'T_max', 's'], example={'w': 50.0, 'L': 100.0, 'h': 10.0},
         keywords=['power line', 'chain', 'sag'],
         assumptions=['Level supports, perfectly flexible inextensible cable.'],
         refs=['Catenary: y = c cosh(x/c)']),

    Calc('composite_centroid', 'Centroid / centre of gravity of a composite body',
         vars=[V('parts', '', 'rows [w, x, y, z]: w = area, volume, mass or weight (one kind; negative for holes); '
                              'coordinates of each part centroid in m', kind='json'),
               V('W_total', '', 'sum of w (same unit as the part w column)', signed=True),
               V('x_bar', 'm', 'x̄', signed=True), V('y_bar', 'm', 'ȳ', signed=True), V('z_bar', 'm', 'z̄', signed=True)],
         procedure=_composite, outputs=['W_total', 'x_bar', 'y_bar', 'z_bar'],
         example={'parts': [[0.02, 0.1, 0.05, 0], [0.01, 0.25, 0.1, 0], [-0.002, 0.1, 0.05, 0]]},
         keywords=['center of gravity', 'centre of mass', 'composite area'],
         refs=['x̄ = Σ w x / Σ w']),

    Calc('mechanical_advantage', 'Mechanical advantage, velocity ratio and efficiency (virtual work)',
         vars=[V('MA', '', 'mechanical advantage F_out/F_in'), V('VR', '', 'velocity ratio d_in/d_out'),
               V('eta', '', 'efficiency', hi=1.0, label='η'), V('F_out', 'N', 'output force (load)'),
               V('F_in', 'N', 'input force (effort)'), V('d_in', 'm', 'input displacement'), V('d_out', 'm', 'output displacement')],
         equations=['MA = F_out/F_in', 'VR = d_in/d_out', 'eta = MA/VR'], outputs=['MA', 'VR', 'eta'],
         example={'F_out': 1000.0, 'F_in': 120.0, 'd_in': 1.0, 'd_out': 0.1},
         keywords=['lever', 'pulley', 'block and tackle', 'virtual work'],
         assumptions=['Ideal machine (virtual work, no friction): F_in·δ_in = F_out·δ_out, so η = 1 and MA = VR.'],
         refs=['Principle of virtual work; machine efficiency']),
]

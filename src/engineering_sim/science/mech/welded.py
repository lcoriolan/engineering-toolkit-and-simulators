# DESCRIPTION: Welded joints section: fillet-weld throat geometry and direct shear, weld-group unit
#              properties (welds idealised as lines), eccentrically loaded weld groups (direct +
#              torsional shear vector sum) and fillet-weld strength / sizing.
"""Welded-joint calculators.

Weld groups use the standard "weld treated as a line" method: the throat of each fillet weld is
idealised as a zero-thickness line along the weld centreline, so the unit properties A_u, I_xu, I_yu,
J_u depend only on the layout geometry. The physical throat area and polar moment follow by multiplying
by the throat t = h/√2 (equal-leg 45° fillet). Allowable stresses are user inputs — this section
computes geometry and induced stress; it does not supply weld-code allowables.

References are the standard relations in machine-design texts (Shigley, *Mechanical Engineering
Design*; Blodgett, *Design of Welded Structures*), cited by relation name and recalled rather than
re-read, so no page numbers are given.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('welded', 'Welded joints', order=130,
                  summary='Fillet throat, weld shear, weld-group unit properties (line method), '
                          'eccentric loads, fillet strength and sizing.')

THROAT = 1.0 / math.sqrt(2.0)   # throat fraction for an equal-leg 45° fillet


def _weld_units(segs):
    """Unit weld-group properties treating each weld as a line along its centreline.

    segs: [[x1, y1, x2, y2], ...] in metres — one straight segment per weld (a polyline weld is split).
    Returns A_u (m, total weld length), the centroid (x_bar, y_bar) and the second moments about the
    centroidal axes I_xu, I_yu and the polar J_u = I_xu + I_yu (all in m^3 per unit throat width).

    For a straight segment from (x1,y1) to (x2,y2) the integrals ∫ y'² dL and ∫ x'² dL evaluate in
    closed form: ∫_0^1 (a + b t)² dt = a² + a b + b²/3, with a the offset of the start point from the
    centroid and b the coordinate change along the segment.
    """
    if not isinstance(segs, list) or not segs:
        raise MechError('segs: give a list of [x1, y1, x2, y2] rows in metres')
    rows = []
    length = 0.0
    sx = sy = 0.0
    for s in segs:
        if len(s) != 4:
            raise MechError('segs: each row must be [x1, y1, x2, y2]')
        x1, y1, x2, y2 = map(float, s)
        li = math.hypot(x2 - x1, y2 - y1)
        if li == 0:
            raise MechError('segs: zero-length segment')
        rows.append((x1, y1, x2, y2, li))
        length += li
        sx += li * (x1 + x2) / 2.0
        sy += li * (y1 + y2) / 2.0
    if length == 0:
        raise MechError('segs: total weld length is zero')
    xb, yb = sx / length, sy / length
    ix = iy = 0.0
    for x1, y1, x2, y2, li in rows:
        a, b = y1 - yb, y2 - y1
        ix += li * (a * a + a * b + b * b / 3.0)
        c, d = x1 - xb, x2 - x1
        iy += li * (c * c + c * d + d * d / 3.0)
    return {'A_u': length, 'x_bar': xb, 'y_bar': yb, 'I_xu': ix, 'I_yu': iy, 'J_u': ix + iy}


def _weld_group_props(values):
    return _weld_units(values['segs'])


def _eccentric_weld(values):
    """In-plane eccentric load on a weld group: direct shear + torsional shear, vector-summed.

    The load P at angle θ applied at (x_P, y_P) is reduced to a force at the centroid plus a torque
    T = (x_P − x̄)Fy − (y_P − ȳ)Fx (CCW positive). Direct shear is uniform (P/A); torsional shear at a
    point is τ = (T/J)(−y', x') with x', y' measured from the centroid. The resultant magnitude is
    evaluated at every weld endpoint — the critical point is the one with the largest resultant.
    """
    u = _weld_units(values['segs'])
    h, P, th = values['h'], values['P'], values['theta']
    xp, yp = values['x_P'], values['y_P']
    t = THROAT * h
    A = t * u['A_u']
    J = t * u['J_u']
    xb, yb = u['x_bar'], u['y_bar']
    Fx, Fy = P * math.cos(th), P * math.sin(th)
    T = (xp - xb) * Fy - (yp - yb) * Fx          # r × F about the centroid, CCW positive
    tdx, tdy = Fx / A, Fy / A                    # uniform direct-shear components
    points = {}
    for s in values['segs']:
        for (px, py) in ((s[0], s[1]), (s[2], s[3])):
            tx = -T * (py - yb) / J               # torsional shear components (T CCW +)
            ty = T * (px - xb) / J
            tau = math.hypot(tdx + tx, tdy + ty)
            points.setdefault((px, py), tau)
    (xc, yc), tau_max = max(points.items(), key=lambda kv: kv[1])
    return {'A': A, 'J': J, 'x_bar': xb, 'y_bar': yb, 'T': T,
            'tau_direct': math.hypot(tdx, tdy), 'tau_max': tau_max,
            'x_crit': xc, 'y_crit': yc,
            'points': [[px, py, points[(px, py)]] for px, py in
                       sorted(points, key=lambda k: -points[k])]}


CALCS = [
    Calc('fillet_throat', 'Fillet weld throat and throat area',
         vars=[V('t', 'm', 'throat dimension (h/√2 for a 45° equal-leg fillet)'),
               V('h', 'm', 'leg size'),
               V('L', 'm', 'weld length'),
               V('A', 'm^2', 'throat area (t·L)')],
         equations=['t = h/sqrt(2)', 'A = t*L'], outputs=['t', 'A'],
         example={'h': 0.006, 'L': 0.1},
         assumptions=['Equal-leg fillet with a 45° throat. For unequal legs or another flank angle use '
                      'the measured throat dimension directly in place of h/√2.'],
         refs=['Fillet weld throat = 0.707 × leg (45° equal-leg fillet; standard)']),

    Calc('weld_direct_shear', 'Direct (primary) shear stress in a weld',
         vars=[V('tau', 'Pa', 'direct shear stress on the throat', label='τ'),
               V('P', 'N', 'load (magnitude)'),
               V('A', 'm^2', 'throat area (0.707·h·L for one fillet)')],
         equations=['tau = P/A'], outputs=['tau'],
         example={'P': 20000.0, 'A': 0.00042},
         assumptions=['Uniform direct shear through the throat area. For a weld group use the total '
                      'throat area A = 0.707·h·A_u from the group calculator.'],
         refs=['τ = P/A (direct weld shear)']),

    Calc('weld_group_props', 'Weld-group unit properties (welds treated as lines)',
         vars=[V('segs', '', 'weld centrelines [[x1, y1, x2, y2] (m), ...] in a local frame', kind='json'),
               V('A_u', 'm', 'unit length (sum of weld lengths)'),
               V('x_bar', 'm', 'centroid x', signed=True, label='x̄'),
               V('y_bar', 'm', 'centroid y', signed=True, label='ȳ'),
               V('I_xu', 'm^3', 'unit second moment about the horizontal centroidal axis'),
               V('I_yu', 'm^3', 'unit second moment about the vertical centroidal axis'),
               V('J_u', 'm^3', 'unit polar second moment (I_xu + I_yu)')],
         procedure=_weld_group_props, outputs=['A_u', 'x_bar', 'y_bar', 'I_xu', 'I_yu', 'J_u'],
         example={'segs': [[0, 0, 0, 0.15], [0.1, 0, 0.1, 0.15], [0, 0, 0.1, 0], [0, 0.15, 0.1, 0.15]]},
         assumptions=['Each weld is idealised as a line along its throat centreline; properties are per '
                      'unit throat width (multiply by 0.707·h for physical area / moments). Single-line, '
                      'two-parallel, box, C and L patterns are all covered by the same segment list.'],
         refs=['Weld-as-a-line unit properties (Blodgett / Shigley; e.g. box J_u = (b+d)³/6)']),

    Calc('eccentric_weld', 'Eccentrically loaded weld group (direct + torsional shear)',
         vars=[V('segs', '', 'weld centrelines [[x1, y1, x2, y2] (m), ...]', kind='json'),
               V('h', 'm', 'leg size'),
               V('P', 'N', 'load magnitude'),
               V('theta', 'rad', 'load direction from +x, CCW positive', lo=-math.pi, hi=math.pi,
                 signed=True, label='θ'),
               V('x_P', 'm', 'load application point x', signed=True),
               V('y_P', 'm', 'load application point y', signed=True),
               V('A', 'm^2', 'total throat area'),
               V('J', 'm^4', 'polar second moment of the throat area'),
               V('x_bar', 'm', 'centroid x', signed=True, label='x̄'),
               V('y_bar', 'm', 'centroid y', signed=True, label='ȳ'),
               V('T', 'N*m', 'torque about the centroid (r × F, CCW +)', signed=True),
               V('tau_direct', 'Pa', 'direct shear magnitude P/A'),
               V('tau_max', 'Pa', 'maximum resultant shear at the critical point'),
               V('x_crit', 'm', 'critical point x', signed=True),
               V('y_crit', 'm', 'critical point y', signed=True),
               V('points', '', '[[x, y, τ (Pa)], ...] at every weld endpoint, largest first', kind='json')],
         procedure=_eccentric_weld,
         outputs=['A', 'J', 'x_bar', 'y_bar', 'T', 'tau_direct', 'tau_max', 'x_crit', 'y_crit', 'points'],
         example={'segs': [[0, 0, 0, 0.15], [0.1, 0, 0.1, 0.15]], 'h': 0.006, 'P': 30000.0,
                  'theta': -math.pi / 2, 'x_P': 0.15, 'y_P': 0.075},
         assumptions=['In-plane eccentric load: uniform direct shear from the load components plus '
                      'torsional shear from the torque T about the weld-group centroid; the resultant is '
                      'the vector sum evaluated at every weld endpoint (the critical point has the '
                      'largest resultant). Out-of-plane bending loads are a separate case not covered '
                      'here.', 'Equal-leg 45° fillets; every stress scales as 1/(0.707 h), so the '
                      'critical point does not depend on the leg size.'],
         refs=['Direct + torsional shear vector sum at weld-group endpoints (Shigley sec. 9-2 form, '
               'recalled)']),

    Calc('fillet_weld_strength', 'Fillet weld allowable strength (per length and total)',
         vars=[V('q', 'N/m', 'allowable load per unit weld length'),
               V('P_allow', 'N', 'allowable total load'),
               V('h', 'm', 'leg size'),
               V('tau_allow', 'Pa', 'allowable shear stress in the weld throat (code/material; user input)'),
               V('L', 'm', 'weld length')],
         equations=['q = h*tau_allow/sqrt(2)', 'P_allow = q*L'], outputs=['q', 'P_allow'],
         example={'h': 0.006, 'tau_allow': 124e6, 'L': 0.1},
         assumptions=['q = 0.707·h·τ_allow is the allowable force per unit weld length of a 45° fillet; '
                      'solve for h to size a weld from a required load and allowable. The allowable '
                      'stress τ_allow is a user input from the governing code, not a built-in table.'],
         refs=['Fillet weld strength per unit length = 0.707·h·τ_allow']),
]

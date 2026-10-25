# DESCRIPTION: Pressure vessels and piping section: ASME Section VIII Div 1 shell/head thickness and
#              MAWP (cylindrical hoop + longitudinal forms, spherical, hemispherical/ellipsoidal/
#              torispherical heads) with thin-shell validity checks, hydrostatic test pressure,
#              simplified nozzle reinforcement, pipe thermal expansion and expansion-loop sizing, and
#              pipe support spacing (simply-supported or continuous spans).
"""Pressure-vessel and piping calculators.

Conventions:
- Design thickness and maximum allowable working pressure follow the ASME Boiler & Pressure Vessel Code,
  Section VIII Division 1, UG-27 (shells) and UG-32 (formed heads): these are the widely published
  closed-form design equations with the allowable stress S and joint efficiency E as USER INPUTS. The
  forms are edition-independent; they are recalled rather than re-read, so no paragraph numbers beyond
  the familiar UG-27/UG-32 names are given. Corrosion allowance is applied by the user to the radii and
  thicknesses before entry (state it in your inputs).
- Hoop / longitudinal / radial STRESS analysis of a shell is NOT duplicated here: use
  materials_mech.thin_cylinder (membrane hoop + longitudinal), materials_mech.thin_sphere, and
  materials_mech.thick_cylinder (Lamé tangential/radial/axial) — the ASME design equations below are the
  membrane-stress limit p·R/t ≤ S·E rearranged for the design variables.
- The thin-shell design equations are valid only inside the code limits (cylinder P ≤ 0.385·S·E and
  t ≤ R/2; sphere/hemispherical head P ≤ 0.665·S·E and t ≤ 0.356·R, recalled). A pressure or thickness
  beyond those limits is computed but reported with a warning pointing at thick-wall rules.
- Piping relations (thermal expansion, guided-cantilever loop, support spacing) are the standard
  pipe-stress analysis forms (Kellogg / ASME B31.3 practice), recalled. The simply-supported and
  continuous span constants are recalled values.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('pressure_vessels', 'Pressure vessels and piping', order=440,
                  summary='ASME shell/head thickness and MAWP, hydrotest pressure, nozzle reinforcement, '
                          'pipe thermal expansion, expansion loops and support spacing.')

PI = math.pi


def _cylindrical_shell(values):
    """ASME UG-27 cylinder: circumferential-stress (longitudinal joint) and longitudinal-stress
    (circumferential joint) thickness and MAWP; the required thickness is the larger, the MAWP the smaller."""
    p, R, S = values['p'], values['R'], values['S']
    E_l, E_c, t = values['E_l'], values['E_c'], values['t']
    if R <= 0 or S <= 0:
        raise MechError('R and S must be positive')
    if p <= 0:
        raise MechError('p (design pressure) must be positive')
    denom_h = S * E_l - 0.6 * p
    if denom_h <= 0:
        raise MechError('p >= S·E_l/0.6: the circumferential-stress form is undefined (pressure beyond the code range)')
    t_hoop = p * R / denom_h                     # circumferential stress, longitudinal joint efficiency E_l
    t_long = p * R / (2 * S * E_c + 0.4 * p)     # longitudinal stress, circumferential joint efficiency E_c
    t_req = max(t_hoop, t_long)
    MAWP_hoop = S * E_l * t / (R + 0.6 * t)
    MAWP_long = 2 * S * E_c * t / (R - 0.4 * t)
    MAWP = min(MAWP_hoop, MAWP_long)
    out = {'t_req': t_req, 'MAWP': MAWP, 't_hoop': t_hoop, 't_long': t_long,
           'MAWP_hoop': MAWP_hoop, 'MAWP_long': MAWP_long}
    warnings = []
    if p > 0.385 * S * E_l:
        warnings.append('design pressure exceeds 0.385·S·E_l: outside the thin-shell range of the ASME '
                        'formula (use thick-wall rules, materials_mech.thick_cylinder)')
    if t > R / 2.0:
        warnings.append('wall thickness t exceeds R/2: outside the thin-shell range of the ASME formula')
    if t_req > R / 2.0:
        warnings.append('required thickness exceeds R/2: outside the thin-shell range of the ASME formula')
    if warnings:
        out['_warnings'] = warnings
    return out


def _spherical_shell(values):
    """ASME UG-27 sphere: required thickness and MAWP with thin-shell validity checks."""
    p, R, S, E, t = values['p'], values['R'], values['S'], values['E'], values['t']
    if R <= 0 or S <= 0:
        raise MechError('R and S must be positive')
    if p <= 0:
        raise MechError('p (design pressure) must be positive')
    denom = 2 * S * E - 0.2 * p
    if denom <= 0:
        raise MechError('p >= 10·S·E: the spherical form is undefined (pressure beyond the code range)')
    t_req = p * R / denom
    MAWP = 2 * S * E * t / (R + 0.2 * t)
    out = {'t_req': t_req, 'MAWP': MAWP}
    warnings = []
    if p > 0.665 * S * E:
        warnings.append('design pressure exceeds 0.665·S·E: outside the thin-shell range of the ASME formula '
                        '(use thick-wall rules)')
    if t > 0.356 * R:
        warnings.append('wall thickness t exceeds 0.356·R: outside the thin-shell range of the ASME formula')
    if t_req > 0.356 * R:
        warnings.append('required thickness exceeds 0.356·R: outside the thin-shell range of the ASME formula')
    if warnings:
        out['_warnings'] = warnings
    return out


def _head_hemispherical(values):
    """ASME UG-32 hemispherical head thickness (same membrane form as a sphere) with validity checks."""
    p, L, S, E = values['p'], values['L'], values['S'], values['E']
    if L <= 0 or S <= 0:
        raise MechError('L and S must be positive')
    if p <= 0:
        raise MechError('p (design pressure) must be positive')
    denom = 2 * S * E - 0.2 * p
    if denom <= 0:
        raise MechError('p >= 10·S·E: the hemispherical form is undefined (pressure beyond the code range)')
    t = p * L / denom
    out = {'t': t}
    warnings = []
    if p > 0.665 * S * E:
        warnings.append('design pressure exceeds 0.665·S·E: outside the thin-shell range of the ASME formula '
                        '(use thick-wall rules)')
    if t > 0.356 * L:
        warnings.append('required thickness exceeds 0.356·L: outside the thin-shell range of the ASME formula')
    if warnings:
        out['_warnings'] = warnings
    return out


def _nozzle_reinforcement(values):
    """Simplified UG-37 area-replacement for a radial nozzle (correction factor F = 1).

    A_req = d·t_r + 2·t_n·t_r·(1 − f_r1) is the area removed from the shell wall (the second term is the
    shell metal displaced by the nozzle wall, credited only in proportion f_r1 = S_n/S). Available
    reinforcement is the shell excess A1 = d·(t − t_r) − 2·t_n·(t − t_r)·(1 − f_r1) over the opening, the
    nozzle-wall excess A2 = 2·h_n·(t_n − t_rn)·f_r2 on both sides of the wall, and the weld area A_weld.
    Negative A1/A2 (too-thin shell/nozzle) contribute 0 to A_avail, with a warning.
    """
    d, t_r, t = values['d'], values['t_r'], values['t']
    t_n, t_rn, h_n = values['t_n'], values['t_rn'], values['h_n']
    f_r1, f_r2, A_weld = values['f_r1'], values['f_r2'], values['A_weld']
    if d <= 0 or t <= 0 or t_n <= 0:
        raise MechError('d, t, t_n must be positive')
    A_req = d * t_r + 2.0 * t_n * t_r * (1.0 - f_r1)
    A1 = d * (t - t_r) - 2.0 * t_n * (t - t_r) * (1.0 - f_r1)   # shell-wall excess (may be negative)
    A2 = 2.0 * h_n * (t_n - t_rn) * f_r2                        # nozzle-wall excess (may be negative)
    A1_c = max(0.0, A1)
    A2_c = max(0.0, A2)
    A_avail = A1_c + A2_c + A_weld
    A_add = max(0.0, A_req - A_avail)
    out = {'A_avail': A_avail, 'A_req': A_req, 'A1': A1, 'A2': A2, 'A_add': A_add}
    warnings = []
    if A1 < 0:
        warnings.append('shell excess A1 is negative (insufficient shell thickness at the opening); '
                        'clamped to 0 in A_avail')
    if A2 < 0:
        warnings.append('nozzle excess A2 is negative (t_n < t_rn); clamped to 0 in A_avail')
    if warnings:
        out['_warnings'] = warnings
    return out


def _pipe_support_spacing(values):
    """Maximum pipe support spacing from bending-stress and mid-span deflection limits.

    The span condition is a user choice: simply supported (M = w·L²/8, δ = 5·w·L⁴/(384·E·I)) or
    continuous (M = w·L²/10, δ = w·L⁴/(185·E·I)) — both recalled beam constants. The bending-stress
    limit σ = M/Z gives L_stress; the deflection limit gives L_defl; the allowable spacing is the smaller.
    """
    w, D_o, t = values['w'], values['D_o'], values['t']
    E, S_allow, d_allow = values['E'], values['S_allow'], values['delta_allow']
    span = values['span']
    D_i = D_o - 2.0 * t
    if D_i <= 0:
        raise MechError('wall thickness t must be less than D_o/2')
    I = PI / 64.0 * (D_o ** 4 - D_i ** 4)
    Z = 2.0 * I / D_o
    if w <= 0:
        raise MechError('w (weight per length) must be positive')
    if span == 'continuous':
        L_stress = math.sqrt(10.0 * Z * S_allow / w)
        L_defl = (185.0 * E * I * d_allow / w) ** 0.25
    else:                                   # simply_supported
        L_stress = math.sqrt(8.0 * Z * S_allow / w)
        L_defl = (384.0 * E * I * d_allow / (5.0 * w)) ** 0.25
    L_max = min(L_stress, L_defl)
    out = {'I': I, 'Z': Z, 'L_stress': L_stress, 'L_defl': L_defl, 'L_max': L_max}
    out['_warnings'] = ['deflection limit governs' if L_defl <= L_stress else 'bending stress limit governs']
    return out


CALCS = [
    Calc('cylindrical_shell', 'Cylindrical shell: hoop + longitudinal forms, required thickness and MAWP',
         vars=[V('t_req', 'm', 'required wall thickness (the larger of the two stress forms)'),
               V('MAWP', 'Pa', 'maximum allowable working pressure (the smaller of the two forms)'),
               V('t_hoop', 'm', 'thickness from circumferential (hoop) stress'),
               V('t_long', 'm', 'thickness from longitudinal stress'),
               V('MAWP_hoop', 'Pa', 'MAWP from circumferential (hoop) stress'),
               V('MAWP_long', 'Pa', 'MAWP from longitudinal stress'),
               V('p', 'Pa', 'design pressure'), V('R', 'm', 'inside radius (corroded)'),
               V('S', 'Pa', 'allowable stress'),
               V('E_l', '', 'longitudinal joint efficiency (circumferential stress)', hi=1.0),
               V('E_c', '', 'circumferential joint efficiency (longitudinal stress)', hi=1.0),
               V('t', 'm', 'nominal wall thickness (corroded)')],
         procedure=_cylindrical_shell, outputs=['t_req', 'MAWP', 't_hoop', 't_long', 'MAWP_hoop', 'MAWP_long'],
         example={'p': 2e6, 'R': 0.5, 'S': 138e6, 'E_l': 0.85, 'E_c': 1.0, 't': 0.010},
         keywords=['asme', 'ug-27', 'boiler', 'tank', 'shell', 'maop', 'longitudinal joint'],
         assumptions=['ASME Sec VIII Div 1 UG-27 cylinder, both stress forms (recalled, edition-independent): '
                      'circumferential stress t = p·R/(S·E_l − 0.6·p), MAWP = S·E_l·t/(R + 0.6·t) with the '
                      'LONGITUDINAL joint efficiency E_l; longitudinal stress t = p·R/(2·S·E_c + 0.4·p), '
                      'MAWP = 2·S·E_c·t/(R − 0.4·t) with the CIRCUMFERENTIAL joint efficiency E_c. The required '
                      'thickness is the LARGER of the two and the MAWP the SMALLER (the hoop form governs for '
                      'most shells since its membrane stress is twice the longitudinal one). R is the corroded '
                      'inside radius. Valid for p ≤ 0.385·S·E_l and t ≤ R/2 — beyond that a warning is shown and '
                      'thick-wall rules apply. For the hoop/longitudinal/radial STRESS at a given pressure use '
                      'materials_mech.thin_cylinder / thick_cylinder.'],
         refs=['ASME Sec VIII Div 1 UG-27 shell thickness and MAWP (circumferential and longitudinal stress forms, recalled)']),

    Calc('spherical_shell', 'Spherical shell: required thickness and MAWP (ASME UG-27)',
         vars=[V('t_req', 'm', 'required wall thickness'), V('MAWP', 'Pa', 'maximum allowable working pressure'),
               V('p', 'Pa', 'design pressure'), V('R', 'm', 'inside radius (corroded)'),
               V('S', 'Pa', 'allowable stress'), V('E', '', 'joint efficiency', hi=1.0),
               V('t', 'm', 'nominal wall thickness (corroded)')],
         procedure=_spherical_shell, outputs=['t_req', 'MAWP'],
         example={'p': 2e6, 'R': 0.5, 'S': 138e6, 'E': 1.0, 't': 0.006},
         keywords=['asme', 'sphere', 'ug-27', 'storage sphere'],
         assumptions=['ASME Sec VIII Div 1 UG-27(d) sphere form (membrane stress p·R/2t), recalled. Valid for '
                      'p ≤ 0.665·S·E and t ≤ 0.356·R — beyond that a warning is shown. See '
                      'materials_mech.thin_sphere for the membrane stress itself.'],
         refs=['ASME Sec VIII Div 1 UG-27 spherical shell thickness and MAWP (recalled form)']),

    Calc('hydro_test', 'Hydrostatic test pressure',
         vars=[V('P_t', 'Pa', 'hydrostatic test pressure'), V('p', 'Pa', 'design pressure or MAWP'),
               V('factor', '', 'test factor (≥ 1)')],
         equations=['P_t = factor*p'], outputs=['P_t'],
         example={'p': 2.318e6, 'factor': 1.3},
         keywords=['hydrotest', 'hydrostatic test', 'ug-99', 'proof test'],
         assumptions=['Test factor is a user input: current ASME Div 1 uses 1.3 × (allowable stress at test '
                      'temperature / allowable stress at design temperature), recalled.'],
         refs=['Hydrostatic test pressure P_t = factor × MAWP']),

    Calc('head_hemispherical', 'Hemispherical head thickness (ASME UG-32)',
         vars=[V('t', 'm', 'required head thickness'), V('p', 'Pa', 'design pressure'),
               V('L', 'm', 'inside spherical radius'), V('S', 'Pa', 'allowable stress'), V('E', '', 'joint efficiency', hi=1.0)],
         procedure=_head_hemispherical, outputs=['t'],
         example={'p': 2e6, 'L': 0.5, 'S': 138e6, 'E': 1.0},
         keywords=['asme', 'hemispherical head', 'ug-32', 'dished head'],
         assumptions=['ASME Sec VIII Div 1 UG-32(f) hemispherical head (same membrane form as a sphere), '
                      'recalled. Valid for p ≤ 0.665·S·E and t ≤ 0.356·L — beyond that a warning is shown.'],
         refs=['ASME Sec VIII Div 1 UG-32 hemispherical head (recalled form)']),

    Calc('head_ellipsoidal', '2:1 ellipsoidal head thickness (ASME UG-32)',
         vars=[V('t', 'm', 'required head thickness'), V('p', 'Pa', 'design pressure'),
               V('D', 'm', 'inside major diameter'), V('S', 'Pa', 'allowable stress'), V('E', '', 'joint efficiency', hi=1.0)],
         equations=['t = p*D/(2*S*E - 0.2*p)'], outputs=['t'],
         example={'p': 2e6, 'D': 1.0, 'S': 138e6, 'E': 1.0},
         keywords=['asme', 'ellipsoidal head', 'ug-32', '2:1'],
         assumptions=['ASME Sec VIII Div 1 UG-32(d) for a 2:1 ellipsoidal head (K = 1), recalled. For other '
                      'D/2h ratios multiply the numerator by K = (2 + (D/2h)²)/6.'],
         refs=['ASME Sec VIII Div 1 UG-32 ellipsoidal head (recalled form)']),

    Calc('head_torispherical', 'Torispherical head thickness (ASME UG-32)',
         vars=[V('t', 'm', 'required head thickness'), V('p', 'Pa', 'design pressure'),
               V('L', 'm', 'inside crown (dish) radius'), V('S', 'Pa', 'allowable stress'), V('E', '', 'joint efficiency', hi=1.0)],
         equations=['t = 0.885*p*L/(S*E - 0.1*p)'], outputs=['t'],
         example={'p': 2e6, 'L': 1.0, 'S': 138e6, 'E': 1.0},
         keywords=['asme', 'torispherical head', 'ug-32', 'flanged and dished'],
         assumptions=['ASME Sec VIII Div 1 UG-32(e) torispherical head, recalled. The 0.885 factor is the '
                      'value of M for the standard ASME proportions — knuckle radius r = 0.06·L and crown '
                      'radius L ≤ D_o (outside diameter). For other proportions use the general form '
                      't = M·p·L/(S·E − 0.1·p) with M = ¼·(3 + √(L/r)), recalled.'],
         refs=['ASME Sec VIII Div 1 UG-32 torispherical head (recalled form)']),

    Calc('nozzle_reinforcement', 'Nozzle reinforcement by area replacement (simplified UG-37)',
         vars=[V('d', 'm', 'finished (corroded) opening diameter'),
               V('t_r', 'm', 'required shell thickness at the opening'),
               V('t', 'm', 'nominal shell thickness'),
               V('t_n', 'm', 'nominal nozzle wall thickness'),
               V('t_rn', 'm', 'required nozzle wall thickness'),
               V('h_n', 'm', 'effective nozzle wall height available for reinforcement (one side)'),
               V('f_r1', '', 'shell-metal strength reduction factor at the opening (S_n/S, ≤ 1)', hi=1.0),
               V('f_r2', '', 'nozzle-wall strength reduction factor (S_n/S, ≤ 1)', hi=1.0),
               V('A_weld', 'm^2', 'available weld (fillet/attachment) area'),
               V('A_avail', 'm^2', 'total available reinforcement area'),
               V('A_req', 'm^2', 'required reinforcement area'),
               V('A1', 'm^2', 'shell-wall excess area', signed=True),
               V('A2', 'm^2', 'nozzle-wall excess area', signed=True),
               V('A_add', 'm^2', 'additional reinforcement area still required (0 if none)')],
         procedure=_nozzle_reinforcement, outputs=['A_avail', 'A_req', 'A1', 'A2', 'A_add'],
         example={'d': 0.15, 't_r': 0.010, 't': 0.012, 't_n': 0.006, 't_rn': 0.002, 'h_n': 0.025,
                  'f_r1': 0.9, 'f_r2': 0.9, 'A_weld': 5e-5},
         keywords=['ug-37', 'nozzle', 'opening', 'reinforcement', 'area replacement'],
         assumptions=['Simplified UG-37 area-replacement for a RADIAL nozzle (correction factor F = 1): '
                      'A_req = d·t_r + 2·t_n·t_r·(1 − f_r1); A1 = d·(t − t_r) − 2·t_n·(t − t_r)·(1 − f_r1); '
                      'A2 = 2·h_n·(t_n − t_rn)·f_r2; the weld area A_weld is a user input. f_r1 and f_r2 are '
                      'the nozzle/shell strength-reduction factors (S_n/S, ≤ 1). A negative A1 or A2 '
                      '(too-thin shell or nozzle) is clamped to 0 inside A_avail, with a warning. h_n is the '
                      'reinforcement limit (≤ 2.5·t_n typical), a user input. The full code method also '
                      'applies the larger-of limit for A1 and F ≠ 1 for non-radial nozzles.'],
         refs=['ASME Sec VIII Div 1 UG-37 area replacement (simplified radial form, recalled)']),

    Calc('thermal_expansion_pipe', 'Thermal expansion of a straight pipe run',
         vars=[V('delta_L', 'm', 'change in length (expansion +, contraction −)', signed=True, label='ΔL'),
               V('alpha', '1/K', 'coefficient of thermal expansion', label='α'),
               V('L', 'm', 'installed length'), V('dT', 'K', 'temperature change (ΔT; enter K or delta_degC)', signed=True, label='ΔT')],
         equations=['delta_L = alpha*L*dT'], outputs=['delta_L'],
         example={'alpha': 12e-6, 'L': 30.0, 'dT': 100.0},
         keywords=['thermal expansion', 'pipe', 'growth'],
         assumptions=['Free (unrestrained) thermal growth; uniform ΔT, constant α. Restrained pipe develops '
                      'stress instead — see materials_mech.thermal_deformation.'],
         refs=['ΔL = α·L·ΔT']),

    Calc('expansion_loop', 'Expansion loop leg length (guided-cantilever method)',
         vars=[V('L', 'm', 'required loop leg length'),
               V('E', 'Pa', "pipe material elastic modulus"), V('D', 'm', 'pipe outside diameter'),
               V('delta', 'm', 'lateral displacement absorbed by this leg', label='δ'),
               V('S_A', 'Pa', 'allowable displacement stress range')],
         equations=['L = sqrt(3*E*D*delta/S_A)'], outputs=['L'],
         example={'E': 200e9, 'D': 0.10, 'delta': 0.05, 'S_A': 150e6},
         keywords=['expansion loop', 'guided cantilever', 'pipe flexibility', 'b31.3'],
         assumptions=['Guided-cantilever model of ONE loop leg: fixed at one end, guided (no rotation) at the '
                      'other, so σ = 3·E·D·δ/L²; setting σ = S_A gives L = √(3·E·D·δ/S_A). A symmetric U-loop '
                      'absorbs total expansion Δ with each leg taking δ = Δ/2 — enter δ per leg. S_A is the '
                      'allowable displacement stress range (e.g. ASME B31.3), a user input.'],
         refs=['Guided-cantilever expansion-loop relation σ = 3EDδ/L² (Kellogg pipe-flexibility practice, recalled)']),

    Calc('pipe_support_spacing', 'Pipe support spacing from bending-stress and deflection limits',
         vars=[V('w', 'N/m', 'pipe weight per unit length (pipe + content + insulation)'),
               V('D_o', 'm', 'pipe outside diameter'), V('t', 'm', 'pipe wall thickness'),
               V('E', 'Pa', 'pipe material elastic modulus'), V('S_allow', 'Pa', 'allowable bending stress'),
               V('delta_allow', 'm', 'allowable mid-span deflection'),
               V('span', '', 'span (support) condition', kind='text',
                 choices=['simply_supported', 'continuous']),
               V('I', 'm^4', 'pipe second moment of area'), V('Z', 'm^3', 'pipe section modulus'),
               V('L_stress', 'm', 'support spacing from the bending-stress limit'),
               V('L_defl', 'm', 'support spacing from the deflection limit'),
               V('L_max', 'm', 'maximum support spacing (the smaller of the two)')],
         procedure=_pipe_support_spacing, outputs=['I', 'Z', 'L_stress', 'L_defl', 'L_max'],
         example={'w': 500.0, 'D_o': 0.168, 't': 0.007, 'E': 200e9, 'S_allow': 138e6,
                  'delta_allow': 0.003, 'span': 'simply_supported'},
         keywords=['pipe support', 'span', 'sag', 'deflection'],
         assumptions=['Beam bending-stress and deflection limits for a uniform pipe span, with a span-condition '
                      'choice: simply supported (M = w·L²/8, δ = 5·w·L⁴/(384·E·I)) or continuous '
                      '(M = w·L²/10, δ = w·L⁴/(185·E·I)) — both recalled. The spacing is the smaller of the '
                      'two limits, with the governing limit reported as a warning.'],
         refs=['Beam bending-stress and deflection limits for a uniform pipe span (span constants recalled)']),
]

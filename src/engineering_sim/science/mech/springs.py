# DESCRIPTION: Springs section: helical compression/tension spring rate, stress-correction factors, shear
#              stress, deflection, geometry (solid/free length), buckling, surge frequency, extension
#              hook stresses, torsion/leaf/Belleville springs, fatigue safety and stored energy.
"""Spring calculators.

Conventions: d = wire diameter, D = mean coil diameter, C = spring index D/d, n_a = active coils. Shear
stresses use the corrected τ = K·8FD/(πd³) with the Wahl or Bergsträsser factor (both given). Extension
hook curvature factors (K)_A, (K)_B are computed from the hook radii (closed forms), not read from
charts. Torsion-spring active coils include the end-leg correction. The Belleville load is given for two
equivalent geometry constants: the DIN 2092 K1 and the original Almen–László M.

References are the standard spring relations in machine-design texts (Shigley, *Mechanical Engineering
Design*, ch. 10; Wahl, *Mechanical Springs*), cited by relation name and recalled rather than re-read.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts

SECTION = section('springs', 'Springs', order=150,
                  summary='Helical, extension, torsion, leaf and Belleville springs: rate, stress, '
                          'deflection, buckling, surge, fatigue and energy.')

PI = math.pi

# End-condition conventions (Shigley Table 10-1, recalled): (coils added to n_a, +1 on the solid-length
# count). plain: n_t = n_a, L_s = d(n_t + 1); plain_ground: n_t = n_a + 1, L_s = d·n_t;
# squared (closed): n_t = n_a + 2, L_s = d(n_t + 1); squared_ground: n_t = n_a + 2, L_s = d·n_t.
_END_OFFSET = {'plain': (0, 1), 'plain_ground': (1, 0), 'squared': (2, 1), 'squared_ground': (2, 0)}


def _spring_geometry(values):
    """Total coils, solid length and free length from active coils and the end condition."""
    d = values['d']
    na = values['n_a']
    end = values['end']
    if end not in _END_OFFSET:
        raise MechError(f'end: {end!r} is not one of {", ".join(sorted(_END_OFFSET))}')
    add, solid_extra = _END_OFFSET[end]
    nt = na + add
    ls = d * (nt + solid_extra)
    return {'n_t': nt, 'L_s': ls, 'L_f': ls + values['delta_max']}


def _spring_buckling(values):
    """Compression-spring buckling: critical deflection, absolute-stability length, stable flag.

    Shigley form δ_cr = C₁′·L0·[1 − √(1 − C₂′/λ_eff²)] with C₁′ = E/(2(E − G)),
    C₂′ = 2π²(E − G)/(2G + E) and λ_eff = α·L0/D. Buckling is possible only when
    L0 ≥ L0_crit = (πD/α)·√(2(E − G)/(2G + E)); for L0 < L0_crit the spring is absolutely
    stable and δ_cr has no real value (reported null, stable = 1).
    """
    L0, D = values['L0'], values['D']
    E, G = values['E'], values['G']
    alpha = values['alpha']
    if alpha <= 0:
        raise MechError('spring_buckling: alpha (end-condition constant) must be positive')
    if E <= G:
        raise MechError("spring_buckling: Young's modulus E must exceed the shear modulus G")
    c1 = E / (2.0 * (E - G))
    c2 = 2.0 * PI ** 2 * (E - G) / (2.0 * G + E)
    L0_crit = (PI * D / alpha) * math.sqrt(2.0 * (E - G) / (2.0 * G + E))
    if L0 < L0_crit:
        return {'L0_crit': L0_crit, 'delta_cr': None, 'stable': 1}
    lam = alpha * L0 / D
    delta_cr = c1 * L0 * (1.0 - math.sqrt(1.0 - c2 / lam ** 2))
    return {'L0_crit': L0_crit, 'delta_cr': delta_cr, 'stable': 0}


def _belleville_load(E, nu, Do, K, t, h, delta):
    """Almen–László load for a coned-disc (Belleville) spring at deflection delta."""
    return (4.0 * E * delta / ((1.0 - nu ** 2) * K * Do ** 2)) * \
        ((h - delta) * (h - delta / 2.0) * t + t ** 3)


def _belleville_k1(C):
    """DIN 2092 geometry constant K1 = (1/π)·((C−1)/C)²/((C+1)/(C−1) − 2/ln C)."""
    return (1.0 / PI) * ((C - 1.0) / C) ** 2 / ((C + 1.0) / (C - 1.0) - 2.0 / math.log(C))


def _belleville_m(C):
    """Original Almen–László geometry constant M = (6/(π·ln C))·((C−1)/C)² (C = D_o/D_i)."""
    return (6.0 / (PI * math.log(C))) * ((C - 1.0) / C) ** 2


def _belleville(values):
    """Belleville (disc) spring load–deflection (Almen–László theory).

    C = D_o/D_i; F = 4E·δ/((1−ν²)·K·D_o²)·[(h−δ)(h−δ/2)·t + t³] with K = K1 (DIN 2092) or
    M (original Almen–László) as selected by ``form``. Valid for 0 ≤ δ ≤ h (flat at δ = h).
    """
    E, nu = values['E'], values['nu']
    Do, Di, t, h = values['D_o'], values['D_i'], values['t'], values['h']
    delta = values['delta']
    form = values['form']
    if form not in ('din_2092', 'almen_laszlo'):
        raise MechError("form: must be 'din_2092' or 'almen_laszlo'")
    if Do <= Di:
        raise MechError('D_o must be greater than D_i (diameter ratio C = D_o/D_i must exceed 1)')
    if t <= 0 or h <= 0:
        raise MechError('t and h must be positive')
    if delta < 0:
        raise MechError('delta must be >= 0')
    if delta > h:
        raise MechError('delta must not exceed the cone height h (the washer is flat at delta = h)')
    C = Do / Di
    K = _belleville_k1(C) if form == 'din_2092' else _belleville_m(C)
    F = _belleville_load(E, nu, Do, K, t, h, delta)
    return {'C': C, 'K': K, 'F': F}


def _belleville_chart(vals):
    E, nu = vals['E'], vals['nu']
    Do, Di, t, h = vals['D_o'], vals['D_i'], vals['t'], vals['h']
    form = vals.get('form', 'din_2092')
    C = Do / Di
    K = _belleville_k1(C) if form == 'din_2092' else _belleville_m(C)
    xs = [h * i / 80.0 for i in range(81)]
    ys = [_belleville_load(E, nu, Do, K, t, h, d) for d in xs]
    return charts.chart('Belleville load–deflection (Almen–László)', 'deflection δ [m]', 'load F [N]',
                        [charts.series('load', xs, ys)],
                        markers=[charts.marker(h, ys[-1], 'flat (δ = h)')])


def _spring_fatigue(values):
    """Spring-wire fatigue: wire strength, torsional endurance and Goodman/Gerber safety factors.

    S_ut = A/d^m with d in mm and A in MPa·mm^m (wire-strength table, Shigley Table 10-4 form); in
    coherent SI this is S_ut = A·(1 mm/d)^m with A = S_ut at d = 1 mm (Pa). S_su = 0.67·S_ut.
    Zimmerli-type test data (S_sa, S_sm measured at a nonzero mean stress) convert to the fully
    reversed endurance with the SAME criterion that is then applied: Goodman S_se = S_sa/(1 − S_sm/S_su) and
    Gerber S_se,G = S_sa/(1 − (S_sm/S_su)²). Goodman and Gerber safety factors then follow for the working
    alternating/mean shear (τ_a, τ_m).
    """
    d, A, m = values['d'], values['A'], values['m']
    S_sa, S_sm = values['S_sa'], values['S_sm']
    tau_a, tau_m = values['tau_a'], values['tau_m']
    if d <= 0:
        raise MechError('spring_fatigue: wire diameter d must be positive')
    if A <= 0:
        raise MechError('spring_fatigue: strength coefficient A must be positive')
    S_ut = A * (1e-3 / d) ** m
    S_su = 0.67 * S_ut
    if S_sm >= S_su:
        raise MechError('spring_fatigue: S_sm must be less than S_su (0.67·S_ut) '
                        'for the Zimmerli conversion')
    if S_sa <= 0:
        raise MechError('spring_fatigue: S_sa must be positive')
    S_se = S_sa / (1.0 - S_sm / S_su)               # Goodman conversion
    S_se_gerber = S_sa / (1.0 - (S_sm / S_su) ** 2)  # Gerber conversion (consistent with the Gerber criterion)
    if tau_a < 0 or tau_m < 0:
        raise MechError('spring_fatigue: tau_a and tau_m must be >= 0')
    if tau_a == 0 and tau_m == 0:
        raise MechError('spring_fatigue: tau_a and tau_m cannot both be zero')
    n_goodman = 1.0 / (tau_a / S_se + tau_m / S_su)
    a = (tau_m / S_su) ** 2
    b = tau_a / S_se_gerber
    if a > 0:
        n_gerber = (math.sqrt(b * b + 4.0 * a) - b) / (2.0 * a)
    else:
        n_gerber = 1.0 / b if b > 0 else float('inf')
    return {'S_ut': S_ut, 'S_su': S_su, 'S_se': S_se, 'S_se_gerber': S_se_gerber,
            'n_goodman': n_goodman, 'n_gerber': n_gerber}


CALCS = [
    # ---- helical compression / tension ------------------------------------------------------------
    Calc('helical_spring_rate', 'Helical spring rate and index',
         vars=[V('k', 'N/m', 'spring rate'),
               V('C', '', 'spring index D/d', lo=4.0),
               V('G', 'Pa', 'shear modulus'),
               V('d', 'm', 'wire diameter'),
               V('D', 'm', 'mean coil diameter'),
               V('n_a', '', 'number of active coils')],
         equations=['k = G*d**4/(8*D**3*n_a)', 'C = D/d'], outputs=['k', 'C'],
         example={'G': 79.3e9, 'd': 0.004, 'D': 0.032, 'n_a': 10.0},
         keywords=['compression spring', 'stiffness', 'coil'],
         assumptions=['Round wire, linear elastic, mean coil diameter D constant. n_a is the number of '
                      'active coils (total minus the inactive end coils). C ≳ 4 is the usual design range.'],
         refs=['k = Gd⁴/(8D³n); C = D/d']),

    Calc('spring_stress', 'Helical spring shear stress with Wahl and Bergsträsser correction',
         vars=[V('C', '', 'spring index D/d', lo=4.0),
               V('K_w', '', 'Wahl correction factor'),
               V('K_b', '', 'Bergsträsser correction factor'),
               V('tau_w', 'Pa', 'corrected shear stress using Wahl factor', label='τw'),
               V('tau_b', 'Pa', 'corrected shear stress using Bergsträsser factor', label='τb'),
               V('F', 'N', 'axial force'),
               V('D', 'm', 'mean coil diameter'),
               V('d', 'm', 'wire diameter')],
         equations=['C = D/d',
                    'K_w = (4*C - 1)/(4*C - 4) + 0.615/C',
                    'K_b = (4*C + 2)/(4*C - 3)',
                    'tau_w = K_w*8*F*D/(pi*d**3)',
                    'tau_b = K_b*8*F*D/(pi*d**3)'],
         outputs=['C', 'K_w', 'K_b', 'tau_w', 'tau_b'],
         example={'F': 500.0, 'D': 0.032, 'd': 0.004},
         keywords=['wahl', 'bergstrasser', 'curvature', 'direct shear', 'correction factor'],
         assumptions=['Correction factors combine the direct (transverse) shear and the wire-curvature '
                      'effect. Naming varies between texts — (4C−1)/(4C−4) + 0.615/C is the Wahl factor '
                      'and (4C+2)/(4C−3) the (simpler) Bergsträsser factor; both are given.',
                      '8FD/(πd³) is the uncorrected shear stress; both corrected stresses are reported.'],
         refs=['Wahl and Bergsträsser stress-correction factors; τ = K·8FD/(πd³) (recalled)']),

    Calc('spring_deflection', 'Helical spring deflection under axial load',
         vars=[V('delta', 'm', 'deflection', label='δ'),
               V('C', '', 'spring index D/d', lo=1.0),
               V('F', 'N', 'axial force'),
               V('D', 'm', 'mean coil diameter'),
               V('d', 'm', 'wire diameter'),
               V('n_a', '', 'number of active coils'),
               V('G', 'Pa', 'shear modulus')],
         equations=['delta = 8*F*D**3*n_a/(G*d**4)', 'C = D/d'], outputs=['delta', 'C'],
         example={'F': 500.0, 'D': 0.032, 'd': 0.004, 'n_a': 10.0, 'G': 79.3e9},
         keywords=['deflection', 'travel'],
         assumptions=['Linear elastic; δ = F/k with k from helical_spring_rate. Physical helical '
                      'geometry needs the spring index C = D/d ≥ 1 (D ≥ d); C is reported so a '
                      'non-physical D < d is rejected rather than silently computed.'],
         refs=['δ = 8FD³n/(Gd⁴); C = D/d']),

    Calc('spring_geometry', 'Helical spring total coils, solid and free length',
         vars=[V('d', 'm', 'wire diameter'),
               V('n_a', '', 'number of active coils'),
               V('end', '', 'end condition', kind='text',
                 choices=['plain', 'plain_ground', 'squared', 'squared_ground']),
               V('delta_max', 'm', 'maximum deflection to solid (working travel)'),
               V('n_t', '', 'total coils'),
               V('L_s', 'm', 'solid length'),
               V('L_f', 'm', 'free length (solid + max deflection)')],
         procedure=_spring_geometry, outputs=['n_t', 'L_s', 'L_f'],
         example={'d': 0.004, 'n_a': 10.0, 'end': 'squared_ground', 'delta_max': 0.06},
         keywords=['solid length', 'free length', 'end condition', 'closed', 'ground'],
         assumptions=['End condition sets the total-coil count and solid length: plain n_t = n_a '
                      '(L_s = d(n_t+1)); plain-and-ground n_t = n_a+1 (L_s = d·n_t); squared (closed) '
                      'n_t = n_a+2 (L_s = d(n_t+1)); squared-and-ground n_t = n_a+2 (L_s = d·n_t).',
                      'L_f = L_s + δ_max is the minimum free length (solid length plus working travel); '
                      'it is not the pitch-based manufactured free length (which follows from the coil '
                      'pitch and end configuration).'],
         refs=['Solid/free length by end condition (Shigley Table 10-1, recalled)']),

    Calc('spring_buckling', 'Compression spring buckling (critical deflection and stability)',
         vars=[V('delta_cr', 'm', 'critical deflection at buckling onset (null when stable)', label='δcr'),
               V('L0_crit', 'm', 'absolute-stability length (no buckling below this)', label='L0,crit'),
               V('stable', '', 'stability flag (1 = absolutely stable, 0 = buckles)', lo=0, hi=1,
                 integer=True),
               V('L0', 'm', 'free length'),
               V('D', 'm', 'mean coil diameter'),
               V('E', 'Pa', "Young's modulus"),
               V('G', 'Pa', 'shear modulus'),
               V('alpha', '', 'end-condition constant (0.5 fixed-fixed, 0.707 hinged, 1.0 free, '
                 '2.0 fixed-free)', label='α')],
         procedure=_spring_buckling, outputs=['delta_cr', 'L0_crit', 'stable'],
         example={'L0': 0.3, 'D': 0.032, 'E': 207e9, 'G': 79.3e9, 'alpha': 0.5},
         keywords=['stability', 'slender', 'column'],
         assumptions=['Spring treated as a column with an equivalent flexural rigidity. δ_cr = C₁′·L0·'
                      '[1 − √(1 − C₂′/λ_eff²)] with C₁′ = E/(2(E−G)), C₂′ = 2π²(E−G)/(2G+E), '
                      'λ_eff = α·L0/D. Buckling is possible only for L0 ≥ L0_crit = (πD/α)·√(2(E−G)/(2G+E)); '
                      'a shorter spring is absolutely stable (stable = 1, δ_cr null). At exactly L0 = L0_crit '
                      'the critical deflection is C₁′·L0_crit.',
                      'Compression springs only; squared-and-ground ends ≈ fixed (α = 0.5).'],
         refs=['Spring buckling / stability criterion (Shigley sec. 10-4 form, recalled)']),

    Calc('spring_surge', 'Helical spring surge (natural) frequency',
         vars=[V('f', 'Hz', 'surge frequency, both ends fixed'),
               V('f_free', 'Hz', 'surge frequency, fixed–free (one end free)'),
               V('d', 'm', 'wire diameter'),
               V('D', 'm', 'mean coil diameter'),
               V('n_a', '', 'number of active coils'),
               V('G', 'Pa', 'shear modulus'),
               V('rho', 'kg/m^3', 'wire material density', label='ρ')],
         equations=['f = d*sqrt(G/(2*rho))/(2*pi*D**2*n_a)', 'f_free = f/2'],
         outputs=['f', 'f_free'],
         example={'d': 0.004, 'D': 0.032, 'n_a': 10.0, 'G': 79.3e9, 'rho': 7850.0},
         keywords=['natural frequency', 'surge', 'resonance', 'valve spring'],
         assumptions=['Fundamental surge mode of the spring alone (its own mass). f is for both ends '
                      'fixed, f = ½√(k/m_s) with m_s = ρ·π²d²D·n_a/4 (active-coil mass) and k = Gd⁴/(8D³n_a); '
                      'one end fixed and the other free gives f_free = ¼√(k/m_s) = f/2. With an attached '
                      'mass m, use f ≈ (1/2π)·√(k/(m + m_s/3)).'],
         refs=['Surge frequency f = (d/(2πD²n_a))·√(G/(2ρ)) (recalled)']),

    # ---- extension springs --------------------------------------------------------------------------
    Calc('extension_hook_stress', 'Extension spring hook stresses (sections A and B)',
         vars=[V('sigma_A', 'Pa', 'maximum stress at hook section A (bending + direct tension)', label='σA'),
               V('tau_B', 'Pa', 'maximum shear stress at hook section B (torsion)', label='τB'),
               V('C1', '', 'curvature index at A, 2·r1/d', lo=1.0),
               V('K_A', '', 'curvature correction factor at A'),
               V('C2', '', 'curvature index at B, 2·r2/d', lo=1.0),
               V('K_B', '', 'curvature correction factor at B'),
               V('F', 'N', 'spring load'),
               V('D', 'm', 'mean coil diameter'),
               V('d', 'm', 'wire diameter'),
               V('r1', 'm', 'inner radius of the hook loop at section A'),
               V('r2', 'm', 'inner radius of the hook loop at section B')],
         equations=['C1 = 2*r1/d',
                    'K_A = (4*C1**2 - C1 - 1)/(4*C1*(C1 - 1))',
                    'sigma_A = K_A*16*F*D/(pi*d**3) + 4*F/(pi*d**2)',
                    'C2 = 2*r2/d',
                    'K_B = (4*C2 - 1)/(4*C2 - 4)',
                    'tau_B = K_B*8*F*D/(pi*d**3)'],
         outputs=['sigma_A', 'tau_B', 'C1', 'K_A', 'C2', 'K_B'],
         example={'F': 50.0, 'D': 0.02, 'd': 0.003, 'r1': 0.006, 'r2': 0.008},
         keywords=['extension spring', 'hook', 'end loop'],
         assumptions=['Section A is where the hook joins the coil body: bending moment F·D/2 (plus the '
                      'direct tension F) with the curvature factor K_A = (4C₁²−C₁−1)/(4C₁(C₁−1)), '
                      'C₁ = 2r₁/d. Section B is the hook end loaded in torsion (torque F·D/2) with '
                      'K_B = (4C₂−1)/(4C₂−4), C₂ = 2r₂/d. Requires C₁, C₂ > 1 (hook bend radius larger '
                      'than the wire radius).'],
         refs=['Extension-spring hook bending and torsion; curvature factors (Shigley sec. 10-9 forms, '
               'recalled)']),

    # ---- torsion / leaf / Belleville ---------------------------------------------------------------
    Calc('torsion_spring', 'Helical torsion spring rate, deflection and stress',
         vars=[V('k_t', 'N*m/rad', 'ideal torsional spring rate per radian'),
               V('k_turn', 'N*m/turn', 'ideal torsional spring rate per turn (2π·k_t)'),
               V('k_prime', 'N*m/turn', 'empirical rate per turn with friction allowance (Shigley)'),
               V('theta', 'rad', 'angular deflection', signed=True, label='θ'),
               V('C', '', 'spring index D/d', lo=4.0),
               V('K_i', '', 'inner-surface stress-correction factor'),
               V('sigma', 'Pa', 'maximum bending stress at the inner surface', signed=True, label='σ'),
               V('N_a', '', 'active coils (body + end-leg correction)'),
               V('M', 'N*m', 'applied moment (torque) at the spring ends', signed=True),
               V('d', 'm', 'wire diameter'),
               V('D', 'm', 'mean coil diameter'),
               V('N_b', '', 'number of body coils'),
               V('l1', 'm', 'length of end leg 1'),
               V('l2', 'm', 'length of end leg 2'),
               V('E', 'Pa', "Young's modulus")],
         equations=['N_a = N_b + (l1 + l2)/(3*pi*D)',
                    'C = D/d',
                    'K_i = (4*C**2 - C - 1)/(4*C*(C - 1))',
                    'k_t = E*d**4/(64*D*N_a)',
                    'k_turn = 2*pi*k_t',
                    'k_prime = E*d**4/(10.8*D*N_a)',
                    'theta = M/k_t',
                    'sigma = K_i*32*M/(pi*d**3)'],
         outputs=['k_t', 'k_turn', 'k_prime', 'theta', 'C', 'K_i', 'sigma', 'N_a'],
         example={'M': 2.0, 'd': 0.004, 'D': 0.03, 'N_b': 5.0, 'l1': 0.04, 'l2': 0.025, 'E': 207e9},
         keywords=['torsion spring', 'clock spring', 'hinge'],
         assumptions=['Wire in bending (coils wind/unwind). k_t = Ed⁴/(64·D·N_a) N·m/rad is the ideal '
                      'straight-wire rate per radian (θ = M·L_w/(EI), L_w = π·D·N_a); the ideal per-turn '
                      'rate is k_turn = 2π·k_t = Ed⁴/(10.19·D·N_a) N·m/turn. Shigley\'s empirical rate '
                      'k_prime = Ed⁴/(10.8·D·N_a) N·m/turn adds a friction allowance. Active coils include '
                      'the end legs: N_a = N_b + (l₁ + l₂)/(3πD). K_i = (4C² − C − 1)/(4C(C−1)) corrects '
                      'the inner (compressive) surface; the outer surface is tensile for a spring wound '
                      'to tighten.'],
         refs=['Torsion spring rate θ = M·L_w/(EI) (straight-wire bending); per-turn rates; end-leg coil '
               'correction; inner-fibre stress factor (Shigley sec. 10-6 forms, recalled)']),

    Calc('leaf_spring', 'Multi-leaf (semi-elliptic) leaf spring deflection and stress',
         vars=[V('sigma', 'Pa', 'maximum bending stress', label='σ'),
               V('delta', 'm', 'central deflection', label='δ'),
               V('F', 'N', 'central load (total)'),
               V('L', 'm', 'span (eye-to-eye)'),
               V('N', '', 'number of leaves', integer=True, lo=1),
               V('b', 'm', 'leaf width'),
               V('t', 'm', 'leaf thickness'),
               V('E', 'Pa', "Young's modulus")],
         equations=['sigma = 3*F*L/(2*N*b*t**2)', 'delta = 3*F*L**3/(8*E*N*b*t**3)'],
         outputs=['sigma', 'delta'],
         example={'F': 3000.0, 'L': 1.0, 'N': 6, 'b': 0.05, 't': 0.008, 'E': 207e9},
         keywords=['laminated spring', 'semi-elliptic', 'suspension'],
         assumptions=['Graduated (equal-stress) leaf idealisation with N leaves sliding without friction '
                      'and equal thickness t. The 3/8 deflection factor is the uniform-strength result '
                      '(1.5× the constant-section beam value); σ is the leaf-surface stress.'],
         refs=['σ = 3FL/(2Nbt²); δ = 3FL³/(8ENbt³) (standard graduated-leaf forms, recalled)']),

    Calc('belleville', 'Belleville (disc) spring load and deflection',
         vars=[V('E', 'Pa', "Young's modulus"),
               V('nu', '', "Poisson's ratio", hi=0.5, label='ν'),
               V('D_o', 'm', 'outside diameter'),
               V('D_i', 'm', 'inside diameter'),
               V('t', 'm', 'disc thickness'),
               V('h', 'm', 'cone height (free height minus thickness)'),
               V('delta', 'm', 'deflection (0 ≤ δ ≤ h)', label='δ'),
               V('form', '', 'geometry constant: K1 (DIN 2092) or M (original Almen–László)',
                 kind='text', choices=['din_2092', 'almen_laszlo']),
               V('C', '', 'diameter ratio D_o/D_i'),
               V('K', '', 'geometry constant (K1 for DIN 2092, M for original Almen–László)'),
               V('F', 'N', 'load at the given deflection')],
         procedure=_belleville, outputs=['F', 'C', 'K'],
         example={'E': 206e9, 'nu': 0.3, 'D_o': 0.1, 'D_i': 0.05, 't': 0.004, 'h': 0.006, 'delta': 0.003,
                  'form': 'din_2092'},
         plots=[{'id': 'load_deflection', 'title': 'Belleville load–deflection', 'fn': _belleville_chart}],
         keywords=['disc spring', 'coned disc', 'almen laszlo', 'washer spring', 'din 2092'],
         assumptions=['Almen–László load–deflection for a coned disc: F = 4E·δ/((1−ν²)·K·D_o²)·'
                      '[(h−δ)(h−δ/2)·t + t³]. The geometry constant K is either K1 (DIN 2092) '
                      '= (1/π)((C−1)/C)²/((C+1)/(C−1) − 2/ln C) or the original Almen–László '
                      'M = (6/(π·ln C))·((C−1)/C)² — the same theory with two conventions for the neutral '
                      'circle; they differ by under 1 % for typical C. Valid for 0 ≤ δ ≤ h (flat at δ = h; '
                      'flattening then snap-through beyond the flat position is not modelled). The '
                      'load–deflection curve is non-monotonic (snap-through) when h/t ≳ √2.'],
         refs=['Almen–László disc-spring equation: F = 4Eδ/((1−ν²)K·D_o²)·[(h−δ)(h−δ/2)t + t³], '
               'DIN 2092 K1 and original Almen–László M (recalled)']),

    # ---- fatigue and energy -------------------------------------------------------------------------
    Calc('spring_energy', 'Elastic energy stored in a spring',
         vars=[V('U', 'J', 'stored energy'),
               V('F', 'N', 'force'),
               V('delta', 'm', 'deflection', label='δ')],
         equations=['U = F*delta/2'], outputs=['U'],
         example={'F': 500.0, 'delta': 0.0646},
         keywords=['stored energy', 'potential energy'],
         assumptions=['Linear spring: U = F·δ/2 = k·δ²/2 = F²/(2k).'],
         refs=['U = Fδ/2 (area under the linear force–deflection curve)']),

    Calc('spring_fatigue', 'Spring-wire fatigue safety factors (Goodman and Gerber)',
         vars=[V('S_ut', 'Pa', 'ultimate tensile strength of the wire (S_ut = A/d^m)'),
               V('S_su', 'Pa', 'ultimate torsional strength (0.67·S_ut)'),
               V('S_se', 'Pa', 'fully reversed torsional endurance, Goodman conversion of Zimmerli-type data'),
               V('S_se_gerber', 'Pa', 'fully reversed torsional endurance, Gerber conversion S_sa/(1 − (S_sm/S_su)²)'),
               V('n_goodman', '', 'Goodman factor of safety'),
               V('n_gerber', '', 'Gerber factor of safety'),
               V('d', 'm', 'wire diameter'),
               V('A', 'Pa', 'wire strength coefficient (S_ut at d = 1 mm; Shigley Table 10-4 A/d^m)'),
               V('m', '', 'wire strength exponent (positive; Shigley Table 10-4)'),
               V('S_sa', 'Pa', 'torsional endurance amplitude from Zimmerli-type data'),
               V('S_sm', 'Pa', 'mean stress of the Zimmerli-type data'),
               V('tau_a', 'Pa', 'alternating shear stress (τmax − τmin)/2', label='τa'),
               V('tau_m', 'Pa', 'mean shear stress (τmax + τmin)/2', label='τm')],
         procedure=_spring_fatigue, outputs=['S_ut', 'S_su', 'S_se', 'S_se_gerber', 'n_goodman', 'n_gerber'],
         example={'d': 0.002, 'A': 2211e6, 'm': 0.145, 'S_sa': 241e6, 'S_sm': 379e6,
                  'tau_a': 100e6, 'tau_m': 150e6},
         keywords=['endurance', 'goodman', 'gerber', 'zimmerli', 'peened', 'alternating stress'],
         assumptions=['Wire strength S_ut = A/d^m with d in mm and A in MPa·mm^m (Shigley Table 10-4 '
                      'form); enter A as the strength at d = 1 mm (Pa) and m the table exponent. '
                      'S_su = 0.67·S_ut. Zimmerli-type test data (S_sa, S_sm — e.g. unpeened ≈ 241/379 MPa, '
                      'peened ≈ 398/534 MPa — are user inputs, not built in) convert to the fully reversed '
                      'endurance S_se = S_sa/(1 − S_sm/S_su). Compute τ_a, τ_m from the load range with '
                      'spring_stress (τ = K·8FD/(πd³) at F_max and F_min). Goodman: 1/n = τ_a/S_se + τ_m/S_su; '
                      'Gerber: n·τ_a/S_se,G + (n·τ_m/S_su)² = 1 with the Gerber-converted S_se,G (solve the quadratic).'],
         refs=['Wire strength S_ut = A/d^m (Shigley Table 10-4 form, recalled); S_su = 0.67·S_ut; '
               'Zimmerli-type fully-reversed conversion S_se = S_sa/(1 − S_sm/S_su); Goodman and Gerber '
               'in torsion (recalled)']),
]

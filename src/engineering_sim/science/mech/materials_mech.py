# DESCRIPTION: Mechanics of materials section (pilot / reference pattern): stress, strain, elastic
#              constants, axial/thermal/torsional deformation, bending and shear, stress transformation
#              and Mohr's circle, failure criteria, vessels, Lamé, rotating disks, curved and composite
#              beams, Hertz contact, stress concentration, energy methods, impact and plasticity.
"""Mechanics of materials calculators.

Sign conventions: tension positive; plane stress components σx, σy, τxy follow the usual right-hand
element convention (τxy positive on the +x face in +y); angles CCW positive from +x. Principal stresses
are ordered σ1 ≥ σ2 (≥ σ3). Failure-criterion factors of safety n are load factors (proportional loading).

Relations are the standard ones in mechanics-of-materials and machine-design texts (e.g. Beer & Johnston,
*Mechanics of Materials*; Budynas & Nisbett, *Shigley's Mechanical Engineering Design*; Young & Budynas,
*Roark's Formulas for Stress and Strain*). They are cited by relation name, recalled rather than re-read.
"""
import math

import numpy as np

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts

SECTION = section('materials_mech', 'Mechanics of materials', order=40,
                  summary='Stress, strain, torsion, bending, transformation, failure theories, vessels, '
                          'contact, stress concentration, energy and plasticity.')

PI = math.pi
# Reusable variable declarations (fresh objects per calculator are created by the helpers).


def E_(desc='elastic (Young\'s) modulus'):
    return V('E', 'Pa', desc)


def nu_(desc='Poisson\'s ratio'):
    return V('nu', '', desc, hi=0.5, label='ν')


def sig(name, desc, label=None):
    return V(name, 'Pa', desc, signed=True, label=label)


# ---- procedures ---------------------------------------------------------------------------------
def _principal_3d(v):
    """Eigenvalues of the symmetric Cauchy stress tensor, sorted σ1 ≥ σ2 ≥ σ3."""
    s = np.array([[v['sx'], v['txy'], v['txz']],
                  [v['txy'], v['sy'], v['tyz']],
                  [v['txz'], v['tyz'], v['sz']]], dtype=float)
    w = np.sort(np.linalg.eigvalsh(s))[::-1]
    s1, s2, s3 = map(float, w)
    vm = math.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2))
    return {'s1': s1, 's2': s2, 's3': s3, 'tau_abs_max': (s1 - s3) / 2, 'sigma_vm': vm}


def _failure(v):
    """Factors of safety from the principal stresses of a plane-stress state (third principal = 0).

    Ductile: von Mises (distortion energy) and Tresca (maximum shear) against S_y.
    Brittle / unequal strength: Rankine (maximum normal), Coulomb–Mohr and modified Mohr against S_ut, S_uc.
    σA, σB are the two in-plane principal stresses with σA ≥ σB.
    """
    sA, sB = max(v['sA'], v['sB']), min(v['sA'], v['sB'])
    Sy, St, Sc = v['S_y'], v['S_ut'], v['S_uc']
    inf = float('inf')
    vm = math.sqrt(sA ** 2 - sA * sB + sB ** 2)
    p = sorted([sA, sB, 0.0], reverse=True)
    tresca_span = p[0] - p[2]
    out = {'n_vm': Sy / vm if vm > 0 else inf, 'n_tresca': Sy / tresca_span if tresca_span > 0 else inf}
    # Rankine: whichever of tension/compression limit is reached first.
    cand = []
    if sA > 0:
        cand.append(St / sA)
    if sB < 0:
        cand.append(Sc / -sB)
    out['n_rankine'] = min(cand) if cand else inf

    def cm():
        if sA >= 0 and sB >= 0:
            return St / sA if sA > 0 else inf
        if sA <= 0 and sB <= 0:
            return Sc / -sB if sB < 0 else inf
        return 1.0 / (sA / St - sB / Sc)

    def mm():
        if sA >= 0 and sB >= 0:
            return St / sA if sA > 0 else inf
        if sA <= 0 and sB <= 0:
            return Sc / -sB if sB < 0 else inf
        if abs(sB) <= sA:  # |σB/σA| ≤ 1: tension governs
            return St / sA
        return 1.0 / ((Sc - St) * sA / (Sc * St) - sB / Sc)
    out['n_cm'] = cm()
    out['n_mm'] = mm()
    return out


def _composite_beam(v):
    """Transformed-section bending of a stacked rectangular composite beam.

    layers: bottom-to-top [[b (m), h (m), E (Pa)], ...]; M bending moment (N·m, sagging positive, puts the
    top in compression). Each layer is transformed to the reference modulus E_ref by width n·b with
    n = E/E_ref; stress in a layer at height y from the neutral axis is σ = −n M y / I_t.
    """
    layers = v['layers']
    M, Eref = v['M'], v['E_ref']
    y0 = 0.0
    rows = []
    for b, h, E in layers:
        n = E / Eref
        rows.append((n * b, h, y0, n))
        y0 += h
    A = sum(bt * h for bt, h, _, _ in rows)
    ybar = sum(bt * h * (yb + h / 2) for bt, h, yb, _ in rows) / A
    I = sum(bt * h ** 3 / 12 + bt * h * (yb + h / 2 - ybar) ** 2 for bt, h, yb, _ in rows)
    stresses = []
    for (bt, h, yb, n) in rows:
        for y in (yb, yb + h):
            stresses.append([y, -n * M * (y - ybar) / I])
    return {'y_na': ybar, 'I_t': I, 'stress_profile': stresses,
            'sigma_max_abs': max(abs(s) for _, s in stresses)}


def _mohr_chart(v):
    sx, sy, t = v['sx'], v['sy'], v['txy']
    c = (sx + sy) / 2
    R = math.hypot((sx - sy) / 2, t)
    return charts.chart("Mohr's circle (plane stress)", 'σ [Pa]', 'τ [Pa] (clockwise-positive plotting)',
                        [charts.circle('circle', c, 0, R),
                         charts.series('x–y diameter', [sx, sy], [t, -t])],
                        markers=[charts.marker(sx, t, 'x face'), charts.marker(sy, -t, 'y face'),
                                 charts.marker(c + R, 0, 'σ1'), charts.marker(c - R, 0, 'σ2'),
                                 charts.marker(c, R, 'τmax')], equal=True, hlines=[0])


CALCS = [
    # ---- basic stress and strain ---------------------------------------------------------------
    Calc('normal_stress', 'Average normal stress (σ = P/A)',
         vars=[sig('sigma', 'normal stress', 'σ'), V('P', 'N', 'axial force (tension +)', signed=True),
               V('A', 'm^2', 'cross-sectional area')],
         equations=['sigma = P/A'], outputs=['sigma'], example={'P': 25e3, 'A': 5e-4},
         keywords=['axial stress', 'tensile', 'compressive'],
         assumptions=['Uniform stress: load through the centroid, away from load application points '
                      '(Saint-Venant) and discontinuities.'],
         refs=['σ = P/A']),

    Calc('shear_stress', 'Average direct shear stress (τ = V/A)',
         vars=[V('tau', 'Pa', 'average shear stress', label='τ'), V('V', 'N', 'shear force'),
               V('A', 'm^2', 'shear area (sum of sheared sections, e.g. 2 for double shear)')],
         equations=['tau = V/A'], outputs=['tau'], example={'V': 8e3, 'A': 2 * PI * 0.005 ** 2},
         keywords=['pin', 'rivet', 'single shear', 'double shear'],
         refs=['τ_avg = V/A']),

    Calc('bearing_stress', 'Bearing stress on a pin or bolt hole (σ_b = P/(t·d))',
         vars=[V('sigma_b', 'Pa', 'average bearing stress', label='σb'), V('P', 'N', 'load transferred'),
               V('t', 'm', 'plate thickness'), V('d', 'm', 'pin diameter')],
         equations=['sigma_b = P/(t*d)'], outputs=['sigma_b'], example={'P': 8e3, 't': 0.006, 'd': 0.01},
         refs=['Projected-area bearing stress']),

    Calc('normal_strain', 'Normal strain (ε = δ/L)',
         vars=[V('eps', '', 'normal strain', signed=True, label='ε'), V('delta', 'm', 'elongation', signed=True, label='δ'),
               V('L', 'm', 'original length')],
         equations=['eps = delta/L'], outputs=['eps'], example={'delta': 0.0006, 'L': 1.2},
         refs=['Engineering strain']),

    Calc('elastic_constants', 'Isotropic elastic constants (G, K from E, ν)',
         vars=[V('G', 'Pa', 'shear modulus'), V('K', 'Pa', 'bulk modulus'), E_(), nu_()],
         equations=['G = E/(2*(1 + nu))', 'K = E/(3*(1 - 2*nu))'], outputs=['G', 'K'],
         example={'E': 200e9, 'nu': 0.29},
         keywords=['modulus of rigidity', 'bulk modulus', 'poisson'],
         assumptions=['Linear elastic, isotropic material; ν < 0.5.'],
         refs=['G = E/2(1+ν); K = E/3(1−2ν)']),

    Calc('hooke_3d', "Generalized Hooke's law (normal strains from triaxial stress)",
         vars=[V('eps_x', '', 'εx', signed=True), V('eps_y', '', 'εy', signed=True), V('eps_z', '', 'εz', signed=True),
               sig('sx', 'σx'), sig('sy', 'σy'), sig('sz', 'σz'), E_(), nu_()],
         equations=['eps_x = (sx - nu*(sy + sz))/E', 'eps_y = (sy - nu*(sx + sz))/E', 'eps_z = (sz - nu*(sx + sy))/E'],
         outputs=['eps_x', 'eps_y', 'eps_z'], example={'sx': 100e6, 'sy': -40e6, 'sz': 20e6, 'E': 200e9, 'nu': 0.3},
         refs=["Generalized Hooke's law (isotropic)"]),

    Calc('volumetric_strain', 'Volumetric (dilatation) strain',
         vars=[V('e_v', '', 'volumetric strain ΔV/V', signed=True), sig('sx', 'σx'), sig('sy', 'σy'), sig('sz', 'σz'), E_(), nu_()],
         equations=['e_v = (1 - 2*nu)*(sx + sy + sz)/E'], outputs=['e_v'],
         example={'sx': 100e6, 'sy': -40e6, 'sz': 20e6, 'E': 200e9, 'nu': 0.3},
         keywords=['dilatation'], assumptions=['Small strain: e = εx + εy + εz.'],
         refs=['Dilatation e = (1−2ν)(σx+σy+σz)/E']),

    Calc('shear_strain', 'Shear strain from shear stress (γ = τ/G)',
         vars=[V('gamma', '', 'shear strain (rad)', signed=True, label='γ'), sig('tau', 'shear stress', 'τ'),
               V('G', 'Pa', 'shear modulus')],
         equations=['gamma = tau/G'], outputs=['gamma'], example={'tau': 60e6, 'G': 77e9},
         refs=["Hooke's law in shear"]),

    # ---- deformation ------------------------------------------------------------------------------
    Calc('axial_deformation', 'Axial deformation (δ = PL/AE)',
         vars=[V('delta', 'm', 'elongation', signed=True, label='δ'), V('P', 'N', 'axial force (tension +)', signed=True),
               V('L', 'm', 'length'), V('A', 'm^2', 'area'), E_()],
         equations=['delta = P*L/(A*E)'], outputs=['delta'], example={'P': 50e3, 'L': 2.0, 'A': 4e-4, 'E': 200e9},
         keywords=['elongation', 'axial stiffness'], refs=['δ = PL/AE']),

    Calc('thermal_deformation', 'Free thermal deformation and fully restrained thermal stress',
         vars=[V('delta_T', 'm', 'free thermal elongation', signed=True, label='δT'),
               sig('sigma_T', 'stress if both ends fully restrained (heating → compression, negative)', 'σT'),
               V('alpha', '1/K', 'coefficient of thermal expansion', label='α'),
               V('dT', 'K', 'temperature change (delta, K = °C difference)', signed=True, label='ΔT'),
               V('L', 'm', 'length'), E_()],
         equations=['delta_T = alpha*dT*L', 'sigma_T = -E*alpha*dT'], outputs=['delta_T', 'sigma_T'],
         example={'alpha': 12e-6, 'dT': 60.0, 'L': 3.0, 'E': 200e9},
         keywords=['thermal stress', 'expansion'],
         assumptions=['Uniform ΔT, α constant, elastic. Enter ΔT in K or delta_degC.'],
         refs=['δT = αΔTL; σ = −EαΔT for full restraint']),

    Calc('torsion_circular', 'Torsion of a solid or hollow circular shaft',
         vars=[V('tau_max', 'Pa', 'maximum shear stress (outer surface)', label='τmax'),
               V('theta', 'rad', 'angle of twist', label='θ'), V('J', 'm^4', 'polar second moment of area'),
               V('T', 'N*m', 'torque'), V('L', 'm', 'length'), V('G', 'Pa', 'shear modulus'),
               V('d_o', 'm', 'outer diameter'), V('d_i', 'm', 'inner diameter (0 for solid)')],
         equations=['tau_max = T*(d_o/2)/J', 'theta = T*L/(G*J)', 'J = pi*(d_o**4 - d_i**4)/32'],
         outputs=['tau_max', 'theta', 'J'], example={'T': 500.0, 'L': 1.0, 'G': 79.3e9, 'd_o': 0.04, 'd_i': 0.0},
         keywords=['shaft', 'twist', 'polar moment'],
         assumptions=['Linear elastic, circular section, plane sections remain plane; away from ends/steps.'],
         refs=['τ = Tr/J; θ = TL/GJ']),

    Calc('shaft_power', 'Shaft power, torque and speed (P = Tω)',
         vars=[V('P', 'W', 'power'), V('T', 'N*m', 'torque'),
               V('omega', 'rad/s', 'angular speed (enter rad/s, rpm or rps)', label='ω')],
         equations=['P = T*omega'], outputs=['P'], example={'T': 150.0, 'omega': 1500 * 2 * PI / 60},
         keywords=['rpm', 'horsepower', 'transmitted power'],
         assumptions=['ω is angular: 1500 rpm = 157.08 rad/s. A Hz value is refused because pint would treat '
                      '1 Hz as 1 rad/s; enter rps instead.'],
         refs=['P = Tω']),

    Calc('torsion_thin_closed', 'Torsion of a thin-walled closed section (Bredt–Batho)',
         vars=[V('tau', 'Pa', 'shear stress at thickness t', label='τ'), V('theta', 'rad', 'angle of twist', label='θ'),
               V('T', 'N*m', 'torque'), V('t', 'm', 'wall thickness (uniform)'),
               V('A_m', 'm^2', 'area enclosed by the wall mid-line'), V('s', 'm', 'mid-line perimeter'),
               V('L', 'm', 'length'), V('G', 'Pa', 'shear modulus')],
         equations=['tau = T/(2*t*A_m)', 'theta = T*L*s/(4*A_m**2*G*t)'], outputs=['tau', 'theta'],
         example={'T': 2000.0, 't': 0.003, 'A_m': 0.1 * 0.05, 's': 0.3, 'L': 2.0, 'G': 26e9},
         keywords=['shear flow', 'tube', 'box section'],
         assumptions=['Thin wall (t ≪ section size), uniform t; for varying t use ∮ds/t in place of s/t.'],
         refs=['Bredt–Batho: q = T/2A_m; θ = TL∮ds/t /(4A_m²G)']),

    Calc('torsion_rectangle', 'Torsion of a solid rectangular bar (Roark approximation)',
         vars=[V('tau_max', 'Pa', 'maximum shear stress (mid long side)', label='τmax'),
               V('K_t', 'm^4', 'torsion constant'), V('theta', 'rad', 'angle of twist', label='θ'),
               V('T', 'N*m', 'torque'), V('w', 'm', 'long side'), V('t', 'm', 'short side (t ≤ w)'),
               V('L', 'm', 'length'), V('G', 'Pa', 'shear modulus')],
         equations=['K_t = w*t**3*(1/3 - 0.21*(t/w)*(1 - (t/w)**4/12))', 'tau_max = T*(3*w + 1.8*t)/(w**2*t**2)',
                    'theta = T*L/(G*K_t)'],
         outputs=['K_t', 'tau_max', 'theta'], example={'T': 100.0, 'w': 0.05, 't': 0.02, 'L': 1.0, 'G': 79.3e9},
         keywords=['non-circular', 'saint-venant', 'bar'],
         assumptions=['Approximate formulas; within a few percent for all aspect ratios (square: K = 0.1408 t⁴ vs '
                      'exact 0.1406; τ = 4.8T/t³ vs 4.81). Requires t ≤ w.'],
         refs=["Roark's Formulas for Stress and Strain, rectangular section torsion (recalled form)"]),

    # ---- bending and shear -------------------------------------------------------------------------
    Calc('bending_stress', 'Flexure formula (σ = My/I)',
         vars=[sig('sigma', 'bending stress at distance y', 'σ'), V('M', 'N*m', 'bending moment', signed=True),
               V('y', 'm', 'distance from neutral axis', signed=True), V('I', 'm^4', 'second moment of area about the neutral axis')],
         equations=['sigma = M*y/I'], outputs=['sigma'], example={'M': 12e3, 'y': 0.1, 'I': 8e-5},
         keywords=['flexure', 'section modulus'],
         assumptions=['Straight, linearly elastic, symmetric bending about a principal axis; magnitude only — '
                      'assign tension/compression side from the moment sense.'],
         refs=['σ = My/I']),

    Calc('beam_shear_stress', 'Transverse shear stress and shear flow (τ = VQ/Ib, q = VQ/I)',
         vars=[V('tau', 'Pa', 'shear stress at the cut', label='τ'), V('q', 'N/m', 'shear flow'),
               V('V', 'N', 'shear force'), V('Q', 'm^3', 'first moment of the area beyond the cut about the NA'),
               V('I', 'm^4', 'second moment of the whole section'), V('b', 'm', 'width at the cut')],
         equations=['tau = V*Q/(I*b)', 'q = V*Q/I'], outputs=['tau', 'q'],
         example={'V': 20e3, 'Q': 1.5e-4, 'I': 8e-5, 'b': 0.01},
         keywords=['shear flow', 'weld', 'fastener spacing', 'jourawski'],
         refs=['Jourawski shear formula']),

    Calc('combined_axial_bending', 'Combined axial load and bending (extreme fibres)',
         vars=[sig('sigma_max', 'maximum fibre stress', 'σmax'), sig('sigma_min', 'minimum fibre stress', 'σmin'),
               V('P', 'N', 'axial force (tension +)', signed=True), V('A', 'm^2', 'area'),
               V('M', 'N*m', 'bending moment magnitude'), V('c', 'm', 'extreme fibre distance'), V('I', 'm^4', 'second moment')],
         equations=['sigma_max = P/A + M*c/I', 'sigma_min = P/A - M*c/I'], outputs=['sigma_max', 'sigma_min'],
         example={'P': 40e3, 'A': 4e-3, 'M': 5e3, 'c': 0.1, 'I': 1.2e-5},
         keywords=['eccentric load', 'superposition'],
         assumptions=['Symmetric section (same c on both sides); small deflections (no P-δ amplification).'],
         refs=['Superposition σ = P/A ± Mc/I']),

    # ---- transformation ---------------------------------------------------------------------------
    Calc('stress_transform', 'Plane stress transformation to a rotated element',
         vars=[sig('sx1', "σx' on the rotated face"), sig('sy1', "σy'"), sig('txy1', "τx'y'"),
               sig('sx', 'σx'), sig('sy', 'σy'), sig('txy', 'τxy'),
               V('theta', 'rad', 'rotation CCW from x', lo=-PI, hi=PI, label='θ')],
         equations=['sx1 = (sx + sy)/2 + (sx - sy)/2*cos(2*theta) + txy*sin(2*theta)',
                    'sy1 = (sx + sy)/2 - (sx - sy)/2*cos(2*theta) - txy*sin(2*theta)',
                    'txy1 = -(sx - sy)/2*sin(2*theta) + txy*cos(2*theta)'],
         outputs=['sx1', 'sy1', 'txy1'], example={'sx': 80e6, 'sy': -40e6, 'txy': 25e6, 'theta': math.radians(30)},
         refs=['Plane stress transformation equations']),

    Calc('principal_stress', "Principal stresses, maximum in-plane shear and Mohr's circle",
         vars=[sig('s1', 'σ1 (max in-plane principal)'), sig('s2', 'σ2 (min in-plane principal)'),
               V('tau_max', 'Pa', 'maximum in-plane shear', label='τmax'),
               V('theta_p', 'rad', 'angle from x to σ1 direction', signed=True, label='θp'),
               sig('sx', 'σx'), sig('sy', 'σy'), sig('txy', 'τxy')],
         equations=['s1 = (sx + sy)/2 + sqrt(((sx - sy)/2)**2 + txy**2)',
                    's2 = (sx + sy)/2 - sqrt(((sx - sy)/2)**2 + txy**2)',
                    'tau_max = sqrt(((sx - sy)/2)**2 + txy**2)',
                    'theta_p = atan2(2*txy, sx - sy)/2'],
         outputs=['s1', 's2', 'tau_max', 'theta_p'], example={'sx': 80e6, 'sy': -40e6, 'txy': 25e6},
         plots=[{'id': 'mohr', 'title': "Mohr's circle", 'fn': _mohr_chart}],
         keywords=['mohr', 'principal angle'],
         assumptions=['In-plane only; for plane stress the third principal stress is 0, so the absolute maximum '
                      'shear is max(τmax, |σ1|/2, |σ2|/2).'],
         refs=['Principal stresses of plane stress; Mohr circle']),

    Calc('principal_stress_3d', 'Principal stresses of a general 3-D stress state',
         vars=[sig('sx', 'σx'), sig('sy', 'σy'), sig('sz', 'σz'), sig('txy', 'τxy'), sig('tyz', 'τyz'), sig('txz', 'τxz'),
               sig('s1', 'σ1'), sig('s2', 'σ2'), sig('s3', 'σ3'),
               V('tau_abs_max', 'Pa', 'absolute maximum shear (σ1−σ3)/2'),
               V('sigma_vm', 'Pa', 'von Mises equivalent stress')],
         procedure=_principal_3d, outputs=['s1', 's2', 's3', 'tau_abs_max', 'sigma_vm'],
         example={'sx': 50e6, 'sy': -20e6, 'sz': 10e6, 'txy': 30e6, 'tyz': 5e6, 'txz': -10e6},
         keywords=['eigenvalue', 'stress tensor', 'invariants', 'von mises'],
         refs=['Eigenvalues of the Cauchy stress tensor']),

    Calc('principal_strain', 'Principal strains and maximum in-plane shear strain',
         vars=[V('e1', '', 'ε1', signed=True), V('e2', '', 'ε2', signed=True), V('gamma_max', '', 'maximum in-plane shear strain γmax'),
               V('eps_x', '', 'εx', signed=True), V('eps_y', '', 'εy', signed=True), V('gamma_xy', '', 'engineering shear strain γxy', signed=True)],
         equations=['e1 = (eps_x + eps_y)/2 + sqrt(((eps_x - eps_y)/2)**2 + (gamma_xy/2)**2)',
                    'e2 = (eps_x + eps_y)/2 - sqrt(((eps_x - eps_y)/2)**2 + (gamma_xy/2)**2)',
                    'gamma_max = 2*sqrt(((eps_x - eps_y)/2)**2 + (gamma_xy/2)**2)'],
         outputs=['e1', 'e2', 'gamma_max'], example={'eps_x': 600e-6, 'eps_y': -200e-6, 'gamma_xy': 400e-6},
         refs=['Plane strain transformation (γ/2 is the tensor shear strain)']),

    Calc('rosette_rectangular', 'Rectangular strain rosette (0°/45°/90°)',
         vars=[V('eps_x', '', 'εx', signed=True), V('eps_y', '', 'εy', signed=True), V('gamma_xy', '', 'γxy', signed=True),
               V('eps_a', '', 'gauge a at 0°', signed=True), V('eps_b', '', 'gauge b at 45°', signed=True),
               V('eps_c', '', 'gauge c at 90°', signed=True)],
         equations=['eps_x = eps_a', 'eps_y = eps_c', 'gamma_xy = 2*eps_b - eps_a - eps_c'],
         outputs=['eps_x', 'eps_y', 'gamma_xy'], example={'eps_a': 500e-6, 'eps_b': 300e-6, 'eps_c': -100e-6},
         keywords=['strain gauge', 'rosette'],
         assumptions=['Feed εx, εy, γxy into principal_strain; gauge angles measured CCW from x.'],
         refs=['ε(θ) = εx cos²θ + εy sin²θ + γxy sinθ cosθ']),

    Calc('rosette_delta', 'Delta strain rosette (0°/60°/120°)',
         vars=[V('eps_x', '', 'εx', signed=True), V('eps_y', '', 'εy', signed=True), V('gamma_xy', '', 'γxy', signed=True),
               V('eps_a', '', 'gauge a at 0°', signed=True), V('eps_b', '', 'gauge b at 60°', signed=True),
               V('eps_c', '', 'gauge c at 120°', signed=True)],
         equations=['eps_x = eps_a', 'eps_y = (2*eps_b + 2*eps_c - eps_a)/3', 'gamma_xy = 2*(eps_b - eps_c)/sqrt(3)'],
         outputs=['eps_x', 'eps_y', 'gamma_xy'], example={'eps_a': 500e-6, 'eps_b': 300e-6, 'eps_c': -100e-6},
         keywords=['strain gauge', 'equiangular rosette'],
         refs=['ε(θ) = εx cos²θ + εy sin²θ + γxy sinθ cosθ at 0°, 60°, 120°']),

    # ---- failure criteria ---------------------------------------------------------------------------
    Calc('von_mises_plane', 'von Mises equivalent stress and factor of safety (plane stress components)',
         vars=[V('sigma_vm', 'Pa', 'von Mises stress', label='σ′'), V('n', '', 'factor of safety S_y/σ′'),
               sig('sx', 'σx'), sig('sy', 'σy'), sig('txy', 'τxy'), V('S_y', 'Pa', 'yield strength')],
         equations=['sigma_vm = sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)', 'n = S_y/sigma_vm'],
         outputs=['sigma_vm', 'n'], example={'sx': 120e6, 'sy': -30e6, 'txy': 45e6, 'S_y': 350e6},
         keywords=['distortion energy', 'equivalent stress', 'ductile'],
         refs=['Distortion-energy (von Mises–Hencky) theory']),

    Calc('failure_criteria', 'Failure criteria comparison: von Mises, Tresca, Rankine, Coulomb–Mohr, modified Mohr',
         vars=[sig('sA', 'in-plane principal stress σA'), sig('sB', 'in-plane principal stress σB'),
               V('S_y', 'Pa', 'yield strength (ductile criteria)'), V('S_ut', 'Pa', 'tensile strength (brittle criteria)'),
               V('S_uc', 'Pa', 'compressive strength magnitude (brittle criteria)'),
               V('n_vm', '', 'von Mises factor of safety'), V('n_tresca', '', 'Tresca factor of safety'),
               V('n_rankine', '', 'Rankine (max normal) factor of safety'), V('n_cm', '', 'Coulomb–Mohr factor of safety'),
               V('n_mm', '', 'modified Mohr factor of safety')],
         procedure=_failure, outputs=['n_vm', 'n_tresca', 'n_rankine', 'n_cm', 'n_mm'],
         example={'sA': 90e6, 'sB': -150e6, 'S_y': 400e6, 'S_ut': 300e6, 'S_uc': 900e6},
         keywords=['tresca', 'max shear', 'brittle', 'mohr', 'rankine', 'safety factor'],
         assumptions=['Plane stress (third principal = 0), proportional loading.',
                      'Use von Mises/Tresca for ductile materials (S_y); Coulomb–Mohr/modified Mohr/Rankine for '
                      'brittle materials (S_ut, S_uc). Coulomb–Mohr with S_yt, S_yc also covers ductile materials '
                      'with unequal yield strengths.'],
         refs=['Failure theories for static loading (Shigley ch. 5 forms, recalled)']),

    # ---- vessels, discs, contact --------------------------------------------------------------------
    Calc('thin_cylinder', 'Thin-walled cylindrical pressure vessel',
         vars=[V('sigma_h', 'Pa', 'hoop (circumferential) stress', label='σh'), V('sigma_l', 'Pa', 'longitudinal stress (closed ends)', label='σl'),
               V('p', 'Pa', 'internal gauge pressure'), V('r', 'm', 'mean radius'), V('t', 'm', 'wall thickness')],
         equations=['sigma_h = p*r/t', 'sigma_l = p*r/(2*t)'], outputs=['sigma_h', 'sigma_l'],
         example={'p': 2e6, 'r': 0.5, 't': 0.01},
         keywords=['hoop stress', 'boiler', 'tank', 'pipe'],
         assumptions=['t/r ≲ 0.1; stresses uniform through the wall; away from heads and nozzles. For ASME design '
                      'thickness see the pressure-vessel section.'],
         refs=['Thin-wall membrane equilibrium']),

    Calc('thin_sphere', 'Thin-walled spherical pressure vessel',
         vars=[V('sigma', 'Pa', 'membrane stress (all directions)', label='σ'), V('p', 'Pa', 'internal gauge pressure'),
               V('r', 'm', 'mean radius'), V('t', 'm', 'wall thickness')],
         equations=['sigma = p*r/(2*t)'], outputs=['sigma'], example={'p': 2e6, 'r': 0.5, 't': 0.008},
         refs=['Thin-wall membrane equilibrium (sphere)']),

    Calc('thick_cylinder', 'Thick-walled cylinder stresses (Lamé)',
         vars=[sig('sigma_t', 'tangential (hoop) stress at radius r', 'σt'), sig('sigma_r', 'radial stress at radius r', 'σr'),
               sig('sigma_a', 'axial stress, closed ends', 'σa'),
               V('p_i', 'Pa', 'internal pressure'), V('p_o', 'Pa', 'external pressure'),
               V('r_i', 'm', 'inner radius'), V('r_o', 'm', 'outer radius'), V('r', 'm', 'evaluation radius (r_i ≤ r ≤ r_o)')],
         equations=['sigma_t = (p_i*r_i**2 - p_o*r_o**2)/(r_o**2 - r_i**2) - r_i**2*r_o**2*(p_o - p_i)/((r_o**2 - r_i**2)*r**2)',
                    'sigma_r = (p_i*r_i**2 - p_o*r_o**2)/(r_o**2 - r_i**2) + r_i**2*r_o**2*(p_o - p_i)/((r_o**2 - r_i**2)*r**2)',
                    'sigma_a = (p_i*r_i**2 - p_o*r_o**2)/(r_o**2 - r_i**2)'],
         outputs=['sigma_t', 'sigma_r', 'sigma_a'], example={'p_i': 100e6, 'p_o': 0.0, 'r_i': 0.05, 'r_o': 0.1, 'r': 0.05},
         keywords=['lame', 'press fit', 'gun barrel', 'hydraulic cylinder'],
         assumptions=['Linear elastic, open or closed ends as stated, plane sections; valid for any wall ratio.'],
         refs=['Lamé thick-cylinder solution']),

    Calc('rotating_disk', 'Rotating annular disk of uniform thickness',
         vars=[V('sigma_t', 'Pa', 'tangential stress at radius r', label='σt'), sig('sigma_r', 'radial stress at radius r', 'σr'),
               V('rho', 'kg/m^3', 'density', label='ρ'), V('omega', 'rad/s', 'angular speed', label='ω'), nu_(),
               V('a', 'm', 'inner radius (0 for a solid disk)'), V('b', 'm', 'outer radius'),
               V('r', 'm', 'evaluation radius (a < r ≤ b; r > 0)')],
         equations=['sigma_t = rho*omega**2*(3 + nu)/8*(a**2 + b**2 + a**2*b**2/r**2 - (1 + 3*nu)/(3 + nu)*r**2)',
                    'sigma_r = rho*omega**2*(3 + nu)/8*(a**2 + b**2 - a**2*b**2/r**2 - r**2)'],
         outputs=['sigma_t', 'sigma_r'],
         example={'rho': 7850.0, 'omega': 1000.0, 'nu': 0.3, 'a': 0.02, 'b': 0.15, 'r': 0.02},
         keywords=['flywheel', 'rotor', 'centrifugal stress', 'hoop'],
         assumptions=['Plane stress, free inner and outer edges, uniform thickness, no shaft interference. '
                      'Peak tangential stress is at the bore (r = a); for a solid disk put a = 0 (peak at centre).'],
         refs=['Rotating disk (Timoshenko / Shigley eq. form, recalled)']),

    Calc('curved_beam', 'Curved beam of rectangular section (Winkler)',
         vars=[V('r_n', 'm', 'neutral-axis radius'), V('e', 'm', 'centroid-to-neutral-axis offset'),
               V('A', 'm^2', 'area'), sig('sigma_i', 'stress at inner fibre', 'σi'), sig('sigma_o', 'stress at outer fibre', 'σo'),
               V('M', 'N*m', 'bending moment (positive decreases curvature: inner fibre tension)', signed=True),
               V('b', 'm', 'section width'), V('r_i', 'm', 'inner radius'), V('r_o', 'm', 'outer radius')],
         equations=['r_n = (r_o - r_i)/log(r_o/r_i)', 'e = (r_i + r_o)/2 - r_n', 'A = b*(r_o - r_i)',
                    'sigma_i = M*(r_n - r_i)/(A*e*r_i)', 'sigma_o = M*(r_n - r_o)/(A*e*r_o)'],
         outputs=['r_n', 'e', 'A', 'sigma_i', 'sigma_o'], example={'M': 500.0, 'b': 0.02, 'r_i': 0.05, 'r_o': 0.09},
         keywords=['crane hook', 'c-frame', 'winkler', 'bach'],
         assumptions=['Pure bending (add P/A for a direct load component), rectangular section, linear elastic.'],
         refs=['Winkler–Bach curved beam theory; r_n = h/ln(r_o/r_i) for a rectangle']),

    Calc('composite_beam', 'Composite (multi-material) beam by transformed section',
         vars=[V('layers', '', 'bottom-to-top layers [[b (m), h (m), E (Pa)], ...]', kind='json'),
               V('M', 'N*m', 'bending moment (sagging +)', signed=True), V('E_ref', 'Pa', 'reference modulus for transformation'),
               V('y_na', 'm', 'neutral axis height above the bottom'), V('I_t', 'm^4', 'transformed second moment'),
               V('stress_profile', '', '[[y (m), σ (Pa)], ...] at each layer bottom/top', kind='json'),
               V('sigma_max_abs', 'Pa', 'largest stress magnitude')],
         procedure=_composite_beam, outputs=['y_na', 'I_t', 'stress_profile', 'sigma_max_abs'],
         example={'layers': [[0.1, 0.01, 200e9], [0.1, 0.15, 10e9]], 'M': 5e3, 'E_ref': 10e9},
         keywords=['flitch', 'sandwich', 'reinforced', 'bimetal', 'transformed section'],
         assumptions=['Perfect bond, plane sections, linear elastic layers of rectangular cross-section.'],
         refs=['Transformed-section method (modular ratio n = E/E_ref)']),

    Calc('hertz_spheres', 'Hertzian contact between two spheres',
         vars=[V('a', 'm', 'contact circle radius'), V('p_max', 'Pa', 'maximum contact pressure'),
               V('F', 'N', 'normal force'), V('d1', 'm', 'diameter of sphere 1'),
               V('d2', 'm', 'diameter of sphere 2 (negative for a concave seat)', signed=True),
               V('E1', 'Pa', 'modulus 1'), V('nu1', '', 'Poisson 1', hi=0.5), V('E2', 'Pa', 'modulus 2'), V('nu2', '', 'Poisson 2', hi=0.5)],
         equations=['a = (3*F/8*((1 - nu1**2)/E1 + (1 - nu2**2)/E2)/(1/d1 + 1/d2))**(1/3)',
                    'p_max = 3*F/(2*pi*a**2)'],
         outputs=['a', 'p_max'], example={'F': 500.0, 'd1': 0.02, 'd2': 0.03, 'E1': 207e9, 'nu1': 0.3, 'E2': 207e9, 'nu2': 0.3},
         keywords=['contact stress', 'ball bearing', 'hertz'],
         assumptions=['Elastic, frictionless, smooth, contact small compared to radii. Flat plate: d2 → ∞ '
                      '(use a very large value). Maximum shear ≈ 0.3 p_max at depth ≈ 0.48a (ν = 0.3).'],
         refs=['Hertz contact of spheres (Shigley eq. form, recalled)']),

    Calc('hertz_cylinders', 'Hertzian contact between two parallel cylinders',
         vars=[V('b', 'm', 'contact half-width'), V('p_max', 'Pa', 'maximum contact pressure'),
               V('F', 'N', 'normal force'), V('l', 'm', 'contact length'), V('d1', 'm', 'diameter of cylinder 1'),
               V('d2', 'm', 'diameter of cylinder 2', signed=True), V('E1', 'Pa', 'modulus 1'), V('nu1', '', 'Poisson 1', hi=0.5),
               V('E2', 'Pa', 'modulus 2'), V('nu2', '', 'Poisson 2', hi=0.5)],
         equations=['b = sqrt(2*F/(pi*l)*((1 - nu1**2)/E1 + (1 - nu2**2)/E2)/(1/d1 + 1/d2))',
                    'p_max = 2*F/(pi*b*l)'],
         outputs=['b', 'p_max'], example={'F': 5000.0, 'l': 0.02, 'd1': 0.05, 'd2': 0.1, 'E1': 207e9, 'nu1': 0.3, 'E2': 207e9, 'nu2': 0.3},
         keywords=['line contact', 'roller', 'gear tooth contact', 'cam'],
         assumptions=['Elastic line contact, uniform load along l, frictionless.'],
         refs=['Hertz contact of cylinders (Shigley eq. form, recalled)']),

    # ---- concentration, energy, impact, plasticity ---------------------------------------------------
    Calc('stress_concentration', 'Stress concentration and fatigue notch factor',
         vars=[V('sigma_max', 'Pa', 'peak stress', label='σmax'), V('K_f', '', 'fatigue notch factor'),
               V('sigma_nom', 'Pa', 'nominal stress'), V('K_t', '', 'theoretical (geometric) concentration factor', lo=1.0),
               V('q', '', 'notch sensitivity', hi=1.0)],
         equations=['sigma_max = K_t*sigma_nom', 'K_f = 1 + q*(K_t - 1)'], outputs=['sigma_max', 'K_f'],
         example={'sigma_nom': 100e6, 'K_t': 2.2, 'q': 0.8},
         keywords=['notch', 'kt', 'kf', 'notch sensitivity', 'fillet'],
         assumptions=['K_t comes from charts (Peterson) for the geometry; q from Neuber/Kuhn–Hardrath data for the '
                      'material and notch radius (fatigue section). These are user inputs, not built-in tables.'],
         refs=['K_f = 1 + q(K_t − 1)']),

    Calc('elliptical_hole', 'Stress concentration at an elliptical hole in a wide plate (Inglis)',
         vars=[V('K_t', '', 'stress concentration factor'), V('a', 'm', 'semi-axis perpendicular to the load'),
               V('b', 'm', 'semi-axis parallel to the load')],
         equations=['K_t = 1 + 2*a/b'], outputs=['K_t'], example={'a': 0.01, 'b': 0.005},
         keywords=['inglis', 'crack', 'circular hole'],
         assumptions=['Infinite plate, uniaxial remote tension; circular hole a = b gives K_t = 3.'],
         refs=['Inglis (1913) elliptical hole solution']),

    Calc('strain_energy', 'Elastic strain energy: axial, torsional and bending members',
         vars=[V('U_axial', 'J', 'axial strain energy P²L/2AE'), V('U_torsion', 'J', 'torsional strain energy T²L/2GJ'),
               V('U_bending', 'J', 'bending strain energy M²L/2EI (constant M)'),
               V('P', 'N', 'axial force'), V('T', 'N*m', 'torque'), V('M', 'N*m', 'bending moment'),
               V('L', 'm', 'length'), V('A', 'm^2', 'area'), E_(), V('G', 'Pa', 'shear modulus'),
               V('J', 'm^4', 'polar second moment'), V('I', 'm^4', 'second moment')],
         equations=['U_axial = P**2*L/(2*A*E)', 'U_torsion = T**2*L/(2*G*J)', 'U_bending = M**2*L/(2*E*I)'],
         outputs=['U_axial', 'U_torsion', 'U_bending'],
         example={'P': 20e3, 'T': 300.0, 'M': 500.0, 'L': 1.0, 'A': 3e-4, 'E': 200e9, 'G': 79e9, 'J': 2.5e-7, 'I': 1.25e-7},
         keywords=['castigliano', 'energy'], refs=['Strain energy of linear elastic members']),

    Calc('resilience_toughness', 'Modulus of resilience and (approximate) modulus of toughness',
         vars=[V('u_r', 'J/m^3', 'modulus of resilience S_y²/2E'), V('u_t', 'J/m^3', 'approximate modulus of toughness'),
               V('S_y', 'Pa', 'yield strength'), V('S_u', 'Pa', 'ultimate strength'), E_(),
               V('eps_f', '', 'strain at fracture', label='εf')],
         equations=['u_r = S_y**2/(2*E)', 'u_t = (S_y + S_u)/2*eps_f'], outputs=['u_r', 'u_t'],
         example={'S_y': 350e6, 'S_u': 450e6, 'E': 200e9, 'eps_f': 0.25},
         keywords=['energy absorption'],
         assumptions=['Toughness uses the ductile-metal approximation (mean of S_y and S_u) × ε_f; integrate the '
                      'measured curve when available.'],
         refs=['Area under the stress–strain curve']),

    Calc('impact_axial', 'Impact from a falling weight on an elastic member',
         vars=[V('delta_max', 'm', 'maximum dynamic deflection'), V('k_impact', '', 'impact factor δmax/δst'),
               V('delta_st', 'm', 'static deflection under W'), V('W', 'N', 'falling weight'),
               V('h', 'm', 'drop height before contact'), V('k', 'N/m', 'member stiffness (AE/L for a bar)')],
         equations=['delta_st = W/k', 'k_impact = 1 + sqrt(1 + 2*h/delta_st)', 'delta_max = k_impact*delta_st'],
         outputs=['delta_st', 'k_impact', 'delta_max'], example={'W': 100.0, 'h': 0.05, 'k': 2e6},
         keywords=['falling load', 'dynamic load factor', 'sudden load'],
         assumptions=['Energy conservation, member mass neglected, no rebound losses; h = 0 (suddenly applied) '
                      'gives factor 2. Stress scales by the same factor: σmax = k_impact σst.'],
         refs=['W(h + δ) = ½kδ²']),

    Calc('plastic_rectangle', 'Elastic–plastic bending of a rectangular beam and residual stress',
         vars=[V('M_y', 'N*m', 'first-yield moment'), V('M_p', 'N*m', 'fully plastic moment'),
               V('M', 'N*m', 'moment for the given elastic-core half-depth'),
               sig('sigma_res', 'surface residual stress after unloading from M', 'σres'),
               V('S_y', 'Pa', 'yield strength'), V('b', 'm', 'width'), V('h', 'm', 'depth'),
               V('y_e', 'm', 'half-depth of the elastic core (h/2 = first yield, 0 = fully plastic)')],
         equations=['M_y = S_y*b*h**2/6', 'M_p = S_y*b*h**2/4', 'M = S_y*b*(h**2/4 - y_e**2/3)',
                    'sigma_res = S_y - M*(h/2)/(b*h**3/12)'],
         outputs=['M_y', 'M_p', 'M', 'sigma_res'], example={'S_y': 250e6, 'b': 0.02, 'h': 0.04, 'y_e': 0.01},
         keywords=['plastic hinge', 'shape factor', 'residual stress', 'elastic perfectly plastic'],
         assumptions=['Elastic–perfectly-plastic material, rectangular section; elastic unloading.'],
         refs=['Elastic–plastic bending of a rectangular section']),
]

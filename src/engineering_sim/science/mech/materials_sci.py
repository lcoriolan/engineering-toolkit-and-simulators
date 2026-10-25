# DESCRIPTION: Materials science section: crystal structure (APF, density, Miller indices, Bragg and
#              interplanar spacing), diffusion, phase fractions, Hall-Petch, stress-strain and
#              ductility, hardness, Charpy, creep, corrosion, composites and laminate (CLT) stiffness.
"""Materials science calculators.

Reuse rules (see common.py): Hall-Petch strength goes through common.hall_petch and isotropic two-phase
elastic bounds through common.elastic_bounds; property calculations use shared implementations
here. Hardness conversions are limited to the formula-based relations that are standard (Brinell
indentation definition and the approximate UTS ≈ 3.45 HB for steels); tabular ASTM E140 conversions are
not recalled. Relations are the standard materials-science ones (e.g. Callister, *Materials Science and
Engineering*; Jones, *Mechanics of Composite Materials*); cited by relation name, recalled rather than
re-read, so no page numbers are given.
"""
import math
from fractions import Fraction

import numpy as np

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts, common

SECTION = section('materials_sci', 'Materials science', order=200,
                  summary='Crystal structure, diffusion, phase fractions, Hall-Petch, stress-strain and '
                          'ductility, hardness, Charpy, creep, corrosion, composites and CLT laminates.')

PI = math.pi
N_A = 6.02214076e23   # Avogadro constant, 1/mol (CODATA 2018)
R_GAS = 8.314462618   # molar gas constant, J/(mol·K) (CODATA 2018)


# ---- procedures ---------------------------------------------------------------------------------
def _miller(v):
    """Miller indices (hkl) from intercepts with the unit-cell axes.

    intercepts: [x, y, z] as fractions of the lattice parameters; 0 means the plane is parallel to that
    axis (intercept at infinity). The indices are the smallest integers proportional to the reciprocal
    intercepts.
    """
    ints = v['intercepts']
    if not isinstance(ints, list) or len(ints) != 3:
        raise MechError('intercepts: [x, y, z] intercepts as fractions of a, b, c (0 = parallel to that axis)')
    recips = []
    for xi in ints:
        if not isinstance(xi, (int, float)):
            raise MechError('intercepts must be numbers (0 = parallel to that axis)')
        if xi == 0:
            recips.append(Fraction(0, 1))
        else:
            recips.append(1 / Fraction(xi).limit_denominator(10 ** 6))
    if all(r == 0 for r in recips):
        raise MechError('a plane cannot be parallel to all three axes')
    lcm = 1
    for r in recips:
        lcm = lcm * r.denominator // math.gcd(lcm, r.denominator)
    scaled = [r * lcm for r in recips]
    g = 0
    for s in scaled:
        g = math.gcd(g, s.numerator)
    g = g or 1
    return {'miller_indices': [s.numerator // g for s in scaled]}


def _hall_petch(v):
    """Hall-Petch yield strength via common.hall_petch (single shared implementation)."""
    return {'sigma_y': common.hall_petch(v['sigma0'], v['k'], v['d'])}


def _fick_second(v):
    """Non-steady diffusion into a semi-infinite solid: C(x,t) = Cs - (Cs - C0) erf(x/(2 sqrt(D t)))."""
    dl = math.sqrt(v['D'] * v['t'])
    if dl <= 0:
        raise MechError('diffusion length sqrt(D*t) must be positive (give D > 0 and t > 0)')
    C = v['Cs'] - (v['Cs'] - v['C0']) * math.erf(v['x'] / (2 * dl))
    return {'C': C, 'diffusion_length': dl}


def _creep(v):
    """Steady-state (secondary) creep rate by the Norton power law: edot = A sigma^n exp(-Q/RT)."""
    return {'edot': v['A'] * v['sigma'] ** v['n'] * math.exp(-v['Q'] / (R_GAS * v['T']))}


def _crystal_density(v):
    """Theoretical density: rho = n A_molar / (N_A a^3)."""
    return {'rho': v['n'] * v['A_am'] / (N_A * v['a'] ** 3)}


def _diffusion(v):
    """Arrhenius diffusion coefficient: D = D0 exp(-Q/(R T))."""
    return {'D': v['D0'] * math.exp(-v['Q'] / (R_GAS * v['T']))}


def _hardness_uts(v):
    """Approximate ultimate tensile strength from Brinell hardness: UTS(MPa) ≈ 3.45 HB (steels)."""
    return {'UTS': 3.45e6 * v['HB']}


def _elastic_bounds(v):
    """Isotropic two-phase elastic bounds via common.elastic_bounds (single shared implementation).

    The E values are the isotropic Young's modulus E = 9KG/(3K+G) corresponding to each K, G bound.
    """
    b = common.elastic_bounds(v['K1'], v['G1'], v['K2'], v['G2'], v['f1'])

    def E(K, G):
        return 9.0 * K * G / (3.0 * K + G)

    return {'E_voigt': E(b['voigt']['K'], b['voigt']['G']),
            'E_reuss': E(b['reuss']['K'], b['reuss']['G']),
            'K_voigt': b['voigt']['K'], 'K_reuss': b['reuss']['K'],
            'K_hs_upper': b['hs_upper']['K'], 'K_hs_lower': b['hs_lower']['K'],
            'G_voigt': b['voigt']['G'], 'G_reuss': b['reuss']['G'],
            'G_hs_upper': b['hs_upper']['G'], 'G_hs_lower': b['hs_lower']['G']}


def _laminate(v):
    """Classical lamination theory: A, B, D matrices and in-plane constants from a ply list.

    plies: [[thickness (m), angle (deg)], ...] bottom-to-top, one material for all plies. Reduced
    stiffness Q from E1, E2, nu12, G12; transformed to global axes by the standard CLT transformation;
    A = sum Q_bar t, B = 1/2 sum Q_bar (z^2 - z0^2), D = 1/3 sum Q_bar (z^3 - z0^3). In-plane (membrane)
    constants come from the compliance a = A^-1: E_x = 1/(h a11), E_y = 1/(h a22), G_xy = 1/(h a66),
    nu_xy = -a12/a11. These are exact for symmetric laminates (B = 0); for unsymmetric layups they come
    from A alone and ignore bending-extension coupling.
    """
    plies = v['plies']
    E1, E2, nu12, G12 = v['E1'], v['E2'], v['nu12'], v['G12']
    if not isinstance(plies, list) or not plies:
        raise MechError('plies: [[thickness (m), angle (deg)], ...] bottom-to-top')
    nu21 = nu12 * E2 / E1
    denom = 1.0 - nu12 * nu21
    if denom <= 0:
        raise MechError('nu12*nu21 must be less than 1')
    Q11, Q22, Q12, Q66 = E1 / denom, E2 / denom, nu12 * E2 / denom, G12
    total = sum(float(row[0]) for row in plies)
    if total <= 0:
        raise MechError('plies: total thickness must be positive')
    A = np.zeros((3, 3))
    B = np.zeros((3, 3))
    D = np.zeros((3, 3))
    z = -total / 2.0  # z measured from the laminate mid-plane
    for row in plies:
        t = float(row[0])
        th = math.radians(float(row[1]))
        zt = z + t
        m, n = math.cos(th), math.sin(th)
        m2, n2 = m * m, n * n
        Qb = np.zeros((3, 3))
        Qb[0, 0] = Q11 * m ** 4 + 2 * (Q12 + 2 * Q66) * m2 * n2 + Q22 * n ** 4
        Qb[0, 1] = (Q11 + Q22 - 4 * Q66) * m2 * n2 + Q12 * (m ** 4 + n ** 4)
        Qb[1, 1] = Q11 * n ** 4 + 2 * (Q12 + 2 * Q66) * m2 * n2 + Q22 * m ** 4
        Qb[0, 2] = (Q11 - Q12 - 2 * Q66) * m ** 3 * n + (Q12 - Q22 + 2 * Q66) * m * n ** 3
        Qb[1, 2] = (Q11 - Q12 - 2 * Q66) * m * n ** 3 + (Q12 - Q22 + 2 * Q66) * m ** 3 * n
        Qb[2, 2] = (Q11 + Q22 - 2 * Q12 - 2 * Q66) * m2 * n2 + Q66 * (m ** 4 + n ** 4)
        Qb[1, 0], Qb[2, 0], Qb[2, 1] = Qb[0, 1], Qb[0, 2], Qb[1, 2]
        A += Qb * (zt - z)
        B += Qb * (zt ** 2 - z ** 2) / 2.0
        D += Qb * (zt ** 3 - z ** 3) / 3.0
        z = zt
    h = total
    if h <= 0:
        raise MechError('plies: total thickness must be positive')
    if A[1, 1] <= 0 or A[0, 0] <= 0:
        raise MechError('A matrix is singular (check E1, E2 and ply angles)')
    a = np.linalg.inv(A)
    return {'E_x': 1.0 / (h * a[0, 0]), 'E_y': 1.0 / (h * a[1, 1]), 'nu_xy': -a[0, 1] / a[0, 0],
            'G_xy': 1.0 / (h * a[2, 2]), 'h': h, 'A': A.tolist(), 'B': B.tolist(), 'D': D.tolist()}


CALCS = [
    Calc('atomic_packing_factor', 'Atomic packing factor of a unit cell',
         vars=[V('APF', '', 'atomic packing factor (fraction of cell volume occupied by atoms)', hi=1.0),
               V('n', '', 'atoms per unit cell', integer=True), V('r', 'm', 'atomic radius'),
               V('a', 'm', 'lattice parameter (cube edge for cubic cells)')],
         equations=['APF = n*(4/3)*pi*r**3/a**3'], outputs=['APF'],
         example={'n': 4, 'r': 0.1278e-9, 'a': 0.3615e-9},
         keywords=['crystal structure', 'fcc', 'bcc', 'packing'],
         assumptions=['APF = (atoms/cell) x (atomic volume) / (cell volume). For FCC r = a/(2 sqrt 2) giving '
                      'APF = pi/(3 sqrt 2) = 0.7405; BCC r = a sqrt(3)/4 gives sqrt(3) pi/8 = 0.6802; simple cubic '
                      'r = a/2 gives pi/6 = 0.5236.'],
         refs=['Atomic packing factor definition']),

    Calc('crystal_density', 'Density from crystal structure',
         vars=[V('rho', 'kg/m^3', 'theoretical density', label='ρ'), V('n', '', 'atoms per unit cell', integer=True),
               V('A_am', 'kg/mol', 'atomic (molar) mass'), V('a', 'm', 'lattice parameter')],
         procedure=_crystal_density, outputs=['rho'],
         example={'n': 4, 'A_am': 0.06355, 'a': 0.3615e-9},
         keywords=['crystal structure', 'unit cell', 'fcc', 'density'],
         assumptions=['rho = (atoms/cell x molar mass) / (Avogadro x cell volume). Example values give copper '
                      '(FCC, a = 0.3615 nm) rho ≈ 8.94 g/cm^3.'],
         refs=['Theoretical density, rho = n A / (N_A V_cell)']),

    Calc('miller_indices', 'Miller indices from intercepts',
         vars=[V('intercepts', '', '[x, y, z] axis intercepts as fractions of the lattice parameters (0 = parallel to that axis)',
                 kind='json'),
               V('miller_indices', '', '[h, k, l] smallest-integer Miller indices', kind='json')],
         procedure=_miller, outputs=['miller_indices'],
         example={'intercepts': [1, 0, 0]},
         keywords=['hkl', 'plane', 'crystallography'],
         assumptions=['Indices are the smallest integers proportional to the reciprocal intercepts; negative '
                      'intercepts give negative indices. Enter intercepts as exact decimals (0.5, 0.25, ...).'],
         refs=['Miller index definition (h:k:l = 1/x : 1/y : 1/z)']),

    Calc('interplanar_spacing_cubic', 'Interplanar spacing of a cubic lattice',
         vars=[V('d', 'm', 'interplanar spacing d_hkl'), V('a', 'm', 'lattice parameter'),
               V('h', '', 'Miller index h', integer=True), V('k', '', 'Miller index k', integer=True),
               V('l', '', 'Miller index l', integer=True)],
         equations=['d = a/sqrt(h**2 + k**2 + l**2)'], outputs=['d'],
         example={'a': 0.3615e-9, 'h': 1, 'k': 1, 'l': 1},
         keywords=['bragg', 'hkl', 'spacing', 'xrd'],
         assumptions=['Cubic lattices only; non-cubic spacing needs the full lattice geometry.'],
         refs=['Cubic interplanar spacing, d = a/sqrt(h^2 + k^2 + l^2)']),

    Calc('bragg_law', "Bragg's law (X-ray diffraction)",
         vars=[V('d', 'm', 'interplanar spacing'), V('n', '', 'order of reflection', integer=True),
               V('lam', 'm', 'X-ray wavelength', label='λ'), V('theta', 'rad', 'Bragg angle (half the 2θ detector angle)', hi=PI / 2, label='θ')],
         equations=['n*lam = 2*d*sin(theta)'], outputs=['d'],
         example={'n': 1, 'lam': 0.154e-9, 'theta': math.radians(20.0)},
         keywords=['xrd', 'diffraction', 'lattice'],
         assumptions=['theta is the Bragg angle; a powder diffractometer reports 2θ, so halve it before use.'],
         refs=["Bragg's law, n lambda = 2 d sin(theta)"]),

    Calc('diffusion_arrhenius', 'Diffusion coefficient (Arrhenius)',
         vars=[V('D', 'm^2/s', 'diffusion coefficient'), V('D0', 'm^2/s', 'pre-exponential (frequency) factor'),
               V('Q', 'J/mol', 'activation energy for diffusion'), V('T', 'K', 'absolute temperature')],
         procedure=_diffusion, outputs=['D'],
         example={'D0': 2.8e-5, 'Q': 251000.0, 'T': 1200.0},
         keywords=['fick', 'interstitial', 'vacancy', 'arrhenius'],
         assumptions=['R = 8.314462618 J/(mol·K) is built in; enter Q in J/mol.'],
         refs=['Arrhenius diffusion, D = D0 exp(-Q/RT)']),

    Calc('fick_first_law', "Fick's first law (steady-state diffusion flux)",
         vars=[V('J', 'kg/(m^2*s)', 'diffusion flux (positive from C1 to C2)', signed=True),
               V('D', 'm^2/s', 'diffusion coefficient'), V('C1', 'kg/m^3', 'concentration at the high side'),
               V('C2', 'kg/m^3', 'concentration at the low side'), V('x', 'm', 'diffusion distance')],
         equations=['J = D*(C1 - C2)/x'], outputs=['J'],
         example={'D': 2e-9, 'C1': 5.0, 'C2': 1.0, 'x': 0.002},
         keywords=['flux', 'steady state', 'carburizing'],
         assumptions=['Steady state (concentration gradient constant); the general vector form is J = -D dC/dx.'],
         refs=["Fick's first law"]),

    Calc('fick_second_law', "Fick's second law (non-steady, semi-infinite solid)",
         vars=[V('C', 'kg/m^3', 'concentration at depth x and time t'), V('diffusion_length', 'm', 'characteristic diffusion length sqrt(D t)'),
               V('C0', 'kg/m^3', 'initial uniform concentration'), V('Cs', 'kg/m^3', 'surface concentration (held constant)'),
               V('D', 'm^2/s', 'diffusion coefficient'), V('x', 'm', 'depth below the surface'), V('t', 's', 'time')],
         procedure=_fick_second, outputs=['C', 'diffusion_length'],
         example={'C0': 0.2, 'Cs': 1.0, 'D': 3e-11, 'x': 0.0005, 't': 3600.0},
         keywords=['carburizing', 'erf', 'transient', 'diffusion'],
         assumptions=['Constant surface concentration, semi-infinite solid: C = Cs - (Cs - C0) erf(x/(2 sqrt(D t))).'],
         refs=["Fick's second law erf solution (standard)"]),

    Calc('lever_rule', 'Lever rule (weight fractions of two phases)',
         vars=[V('W_alpha', '', 'weight fraction of phase α', hi=1.0, label='Wα'),
               V('W_beta', '', 'weight fraction of phase β', hi=1.0, label='Wβ'),
               V('C_0', '', 'overall alloy composition', hi=1.0, label='C0'),
               V('C_alpha', '', 'composition of phase α (tie-line end)', hi=1.0, label='Cα'),
               V('C_beta', '', 'composition of phase β (tie-line end)', hi=1.0, label='Cβ')],
         equations=['W_alpha = (C_beta - C_0)/(C_beta - C_alpha)', 'W_beta = (C_0 - C_alpha)/(C_beta - C_alpha)'],
         outputs=['W_alpha', 'W_beta'],
         example={'C_0': 0.35, 'C_alpha': 0.1, 'C_beta': 0.9},
         keywords=['phase diagram', 'tie line', 'fraction'],
         assumptions=['Two-phase region of a binary phase diagram; C_alpha and C_beta are read from the tie line at '
                      'the temperature of interest. Compositions are dimensionless fractions (wt fraction or at fraction).'],
         refs=['Lever rule']),

    Calc('hall_petch', 'Hall-Petch yield strength vs grain size',
         vars=[V('sigma_y', 'Pa', 'yield strength', label='σy'), V('sigma0', 'Pa', 'friction stress (single-crystal strength)'),
               V('k', 'Pa*m^0.5', 'Hall-Petch strengthening coefficient', label='k'),
               V('d', 'm', 'mean grain diameter')],
         procedure=_hall_petch, outputs=['sigma_y'],
         example={'sigma0': 50e6, 'k': 0.74e6, 'd': 100e-6},
         keywords=['grain size', 'grain boundary', 'strengthening'],
         assumptions=['sigma_y = sigma0 + k/sqrt(d). Empirical; breaks down (inverse Hall-Petch) below ~10-20 nm, '
                      'which is not modelled.'],
         refs=['Hall-Petch relation (via common.hall_petch / materials_props.hall_petch)']),

    Calc('true_stress_strain', 'True stress and true strain from engineering values',
         vars=[V('sigma_true', 'Pa', 'true stress', label='σt'), V('eps_true', '', 'true (logarithmic) strain', label='εt'),
               V('sigma_eng', 'Pa', 'engineering stress', label='σ'), V('eps_eng', '', 'engineering strain', label='ε')],
         equations=['sigma_true = sigma_eng*(1 + eps_eng)', 'eps_true = log(1 + eps_eng)'],
         outputs=['sigma_true', 'eps_true'],
         example={'sigma_eng': 300e6, 'eps_eng': 0.1},
         keywords=['plastic', 'logarithmic strain', 'necking'],
         assumptions=['Valid up to necking (constant-volume plastic flow). Beyond necking the true stress needs a '
                      'Bridgman correction for triaxiality.'],
         refs=['True stress/strain conversion']),

    Calc('strain_hardening', 'Strain-hardening exponent and strength coefficient (power law)',
         vars=[V('n', '', 'strain-hardening exponent', hi=1.0), V('K', 'Pa', 'strength coefficient'),
               V('sigma1', 'Pa', 'true stress at point 1'), V('eps1', '', 'true strain at point 1'),
               V('sigma2', 'Pa', 'true stress at point 2'), V('eps2', '', 'true strain at point 2')],
         equations=['n = log(sigma2/sigma1)/log(eps2/eps1)', 'K = sigma1/eps1**n'], outputs=['n', 'K'],
         example={'sigma1': 250e6, 'eps1': 0.05, 'sigma2': 350e6, 'eps2': 0.2},
         keywords=['hollomon', 'power law', 'flow curve', 'ludwik'],
         assumptions=['Power-law true stress-strain curve sigma = K epsilon^n fitted between two points; '
                      '0 < n < 1 for most metals.'],
         refs=['Hollomon power-law fit, sigma = K epsilon^n']),

    Calc('ductility', 'Percent elongation and reduction of area',
         vars=[V('EL', '', 'elongation (fraction of the original gauge length; ×100 = %)'),
               V('RA', '', 'reduction of area (fraction; ×100 = %)', hi=1.0),
               V('L_0', 'm', 'original gauge length'), V('L_f', 'm', 'final gauge length at fracture'),
               V('A_0', 'm^2', 'original cross-sectional area'), V('A_f', 'm^2', 'final (necked) cross-sectional area')],
         equations=['EL = (L_f - L_0)/L_0', 'RA = (A_0 - A_f)/A_0'], outputs=['EL', 'RA'],
         example={'L_0': 0.05, 'L_f': 0.06, 'A_0': 1e-4, 'A_f': 0.6e-4},
         keywords=['elongation', 'necking', 'ductility'],
         assumptions=['Reported as fractions here; multiply by 100 for the usual percentage. Elongation depends on '
                      'gauge length for a necked specimen.'],
         refs=['Percent elongation and reduction of area definitions']),

    Calc('brinell_hardness', 'Brinell hardness from indentation',
         vars=[V('HB', 'Pa', 'Brinell hardness as a pressure (the conventional HB number = HB/9.80665e6 kgf/mm^2)'),
               V('P', 'N', 'indenter load'), V('D', 'm', 'ball diameter'), V('d', 'm', 'indentation diameter')],
         equations=['HB = 2*P/(pi*D*(D - sqrt(D**2 - d**2)))'], outputs=['HB'],
         example={'P': 30000.0, 'D': 0.01, 'd': 0.005},
         keywords=['hardness', 'indentation', 'hb'],
         assumptions=['Standard Brinell definition with P in N and D, d in m; the conventional HB number is this '
                      'pressure in kgf/mm^2 (divide by 9.80665e6).'],
         refs=['Brinell hardness definition']),

    Calc('hardness_uts', 'Approximate UTS from Brinell hardness (steels)',
         vars=[V('UTS', 'Pa', 'ultimate tensile strength (approximate)'),
               V('HB', '', 'Brinell hardness number (dimensionless)')],
         procedure=_hardness_uts, outputs=['UTS'],
         example={'HB': 200.0},
         keywords=['hardness', 'conversion', 'tensile strength'],
         assumptions=['UTS(MPa) ≈ 3.45 × HB for carbon and alloy steels (HB ≲ 400). Approximate only — use a real '
                      'tensile test for design. Tabular ASTM E140 conversions are NOT provided (not recalled).'],
         refs=['Rule of thumb UTS ≈ 3.45 HB MPa (approximate; recalled)']),

    Calc('charpy_impact', 'Charpy impact toughness (energy per fracture area)',
         vars=[V('toughness', 'J/m^2', 'impact toughness'), V('E', 'J', 'Charpy impact energy (CVN)'),
               V('A', 'm^2', 'cross-sectional area at the notch (or ligament area)')],
         equations=['toughness = E/A'], outputs=['toughness'],
         example={'E': 50.0, 'A': 0.8e-4},
         keywords=['charpy', 'izod', 'impact', 'notch'],
         assumptions=['Normalizes the measured impact energy by the fractured area; the value depends on notch '
                      'geometry and test temperature (the ductile-brittle transition is empirical — see report).'],
         refs=['Impact toughness definition']),

    Calc('creep_rate', 'Steady-state creep rate (Norton power law)',
         vars=[V('edot', '1/s', 'steady-state creep strain rate', label='ε̇'),
               V('A', '', 'creep constant in 1/(s·Pa^n) — its value must match the Pa you use for sigma'),
               V('sigma', 'Pa', 'applied stress (Pa; A must be consistent with this)', label='σ'),
               V('n', '', 'stress exponent'),
               V('Q', 'J/mol', 'activation energy for creep'), V('T', 'K', 'absolute temperature')],
         procedure=_creep, outputs=['edot'],
         example={'A': 1e-15, 'sigma': 100e6, 'n': 3.0, 'Q': 280000.0, 'T': 1000.0},
         keywords=['creep', 'norton', 'secondary creep', 'arrhenius'],
         assumptions=['Steady-state (secondary) creep only: edot = A sigma^n exp(-Q/RT) with R = 8.314462618 '
                      'J/(mol·K). sigma is entered in Pa, so A must carry units of 1/(s·Pa^n) (its value depends on n '
                      'and on the stress unit); enter a bare number with those units kept consistent.'],
         refs=['Norton creep law']),

    Calc('larson_miller', 'Larson-Miller parameter and rupture time',
         vars=[V('LMP', 'K', 'Larson-Miller parameter'), V('T', 'K', 'absolute temperature'),
               V('C', '', 'Larson-Miller material constant (≈ 20 for many alloys)'),
               V('t_r', '', 'time to rupture in hours (bare number of hours)')],
         equations=['LMP = T*(C + log(t_r)/log(10))'], outputs=['LMP'],
         example={'T': 900.0, 'C': 20.0, 't_r': 1000.0},
         keywords=['creep', 'rupture', 'life', 'lmp'],
         assumptions=['LMP = T (C + log10 t) with T in kelvin and t_r the rupture time in hours; C is a material '
                      'constant. Solve for t_r to get rupture time at a given LMP and T.'],
         refs=['Larson-Miller parameter (recalled)']),

    Calc('corrosion_rate', 'Corrosion (penetration) rate from mass loss',
         vars=[V('CR', 'm/s', 'corrosion penetration rate'), V('W', 'kg', 'mass lost'),
               V('rho', 'kg/m^3', 'material density', label='ρ'), V('A', 'm^2', 'exposed surface area'),
               V('t', 's', 'exposure time')],
         equations=['CR = W/(rho*A*t)'], outputs=['CR'],
         example={'W': 1e-3, 'rho': 7800.0, 'A': 0.01, 't': 3.1536e7},
         keywords=['astm g1', 'mils per year', 'mpy', 'penetration'],
         assumptions=['Uniform corrosion: penetration rate = mass loss / (density x area x time). Convert to mm/yr '
                      'by multiplying by 3.1536e7 s/yr x 1000 mm/m (or enter t in years).'],
         refs=['Corrosion rate from mass loss (ASTM G1 form, unit-consistent)']),

    Calc('galvanic_potential', 'Galvanic couple potential difference',
         vars=[V('E_cell', 'V', 'cell (driving) potential of the couple', signed=True),
               V('E_cathode', 'V', 'potential of the cathodic (noble) member', signed=True),
               V('E_anode', 'V', 'potential of the anodic (active) member', signed=True)],
         equations=['E_cell = E_cathode - E_anode'], outputs=['E_cell'],
         example={'E_cathode': -0.13, 'E_anode': -0.76},
         keywords=['galvanic series', 'corrosion', 'sacrificial anode', 'emf'],
         assumptions=['Potentials are standard electrode potentials (vs SHE) or galvanic-series potentials from a '
                      'table — user inputs, not built in. A larger positive E_cell means a stronger driving force.'],
         refs=['Galvanic cell potential, E = E_cathode - E_anode']),

    Calc('composite_modulus', 'Composite rule-of-mixtures moduli (fibre/matrix)',
         vars=[V('E_L', 'Pa', 'longitudinal modulus (isostrain, upper bound)'),
               V('E_T', 'Pa', 'transverse modulus (isostress, lower-bound estimate)'),
               V('V_f', '', 'fibre volume fraction', hi=1.0), V('V_m', '', 'matrix volume fraction (= 1 - V_f)', hi=1.0),
               V('E_f', 'Pa', 'fibre modulus'), V('E_m', 'Pa', 'matrix modulus')],
         equations=['E_L = V_f*E_f + V_m*E_m', 'E_T = 1/(V_f/E_f + V_m/E_m)', 'V_m = 1 - V_f'],
         outputs=['E_L', 'E_T', 'V_m'],
         example={'V_f': 0.4, 'E_f': 230e9, 'E_m': 3.5e9},
         keywords=['composite', 'rule of mixtures', 'voigt', 'reuss'],
         assumptions=['Longitudinal modulus is the isostrain (Voigt) rule of mixtures; transverse is the isostress '
                      '(Reuss) bound (it underestimates the true transverse modulus).'],
         refs=['Rule of mixtures for fibre composites']),

    Calc('composite_strength', 'Composite longitudinal strength (rule of mixtures)',
         vars=[V('sigma_L', 'Pa', 'longitudinal tensile strength', label='σL'),
               V('V_f', '', 'fibre volume fraction', hi=1.0), V('V_m', '', 'matrix volume fraction (= 1 - V_f)', hi=1.0),
               V('sigma_f', 'Pa', 'fibre tensile strength'), V('sigma_m', 'Pa', 'matrix stress at fibre fracture strain')],
         equations=['sigma_L = V_f*sigma_f + V_m*sigma_m', 'V_m = 1 - V_f'], outputs=['sigma_L', 'V_m'],
         example={'V_f': 0.4, 'sigma_f': 3.5e9, 'sigma_m': 30e6},
         keywords=['composite', 'strength', 'rule of mixtures'],
         assumptions=['Longitudinal (isostrain) strength. Transverse strength is matrix-limited and not a simple rule '
                      'of mixtures, so it is not provided.'],
         refs=['Longitudinal strength rule of mixtures']),

    Calc('elastic_bounds', 'Isotropic two-phase elastic bounds (Voigt, Reuss, Hashin-Shtrikman)',
         vars=[V('K1', 'Pa', 'bulk modulus of phase 1'), V('G1', 'Pa', 'shear modulus of phase 1'),
               V('K2', 'Pa', 'bulk modulus of phase 2'), V('G2', 'Pa', 'shear modulus of phase 2'),
               V('f1', '', 'volume fraction of phase 1', hi=1.0),
               V('E_voigt', 'Pa', 'isotropic E from the Voigt (isostrain) K and G'),
               V('E_reuss', 'Pa', 'isotropic E from the Reuss (isostress) K and G'),
               V('K_voigt', 'Pa', 'Voigt bulk bound'), V('K_reuss', 'Pa', 'Reuss bulk bound'),
               V('K_hs_upper', 'Pa', 'Hashin-Shtrikman upper bulk bound'), V('K_hs_lower', 'Pa', 'Hashin-Shtrikman lower bulk bound'),
               V('G_voigt', 'Pa', 'Voigt shear bound'), V('G_reuss', 'Pa', 'Reuss shear bound'),
               V('G_hs_upper', 'Pa', 'Hashin-Shtrikman upper shear bound'), V('G_hs_lower', 'Pa', 'Hashin-Shtrikman lower shear bound')],
         procedure=_elastic_bounds,
         outputs=['E_voigt', 'E_reuss', 'K_voigt', 'K_reuss', 'K_hs_upper', 'K_hs_lower',
                  'G_voigt', 'G_reuss', 'G_hs_upper', 'G_hs_lower'],
         example={'K1': 165e9, 'G1': 80e9, 'K2': 2.5e9, 'G2': 1.0e9, 'f1': 0.4},
         keywords=['hashin-shtrikman', 'voigt', 'reuss', 'bounds', 'composite'],
         assumptions=['Isotropic two-phase bounds on the effective bulk and shear moduli; E values are the isotropic '
                      'E = 9KG/(3K+G) for each bound pair. Voigt/Reuss are isostrain/isostress; Hashin-Shtrikman is '
                      'the tightest bound for well-ordered isotropic phases.'],
         refs=['Voigt/Reuss/Hashin-Shtrikman bounds (via common.elastic_bounds / materials_props.rule_of_mixtures)']),

    Calc('laminate_clt', 'Composite laminate stiffness (classical lamination theory)',
         vars=[V('plies', '', '[[thickness (m), angle (deg)], ...] bottom-to-top', kind='json'),
               V('E1', 'Pa', 'ply longitudinal modulus'), V('E2', 'Pa', 'ply transverse modulus'),
               V('nu12', '', 'ply major Poisson ratio', hi=0.5, label='ν12'),
               V('G12', 'Pa', 'ply in-plane shear modulus'),
               V('E_x', 'Pa', 'laminate in-plane modulus in x'), V('E_y', 'Pa', 'laminate in-plane modulus in y'),
               V('nu_xy', '', 'laminate in-plane Poisson ratio'), V('G_xy', 'Pa', 'laminate in-plane shear modulus'),
               V('h', 'm', 'total laminate thickness'),
               V('A', '', 'A (extensional) matrix, N/m, 3x3', kind='json'),
               V('B', '', 'B (coupling) matrix, N, 3x3', kind='json'),
               V('D', '', 'D (bending) matrix, N·m, 3x3', kind='json')],
         procedure=_laminate, outputs=['E_x', 'E_y', 'nu_xy', 'G_xy', 'h', 'A', 'B', 'D'],
         example={'plies': [[0.001, 45], [0.001, -45]], 'E1': 100e9, 'E2': 10e9, 'nu12': 0.3, 'G12': 5e9},
         keywords=['clt', 'laminate', 'composite', 'abd', 'plate'],
         assumptions=['One material for all plies; Q transformed to global axes by the standard CLT transformation. '
                      'A = sum(Q_bar t), B = 1/2 sum(Q_bar (z2^2 - z1^2)), D = 1/3 sum(Q_bar (z2^3 - z1^3)). The '
                      'membrane constants come from the compliance a = A^-1 (E_x = 1/(h a11), E_y = 1/(h a22), '
                      'G_xy = 1/(h a66), nu_xy = -a12/a11) and are exact for symmetric laminates (B = 0); for '
                      'unsymmetric laminates bending-extension coupling makes the true laminate moduli differ.'],
         refs=['Classical lamination theory (Jones, Mechanics of Composite Materials; recalled)']),
]

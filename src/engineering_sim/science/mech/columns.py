# DESCRIPTION: Columns and stability section: Euler and Johnson column formulas with the slenderness
#              transition, secant formula for eccentric loading, lateral-torsional buckling of I-beams,
#              and plate/shell elastic buckling.
"""Columns and stability calculators.

Sign and convention notes:
- Compressive axial load P is positive in magnitude; buckling is a stability limit, not a material
  stress limit. The critical load is the *same* for compression in either sense.
- The effective-length factor K is a user input (1.0 pinned–pinned, 0.5 fixed–fixed, 0.7 fixed–pinned,
  2.0 fixed–free); it folds the end conditions into an equivalent pin-ended length L_e = K·L.
- All relations here are elastic-instability landmarks. A design must still apply a safety factor and
  account for imperfections, residual stress and inelasticity.

Relations are the standard ones in column/plate theory (Timoshenko & Gere, *Theory of Elastic
Stability*; Beer & Johnston; Shigley). They are cited by relation name, recalled rather than re-read.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('columns', 'Columns and stability', order=50,
                  summary='Euler and Johnson columns, slenderness, secant formula, lateral-torsional '
                          'buckling, plate and shell buckling.')

PI = math.pi

# Shared declarations -------------------------------------------------------------------------------
K_FACTOR = "effective length factor (1.0 pinned–pinned; 0.5 fixed–fixed; 0.7 fixed–pinned; 2.0 fixed–free)"


def _column_critical(values):
    """Critical load/stress of a concentrically loaded column, Euler or Johnson by slenderness.

    Returns radius of gyration r, slenderness λ = KL/r, the Euler–Johnson transition slenderness
    λ_t = π·sqrt(2E/Sy), the critical stress σ_cr (Johnson below λ_t, Euler at/above) and P_cr = σ_cr·A.
    """
    E, Sy, K, L, I, A = values['E'], values['Sy'], values['K'], values['L'], values['I'], values['A']
    r = math.sqrt(I / A)
    lam = K * L / r
    lam_t = math.sqrt(2 * PI ** 2 * E / Sy)
    if lam < lam_t:
        sigma_cr = Sy - Sy ** 2 * lam ** 2 / (4 * PI ** 2 * E)  # Johnson (short/intermediate)
    else:
        sigma_cr = PI ** 2 * E / lam ** 2                     # Euler (long/slender)
    return {'r': r, 'lam': lam, 'lam_t': lam_t, 'sigma_cr': sigma_cr, 'P_cr': sigma_cr * A}


def _secant_column(values):
    """Maximum compressive stress of an eccentrically loaded column (secant formula).

    σ_max = (P/A)[1 + (ec/r²)sec(arg)] with arg = KL/(2r)·√(P/(EA)). Requires arg < π/2: beyond it
    the column is past its first instability and the formula is not physical (this also keeps the
    numeric inversion unique, since σ_max is monotonic in every input over the physical range).
    """
    P, A, e, c, r, E, K, L = (values[k] for k in ('P', 'A', 'e', 'c', 'r', 'E', 'K', 'L'))
    arg = K * L / (2 * r) * math.sqrt(P / (E * A))
    if arg >= PI / 2:
        raise MechError('secant argument ≥ π/2: the column is past its instability limit')
    sigma_max = (P / A) * (1 + e * c / (r ** 2 * math.cos(arg)))
    return {'sigma_max': sigma_max}


CALCS = [
    Calc('euler_critical_load', 'Euler critical buckling load (P_cr = π²EI/(KL)²)',
         vars=[V('P_cr', 'N', 'critical (Euler) buckling load', label='Pcr'),
               V('E', 'Pa', "Young's modulus"), V('I', 'm^4', 'second moment of area about the buckling axis'),
               V('K', '', K_FACTOR), V('L', 'm', 'actual unsupported length')],
         equations=['P_cr = pi**2*E*I/(K*L)**2'], outputs=['P_cr'],
         example={'E': 200e9, 'I': 1e-6, 'K': 1.0, 'L': 2.0},
         keywords=['buckling', 'euler', 'slender column', 'critical load'],
         assumptions=['Elastic buckling of a long, slender column: the material stays below its '
                      'proportional limit. I must be the minimum second moment for the buckling axis. '
                      'Short/intermediate columns buckle inelastically — use the Johnson or column_critical '
                      'calculator.'],
         refs=['Euler critical load P_cr = π²EI/(KL)²']),

    Calc('slenderness_ratio', 'Slenderness ratio and radius of gyration (λ = KL/r, r = √(I/A))',
         vars=[V('lam', '', 'slenderness ratio KL/r', label='λ'),
               V('K', '', K_FACTOR), V('L', 'm', 'unsupported length'),
               V('r', 'm', 'radius of gyration about the buckling axis'),
               V('I', 'm^4', 'second moment of area about the buckling axis'),
               V('A', 'm^2', 'cross-sectional area')],
         equations=['lam = K*L/r', 'r = sqrt(I/A)'], outputs=['lam', 'r'],
         example={'K': 1.0, 'L': 2.0, 'I': 1e-6, 'A': 1e-3},
         keywords=['slenderness', 'radius of gyration', 'lambda'],
         assumptions=['λ is dimensionless (radians are not involved); enter it as a bare number.'],
         refs=['Slenderness ratio λ = KL/r; radius of gyration r = √(I/A)']),

    Calc('euler_critical_stress', 'Euler critical stress (σ_cr = π²E/λ²)',
         vars=[V('sigma_cr', 'Pa', 'critical (Euler) buckling stress', label='σcr'),
               V('E', 'Pa', "Young's modulus"), V('lam', '', 'slenderness ratio KL/r', label='λ')],
         equations=['sigma_cr = pi**2*E/lam**2'], outputs=['sigma_cr'],
         example={'E': 200e9, 'lam': 100.0},
         keywords=['critical stress', 'euler', 'hyperbola'],
         assumptions=['Valid for the elastic range (λ at/above the transition; σ_cr ≤ proportional limit).'],
         refs=['Euler critical stress σ_cr = π²E/λ²']),

    Calc('johnson_column', 'Johnson parabolic critical stress (intermediate columns)',
         vars=[V('sigma_cr', 'Pa', 'Johnson critical stress', label='σcr'),
               V('Sy', 'Pa', 'yield (or compressive yield) strength'),
               V('E', 'Pa', "Young's modulus"), V('lam', '', 'slenderness ratio KL/r', label='λ')],
         equations=['sigma_cr = Sy - Sy**2*lam**2/(4*pi**2*E)'], outputs=['sigma_cr'],
         example={'Sy': 250e6, 'E': 200e9, 'lam': 50.0},
         keywords=['parabolic formula', 'intermediate column', 'johnson', 'inelastic'],
         assumptions=['Johnson (parabolic) approximation for short/intermediate columns with λ below the '
                      'Euler–Johnson transition λ_t = π√(2E/Sy). σ_cr ≤ Sy; do not apply beyond λ_t '
                      '(the formula turns negative) — use Euler there.'],
         refs=['Johnson parabolic formula (J. B. Johnson, tangent-parabola for intermediate columns)']),

    Calc('column_critical', 'Column critical load/stress with the Euler–Johnson transition',
         vars=[V('E', 'Pa', "Young's modulus"), V('Sy', 'Pa', 'yield strength'), V('K', '', K_FACTOR),
               V('L', 'm', 'unsupported length'), V('I', 'm^4', 'second moment of area about the buckling axis'),
               V('A', 'm^2', 'cross-sectional area'),
               V('r', 'm', 'radius of gyration'), V('lam', '', 'slenderness ratio KL/r', label='λ'),
               V('lam_t', '', 'transition slenderness π√(2E/Sy)', label='λt'),
               V('sigma_cr', 'Pa', 'critical stress (Johnson or Euler)', label='σcr'),
               V('P_cr', 'N', 'critical load = σ_cr·A', label='Pcr')],
         procedure=_column_critical, outputs=['r', 'lam', 'lam_t', 'sigma_cr', 'P_cr'],
         example={'E': 200e9, 'Sy': 250e6, 'K': 1.0, 'L': 2.0, 'I': 1e-6, 'A': 1e-3},
         keywords=['transition', 'short column', 'long column', 'design load'],
         assumptions=['Chooses Johnson for λ < λ_t = π√(2E/Sy) and Euler for λ ≥ λ_t; the two curves meet '
                      'at σ_cr = Sy/2. Inelastic behaviour is idealized by the Johnson parabola — apply a '
                      'safety factor and account for imperfections.'],
         refs=['Euler and Johnson column formulas with the tangent transition λ_t = π√(2E/Sy)']),

    Calc('secant_column', 'Secant formula: eccentrically loaded column maximum stress',
         vars=[V('sigma_max', 'Pa', 'maximum compressive stress (extreme fibre)', label='σmax'),
               V('P', 'N', 'compressive load'), V('A', 'm^2', 'cross-sectional area'),
               V('e', 'm', 'load eccentricity from the centroid'), V('c', 'm', 'extreme-fibre distance from the neutral axis'),
               V('r', 'm', 'radius of gyration'), V('E', 'Pa', "Young's modulus"),
               V('K', '', K_FACTOR), V('L', 'm', 'unsupported length')],
         procedure=_secant_column, outputs=['sigma_max'],
         example={'P': 50e3, 'A': 2e-3, 'e': 0.010, 'c': 0.020, 'r': 0.025, 'E': 200e9, 'K': 1.0, 'L': 1.5},
         keywords=['eccentric load', 'secant', 'amplification', 'imperfection'],
         assumptions=['Effective length KL used in the secant argument (K = 1 for pinned–pinned), all eccentricity in one plane, elastic material; '
                      'σ_max is at the concave extreme fibre. The secant argument must stay below π/2; '
                      'beyond it the column is unstable. For e = 0 this does not reduce to Euler (it is a '
                      'stress, not a buckling) — use column_critical for concentric load.'],
         refs=['Secant formula for an eccentrically loaded column']),

    Calc('lateral_torsional_buckling', 'Elastic lateral-torsional buckling moment of a doubly symmetric I-beam',
         vars=[V('M_cr', 'N*m', 'elastic critical moment', label='Mcr'),
               V('C_b', '', 'moment-gradient factor (1.0 for uniform moment)'),
               V('E', 'Pa', "Young's modulus"), V('G', 'Pa', 'shear modulus'),
               V('I_y', 'm^4', 'second moment of area about the weak (y) axis'),
               V('J', 'm^4', 'St-Venant torsion constant'),
               V('C_w', 'm^6', 'warping constant', label='Cw'),
               V('L', 'm', 'unbraced length (between lateral restraints)')],
         equations=['M_cr = C_b*pi/L*sqrt(E*I_y*G*J)*sqrt(1 + pi**2*E*C_w/(G*J*L**2))'],
         outputs=['M_cr'],
         example={'C_b': 1.0, 'E': 200e9, 'G': 77e9, 'I_y': 1e-6, 'J': 1e-7, 'C_w': 1e-9, 'L': 4.0},
         keywords=['ltb', 'beam buckling', 'warping', 'timoshenko'],
         assumptions=['Doubly symmetric I-section under uniform moment about the strong axis, elastic, '
                      'ends laterally supported but free to warp. C_b (user input) accounts for a '
                      'non-uniform moment diagram. J and C_w are section properties the user must supply.'],
         refs=['Elastic lateral-torsional buckling of a doubly symmetric I-section under uniform moment '
               '(Timoshenko/Gere closed form, recalled)']),

    Calc('plate_buckling', 'Elastic buckling of a flat plate (σ_cr = kπ²E/(12(1−ν²))(t/b)²)',
         vars=[V('sigma_cr', 'Pa', 'critical plate buckling stress', label='σcr'),
               V('k', '', 'plate buckling coefficient (4.0 for simply supported edges, uniaxial compression)'),
               V('E', 'Pa', "Young's modulus"), V('nu', '', 'Poisson ratio', hi=0.5, label='ν'),
               V('t', 'm', 'plate thickness'), V('b', 'm', 'loaded-edge width')],
         equations=['sigma_cr = k*pi**2*E/(12*(1 - nu**2))*(t/b)**2'], outputs=['sigma_cr'],
         example={'k': 4.0, 'E': 200e9, 'nu': 0.3, 't': 0.010, 'b': 0.500},
         keywords=['plate buckling', 'stiffener', 'panel', 'buckling coefficient'],
         assumptions=['Rectangular plate, elastic; k is a user input from the edge-support and loading '
                      'condition (standard tables). σ_cr ∝ (t/b)²: halving thickness quarters the capacity. '
                      'Plates carry load beyond the critical stress (post-buckling), unlike columns.'],
         refs=['Elastic plate buckling σ_cr = kπ²E/(12(1−ν²))(t/b)²']),

    Calc('shell_buckling_axial', 'Axial buckling of a thin cylindrical shell (with knock-down factor)',
         vars=[V('sigma_cr', 'Pa', 'classical (linear elastic) axial buckling stress', label='σcr'),
               V('sigma_des', 'Pa', 'knocked-down design buckling stress', label='σdes'),
               V('gamma', '', 'knock-down factor (empirical, from test data)', hi=1.0, label='γ'),
               V('E', 'Pa', "Young's modulus"), V('nu', '', 'Poisson ratio', hi=0.5, label='ν'),
               V('t', 'm', 'wall thickness'), V('r', 'm', 'mean radius')],
         equations=['sigma_cr = E*t/(r*sqrt(3*(1 - nu**2)))', 'sigma_des = gamma*sigma_cr'],
         outputs=['sigma_cr', 'sigma_des'],
         example={'gamma': 0.5, 'E': 200e9, 'nu': 0.3, 't': 0.002, 'r': 0.200},
         keywords=['cylindrical shell', 'knock-down', 'axial buckling', 'monocoque'],
         assumptions=['Classical linear-elastic axial buckling of a perfect thin cylinder. Real shells '
                      'buckle far below this because of imperfections, so a knock-down factor γ '
                      '(typically 0.1–0.7 from test data) must be applied. Thin wall only (t/r small).'],
         refs=['Classical axial buckling of a thin cylinder σ_cr = Et/(r√(3(1−ν²)))']),
]

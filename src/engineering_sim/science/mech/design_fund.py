# DESCRIPTION: Machine design fundamentals section: factor of safety and stress-strength reliability,
#              tolerance stack-up (worst-case and RSS), limits-and-fits with clearance/interference,
#              shrink-fit pressure / press force / holding torque, and preferred numbers (Renard).
"""Machine design fundamentals calculators.

Conventions (read before using these relations):
- Factor of safety n = S/sigma with S the relevant strength and sigma the working stress; the design
  factor is the value of n the designer requires, so the same relation serves both.
- Fits use the ISO 286 deviation convention: the hole has fundamental deviations EI (lower) and ES
  (upper), the shaft ei (lower) and es (upper). These four deviations are USER INPUTS (not tabulated
  here); a negative min clearance means an interference fit.
- Shrink-fit pressure uses the thick-cylinder (Lamé) interference relation for hub (outer cylinder, E_o,
  nu_o, outer radius d_o) and shaft (inner cylinder, E_i, nu_i, inner radius d_i, 0 for solid).
- Renard preferred numbers are returned UNROUNDED as 10^(i/n); the ISO-standard rounded values (R5: 1.0,
  1.6, 2.5, ...) are not tabulated here.

Reference relations are the standard ones in machine-design texts (e.g. Budynas & Nisbett, *Shigley's
Mechanical Engineering Design*; Juvinall & Marshek, *Fundamentals of Machine Component Design*). They
are cited by relation name, recalled rather than re-read, so no page numbers are given.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('design_fund', 'Machine design fundamentals', order=90,
                  summary='Factor of safety, stress-strength reliability, tolerance stack-up, limits and '
                          'fits, interference fits and preferred (Renard) numbers.')

PI = math.pi


# ---------------------------------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------------------------------
def _tolerance_stackup(values):
    """Worst-case and RSS (statistical) tolerance stack of dimensions [[nominal, +/-tol], ...]."""
    parts = values['parts']
    if not isinstance(parts, list) or not parts:
        raise MechError('parts: give [[nominal, +/-tol], ...]')
    total = worst_max = worst_min = rss2 = 0.0
    for row in parts:
        if len(row) != 2:
            raise MechError('each part row must be [nominal, +/-tol]')
        nom, tol = float(row[0]), float(row[1])
        total += nom
        worst_max += nom + tol
        worst_min += nom - tol
        rss2 += tol * tol
    return {'total_nominal': total, 'worst_max': worst_max, 'worst_min': worst_min,
            'rss_tol': math.sqrt(rss2)}


def _renard_series(values):
    """Unrounded Renard preferred numbers of series R(n) over one decade: 10^(i/n) for i = 0..n-1."""
    n = int(round(values['n']))
    if n not in (5, 10, 20, 40, 80):
        raise MechError('n must be 5, 10, 20, 40 or 80 (Renard R5/R10/R20/R40/R80)')
    vals = [10.0 ** (i / n) for i in range(n)]
    return {'values': vals, 'count': float(n)}


CALCS = [
    # ---- factor of safety and reliability --------------------------------------------------------
    Calc('factor_of_safety', 'Factor of safety (n = strength / stress)',
         vars=[V('n', '', 'factor of safety (design factor)'),
               V('S', 'Pa', 'material strength (e.g. yield S_y or ultimate S_ut)'),
               V('sigma', 'Pa', 'working stress', label='σ')],
         equations=['n = S/sigma'], outputs=['n'], example={'S': 350e6, 'sigma': 120e6},
         keywords=['factor of safety', 'design factor', 'margin'],
         assumptions=['Probabilistic stress–strength reliability lives in reliability.reliability_index (not duplicated here).',
                      'Static loading. The same relation is the design factor (the required n_d) and the '
                      'realized factor of safety (achieved n). Use the strength that matches the failure '
                      'mode (yield for ductile, ultimate for brittle).'],
         refs=['n = S/sigma']),

    Calc('tolerance_stackup', 'Tolerance stack-up: worst case and RSS (statistical)',
         vars=[V('parts', '', 'rows [nominal, +/-tol] of each dimension in m', kind='json'),
               V('total_nominal', 'm', 'sum of the nominal dimensions', signed=True),
               V('worst_max', 'm', 'worst-case maximum sum'),
               V('worst_min', 'm', 'worst-case minimum sum'),
               V('rss_tol', 'm', 'root-sum-square tolerance (statistical)')],
         procedure=_tolerance_stackup, outputs=['total_nominal', 'worst_max', 'worst_min', 'rss_tol'],
         example={'parts': [[10.0, 0.1], [20.0, 0.2], [5.0, 0.05]]},
         keywords=['tolerance', 'stack up', 'rss', 'worst case', 'accumulation'],
         assumptions=['Arithmetic (worst-case) stack adds all +/-tolerances; the RSS (root-sum-square) value '
                      'assumes independent, normally distributed dimensions. The parts rows and the outputs '
                      'are in m (enter any compatible length unit).'],
         refs=['Worst-case and root-sum-square tolerance accumulation']),

    Calc('limits_fit', 'Limits and fits: clearance or interference from hole and shaft deviations',
         vars=[V('D', 'm', 'basic (nominal) size'),
               V('hole_es', 'm', 'hole upper deviation ES', signed=True),
               V('hole_ei', 'm', 'hole lower deviation EI', signed=True),
               V('shaft_es', 'm', 'shaft upper deviation es', signed=True),
               V('shaft_ei', 'm', 'shaft lower deviation ei', signed=True),
               V('D_max', 'm', 'maximum hole size'), V('D_min', 'm', 'minimum hole size'),
               V('d_max', 'm', 'maximum shaft size'), V('d_min', 'm', 'minimum shaft size'),
               V('max_clearance', 'm', 'maximum clearance (D_max - d_min)', signed=True),
               V('min_clearance', 'm', 'minimum clearance (D_min - d_max; negative = interference)', signed=True)],
         equations=['D_max = D + hole_es', 'D_min = D + hole_ei', 'd_max = D + shaft_es', 'd_min = D + shaft_ei',
                    'max_clearance = D_max - d_min', 'min_clearance = D_min - d_max'],
         outputs=['D_max', 'D_min', 'd_max', 'd_min', 'max_clearance', 'min_clearance'],
         example={'D': 0.05, 'hole_es': 25e-6, 'hole_ei': 0.0, 'shaft_es': -25e-6, 'shaft_ei': -50e-6},
         keywords=['iso 286', 'hole basis', 'shaft basis', 'clearance fit', 'interference fit', 'transition fit'],
         assumptions=['ISO 286 deviations are entered explicitly (not tabulated). min_clearance > 0 is a '
                      'clearance fit, < 0 an interference fit, straddling zero a transition fit.'],
         refs=['Hole/shaft limit sizes from the four ISO 286 fundamental deviations']),

    # ---- interference (shrink) fits --------------------------------------------------------------
    Calc('interference_pressure', 'Shrink-fit contact pressure (thick-cylinder / Lamé)',
         vars=[V('p', 'Pa', 'contact pressure'), V('delta', 'm', 'diametral interference'),
               V('d', 'm', 'nominal (joint) diameter'), V('d_o', 'm', 'hub outer diameter'),
               V('d_i', 'm', 'shaft inner diameter (0 for solid)'),
               V('E_o', 'Pa', 'hub modulus'), V('nu_o', '', 'hub Poisson ratio', hi=0.5),
               V('E_i', 'Pa', 'shaft modulus'), V('nu_i', '', 'shaft Poisson ratio', hi=0.5)],
         equations=['p = delta/(d*((1/E_o)*((d_o**2 + d**2)/(d_o**2 - d**2) + nu_o) '
                    '+ (1/E_i)*((d**2 + d_i**2)/(d**2 - d_i**2) - nu_i)))'],
         outputs=['p'], example={'delta': 40e-6, 'd': 0.05, 'd_o': 0.09, 'd_i': 0.0,
                                 'E_o': 207e9, 'nu_o': 0.3, 'E_i': 207e9, 'nu_i': 0.3},
         keywords=['shrink fit', 'press fit', 'interference', 'lame', 'hub', 'contact pressure'],
         assumptions=['Linear elastic, uniform interference, both members thick-walled; the Lamé radial-'
                      'displacement sums equal the diametral interference. Solid shaft: d_i = 0.'],
         refs=['Thick-cylinder interference fit: delta = p d [(1/E_o)((d_o^2+d^2)/(d_o^2-d^2)+nu_o) '
               '+ (1/E_i)((d^2+d_i^2)/(d^2-d_i^2)-nu_i)] (recalled form)']),

    Calc('press_force', 'Axial force to press a shrink fit together',
         vars=[V('F', 'N', 'maximum press (assembly) force'), V('d', 'm', 'joint diameter'),
               V('L', 'm', 'engagement length'), V('p', 'Pa', 'contact pressure'), V('mu', '', 'friction coefficient', label='μ')],
         equations=['F = pi*d*L*p*mu'], outputs=['F'],
         example={'d': 0.05, 'L': 0.03, 'p': 57.244e6, 'mu': 0.15},
         keywords=['press force', 'assembly', 'shrink fit'],
         assumptions=['Coulomb friction over the whole engagement; use the peak contact pressure.'],
         refs=['F = pi d L p mu (axial press force)']),

    Calc('holding_torque', 'Torque capacity of a shrink (interference) fit',
         vars=[V('T', 'N*m', 'holding (friction) torque capacity'), V('d', 'm', 'joint diameter'),
               V('L', 'm', 'engagement length'), V('p', 'Pa', 'contact pressure'), V('mu', '', 'friction coefficient', label='μ')],
         equations=['T = pi/2*d**2*L*p*mu'], outputs=['T'],
         example={'d': 0.05, 'L': 0.03, 'p': 57.244e6, 'mu': 0.15},
         keywords=['holding torque', 'friction torque', 'shrink fit'],
         assumptions=['Torque transmitted by friction on the cylindrical interface; T = F d/2 with F the '
                      'press force.'],
         refs=['T = (pi/2) d^2 L p mu']),

    # ---- preferred numbers -----------------------------------------------------------------------
    Calc('preferred_number', 'Preferred number from a Renard series (10^(i/n))',
         vars=[V('value', '', 'preferred number'), V('i', '', 'index within the decade', lo=0.0, hi=80.0),
               V('n', '', 'steps per decade (5, 10, 20, 40 for R5/R10/R20/R40)', lo=5.0, hi=80.0)],
         equations=['value = 10**(i/n)'], outputs=['value'], example={'i': 6, 'n': 20},
         keywords=['renard', 'preferred size', 'standard number'],
         assumptions=['Unrounded Renard value 10^(i/n) in the decade [1, 10). The ISO rounded preferred '
                      'numbers (e.g. R20: 1.12, 1.25, 1.4, ...) differ slightly from these exact values.'],
         refs=['Renard series value = 10^(i/n)']),

    Calc('renard_series', 'Renard preferred number series over one decade (unrounded)',
         vars=[V('n', '', 'series steps per decade: 5, 10, 20, 40 or 80', lo=5.0, hi=80.0, integer=True),
               V('values', '', 'preferred numbers 10^(i/n) for i = 0..n-1 (decade [1, 10))', kind='json'),
               V('count', '', 'number of values (= n)')],
         procedure=_renard_series, outputs=['values', 'count'],
         example={'n': 20},
         keywords=['renard', 'r5', 'r10', 'r20', 'r40', 'preferred size'],
         assumptions=['Returns the UNROUNDED series 10^(i/n); ISO R-series tables use rounded values '
                      '(R5: 1.0, 1.6, 2.5, 4.0, 6.3, 10). Scale the decade as needed (multiply by 10^k).'],
         refs=['Renard series 10^(i/n), i = 0..n-1']),
]

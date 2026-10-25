# DESCRIPTION: Fatigue section: endurance limits, Marin factors, S–N curves, mean/alternating stress,
#              Goodman/Gerber/Soderberg/ASME-elliptic/Langer safety factors, Miner's rule, strain-life
#              (Coffin–Manson), Neuber's rule and welded-joint S–N fatigue.
"""Fatigue calculators.

Conventions: alternating stress σa and midrange (mean) stress σm are defined from σmax ≥ σmin as
σm = (σmax + σmin)/2 and σa = (σmax − σmin)/2; stress ratio R = σmin/σmax. A safety factor n is a load
factor (proportional loading). Chart/table coefficients (Marin surface a, b; reliability factor ke;
S–N intercept/slope; weld detail class FAT and slope m; Wileman A, B) are user inputs whose desc names
the source table — none is a recalled table embedded as a normative value.

Relations are the standard fatigue and machine-design forms (Shigley, *Mechanical Engineering Design*;
Bannantine, Comer & Handrock, *Fundamentals of Metal Fatigue Analysis*; IIW recommendations for the
fatigue of welded joints). They are cited by relation name, recalled rather than re-read, so no page or
equation numbers are given.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts

SECTION = section('fatigue', 'Fatigue', order=100,
                  summary='Endurance limits, Marin factors, S–N curves, mean/alternating stress, '
                          'fatigue failure criteria, Miner damage, strain-life, Neuber and welded joints.')

PI = math.pi


# ---- procedures ---------------------------------------------------------------------------------
def _marin_surface(values):
    """Surface (Marin) factor ka = a·(Sut/MPa)^b with a, b from the Marin surface table (Sut in MPa).

    Example pairs (a, b): ground 1.58, −0.085; machined/cold-drawn 4.51, −0.265; hot-rolled 57.7,
    −0.718; as-forged 272, −0.995. These are the user's inputs, not built-in values.
    """
    Sut_MPa = values['Sut'] / 1e6
    return {'ka': values['a'] * Sut_MPa ** values['b']}


def _marin_size(values):
    """Size (Marin) factor kb for a rotating round section (d in mm).

    kb = 1.24 d^−0.107 for 2.79 ≤ d ≤ 51 mm; kb = 1.51 d^−0.157 for 51 < d ≤ 254 mm. Axial load gives
    kb = 1 regardless of size (enter it directly in the corrected-endurance calculator).
    """
    d_mm = values['d'] * 1e3
    if 2.79 <= d_mm <= 51.0:
        kb = 1.24 * d_mm ** -0.107
    elif 51.0 < d_mm <= 254.0:
        kb = 1.51 * d_mm ** -0.157
    else:
        raise MechError('marin_size: d must be 2.79–254 mm for the rotating-round formula; for axial '
                        'load kb = 1 — enter it directly in the corrected endurance limit')
    return {'kb': kb}


def _sn_constants(values):
    """S–N power-law constants a, b from two points: S = a·N^b (b = ln(S1/S2)/ln(N1/N2))."""
    S1, N1 = values['S1'], values['N1']
    S2, N2 = values['S2'], values['N2']
    if S1 <= 0 or S2 <= 0 or N1 <= 0 or N2 <= 0:
        raise MechError('sn_constants: S and N must be positive')
    if S1 == S2 or N1 == N2:
        raise MechError('sn_constants: the two (S, N) points must be distinct')
    b = math.log(S1 / S2) / math.log(N1 / N2)
    a = S1 / N1 ** b
    return {'a': a, 'b': b}


def _miner(values):
    """Miner's linear cumulative-damage rule: D = Σ n_i/N_i over blocks [[n_i, N_i], ...]."""
    blocks = values['blocks']
    if not isinstance(blocks, list) or not blocks:
        raise MechError('miner_rule: give blocks as [[n_i (cycles), N_i (cycles)], ...]')
    D = 0.0
    for row in blocks:
        if len(row) != 2:
            raise MechError('miner_rule: each block must be [n_i, N_i]')
        n, N = float(row[0]), float(row[1])
        if N <= 0 or n < 0:
            raise MechError('miner_rule: need N_i > 0 and n_i >= 0')
        D += n / N
    if D <= 0:
        raise MechError('miner_rule: total damage is zero, so life is unbounded')
    return {'D': D, 'life': 1.0 / D}


def _neuber(values):
    """Neuber's rule with a Ramberg–Osgood cyclic stress–strain curve, solved by bisection.

    Finds the local (notch-root) stress σ and strain ε satisfying the Neuber hyperbola
    σ·ε = (Kf·S)²/E together with ε = σ/E + (σ/K')^(1/n'). The nominal stress S is elastic, so the
    nominal strain is e = S/E; Kf is the fatigue notch factor (Kt may be used for a sharp elastic notch).
    """
    Kf, S, E, Kp, np = values['Kf'], values['S'], values['E'], values['Kp'], values['np']
    if S <= 0 or E <= 0 or Kp <= 0 or np <= 0:
        raise MechError('neuber: need S, E, Kp, np > 0')
    if Kf < 1.0:
        raise MechError('neuber: Kf must be >= 1')
    target = (Kf * S) ** 2 / E

    def g(sigma):
        return sigma * (sigma / E + (sigma / Kp) ** (1.0 / np)) - target

    # g(0) = -target < 0 and g(Kf·S) = Kf·S·(Kf·S/Kp)^(1/np) > 0, so exactly one root in (0, Kf·S].
    lo, hi = 0.0, Kf * S
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)
    e_nom = S / E
    eps = sigma / E + (sigma / Kp) ** (1.0 / np)
    return {'sigma': sigma, 'eps': eps, 'K_sigma': sigma / S, 'K_eps': eps / e_nom, 'e_nom': e_nom}


def _fatigue_diagram(values):
    """Failure loci in the σm–σa plane (Goodman, Gerber, Soderberg, ASME elliptic, Langer yield)."""
    Se, Sut, Sy = values['Se'], values['Sut'], values['Sy']
    sm, sa = values['sigma_m'], values['sigma_a']
    xmax = max(Sut, Sy)
    x = charts.linspace(0.0, xmax, 200)
    goodman = [Se * (1 - t / Sut) for t in x]
    gerber = [Se * (1 - (t / Sut) ** 2) for t in x]
    soderberg = [Se * (1 - t / Sy) for t in x]
    asme = [Se * math.sqrt(max(0.0, 1 - (t / Sy) ** 2)) for t in x]
    langer = [Sy - t for t in x]
    return charts.chart('Fatigue failure diagram (σa vs σm)', 'σm [Pa]', 'σa [Pa]',
                        [charts.series('Goodman', x, goodman),
                         charts.series('Gerber', x, gerber),
                         charts.series('Soderberg', x, soderberg),
                         charts.series('ASME elliptic', x, asme),
                         charts.series('Langer (yield)', x, langer)],
                        markers=[charts.marker(sm, sa, 'load point')])


def _endurance_prime(values):
    """Endurance limit of a steel from Sut: Se′ = 0.5 Sut up to 1400 MPa, capped at 700 MPa above.

    The cap applies to steels only; wrought aluminium alloys (Se′ ≈ 0.4 Sut) are not handled here.
    """
    Sut = values['Sut']
    if Sut <= 0:
        raise MechError('endurance_prime: Sut must be positive')
    return {'Se_prime': 0.5 * Sut if Sut <= 1400e6 else 700e6}


def _sn_power(values):
    """Fatigue strength S = a·N^b (Basquin power law); warns outside the 10³–10⁶ cycle validity range."""
    a, b, N = values['a'], values['b'], values['N']
    if a <= 0:
        raise MechError('sn_power: the S–N intercept a must be positive')
    S = a * N ** b
    out = {'S': S}
    if N < 1e3 or N > 1e6:
        out['_warnings'] = ['sn_power: N is outside the 10^3–10^6 cycle range where the power-law '
                            'S–N fit is usually valid']
    return out


def _fatigue_criteria(values):
    """Goodman, Gerber, Soderberg, ASME-elliptic and Langer factors of safety.

    For a compressive mean stress (sigma_m < 0) the mean stress does not reduce fatigue life, so the
    four fatigue criteria all reduce to n = Se/sigma_a (the fully-reversed value); the first-cycle-yield
    guard becomes n_y = Sy/(sigma_a + |sigma_m|), i.e. Sy/(sigma_a − sigma_m) with sigma_m < 0. The
    ASME-elliptic relation is even in sigma_m; treating its compressive branch as non-damaging (the same
    Se/sigma_a reduction as Goodman/Gerber/Soderberg) is the documented convention here.
    """
    sa, sm = values['sigma_a'], values['sigma_m']
    Se, Sut, Sy = values['Se'], values['Sut'], values['Sy']
    if sa <= 0:
        raise MechError('fatigue_criteria: sigma_a must be positive')
    if Se <= 0 or Sut <= 0 or Sy <= 0:
        raise MechError('fatigue_criteria: Se, Sut and Sy must be positive')
    if sm < 0:
        nf = Se / sa
        n_goodman = n_gerber = n_soderberg = n_asme = nf
        n_langer = Sy / (sa - sm)  # sa + |sm|
    else:
        n_goodman = 1.0 / (sa / Se + sm / Sut)
        a = (sm / Sut) ** 2
        b = sa / Se
        n_gerber = (-b + math.sqrt(b * b + 4.0 * a)) / (2.0 * a) if a > 0 else 1.0 / b
        n_soderberg = 1.0 / (sa / Se + sm / Sy)
        n_asme = 1.0 / math.sqrt((sa / Se) ** 2 + (sm / Sy) ** 2)
        n_langer = Sy / (sa + sm)
    return {'n_goodman': n_goodman, 'n_gerber': n_gerber, 'n_soderberg': n_soderberg,
            'n_asme': n_asme, 'n_langer': n_langer}


CALCS = [
    Calc('endurance_prime', 'Endurance limit of a steel (Se′ = 0.5 Sut, capped at 700 MPa)',
         vars=[V('Se_prime', 'Pa', 'theoretical endurance limit', label='Se′'),
               V('Sut', 'Pa', 'ultimate tensile strength')],
         procedure=_endurance_prime, outputs=['Se_prime'], example={'Sut': 600e6},
         keywords=['endurance limit', 'rotating beam'],
         assumptions=['Steels: Se′ = 0.5 Sut for Sut ≤ 1400 MPa, and Se′ = 700 MPa above 1400 MPa. '
                      'Wrought aluminium alloys: Se′ ≈ 0.4 Sut — not handled by this steel branch; enter '
                      'the ratio yourself. Based on fully reversed rotating-beam specimens.'],
         refs=['Se′ = 0.5 Sut capped at 700 MPa (steel; recalled)']),

    Calc('marin_surface', 'Marin surface (finish) factor (ka = a·Sut^b)',
         vars=[V('ka', '', 'surface factor'),
               V('a', '', 'surface coefficient a (Marin surface table, Sut in MPa)'),
               V('b', '', 'surface exponent b (Marin surface table)', signed=True),
               V('Sut', 'Pa', 'ultimate tensile strength')],
         procedure=_marin_surface, outputs=['ka'],
         example={'a': 4.51, 'b': -0.265, 'Sut': 600e6},
         keywords=['marin', 'finish', 'surface finish'],
         assumptions=['a, b are tabulated for Sut in MPa (machined 4.51, −0.265; ground 1.58, −0.085; '
                      'hot-rolled 57.7, −0.718; as-forged 272, −0.995). Enter the pair for your finish.'],
         refs=['Marin surface factor ka = a·Sut^b (recalled)']),

    Calc('marin_size', 'Marin size factor (kb, rotating round section)',
         vars=[V('kb', '', 'size factor'),
               V('d', 'm', 'section diameter (rotating round bar)')],
         procedure=_marin_size, outputs=['kb'], example={'d': 0.02},
         keywords=['marin', 'size factor', 'diameter'],
         assumptions=['kb = 1.24 d^−0.107 for 2.79 ≤ d ≤ 51 mm; kb = 1.51 d^−0.157 for 51 < d ≤ 254 mm. '
                      'Bending or torsion of a rotating round section. Axial load: kb = 1 (enter directly).'],
         refs=['Marin size factor kb (rotating round section, recalled)']),

    Calc('endurance_limit_corrected', 'Corrected endurance limit (Marin)',
         vars=[V('Se', 'Pa', 'fully corrected endurance limit'),
               V('Se_prime', 'Pa', 'theoretical endurance limit', label='Se′'),
               V('ka', '', 'surface factor (Marin surface table)'),
               V('kb', '', 'size factor (rotating-round size relation)'),
               V('kc', '', 'load factor (1.0 bend, 0.85 axial, 0.59 torsion typical)'),
               V('kd', '', 'temperature factor'),
               V('ke', '', 'reliability factor'),
               V('kf', '', 'miscellaneous-effects factor')],
         equations=['Se = ka*kb*kc*kd*ke*kf*Se_prime'], outputs=['Se'],
         example={'ka': 0.828, 'kb': 0.9, 'kc': 0.85, 'kd': 1.0, 'ke': 0.868, 'kf': 1.0,
                  'Se_prime': 300e6},
         keywords=['marin', 'corrected endurance limit', 'modified endurance'],
         assumptions=['Se = ka·kb·kc·kd·ke·kf·Se′. Every k factor is a user input (the surface, size, '
                      'load, temperature and reliability factors are tabulated values the user supplies); '
                      'kf covers residual stress, plating, corrosion and any other effect.'],
         refs=['Marin equation Se = ka kb kc kd ke kf Se′ (recalled)']),

    Calc('sn_power', 'S–N curve (power law): fatigue strength at N cycles',
         vars=[V('S', 'Pa', 'fatigue strength at N cycles'),
               V('a', 'Pa', 'S–N curve intercept (coefficient a in S = a·N^b)'),
               V('b', '', 'S–N curve slope (exponent b, negative)', signed=True),
               V('N', '', 'number of cycles', lo=1.0)],
         procedure=_sn_power, outputs=['S'], example={'a': 1.25e9, 'b': -0.13265, 'N': 1e5},
         keywords=['s-n curve', 'wöhler', 'high-cycle', 'fatigue strength'],
         assumptions=['High-cycle fatigue (no plasticity): S = a·N^b on a log–log plot, N ≥ 1. A warning '
                      'is emitted outside the 10³–10⁶ cycle range where the power law is usually valid; '
                      'use strain-life below ~10³ cycles.'],
         refs=['Basquin / power-law S–N relation S = a N^b']),

    Calc('sn_constants', 'S–N curve constants from two points',
         vars=[V('a', 'Pa', 'S–N intercept (a in S = a·N^b)'),
               V('b', '', 'S–N slope (exponent b)', signed=True),
               V('S1', 'Pa', 'stress at N1'), V('N1', '', 'cycles at S1', lo=1.0),
               V('S2', 'Pa', 'stress at N2'), V('N2', '', 'cycles at S2', lo=1.0)],
         procedure=_sn_constants, outputs=['a', 'b'],
         example={'S1': 500e6, 'N1': 1e3, 'S2': 200e6, 'N2': 1e6},
         keywords=['s-n curve', 'curve fit', 'basquin'],
         assumptions=['Fits S = a·N^b through two points: b = ln(S1/S2)/ln(N1/N2), a = S1/N1^b. '
                      'Feed a, b into sn_power to read off the strength at any N.'],
         refs=['Two-point fit of the power-law S–N curve']),

    Calc('mean_alternating', 'Mean and alternating stress, stress ratio',
         vars=[V('sigma_m', 'Pa', 'midrange (mean) stress', signed=True, label='σm'),
               V('sigma_a', 'Pa', 'alternating stress', label='σa'),
               V('R', '', 'stress ratio σmin/σmax', signed=True),
               V('sigma_max', 'Pa', 'maximum stress', signed=True, label='σmax'),
               V('sigma_min', 'Pa', 'minimum stress', signed=True, label='σmin')],
         equations=['sigma_m = (sigma_max + sigma_min)/2', 'sigma_a = (sigma_max - sigma_min)/2',
                    'R = sigma_min/sigma_max'],
         outputs=['sigma_m', 'sigma_a', 'R'],
         example={'sigma_max': 200e6, 'sigma_min': -100e6},
         keywords=['stress ratio', 'fully reversed', 'pulsating', 'mean stress'],
         assumptions=['Requires σmax ≥ σmin (then σa ≥ 0). Fully reversed: R = −1; pulsating from zero: '
                      'R = 0; static: R = 1.'],
         refs=['σm = (σmax+σmin)/2; σa = (σmax−σmin)/2; R = σmin/σmax']),

    Calc('fatigue_criteria', 'Fatigue safety factors: Goodman, Gerber, Soderberg, ASME elliptic, Langer',
         vars=[V('sigma_a', 'Pa', 'alternating stress', label='σa'),
               V('sigma_m', 'Pa', 'midrange (mean) stress', signed=True, label='σm'),
               V('Se', 'Pa', 'fully corrected endurance limit'),
               V('Sut', 'Pa', 'ultimate tensile strength'),
               V('Sy', 'Pa', 'yield strength'),
               V('n_goodman', '', 'Goodman factor of safety'),
               V('n_gerber', '', 'Gerber factor of safety'),
               V('n_soderberg', '', 'Soderberg factor of safety'),
               V('n_asme', '', 'ASME-elliptic factor of safety'),
               V('n_langer', '', 'Langer (static yield) factor of safety')],
         procedure=_fatigue_criteria,
         outputs=['n_goodman', 'n_gerber', 'n_soderberg', 'n_asme', 'n_langer'],
         example={'sigma_a': 100e6, 'sigma_m': 150e6, 'Se': 180e6, 'Sut': 600e6, 'Sy': 400e6},
         plots=[{'id': 'diagram', 'title': 'Fatigue failure diagram', 'fn': _fatigue_diagram}],
         keywords=['goodman', 'gerber', 'soderberg', 'asme', 'langer', 'modified goodman', 'safety factor'],
         assumptions=['Proportional loading (all stress components scale together); σm ≥ 0 for the '
                      'diagram. Goodman is the usual design line; Soderberg is conservative (yield-based); '
                      'Gerber fits test data best for ductile steels; ASME-elliptic matches data and '
                      'includes yield; Langer guards first-cycle yielding (σa + σm ≤ Sy). For a '
                      'compressive mean stress (σm < 0) Goodman, Gerber, Soderberg and ASME-elliptic all '
                      'reduce to Se/σa, and Langer uses σa + |σm|. Check that n_langer ≥ n to avoid yield '
                      'on the first cycle.'],
         refs=['Goodman, Gerber, Soderberg, ASME-elliptic and Langer fatigue criteria (recalled)'],
         no_invert={'Sy': 'Sy enters only the yield-based criteria (Soderberg/ASME/Langer), not the '
                          'n_goodman output used for procedure inversion; invert Sy from n_langer instead'}),

    Calc('miner_rule', "Miner's rule: cumulative damage and predicted life",
         vars=[V('blocks', '', 'load blocks [[n_i (cycles), N_i (cycles to failure at that amplitude)], ...]',
                 kind='json'),
               V('D', '', 'cumulative damage Σ n_i/N_i (failure at D = 1)'),
               V('life', '', 'predicted repetitions of the block sequence to failure (1/D)')],
         procedure=_miner, outputs=['D', 'life'],
         example={'blocks': [[5000, 1e5], [20000, 2e5]]},
         keywords=['palmgren-miner', 'cumulative damage', 'block loading'],
         assumptions=['Linear damage accumulation; D = 1 predicts failure (some codes use D = 0.7). '
                      'Ignores load sequence/order and mean-stress interaction.'],
         refs=["Palmgren–Miner linear damage rule"]),

    Calc('coffin_manson', 'Strain-life (Coffin–Manson) total strain amplitude',
         vars=[V('eps_a', '', 'total strain amplitude Δε/2', label='εa'),
               V('sigma_fp', 'Pa', 'fatigue strength coefficient σf′', label='σf′'),
               V('b', '', 'fatigue strength exponent (negative)', signed=True),
               V('eps_fp', '', 'fatigue ductility coefficient εf′', label='εf′'),
               V('c', '', 'fatigue ductility exponent (negative)', signed=True),
               V('E', 'Pa', "Young's modulus"),
               V('Nf', '', 'cycles to failure (reversals 2Nf = 2·Nf)', lo=0.5)],
         equations=['eps_a = (sigma_fp/E)*(2*Nf)**b + eps_fp*(2*Nf)**c'],
         outputs=['eps_a'],
         example={'sigma_fp': 1000e6, 'b': -0.09, 'eps_fp': 0.6, 'c': -0.6, 'E': 200e9, 'Nf': 1000.0},
         keywords=['strain-life', 'low-cycle fatigue', 'coffin-manson', 'basquin', 'manson-hirschberg'],
         assumptions=['Total strain amplitude = elastic (Basquin) + plastic (Coffin–Manson) parts. '
                      'σf′ and εf′ are the cyclic material constants; b ≈ −0.05…−0.12, c ≈ −0.5…−0.7 for '
                      'metals. Solving for Nf is numeric (the relation is transcendental in Nf).'],
         refs=['Coffin–Manson / Basquin strain–life relation']),

    Calc('neuber', "Neuber's rule with a Ramberg–Osgood cyclic curve (local stress/strain)",
         vars=[V('Kf', '', 'fatigue notch factor (or Kt for a sharp elastic notch)', lo=1.0),
               V('S', 'Pa', 'nominal (net-section) stress'),
               V('E', 'Pa', "Young's modulus"),
               V('Kp', 'Pa', "cyclic strength coefficient K'", label='K′'),
               V('np', '', "cyclic strain-hardening exponent n'", label='n′'),
               V('sigma', 'Pa', 'local (notch-root) stress', label='σ'),
               V('eps', '', 'local (notch-root) strain', label='ε'),
               V('K_sigma', '', 'stress concentration factor on stress σ/S', label='Kσ'),
               V('K_eps', '', 'strain concentration factor on strain ε/e', label='Kε'),
               V('e_nom', '', 'nominal strain S/E', label='e')],
         procedure=_neuber, outputs=['sigma', 'eps', 'K_sigma', 'K_eps', 'e_nom'],
         example={'Kf': 2.0, 'S': 300e6, 'E': 200e9, 'Kp': 1200e6, 'np': 0.15},
         keywords=['neuber', 'notch root', 'local strain', 'ramberg-osgood', 'cyclic stress-strain'],
         assumptions=['Neuber hyperbola σ·ε = (Kf·S)²/E solved together with the cyclic stress–strain '
                      'curve ε = σ/E + (σ/K′)⁽¹/ⁿ′⁾ (Ramberg–Osgood). Nominal behaviour is elastic. '
                      'K′ and n′ are cyclic material constants (user inputs).'],
         refs=["Neuber's rule; Ramberg–Osgood cyclic stress–strain curve (recalled)"]),

    Calc('welded_fatigue', 'Welded-joint fatigue (nominal stress-range S–N)',
         vars=[V('N', '', 'cycles to failure'),
               V('FAT', 'Pa', 'detail category: characteristic stress range at 2e6 cycles (FAT class)'),
               V('Delta_sigma', 'Pa', 'applied nominal stress range Δσ', label='Δσ'),
               V('m', '', 'S–N slope (m = 3 for most details, 5 for some)')],
         equations=['N = 2e6*(FAT/Delta_sigma)**m'], outputs=['N'],
         example={'FAT': 80e6, 'Delta_sigma': 100e6, 'm': 3.0},
         keywords=['iiw', 'eurocode', 'detail category', 'fat class', 'stress range', 'weld'],
         assumptions=['Nominal-stress-range method (IIW / Eurocode 3): Δσ^m·N = FAT^m·2×10⁶. FAT and '
                      'the slope m are the detail-category inputs supplied by the user (FAT 36–160 MPa '
                      'typical). Valid for the design S–N curve; the constant-amplitude fatigue limit '
                      '(the horizontal knee, typically at 1×10⁷ cycles) is not modelled and no '
                      'stress-ratio correction is applied.'],
         refs=['IIW / Eurocode 3 nominal stress-range S–N: Δσ^m·N = Δσc^m·2e6 (recalled)']),
]

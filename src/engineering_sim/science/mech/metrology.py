# DESCRIPTION: Metrology and quality section: measurement statistics and uncertainty, gauge R&R,
#              process capability (Cp/Cpk/Pp/Ppk), control charts (X-bar/R constants computed from the
#              range distribution, p-chart), sigma level / DPMO, acceptance sampling (OC curve), sine
#              bar and gauge-block combination.
"""Metrology and quality calculators.

Conventions:
- Measured quantities use 'm' as the generic measurement unit (enter mm, in, …); the arithmetic is
  unit-agnostic. Means and limits may legitimately be negative (signed=True). Cpk/Ppk are signed: a
  process mean outside the specification gives a negative index.
- Control-chart constants A2, D3, D4, d2 (and d3) are COMPUTED from their statistical definitions:
  d2 and d3 come from a numeric integration of the range distribution of n iid standard normals, never
  from a recalled table. Gauge R&R follows the AIAG method: the repeatability constant K1 uses d2, while
  the single-range constants K2/K3 use d2* = √(d2² + d3²).
- Sigma level: Φ⁻¹(yield) of long-term (shifted) data is Z_lt; the short-term level removes the 1.5σ
  shift, Z_st = Z_lt + 1.5.
- Fits/limits and worst-case/RSS tolerance stack-up live in the design_fund section; this section owns
  the statistical (process-quality) side only.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('metrology', 'Metrology and quality', order=350,
                  summary='Measurement stats/uncertainty, gauge R&R, process capability, control charts, '
                          'sigma level and DPMO, acceptance sampling, sine bar, gauge blocks.')

PI = math.pi

_D2_D3_CACHE = {}


def _d2_d3(n):
    """d2 = E[range] and d3 = std[range] of n iid standard normals, by numeric integration.

    P(range ≤ x) = n ∫ [Φ(u) − Φ(u − x)]^{n−1} φ(u) du; then d2 = ∫(1 − F_R) dx and
    E[R²] = ∫ 2x(1 − F_R) dx, both evaluated with fixed Gauss–Legendre quadrature (80 nodes over
    u ∈ [−10, 10] and x ∈ [0, 10]). Reproduces the published d2 table (d2(2) = 2/√π, d2(5) = 2.3259).
    """
    n = int(n)
    if n < 2:
        raise MechError('subgroup size n must be at least 2')
    if n in _D2_D3_CACHE:
        return _D2_D3_CACHE[n]
    import numpy as np
    from scipy import stats
    xu, wu = np.polynomial.legendre.leggauss(80)
    xu = xu * 10.0
    wu = wu * 10.0
    xx, wx = np.polynomial.legendre.leggauss(80)
    xx = (xx + 1.0) * 5.0
    wx = wx * 5.0
    Phiu = stats.norm.cdf(xu)
    phiu = stats.norm.pdf(xu)
    Fr = np.empty_like(xx)
    for i, x in enumerate(xx):
        Fr[i] = n * np.dot((Phiu - stats.norm.cdf(xu - x)) ** (n - 1), wu * phiu)
    d2 = float(np.dot(1.0 - Fr, wx))
    E2 = float(np.dot(2.0 * xx * (1.0 - Fr), wx))
    d3 = math.sqrt(max(0.0, E2 - d2 * d2))
    _D2_D3_CACHE[n] = (d2, d3)
    return d2, d3


def _d2_star(n):
    """d2* = √(d2² + d3²): the range correction for a SINGLE range (g = 1), used by K2/K3 in gauge R&R."""
    d2, d3 = _d2_d3(n)
    return math.sqrt(d2 * d2 + d3 * d3)


# ---- procedures ----------------------------------------------------------------------------------
def _measurement_stats(values):
    """Summary statistics of repeated measurements (repeatability basis)."""
    data = values['measurements']
    if not isinstance(data, list) or len(data) < 2:
        raise MechError('measurements: give a list of at least two numbers')
    xs = [float(x) for x in data]
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return {'mean': mean, 'sd': math.sqrt(var), 'range': max(xs) - min(xs),
            'n': n, 'min': min(xs), 'max': max(xs)}


def _control_constants(values):
    n = int(values['n'])
    d2, d3 = _d2_d3(n)
    A2 = 3.0 / (d2 * math.sqrt(n))
    D3 = max(0.0, 1.0 - 3.0 * d3 / d2)
    D4 = 1.0 + 3.0 * d3 / d2
    return {'d2': d2, 'd3': d3, 'A2': A2, 'D3': D3, 'D4': D4}


def _xbar_r_limits(values):
    n = int(values['n'])
    d2, d3 = _d2_d3(n)
    A2 = 3.0 / (d2 * math.sqrt(n))
    D3 = max(0.0, 1.0 - 3.0 * d3 / d2)
    D4 = 1.0 + 3.0 * d3 / d2
    Xbb, Rbar = values['X_bar_bar'], values['R_bar']
    return {'UCL_x': Xbb + A2 * Rbar, 'LCL_x': Xbb - A2 * Rbar, 'CL_x': Xbb,
            'UCL_R': D4 * Rbar, 'LCL_R': D3 * Rbar, 'CL_R': Rbar}


def _gauge_rr(values):
    r = int(values['n_trials'])
    a = int(values['n_appraisers'])
    p_ = int(values['n_parts'])
    k = values['k_sigma']
    # K1 uses the plain d2 (R̄ is an average of many trial ranges, g = appraisers × parts).
    K1 = k / _d2_d3(r)[0]
    # K2 and K3 come from a SINGLE range (the appraiser-average range and the part-average range, g = 1),
    # so they use d2* = √(d2² + d3²).
    K2 = k / _d2_star(a)
    K3 = k / _d2_star(p_)
    Rbar = values['R_bar']
    EV = Rbar * K1
    # Reproducibility: appraiser variation, corrected for the repeatability embedded in X̄_diff.
    AV2 = (values['X_bar_diff'] * K2) ** 2 - EV ** 2 / (r * p_)
    AV = math.sqrt(max(0.0, AV2))
    GRR = math.sqrt(EV ** 2 + AV ** 2)
    PV = values['R_p'] * K3
    TV = math.sqrt(GRR ** 2 + PV ** 2)
    tol = values['tol']
    if tol <= 0:
        raise MechError('tol must be positive')
    return {'EV': EV, 'AV': AV, 'GRR': GRR, 'PV': PV, 'TV': TV,
            'pct_GRR_TV': 100.0 * GRR / TV, 'pct_GRR_Tol': 100.0 * GRR / tol,
            'ndc': 1.41 * PV / GRR}


def _cp_cpk(values):
    mu, sigma, USL, LSL = values['mu'], values['sigma'], values['USL'], values['LSL']
    if USL <= LSL:
        raise MechError('USL must be greater than LSL')
    if sigma <= 0:
        raise MechError('sigma must be positive')
    return {'Cpk': min(USL - mu, mu - LSL) / (3.0 * sigma), 'Cp': (USL - LSL) / (6.0 * sigma)}


def _pp_ppk(values):
    mu, sigma_t, USL, LSL = values['mu'], values['sigma_total'], values['USL'], values['LSL']
    if USL <= LSL:
        raise MechError('USL must be greater than LSL')
    if sigma_t <= 0:
        raise MechError('sigma_total must be positive')
    return {'Ppk': min(USL - mu, mu - LSL) / (3.0 * sigma_t), 'Pp': (USL - LSL) / (6.0 * sigma_t)}


def _sigma_level(values):
    from scipy import stats
    y = values['yield']
    if not (0.0 < y < 1.0):
        raise MechError('yield must be strictly between 0 and 1')
    # The yield of long-term (shifted) data gives the long-term sigma level directly; the short-term
    # level removes the conventional 1.5σ shift (3.4 DPMO → Z_lt = 4.5 → Z_st = 6.0).
    Z_lt = float(stats.norm.ppf(y))
    return {'Z_lt': Z_lt, 'Z_st': Z_lt + 1.5}


def _sigma_level_dpmo(values):
    from scipy import stats
    dpmo = values['dpmo']
    if not (0.0 < dpmo < 1e6):
        raise MechError('dpmo must be strictly between 0 and 1e6')
    y = 1.0 - dpmo / 1e6
    Z_lt = float(stats.norm.ppf(y))
    return {'Z_lt': Z_lt, 'Z_st': Z_lt + 1.5}


def _control_limits_p(values):
    p_bar, n = values['p_bar'], values['n']
    if n <= 0:
        raise MechError('subgroup size n must be positive')
    sigma = math.sqrt(p_bar * (1.0 - p_bar) / n)
    UCL = min(1.0, p_bar + 3.0 * sigma)
    LCL = max(0.0, p_bar - 3.0 * sigma)
    return {'UCL': UCL, 'LCL': LCL, 'CL': p_bar}


def _acceptance(values):
    """Probability of acceptance for a single sampling plan, hypergeometric (finite lot)."""
    from scipy import stats
    N, n, c = int(values['N']), int(values['n']), int(values['c'])
    p = values['p']
    if not (0 <= c <= n <= N):
        raise MechError('need 0 <= c <= n <= N')
    if not (0.0 <= p <= 1.0):
        raise MechError('p must be between 0 and 1')
    D = int(round(p * N))
    Pa = float(stats.hypergeom(N, D, n).cdf(c))
    return {'P_a': Pa, 'AOQ': p * Pa * (N - n) / N}


def _oc_binomial(values):
    """Probability of acceptance, binomial (infinite lot / with replacement)."""
    from scipy import stats
    n, c, p = int(values['n']), int(values['c']), values['p']
    if not (0 <= c <= n):
        raise MechError('need 0 <= c <= n')
    if not (0.0 <= p <= 1.0):
        raise MechError('p must be between 0 and 1')
    return {'P_a': float(stats.binom(n, p).cdf(c))}


def _gauge_blocks(values):
    """Gauge-block combination by minimum-block-count dynamic programming.

    Sizes are scaled to 0.5 µm so 0.0005 mm blocks are representable exactly. The 0/1 knapsack DP stores
    the fewest blocks that reach each length, so among several exact stacks the one using the fewest
    blocks is returned (e.g. target 5 mm from {1..5} mm → {5}, not {3+2}).
    """
    import numpy as np
    target = values['target']
    if target < 0:
        raise MechError('target must be non-negative')
    scale = 2e6  # 0.5 um per unit
    T = int(round(target * scale))
    if T > 4_000_000:
        raise MechError('target too large for an exact combination (limit 2 m)')
    raw = [float(b) for b in values['blocks'] if float(b) > 0]
    if not raw:
        raise MechError('blocks: give a non-empty list of positive block sizes')
    # Largest blocks first: a DP stage only claims a length when it strictly improves it, so on ties between equally
    # short stacks the larger blocks win (conventional wringing practice, e.g. 6 mm → {5, 1} rather than {4, 2}).
    blocks = sorted(set(int(round(b * scale)) for b in raw), reverse=True)
    INF = 1 << 30
    dp = np.full(T + 1, INF, dtype=np.int64)
    dp[0] = 0
    # 0/1 knapsack: the right-hand side is evaluated from the previous stage before assignment, so each block is
    # used at most once. For reconstruction keep, per block stage, a bit-packed mask of the lengths that stage

    # could reuse a block, e.g. {1,2,3,4} → 9 gave [4, 4, 1].]
    improved = []
    for b in blocks:
        cand = np.full(T + 1, INF, dtype=np.int64)
        if b <= T:
            cand[b:] = dp[:-b] + 1
        better = cand < dp
        improved.append(np.packbits(better))
        dp = np.where(better, cand, dp)
    if dp[T] >= INF:
        raise MechError('no exact combination of the given blocks reaches the target')
    selected = []
    r = T
    for i in range(len(blocks) - 1, -1, -1):
        if r == 0:
            break
        if np.unpackbits(improved[i], count=T + 1)[r]:
            selected.append(blocks[i] / scale)
            r -= blocks[i]
    if r != 0:
        raise MechError('internal reconstruction error')
    return {'selected': sorted(selected, reverse=True), 'residual': r / scale, 'n_blocks': len(selected)}


CALCS = [
    # ---- measurement --------------------------------------------------------------------------------
    Calc('measurement_stats', 'Summary statistics of repeated measurements (repeatability)',
         vars=[V('measurements', '', 'list of repeated measurements in the same unit (m)', kind='json'),
               V('mean', 'm', 'arithmetic mean', signed=True),
               V('sd', 'm', 'sample standard deviation (n − 1)'),
               V('range', 'm', 'max − min'), V('n', '', 'number of measurements', integer=True),
               V('min', 'm', 'minimum', signed=True), V('max', 'm', 'maximum', signed=True)],
         procedure=_measurement_stats, outputs=['mean', 'sd', 'range', 'n', 'min', 'max'],
         example={'measurements': [0.01002, 0.01005, 0.00999, 0.01003, 0.01001]},
         keywords=['repeatability', 'standard deviation', 'range', 'mean'],
         assumptions=['Sample standard deviation with the n − 1 denominator.'],
         refs=['Sample mean, standard deviation and range']),

    Calc('measurement_uncertainty', 'Combined and expanded measurement uncertainty',
         vars=[V('u_c', 'm', 'combined standard uncertainty'), V('U', 'm', 'expanded uncertainty k·u_c'),
               V('u_A', 'm', 'type A (random) standard uncertainty'),
               V('u_B', 'm', 'type B (systematic) standard uncertainty'),
               V('k', '', 'coverage factor (k = 2 for ~95%)', lo=1.0)],
         equations=['u_c = sqrt(u_A**2 + u_B**2)', 'U = k*u_c'], outputs=['u_c', 'U'],
         example={'u_A': 0.0005, 'u_B': 0.0003, 'k': 2.0},
         keywords=['uncertainty', 'gum', 'coverage factor', 'type a', 'type b'],
         assumptions=['Independent contributions combined in quadrature; k = 2 gives ~95% coverage for a '
                      'normal distribution.'],
         refs=['GUM combined uncertainty u_c = √(u_A² + u_B²); U = k·u_c (recalled)']),

    Calc('gauge_rr', 'Gauge R&R study (AIAG method, K-constants from computed d2/d2*)',
         vars=[V('n_trials', '', 'number of repeated trials per appraiser per part', lo=2.0, hi=10.0, integer=True),
               V('n_appraisers', '', 'number of appraisers (operators)', lo=2.0, hi=10.0, integer=True),
               V('n_parts', '', 'number of parts', lo=2.0, hi=25.0, integer=True),
               V('k_sigma', '', 'sigma-spread multiplier for the range (5.15 for 99 %, 6 for 99.73 %)', lo=1.0),
               V('R_bar', 'm', 'average of the trial ranges (over all appraisers and parts)'),
               V('X_bar_diff', 'm', 'range of the appraiser part-averages (max − min)'),
               V('R_p', 'm', 'range of the part averages'),
               V('tol', 'm', 'tolerance (USL − LSL) or process spread for %Tolerance'),
               V('EV', 'm', 'equipment variation (repeatability)'),
               V('AV', 'm', 'appraiser variation (reproducibility)'),
               V('GRR', 'm', 'gauge repeatability & reproducibility (EV² + AV²)^½'),
               V('PV', 'm', 'part variation'), V('TV', 'm', 'total study variation'),
               V('pct_GRR_TV', '', 'GRR as % of total variation'),
               V('pct_GRR_Tol', '', 'GRR as % of tolerance'),
               V('ndc', '', 'number of distinct categories (1.41·PV/GRR)')],
         procedure=_gauge_rr,
         outputs=['EV', 'AV', 'GRR', 'PV', 'TV', 'pct_GRR_TV', 'pct_GRR_Tol', 'ndc'],
         example={'n_trials': 2, 'n_appraisers': 3, 'n_parts': 5, 'k_sigma': 5.15, 'R_bar': 5e-6,
                  'X_bar_diff': 1e-5, 'R_p': 5e-5, 'tol': 5e-4},
         keywords=['gauge r&r', 'msa', 'repeatability', 'reproducibility', 'aiag', 'ndc'],
         assumptions=['AIAG method: K1 = k_sigma/d2(trials) uses the plain d2 (R̄ averages many ranges), while '
                      'K2 = k_sigma/d2*(appraisers) and K3 = k_sigma/d2*(parts) use d2* = √(d2² + d3²) because '
                      'X̄_diff and R_p are each a single range (g = 1). d2 and d3 are computed by numeric '
                      'integration of the range distribution. AV is corrected for the repeatability embedded in '
                      'X̄_diff. %GRR < 10% acceptable, 10–30% marginal, >30% unacceptable.'],
         refs=['AIAG Measurement Systems Analysis (MSA), gauge R&R (recalled)']),

    # ---- process capability -------------------------------------------------------------------------
    Calc('cp_cpk', 'Process capability Cp and Cpk (short-term sigma)',
         vars=[V('mu', 'm', 'process mean', signed=True, label='μ'),
               V('sigma', 'm', 'within-subgroup (short-term) standard deviation'),
               V('USL', 'm', 'upper specification limit', signed=True),
               V('LSL', 'm', 'lower specification limit', signed=True),
               V('Cp', '', 'process capability (potential) Cp = (USL − LSL)/(6σ)'),
               V('Cpk', '', 'process capability index Cpk = min(USL − μ, μ − LSL)/(3σ)', signed=True)],
         procedure=_cp_cpk, outputs=['Cpk', 'Cp'],
         example={'mu': 0.010, 'sigma': 0.0001, 'USL': 0.0104, 'LSL': 0.0096},
         keywords=['process capability', 'cp', 'cpk', 'six sigma'],
         assumptions=['Cp measures potential (centered); Cpk reflects the nearer limit and is NEGATIVE when the '
                      'process mean falls outside the specification (a signed index). Requires USL > LSL.'],
         refs=['Cp = (USL − LSL)/(6σ); Cpk = min(USL − μ, μ − LSL)/(3σ) (recalled)']),

    Calc('pp_ppk', 'Process performance Pp and Ppk (overall sigma)',
         vars=[V('mu', 'm', 'process mean', signed=True, label='μ'),
               V('sigma_total', 'm', 'overall (long-term) standard deviation'),
               V('USL', 'm', 'upper specification limit', signed=True),
               V('LSL', 'm', 'lower specification limit', signed=True),
               V('Pp', '', 'process performance (potential) Pp = (USL − LSL)/(6σ_t)'),
               V('Ppk', '', 'process performance index Ppk = min(USL − μ, μ − LSL)/(3σ_t)', signed=True)],
         procedure=_pp_ppk, outputs=['Ppk', 'Pp'],
         example={'mu': 0.010, 'sigma_total': 0.00015, 'USL': 0.0104, 'LSL': 0.0096},
         keywords=['process performance', 'pp', 'ppk', 'long term'],
         assumptions=['Pp/Ppk use the overall (long-term) sigma; Cp/Cpk use the within-subgroup sigma. Ppk is '
                      'signed (negative when the mean is outside the specification).'],
         refs=['Pp = (USL − LSL)/(6σ_t); Ppk = min(USL − μ, μ − LSL)/(3σ_t) (recalled)']),

    # ---- control charts ----------------------------------------------------------------------------
    Calc('control_chart_constants', 'Control chart constants A2, D3, D4, d2, d3 (computed from the range distribution)',
         vars=[V('n', '', 'subgroup (sample) size', lo=2.0, hi=25.0, integer=True),
               V('d2', '', 'mean of the range of n standard normals (E[R])'),
               V('d3', '', 'standard deviation of the range of n standard normals'),
               V('A2', '', 'X̄-chart factor 3/(d2·√n)'),
               V('D3', '', 'R-chart lower factor max(0, 1 − 3·d3/d2)'),
               V('D4', '', 'R-chart upper factor 1 + 3·d3/d2')],
         procedure=_control_constants, outputs=['d2', 'd3', 'A2', 'D3', 'D4'],
         example={'n': 5},
         keywords=['shewhart', 'xbar', 'r chart', 'control limits', 'd2', 'a2'],
         assumptions=['d2 and d3 are computed by numeric integration of the range distribution of n iid '
                      'standard normals (not from a recalled table): d2(2) = 2/√π exactly.'],
         refs=['Statistical definitions d2 = E[R], d3 = √Var[R]; A2 = 3/(d2√n), D3/D4 = 1 ∓ 3d3/d2 (recalled)']),

    Calc('control_limits_xbar_r', 'X̄ and R chart control limits (from computed A2/D3/D4)',
         vars=[V('X_bar_bar', 'm', 'grand mean (center of the X̄ chart)', signed=True),
               V('R_bar', 'm', 'average range (center of the R chart)'),
               V('n', '', 'subgroup size', lo=2.0, hi=25.0, integer=True),
               V('UCL_x', 'm', 'X̄-chart upper control limit', signed=True),
               V('LCL_x', 'm', 'X̄-chart lower control limit', signed=True),
               V('CL_x', 'm', 'X̄-chart center line', signed=True),
               V('UCL_R', 'm', 'R-chart upper control limit'),
               V('LCL_R', 'm', 'R-chart lower control limit'),
               V('CL_R', 'm', 'R-chart center line')],
         procedure=_xbar_r_limits, outputs=['UCL_x', 'LCL_x', 'CL_x', 'UCL_R', 'LCL_R', 'CL_R'],
         example={'X_bar_bar': 0.010, 'R_bar': 0.0005, 'n': 5},
         keywords=['shewhart', 'xbar', 'r chart', 'control limits'],
         assumptions=['UCL_x = X̄̄ + A2·R̄, LCL_x = X̄̄ − A2·R̄; R chart UCL/LCL = D4·R̄ / D3·R̄. A2, D3, D4 are '
                      'computed (see control_chart_constants). A negative LCL_x means no lower limit.'],
         refs=['Shewhart X̄ and R chart limits (recalled)']),

    Calc('control_limits_p', 'p-chart (fraction nonconforming) control limits',
         vars=[V('p_bar', '', 'average fraction nonconforming', hi=1.0),
               V('n', '', 'subgroup size', lo=1.0),
               V('UCL', '', 'upper control limit (capped at 1)'),
               V('LCL', '', 'lower control limit (floored at 0)'),
               V('CL', '', 'center line (= p̄)')],
         procedure=_control_limits_p, outputs=['UCL', 'LCL', 'CL'],
         example={'p_bar': 0.1, 'n': 100.0},
         keywords=['p chart', 'attribute', 'fraction nonconforming', 'control limits'],
         assumptions=['Control limits p̄ ± 3√(p̄(1−p̄)/n), with LCL = max(0, …) and UCL = min(1, …) — a p-chart '
                      'cannot have a negative lower limit or an upper limit above 1.'],
         refs=['p-chart control limits (recalled)']),

    # ---- sigma level / DPMO --------------------------------------------------------------------------
    Calc('dpmo', 'Defects per million opportunities (DPMO)',
         vars=[V('dpmo', '', 'defects per million opportunities'),
               V('D', '', 'number of defects', lo=0.0), V('N', '', 'number of units inspected', lo=1.0),
               V('O', '', 'opportunities per unit', lo=1.0)],
         equations=['dpmo = D/(N*O)*1e6'], outputs=['dpmo'],
         example={'D': 5.0, 'N': 1000.0, 'O': 10.0},
         keywords=['dpmo', 'defects', 'ppm', 'yield'],
         assumptions=['DPMO = (defects/(units × opportunities)) × 10⁶.'],
         refs=['DPMO definition (recalled)']),

    Calc('sigma_level', 'Process sigma level from long-term yield (Z_lt and Z_st)',
         vars=[V('yield', '', 'first-pass yield of LONG-TERM (shifted) data (fraction conforming, 0..1)',
                 lo=0.0, hi=1.0),
               V('Z_lt', '', 'long-term sigma level Z_lt = Φ⁻¹(yield)', signed=True),
               V('Z_st', '', 'short-term sigma level (Z_lt + 1.5)', signed=True)],
         procedure=_sigma_level, outputs=['Z_lt', 'Z_st'],
         example={'yield': 0.9986501},
         keywords=['sigma level', 'z score', 'yield', 'six sigma', 'dpmo'],
         assumptions=['The yield of long-term (shifted) process data gives the long-term sigma level directly, '
                      'Z_lt = Φ⁻¹(yield); the short-term level removes the conventional 1.5σ shift, '
                      'Z_st = Z_lt + 1.5. Landmark: yield = 0.99865 (1350 DPMO) → Z_lt = 3, Z_st = 4.5.'],
         refs=['Z_lt = Φ⁻¹(yield); Z_st = Z_lt + 1.5 (Six Sigma convention, recalled)']),

    Calc('sigma_level_dpmo', 'Process sigma level from DPMO (Z_lt and Z_st)',
         vars=[V('dpmo', '', 'defects per million opportunities', lo=0.0, hi=1e6),
               V('Z_lt', '', 'long-term sigma level Z_lt = Φ⁻¹(1 − dpmo/1e6)', signed=True),
               V('Z_st', '', 'short-term sigma level (Z_lt + 1.5)', signed=True)],
         procedure=_sigma_level_dpmo, outputs=['Z_lt', 'Z_st'],
         example={'dpmo': 1350.0},
         keywords=['sigma level', 'dpmo', 'z score', 'six sigma'],
         assumptions=['DPMO → yield = 1 − dpmo/10⁶, then Z_lt = Φ⁻¹(yield) and Z_st = Z_lt + 1.5. Landmark: '
                      '3.4 DPMO → Z_lt = 4.5, Z_st = 6.0.'],
         refs=['DPMO → sigma level; Z_st = Z_lt + 1.5 (Six Sigma convention, recalled)']),

    # ---- acceptance sampling ------------------------------------------------------------------------
    Calc('acceptance_sampling', 'Acceptance probability and AOQ for a single sampling plan (finite lot, hypergeometric)',
         vars=[V('N', '', 'lot size', lo=1.0, integer=True), V('n', '', 'sample size', lo=1.0, integer=True),
               V('c', '', 'acceptance number', lo=0.0, integer=True),
               V('p', '', 'fraction nonconforming in the lot', hi=1.0),
               V('P_a', '', 'probability of acceptance'), V('AOQ', '', 'average outgoing quality (with rectification)')],
         procedure=_acceptance, outputs=['P_a', 'AOQ'],
         example={'N': 1000, 'n': 80, 'c': 3, 'p': 0.05},
         plots=[{'id': 'oc', 'title': 'OC curve', 'x': 'p', 'from': 0.0, 'to': 0.2, 'n': 60, 'y': ['P_a']}],
         keywords=['oc curve', 'acceptance sampling', 'hypergeometric', 'aql', 'lot'],
         assumptions=['Hypergeometric model (sampling without replacement); D = round(p·N). AOQ assumes '
                      'rejected lots are 100% inspected (rectification).'],
         refs=['Hypergeometric operating-characteristic (OC) curve (recalled)']),

    Calc('oc_binomial', 'Acceptance probability for a single sampling plan (infinite lot, binomial)',
         vars=[V('n', '', 'sample size', lo=1.0, integer=True), V('c', '', 'acceptance number', lo=0.0, integer=True),
               V('p', '', 'fraction nonconforming', hi=1.0), V('P_a', '', 'probability of acceptance')],
         procedure=_oc_binomial, outputs=['P_a'],
         example={'n': 80, 'c': 3, 'p': 0.05},
         keywords=['oc curve', 'binomial', 'acceptance sampling', 'aql'],
         assumptions=['Binomial model (large/infinite lot or sampling with replacement).'],
         refs=['Binomial operating-characteristic (OC) curve (recalled)']),

    # ---- dimensional metrology ----------------------------------------------------------------------
    Calc('sine_bar', 'Sine bar angle from gauge-block stack height',
         vars=[V('theta', 'rad', 'angle set by the sine bar', hi=PI / 2, label='θ'),
               V('H', 'm', 'gauge-block stack height (difference between the roll supports)'),
               V('L', 'm', 'centre distance between the sine-bar rolls')],
         equations=['theta = asin(H/L)'], outputs=['theta'],
         example={'H': 0.025, 'L': 0.1},
         keywords=['sine bar', 'angle', 'gauge blocks', 'inspection'],
         assumptions=['Requires H ≤ L. The stack height H is the difference between the two roll supports.'],
         refs=['sin θ = H/L (sine bar, recalled)']),

    Calc('gauge_blocks', 'Gauge-block combination for a target length (minimum block count)',
         vars=[V('target', 'm', 'target length to build'),
               V('blocks', '', 'available gauge-block sizes in m (e.g. [0.020, 0.001001])', kind='json'),
               V('selected', '', 'block sizes that sum to the target, in m (largest first)', kind='json'),
               V('residual', 'm', 'residual (0 for an exact combination)'),
               V('n_blocks', '', 'number of blocks used', integer=True)],
         procedure=_gauge_blocks, outputs=['selected', 'residual', 'n_blocks'],
         example={'target': 0.025231,
                  'blocks': [0.020, 0.010, 0.005, 0.002, 0.0012, 0.00103, 0.001001]},
         keywords=['gauge blocks', 'jo blocks', 'wringing', 'stack', 'combination'],
         assumptions=['Each block used at most once; sizes resolved to 0.5 µm. Among the exact stacks, the one '
                      'using the FEWEST blocks is returned (target 5 mm from {1..5} mm → {5}, not {3+2}). '
                      'Raises an error when no exact combination exists.'],
         refs=['Gauge-block stacking by minimum-count subset-sum dynamic programming']),
]

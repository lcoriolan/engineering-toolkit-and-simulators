# DESCRIPTION: Reliability and maintenance section: exponential and Weibull reliability, hazard rate,
#              MTBF/MTTF/MTTR and availability, Weibull parameter estimation (median-rank regression and
#              MLE), series/parallel system reliability, and the normal stress/strength reliability index.
"""Reliability and maintenance calculators.

Conventions: failure times are in seconds (enter '1 h', '5000 s', etc.); failure rate λ is in 1/s
(enter '0.0001 1/s', '0.36 1/h'). Reliability R and cumulative failure F are dimensionless in [0, 1].
The Weibull scale parameter η has units of time; the shape parameter β is dimensionless.

Relations are the standard ones in reliability-engineering texts (e.g. O'Connor & Kleyner, *Practical
Reliability Engineering*; Ebeling, *An Introduction to Reliability and Maintainability Engineering*;
Meeker & Escobar, *Statistical Methods for Reliability Data*). They are cited by relation name, recalled
rather than re-read, so no page numbers are given.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('reliability', 'Reliability and maintenance', order=360,
                  summary='Reliability functions, failure and hazard rates, MTBF/MTTF/MTTR, availability, '
                          'Weibull fitting, system reliability and the reliability index.')

PI = math.pi


# ---- procedures -----------------------------------------------------------------------------------
def _weibull_fit(v):
    """Weibull β, η from a complete (uncensored) failure-time sample, two ways:

    - Median-rank regression (Bernard median ranks F_i = (i − 0.3)/(n + 0.4)) on the linearised CDF
      ln(ln(1/(1−F))) = β ln t − β ln η: ordinary least squares, slope = β, η = exp(−intercept/β).
    - Maximum likelihood via scipy.stats.weibull_min.fit with location fixed at 0 (2-parameter Weibull).
    """
    from scipy import stats
    data = v['data']
    if not isinstance(data, (list, tuple)) or len(data) < 3:
        raise MechError('data: give at least three failure times (a JSON list)')
    x = sorted(float(d) for d in data)
    n = len(x)
    if any(d <= 0 for d in x) or not all(math.isfinite(d) for d in x):
        raise MechError('data: failure times must be positive and finite')
    lx = [math.log(d) for d in x]
    y = [math.log(-math.log(1.0 - (i - 0.3) / (n + 0.4))) for i in range(1, n + 1)]
    mx = sum(lx) / n
    my = sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in lx)
    if sxx == 0:
        raise MechError('data: all failure times are equal; cannot fit a Weibull slope')
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(lx, y))
    beta_mr = sxy / sxx
    if beta_mr <= 0 or not math.isfinite(beta_mr):
        raise MechError('data: median-rank fit gives a non-positive shape (data not Weibull-like)')
    eta_mr = math.exp(-(my - beta_mr * mx) / beta_mr)
    c, _loc, scale = stats.weibull_min.fit(x, floc=0.0)
    beta_mle = float(c)
    eta_mle = float(scale)
    return {'beta_MR': beta_mr, 'eta_MR': eta_mr, 'beta_MLE': beta_mle, 'eta_MLE': eta_mle,
            'mu_MR': eta_mr * math.gamma(1.0 + 1.0 / beta_mr),
            'mu_MLE': eta_mle * math.gamma(1.0 + 1.0 / beta_mle)}


def _weibull_hazard(v):
    """Weibull hazard rate and pdf at time t: h(t) = (β/η)(t/η)^(β−1), f(t) = h(t)·R(t)."""
    beta, eta, t = v['beta'], v['eta'], v['t']
    if beta <= 0 or eta <= 0 or t < 0:
        raise MechError('Weibull needs β > 0, η > 0 and t ≥ 0')
    if t == 0.0:
        # h(0) = 0 for β > 1, = 1/η for β = 1, and is unbounded for β < 1 (wear-in).
        if beta < 1.0:
            raise MechError('hazard rate is unbounded at t = 0 for β < 1')
        h = 0.0 if beta > 1.0 else 1.0 / eta
        return {'h': h, 'f': h}
    ratio = t / eta
    h = (beta / eta) * ratio ** (beta - 1.0)
    f = h * math.exp(-(ratio ** beta))
    return {'h': h, 'f': f}


def _weibull_mttf(v):
    """Weibull mean time to failure: MTTF = η·Γ(1 + 1/β)."""
    beta, eta = v['beta'], v['eta']
    if beta <= 0 or eta <= 0:
        raise MechError('Weibull MTTF needs β > 0 and η > 0')
    return {'MTTF': eta * math.gamma(1.0 + 1.0 / beta)}


def _series(v):
    """Reliability of n independent components in series (system survives only if every component does)."""
    rs = v['R_list']
    if not isinstance(rs, (list, tuple)) or not rs:
        raise MechError('R_list: give a JSON list of component reliabilities (each in [0, 1])')
    rs = [float(r) for r in rs]
    if any(not 0 <= r <= 1 for r in rs):
        raise MechError('R_list: reliabilities must be in [0, 1]')
    R = 1.0
    for r in rs:
        R *= r
    return {'R_s': R}


def _parallel(v):
    """Reliability of n independent components in parallel (system fails only if every component fails)."""
    rs = v['R_list']
    if not isinstance(rs, (list, tuple)) or not rs:
        raise MechError('R_list: give a JSON list of component reliabilities (each in [0, 1])')
    rs = [float(r) for r in rs]
    if any(not 0 <= r <= 1 for r in rs):
        raise MechError('R_list: reliabilities must be in [0, 1]')
    Q = 1.0
    for r in rs:
        Q *= (1.0 - r)
    return {'R_p': 1.0 - Q}


def _reliability_index(v):
    """Reliability index β and probability of failure for normally distributed load and strength.

    Strength S ~ N(μ_S, σ_S) and load L ~ N(μ_L, σ_L) are independent, so the margin M = S − L is
    N(μ_S − μ_L, sqrt(σ_S² + σ_L²)). β = (μ_S − μ_L)/sqrt(σ_S² + σ_L²); p_f = P(M < 0) = Φ(−β).
    """
    muS, sS, muL, sL = v['mu_S'], v['sigma_S'], v['mu_L'], v['sigma_L']
    if sS <= 0 or sL <= 0:
        raise MechError('sigma_S and sigma_L must be positive')
    if muL <= 0:
        raise MechError('mu_L (mean load) must be positive')
    denom = math.sqrt(sS * sS + sL * sL)
    beta = (muS - muL) / denom
    p_f = 0.5 * math.erfc(beta / math.sqrt(2.0))
    return {'beta': beta, 'p_f': p_f, 'n_c': muS / muL}


CALCS = [
    Calc('exp_reliability', 'Exponential reliability and constant failure rate',
         vars=[V('R', '', 'reliability (probability of surviving to time t)', hi=1.0),
               V('lam', '1/s', 'constant failure rate (enter e.g. 0.36 1/h)', label='λ'),
               V('t', 's', 'mission time'),
               V('MTTF', 's', 'mean time to failure')],
         equations=['R = exp(-lam*t)', 'MTTF = 1/lam'], outputs=['R', 'MTTF'],
         example={'lam': 1e-4, 't': 1e4},
         keywords=['failure rate', 'hazard', 'exponential', 'mttf', 'reliability function'],
         assumptions=['Constant (random) failure rate over the mission — the exponential law holds in the '
                      'useful life of a bathtub curve. h(t) = λ (constant hazard).'],
         refs=['R(t) = exp(−λt); MTTF = 1/λ for the exponential distribution']),

    Calc('weibull_reliability', 'Weibull reliability and cumulative failure probability',
         vars=[V('R', '', 'reliability R(t)', hi=1.0),
               V('F', '', 'cumulative failure probability F(t) = 1 − R(t)', hi=1.0),
               V('beta', '', 'Weibull shape parameter (β < 1 wear-in, β = 1 exponential, β > 1 wear-out)', label='β'),
               V('eta', 's', 'Weibull scale parameter (characteristic life, R(η) = 36.8 %)', label='η'),
               V('t', 's', 'time')],
         equations=['R = exp(-(t/eta)**beta)', 'F = 1 - exp(-(t/eta)**beta)'],
         outputs=['R', 'F'], example={'beta': 2.0, 'eta': 1000.0, 't': 500.0},
         keywords=['weibull', 'reliability function', 'cdf', 'bathtub'],
         assumptions=['Two-parameter Weibull, t ≥ 0, β > 0, η > 0. At t = η the reliability is e⁻¹ ≈ 0.368 '
                      'for any β. β = 1 recovers the exponential law.'],
         refs=['Weibull distribution R(t) = exp(−(t/η)^β); F(t) = 1 − R(t)']),

    Calc('weibull_hazard', 'Weibull hazard rate and failure probability density',
         vars=[V('h', '1/s', 'hazard (instantaneous failure) rate h(t) = f(t)/R(t)'),
               V('f', '1/s', 'failure probability density f(t)'),
               V('beta', '', 'Weibull shape parameter', label='β'),
               V('eta', 's', 'Weibull scale parameter (characteristic life)', label='η'),
               V('t', 's', 'time')],
         procedure=_weibull_hazard, outputs=['h', 'f'],
         example={'beta': 2.0, 'eta': 1000.0, 't': 1000.0},
         keywords=['weibull', 'hazard rate', 'failure rate', 'pdf', 'bathtub'],
         assumptions=['Two-parameter Weibull, t ≥ 0, β > 0, η > 0. h(t) is the instantaneous failure rate '
                      '(increasing for β > 1 wear-out, decreasing for β < 1 wear-in, constant β = 1).'],
         refs=['Weibull h(t) = (β/η)(t/η)^(β−1); f(t) = h(t)·R(t)']),

    Calc('availability', 'MTBF, MTTF, MTTR and steady-state availability',
         vars=[V('A', '', 'availability (fraction of time the system is up)', hi=1.0),
               V('MTBF', 's', 'mean time between failures = MTTF + MTTR (strict convention: includes repair time)'),
               V('MTTF', 's', 'mean time to failure = mean UP time between failures'),
               V('MTTR', 's', 'mean time to repair (downtime per failure)')],
         equations=['A = MTTF/MTBF', 'MTBF = MTTF + MTTR'], outputs=['A', 'MTBF'],
         example={'MTTF': 950 * 3600.0, 'MTTR': 50 * 3600.0},
         keywords=['uptime', 'downtime', 'inherent availability', 'maintenance'],
         assumptions=['Steady-state (inherent) availability of a repairable item, one failure/repair cycle at '
                      'a time: A = MTTF/(MTTF + MTTR) = MTTF/MTBF, where MTBF = MTTF + MTTR (mean time '
                      'BETWEEN failures includes repair time). Some datasheets call the mean up time "MTBF"; '
                      'that datasheet form is the availability_datasheet calculator. For a non-repairable item '
                      'MTBF = MTTF.'],
         refs=['A = MTTF/MTBF; MTBF = MTTF + MTTR']),

    Calc('availability_datasheet', 'Datasheet-form availability (MTBF as mean up time)',
         vars=[V('A', '', 'availability (fraction of time up)', hi=1.0),
               V('MTBF_up', 's', 'mean up time between failures (the datasheet "MTBF" convention)'),
               V('MTTR', 's', 'mean time to repair (downtime per failure)')],
         equations=['A = MTBF_up/(MTBF_up + MTTR)'], outputs=['A'],
         example={'MTBF_up': 1000 * 3600.0, 'MTTR': 10 * 3600.0},
         keywords=['uptime', 'downtime', 'availability', 'datasheet', 'mtbf'],
         assumptions=['Datasheet convention: "MTBF" is quoted as the mean UP time between failures, so '
                      'MTBF_up ≡ MTTF and A = MTBF_up/(MTBF_up + MTTR). Equivalent to the inherent '
                      'availability of the availability calculator (which keeps the strict MTBF = MTTF + MTTR '
                      'convention).'],
         refs=['A = MTBF_up/(MTBF_up + MTTR) (datasheet MTBF convention)']),

    Calc('weibull_mttf', 'Weibull mean time to failure (MTTF)',
         vars=[V('MTTF', 's', 'mean time to failure = η·Γ(1 + 1/β)'),
               V('beta', '', 'Weibull shape parameter', label='β'),
               V('eta', 's', 'Weibull scale parameter (characteristic life)', label='η')],
         procedure=_weibull_mttf, outputs=['MTTF'],
         example={'beta': 2.0, 'eta': 1000.0},
         keywords=['weibull', 'mttf', 'mean time to failure', 'gamma'],
         assumptions=['Two-parameter Weibull mean life MTTF = η·Γ(1 + 1/β). β = 1 (exponential) gives '
                      'MTTF = η; β = 2 (Rayleigh) gives MTTF = η·√π/2.'],
         refs=['Weibull MTTF = η Γ(1 + 1/β)']),

    Calc('weibull_fit', 'Weibull parameter estimation (median-rank regression and MLE)',
         vars=[V('data', '', 'complete (uncensored) failure-time sample, seconds, e.g. [1000, 2000, 3100]', kind='json'),
               V('beta_MR', '', 'shape β from median-rank regression (Bernard)', label='βMR'),
               V('eta_MR', 's', 'scale η from median-rank regression (Bernard)', label='ηMR'),
               V('beta_MLE', '', 'shape β from maximum likelihood (scipy)', label='βMLE'),
               V('eta_MLE', 's', 'scale η from maximum likelihood (scipy)', label='ηMLE'),
               V('mu_MR', 's', 'mean life μ = η·Γ(1 + 1/β) (median-rank fit)'),
               V('mu_MLE', 's', 'mean life μ = η·Γ(1 + 1/β) (MLE fit)')],
         procedure=_weibull_fit, outputs=['beta_MR', 'eta_MR', 'beta_MLE', 'eta_MLE', 'mu_MR', 'mu_MLE'],
         example={'data': [500.0, 900.0, 1400.0, 2200.0, 3600.0, 6000.0]},
         keywords=['weibull', 'parameter estimation', 'probability plot', 'median rank', 'mle'],
         assumptions=['Complete (no censoring), two-parameter Weibull. Median-rank regression is a probability-'
                      'plot least-squares fit using Bernard ranks F_i = (i − 0.3)/(n + 0.4); MLE uses '
                      'scipy.stats.weibull_min.fit with location 0. Report which estimate you used.'],
         refs=["Bernard's median rank F_i = (i − 0.3)/(n + 0.4); Weibull MLE (scipy.stats.weibull_min.fit)"]),

    Calc('series_reliability', 'Series system reliability (n independent components)',
         vars=[V('R_list', '', 'component reliabilities, e.g. [0.9, 0.95, 0.99]', kind='json'),
               V('R_s', '', 'system reliability (product of the component reliabilities)', hi=1.0)],
         procedure=_series, outputs=['R_s'],
         example={'R_list': [0.95, 0.9, 0.98]},
         keywords=['series', 'system reliability', 'rbd'],
         assumptions=['Independent components; the system survives only if every component survives, so '
                      'R_s = R1·R2·…·Rn. Series systems are only as good as their weakest link.'],
         refs=['R_s = Π R_i for independent series components']),

    Calc('parallel_reliability', 'Parallel (redundant) system reliability',
         vars=[V('R_list', '', 'component reliabilities, e.g. [0.9, 0.9]', kind='json'),
               V('R_p', '', 'system reliability (one minus the product of failure probabilities)', hi=1.0)],
         procedure=_parallel, outputs=['R_p'],
         example={'R_list': [0.9, 0.9]},
         keywords=['parallel', 'redundant', 'system reliability', 'rbd'],
         assumptions=['Independent components in active parallel (redundancy); the system fails only if every '
                      'component fails, so R_p = 1 − (1−R1)(1−R2)⋯(1−Rn).'],
         refs=['R_p = 1 − Π (1 − R_i) for independent parallel components']),

    Calc('reliability_index', 'Reliability index β and probability of failure (normal stress/strength)',
         vars=[V('beta', '', 'reliability index (number of combined standard deviations of margin)', signed=True, label='β'),
               V('p_f', '', 'probability of failure P(strength < load)', hi=1.0),
               V('n_c', '', 'central factor of safety μ_S/μ_L'),
               V('mu_S', 'Pa', 'mean strength', label='μS'),
               V('sigma_S', 'Pa', 'standard deviation of strength', label='σS'),
               V('mu_L', 'Pa', 'mean load (stress)', label='μL'),
               V('sigma_L', 'Pa', 'standard deviation of load', label='σL')],
         procedure=_reliability_index, outputs=['beta', 'p_f', 'n_c'],
         example={'mu_S': 400e6, 'sigma_S': 40e6, 'mu_L': 250e6, 'sigma_L': 30e6},
         keywords=['stress strength', 'interference', 'probability of failure', 'six sigma'],
         assumptions=['Independent normal strength and load; the margin M = S − L is normal, so β = '
                      '(μ_S − μ_L)/sqrt(σ_S² + σ_L²) and p_f = Φ(−β). Assumes a linear (stress) limit state; '
                      'for load multipliers use the reliability-index form with the limit-state function.'],
         refs=['Load–strength interference; reliability index β; p_f = Φ(−β)']),
]

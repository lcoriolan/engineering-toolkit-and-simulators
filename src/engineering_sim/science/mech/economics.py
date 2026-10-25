# DESCRIPTION: Engineering economics section: present/future/annual worth, simple and compound interest,
#              effective rate, IRR (all real roots of the NPV polynomial), cash-flow NPV/FW/AW and discounted payback, benefit-cost ratio, depreciation
#              (straight-line, declining-balance, SOYD, table/MACRS-as-input), break-even and life-cycle cost.
"""Engineering economics calculators.

Currency is dimensionless (unit ''); interest rates are per period (dimensionless). Cash-flow periods are
counted from t = 0 (the first cash-flow entry is the period-0 flow). Depreciation methods are the standard
ones; MACRS percentages are user inputs (no recalled tables are embedded). IRR is found from the NPV
polynomial in x = 1 + r (numpy.roots) and reports all real roots r > −1, including tangent/even-multiplicity
roots and rates near −100 %.
"""
import math

import numpy as np
from scipy import optimize

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('economics', 'Engineering economics', order=470,
                  summary='Present/future/annual worth, interest, effective rate, IRR, payback, benefit-cost '
                          'ratio, depreciation, break-even and life-cycle cost.')


def _number(x, what):
    """Coerce a JSON scalar to a finite float, or raise MechError (never a bare ValueError)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise MechError(f'{what}: {x!r} is not a finite number')
    return float(x)


def _polish_root(P, dP, x0):
    """Refine a real root x0 of the NPV polynomial P (in x = 1 + r) to machine precision.

    A simple root is bracketed on P itself (sign change). A tangent/even-multiplicity root does not
    change sign in P, so it is bracketed on the derivative P′ (which has a simple root there).
    """
    for delta in (1e-3, 1e-2, 1e-1):
        lo, hi = x0 - delta, x0 + delta
        if lo <= 0:
            continue
        try:
            vlo, vhi = P(lo), P(hi)
        except (OverflowError, ZeroDivisionError):
            continue
        if math.isfinite(vlo) and math.isfinite(vhi) and vlo * vhi < 0:
            try:
                return float(optimize.brentq(P, lo, hi, xtol=1e-15, rtol=1e-14, maxiter=200))
            except (ValueError, RuntimeError):
                pass
    for delta in (1e-3, 1e-2, 1e-1):
        lo, hi = x0 - delta, x0 + delta
        if lo <= 0:
            continue
        try:
            vlo, vhi = dP(lo), dP(hi)
        except (OverflowError, ZeroDivisionError):
            continue
        if math.isfinite(vlo) and math.isfinite(vhi) and vlo * vhi < 0:
            try:
                return float(optimize.brentq(dP, lo, hi, xtol=1e-15, rtol=1e-14, maxiter=200))
            except (ValueError, RuntimeError):
                pass
    return float(x0)


def _irr(values):
    """All real internal rates of return from a cash-flow list [CF0, CF1, ...].

    NPV(r) = Σ CF_t (1+r)^-t. Multiplying by (1+r)^(n-1) turns this into a polynomial in x = 1 + r with
    the cash flows as coefficients, so numpy.roots gives every root (including tangent/even-multiplicity
    roots and rates near −100 %); each real x > 0 is polished against the polynomial directly (evaluated
    by Horner in x to avoid the cancellation the sum Σ CF_t/(1+r)^t suffers near a multiple root).
    """
    cf_raw = values['cash_flows']
    if not isinstance(cf_raw, list) or len(cf_raw) < 2:
        raise MechError('cash_flows: give [CF0, CF1, ...] with at least two entries')
    cf = [_number(x, 'cash_flows') for x in cf_raw]
    poly = np.asarray(cf, dtype=float)       # P(x) = sum cf_t x^(n-1-t), x = 1 + r
    dpoly = np.polyder(poly)                 # P'(x)

    def P(x):
        return float(np.polyval(poly, x))

    def dP(x):
        return float(np.polyval(dpoly, x))

    xs = np.roots(poly)
    roots = []
    for x in xs:
        if abs(x.imag) > 1e-8 * max(1.0, abs(x.real)):
            continue
        xr = float(x.real)
        if xr <= 1e-12:  # r <= -1 (the NPV pole) or x = 0
            continue
        xstar = _polish_root(P, dP, xr)
        if not math.isfinite(xstar) or xstar <= 1e-12:
            continue
        roots.append(xstar - 1.0)
    roots.sort()
    dedup = []
    for r in roots:
        if not any(math.isclose(r, o, rel_tol=1e-9, abs_tol=1e-12) for o in dedup):
            dedup.append(r)
    if not dedup:
        raise MechError('no real IRR (rate > -1) for these cash flows')
    return {'irr': dedup}


def _depreciation_table(values):
    """Depreciation from a user-supplied percentage schedule (e.g. MACRS rates)."""
    rates = values['rates']
    P = values['P']
    if not isinstance(rates, list) or not rates:
        raise MechError('rates: give a non-empty list of per-period percentages (e.g. [20, 32, ...])')
    rates = [_number(r, 'rates') / 100.0 for r in rates]  # entered as percent
    if any(r < 0 or r > 1 for r in rates):
        raise MechError('rates: each percentage must be in 0..100')
    if sum(rates) > 1.0 + 1e-9:
        raise MechError('rates: percentages sum to more than 100 % (book value would go negative)')
    dep = [P * r for r in rates]
    bv, acc = [], 0.0
    for d in dep:
        acc += d
        bv.append(P - acc)
    if min(bv) < -1e-9:
        raise MechError('book value went negative; check the rate schedule')
    return {'first_year': dep[0], 'depreciation': dep, 'book_values': bv}


def _annual_worth(values):
    """Annual worth from a present amount (capital-recovery factor), with the i = 0 limit A = P/n."""
    P, i, n = values['P'], values['i'], values['n']
    if n <= 0:
        raise MechError('n: number of periods must be positive')
    if i == 0:
        return {'A': P / n}
    return {'A': P * i * (1.0 + i) ** n / ((1.0 + i) ** n - 1.0)}


def _sinking_fund(values):
    """Annual deposit to accumulate a future sum (sinking-fund factor), with the i = 0 limit A = F/n."""
    F, i, n = values['F'], values['i'], values['n']
    if n <= 0:
        raise MechError('n: number of periods must be positive')
    if i == 0:
        return {'A': F / n}
    return {'A': F * i / ((1.0 + i) ** n - 1.0)}


def _life_cycle_cost(values):
    """Life-cycle cost (present worth) with the i = 0 limit LCC = C0 + n·C_annual."""
    C0, C_annual, i, n = values['C0'], values['C_annual'], values['i'], values['n']
    if n <= 0:
        raise MechError('n: life in periods must be positive')
    if i == 0:
        return {'LCC': C0 + n * C_annual}
    return {'LCC': C0 + C_annual * ((1.0 + i) ** n - 1.0) / (i * (1.0 + i) ** n)}


def _depreciation_sl(values):
    """Straight-line depreciation D = (P − S)/n and book value, with k > n rejected."""
    P, S, n, k = values['P'], values['S'], values['n'], values['k']
    if n <= 0:
        raise MechError('n: useful life must be positive')
    if k > n + 1e-9:
        raise MechError('k: period index cannot exceed the useful life n')
    if S < 0:
        raise MechError('S: salvage value cannot be negative')
    D = (P - S) / n
    BV = P - k * D
    if BV < -1e-9 * max(1.0, abs(P)):
        raise MechError('book value would be negative')
    return {'BV_k': BV, 'D': D}


def _depreciation_soyd(values):
    """Sum-of-years-digits depreciation and book value, with k > n rejected."""
    P, S, n, k = values['P'], values['S'], values['n'], values['k']
    if n <= 0:
        raise MechError('n: useful life must be positive')
    if k > n + 1e-9:
        raise MechError('k: period index cannot exceed the useful life n')
    if S < 0:
        raise MechError('S: salvage value cannot be negative')
    SOYD = n * (n + 1) / 2
    D_k = (P - S) * (n - k + 1) / SOYD
    BV_k = P - (P - S) * (k * (n + 1) - k * (k + 1) / 2) / SOYD
    return {'D_k': D_k, 'BV_k': BV_k, 'SOYD': SOYD}


def _cash_flow_series(values):
    """Present/future/annual worth and discounted payback of a cash-flow series at rate i.

    Cash flows are [CF0, CF1, ...] with CF0 at t = 0 (negative for an outlay). NPV = Σ CF_t/(1+i)^t;
    future worth is compounded to the end of the last period; annual worth spreads NPV uniformly;
    discounted payback is the first time the cumulative discounted flow reaches zero (linearly
    interpolated within the crossing period).
    """
    cf_raw = values['cash_flows']
    i = values['i']
    if not isinstance(cf_raw, list) or len(cf_raw) < 2:
        raise MechError('cash_flows: give [CF0, CF1, ...] with at least two entries')
    cf = [_number(x, 'cash_flows') for x in cf_raw]
    n = len(cf)
    if i < 0:
        raise MechError('i: interest rate must be >= 0')
    pv = [cf[t] / (1.0 + i) ** t for t in range(n)]
    npv = sum(pv)
    m = n - 1  # number of periods in the horizon (n cash flows span n − 1 periods)
    fw = npv * (1.0 + i) ** m
    aw = npv / m if i == 0 else npv * i * (1.0 + i) ** m / ((1.0 + i) ** m - 1.0)
    payback = None
    cum = 0.0
    for t in range(n):
        prev = cum
        cum += pv[t]
        if prev < 0 and cum >= 0:
            payback = (t - 1) + (-prev) / pv[t] if pv[t] != 0 else float(t - 1)
            break
    out = {'npv': npv, 'fw': fw, 'aw': aw, 'payback': payback}
    if payback is None:
        if cf[0] >= 0:
            out['payback'] = 0.0
        else:
            # A project that never pays back still has a (negative) worth — the main reason to run NPV.

            out['_warnings'] = ['the project is not paid back within the given cash-flow series (payback undefined)']
    return out


CALCS = [
    Calc('simple_interest', 'Simple interest: future value and interest (F = P(1 + n·i))',
         vars=[V('P', '', 'principal'), V('i', '', 'interest rate per period'), V('n', '', 'number of periods'),
               V('F', '', 'future value'), V('I', '', 'total interest')],
         equations=['F = P*(1 + n*i)', 'I = P*i*n'], outputs=['F', 'I'],
         example={'P': 1000.0, 'i': 0.05, 'n': 3.0},
         keywords=['simple interest', 'future value'],
         refs=['F = P(1 + ni) (simple interest)']),

    Calc('compound_interest', 'Compound interest: future value (F = P(1 + i)^n)',
         vars=[V('P', '', 'principal'), V('i', '', 'interest rate per period'), V('n', '', 'number of periods'),
               V('F', '', 'future value')],
         equations=['F = P*(1 + i)**n'], outputs=['F'],
         example={'P': 1000.0, 'i': 0.05, 'n': 3.0},
         keywords=['compound interest', 'future value', 'present worth'],
         assumptions=['Solving for P gives the present worth P = F/(1+i)^n.'],
         refs=['F = P(1 + i)^n (compound interest)']),

    Calc('effective_rate', 'Effective annual interest rate from a nominal rate (i_eff = (1 + r/m)^m − 1)',
         vars=[V('r', '', 'nominal annual rate'), V('m', '', 'compounding periods per year', lo=0.001),
               V('i_eff', '', 'effective annual rate')],
         equations=['i_eff = (1 + r/m)**m - 1'], outputs=['i_eff'],
         example={'r': 0.12, 'm': 12.0},
         keywords=['effective rate', 'nominal rate', 'compounding', 'apy'],
         assumptions=['m = 12 monthly, 4 quarterly, 365 daily. As m → ∞ the effective rate approaches e^r − 1.'],
         refs=['i_eff = (1 + r/m)^m − 1']),

    Calc('annual_worth', 'Annual worth from a present amount (capital-recovery factor)',
         vars=[V('A', '', 'annual worth (uniform series)'), V('P', '', 'present worth'),
               V('i', '', 'interest rate per period'), V('n', '', 'number of periods')],
         procedure=_annual_worth, outputs=['A'],
         example={'P': 1000.0, 'i': 0.05, 'n': 3.0},
         keywords=['capital recovery', 'crf', 'annual worth', 'equivalent annual'],
         assumptions=['At i = 0 the capital-recovery factor is the 0/0 limit A = P/n (principal spread over n periods).'],
         refs=['A = P·i(1+i)^n/((1+i)^n − 1) (capital recovery); A = P/n at i = 0']),

    Calc('sinking_fund', 'Annual amount to accumulate a future sum (sinking-fund factor)',
         vars=[V('A', '', 'annual deposit'), V('F', '', 'future value'), V('i', '', 'interest rate per period'),
               V('n', '', 'number of periods')],
         procedure=_sinking_fund, outputs=['A'],
         example={'F': 1000.0, 'i': 0.05, 'n': 3.0},
         keywords=['sinking fund', 'series compound amount'],
         assumptions=['At i = 0 the sinking-fund factor is the 0/0 limit A = F/n (no interest accrues).'],
         refs=['A = F·i/((1+i)^n − 1) (sinking fund); A = F/n at i = 0']),

    Calc('irr', 'Internal rate of return from a cash-flow series (all real roots)',
         vars=[V('cash_flows', '', 'cash flows [CF0, CF1, ...] (CF0 at t = 0; negative = outflow)', kind='json'),
               V('irr', '', 'all real IRRs as decimals (rate > −1), ascending', kind='json')],
         procedure=_irr, outputs=['irr'],
         example={'cash_flows': [-100.0, 60.0, 60.0]},
         keywords=['rate of return', 'npv', 'discounted', 'ror'],
         assumptions=['IRR solves NPV(r) = Σ CF_t/(1+r)^t = 0 for r > −1. All real roots are found from the '
                      'NPV polynomial in x = 1 + r (including tangent/even-multiplicity roots and rates near '
                      '−100 %); non-conventional flows (several sign changes) can therefore have several IRRs, '
                      'all reported ascending.'],
         refs=['Roots of NPV(r) = 0 via the NPV polynomial']),

    Calc('payback', 'Simple payback period (PP = C0/A)',
         vars=[V('PP', '', 'payback period (periods)'), V('C0', '', 'initial investment'),
               V('A', '', 'uniform net annual savings')],
         equations=['PP = C0/A'], outputs=['PP'],
         example={'C0': 10000.0, 'A': 2000.0},
         keywords=['payback', 'simple payback'],
         assumptions=['Ignores the time value of money; discounted payback is in cash_flow_series.'],
         refs=['PP = C0/A']),

    Calc('benefit_cost_ratio', 'Benefit-cost ratio (B/C)',
         vars=[V('bc', '', 'benefit-cost ratio'), V('B', '', 'present worth of benefits'),
               V('C', '', 'present worth of costs')],
         equations=['bc = B/C'], outputs=['bc'],
         example={'B': 120000.0, 'C': 100000.0},
         keywords=['benefit cost', 'b/c ratio'],
         assumptions=['B/C > 1 favours the project; benefits and costs are both present worths.'],
         refs=['B/C = PW(benefits)/PW(costs)']),

    Calc('depreciation_sl', 'Straight-line depreciation and book value',
         vars=[V('P', '', 'first cost'), V('S', '', 'salvage value'), V('n', '', 'useful life (periods)', lo=0.001),
               V('k', '', 'period index (year k)', lo=0.0),
               V('BV_k', '', 'book value after k periods', signed=True),
               V('D', '', 'annual depreciation (P − S)/n')],
         procedure=_depreciation_sl, outputs=['BV_k', 'D'],
         example={'P': 10000.0, 'S': 1000.0, 'n': 5.0, 'k': 2.0},
         keywords=['depreciation', 'straight line', 'book value'],
         assumptions=['k must not exceed the useful life n (a period beyond the life is rejected); at k = n the '
                      'book value equals the salvage value.'],
         refs=['D = (P − S)/n; BV_k = P − kD']),

    Calc('depreciation_db', 'Declining-balance depreciation and book value',
         vars=[V('P', '', 'first cost'), V('d', '', 'declining-balance rate per period', lo=0.0, hi=1.0),
               V('k', '', 'period index (year k)', lo=0.0),
               V('BV_k', '', 'book value after k periods', signed=True),
               V('D_k', '', 'depreciation in period k', signed=True)],
         equations=['BV_k = P*(1 - d)**k', 'D_k = P*(1 - d)**(k - 1)*d'], outputs=['BV_k', 'D_k'],
         example={'P': 10000.0, 'd': 0.4, 'k': 2.0},
         keywords=['depreciation', 'declining balance', 'double declining', 'book value'],
         assumptions=['d = 2/n is double-declining balance. The book value is not floored at the salvage value '
                      '(apply S as a floor in practice).'],
         refs=['BV_k = P(1 − d)^k; D_k = P(1 − d)^(k−1)d']),

    Calc('depreciation_soyd', 'Sum-of-years-digits depreciation and book value',
         vars=[V('P', '', 'first cost'), V('S', '', 'salvage value'), V('n', '', 'useful life (periods)', lo=0.001),
               V('k', '', 'period index (year k)', lo=0.0),
               V('D_k', '', 'depreciation in period k', signed=True), V('BV_k', '', 'book value after k periods', signed=True),
               V('SOYD', '', 'sum of years digits n(n+1)/2')],
         procedure=_depreciation_soyd, outputs=['D_k', 'BV_k', 'SOYD'],
         example={'P': 10000.0, 'S': 1000.0, 'n': 5.0, 'k': 2.0},
         keywords=['depreciation', 'soyd', 'book value'],
         assumptions=['k must not exceed the useful life n; at k = n the book value equals the salvage value.'],
         refs=['D_k = (P − S)(n − k + 1)/SOYD; SOYD = n(n+1)/2']),

    Calc('depreciation_table', 'Depreciation from a user-supplied percentage schedule (e.g. MACRS)',
         vars=[V('P', '', 'first cost'),
               V('rates', '', 'per-period percentages [r1, r2, ...] (MACRS rates are user inputs)', kind='json'),
               V('first_year', '', 'first-period depreciation'),
               V('depreciation', '', 'per-period depreciation amounts', kind='json'),
               V('book_values', '', 'book value at the end of each period', kind='json')],
         procedure=_depreciation_table, outputs=['first_year', 'depreciation', 'book_values'],
         example={'P': 10000.0, 'rates': [20.0, 32.0, 19.2, 11.52, 11.52, 5.76]},
         keywords=['macrs', 'depreciation', 'book value', 'schedule'],
         assumptions=['Rates are entered as percentages and applied to the first cost P (MACRS convention, no salvage). '
                      'No rate table is built in — the user supplies the percentages from the applicable code.'],
         refs=['D_k = r_k·P (rate-schedule depreciation)']),

    Calc('break_even', 'Break-even point (quantity and revenue)',
         vars=[V('Q', '', 'break-even quantity'), V('R', '', 'break-even revenue'),
               V('FC', '', 'fixed cost'), V('p', '', 'unit selling price'), V('v', '', 'unit variable cost')],
         equations=['Q = FC/(p - v)', 'R = p*Q'], outputs=['Q', 'R'],
         example={'FC': 10000.0, 'p': 50.0, 'v': 30.0},
         keywords=['break even', 'breakeven', 'cost volume profit'],
         assumptions=['Requires p > v for a positive break-even quantity.'],
         refs=['Q = FC/(p − v); R = p·Q']),

    Calc('life_cycle_cost', 'Life-cycle cost: first cost plus present worth of uniform annual costs',
         vars=[V('LCC', '', 'life-cycle cost (present worth)'), V('C0', '', 'initial (first) cost'),
               V('C_annual', '', 'uniform annual operating/maintenance cost'),
               V('i', '', 'interest rate per period'), V('n', '', 'life in periods')],
         procedure=_life_cycle_cost, outputs=['LCC'],
         example={'C0': 100000.0, 'C_annual': 5000.0, 'i': 0.05, 'n': 10.0},
         keywords=['life cycle', 'lcc', 'present worth', 'total cost'],
         assumptions=['C_annual is a uniform end-of-period series discounted at rate i over n periods. '
                      'At i = 0 the cost is simply C0 + n·C_annual.'],
         refs=['LCC = C0 + C_annual·((1+i)^n − 1)/(i(1+i)^n); LCC = C0 + n·C_annual at i = 0']),

    Calc('cash_flow_series', 'Present/future/annual worth and discounted payback of a cash-flow series',
         vars=[V('cash_flows', '', 'cash flows [CF0, CF1, ...] (CF0 at t = 0; negative = outflow)', kind='json'),
               V('i', '', 'interest rate per period'),
               V('npv', '', 'net present value (present worth)', signed=True),
               V('fw', '', 'future worth at the end of the last period', signed=True),
               V('aw', '', 'equivalent uniform annual worth', signed=True),
               V('payback', '', 'discounted payback period (periods)')],
         procedure=_cash_flow_series, outputs=['npv', 'fw', 'aw', 'payback'],
         example={'cash_flows': [-1000.0, 300.0, 400.0, 500.0], 'i': 0.05},
         keywords=['npv', 'net present value', 'present worth', 'future worth', 'annual worth', 'discounted payback'],
         assumptions=['Cash flows are end-of-period with CF0 at t = 0; n flows span an n − 1 period horizon. '
                      'FW is compounded to the end of the last period; AW is the uniform series equivalent of the '
                      'NPV over that n − 1 period horizon. Discounted payback is the first time the cumulative '
                      'discounted flow reaches zero (interpolated within the crossing period); it errors if the '
                      'project never pays back.'],
         refs=['NPV = Σ CF_t/(1+i)^t; FW = NPV(1+i)^(n−1); AW = NPV·CRF; discounted payback']),

    Calc('modified_bc', 'Modified benefit-cost ratio ((B − O&M)/C)',
         vars=[V('mbc', '', 'modified benefit-cost ratio', signed=True),
               V('B', '', 'present worth of benefits'),
               V('O', '', 'present worth of operating and maintenance costs'),
               V('C', '', 'present worth of initial investment', lo=0.001)],
         equations=['mbc = (B - O)/C'], outputs=['mbc'],
         example={'B': 120000.0, 'O': 30000.0, 'C': 80000.0},
         keywords=['benefit cost', 'modified b/c', 'public sector'],
         assumptions=['Operating and maintenance costs are subtracted from benefits in the numerator and only the '
                      'initial investment appears in the denominator (the modified B/C convention). A project is '
                      'attractive when mbc > 1.'],
         refs=['Modified B/C = (PW(benefits) − PW(O&M))/PW(initial investment)']),

    Calc('continuous_compounding', 'Effective rate under continuous compounding (i_eff = e^r − 1)',
         vars=[V('r', '', 'nominal annual rate (continuously compounded)'),
               V('i_eff', '', 'effective annual rate')],
         equations=['i_eff = exp(r) - 1'], outputs=['i_eff'],
         example={'r': 0.1},
         keywords=['continuous compounding', 'effective rate', 'e', 'nominal rate'],
         assumptions=['This is the m → ∞ limit of the nominal-rate formula i_eff = (1 + r/m)^m − 1.'],
         refs=['i_eff = e^r − 1 (continuous compounding)']),
]

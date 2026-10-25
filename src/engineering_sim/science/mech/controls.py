# DESCRIPTION: Control systems section: transfer functions and block reduction, poles/zeros and the
#              characteristic equation, second-order step metrics, steady-state error and error constants,
#              Routh–Hurwitz stability, frequency response/Bode and root locus, gain/phase margins and
#              bandwidth, Ziegler–Nichols PID tuning, and state-space controllability/observability.
"""Control systems calculators.

Transfer functions are given as JSON coefficient lists in descending powers: num = [b0, b1, …] for
b0 s^m + b1 s^(m−1) + …, den = [a0, a1, …] for a0 s^n + …. Frequency response uses scipy.signal (no
python-control). Complex poles/zeros are returned as [real, imaginary] pairs, sorted by real part then
imaginary part. Angles are radians internally (phase, phase margin); the UI shows degrees.

Relations are the standard ones in classical-control texts (e.g. Nise, *Control Systems Engineering*;
Ogata, *Modern Control Engineering*; Dorf & Bishop, *Modern Control Systems*). They are cited by relation
name, recalled rather than re-read, so no page numbers are given.
"""
import math

import numpy as np
import sympy as sp

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts

SECTION = section('controls', 'Control systems', order=380,
                  summary='Transfer functions, block reduction, poles/zeros, step metrics, steady-state error, '
                          'Routh–Hurwitz, Bode and root locus, stability margins, PID tuning and state space.')

PI = math.pi


# ---- helpers --------------------------------------------------------------------------------------
def _clist(values):
    """Sorted [real, imag] pairs from a sequence of complex roots/values (deterministic order)."""
    out = sorted(values, key=lambda c: (float(c.real), float(c.imag)))
    return [[float(c.real), float(c.imag)] for c in out]


def _coeffs(x, name):
    if not isinstance(x, (list, tuple)) or not x:
        raise MechError(f'{name}: give a coefficient list (descending powers)')
    return [float(c) for c in x]


def _tf(num, den):
    """Coefficient lists with leading zeros stripped; denominator must be non-degenerate."""
    num = _coeffs(num, 'num')
    den = _coeffs(den, 'den')
    while len(num) > 1 and num[0] == 0.0:
        num.pop(0)
    while len(den) > 1 and den[0] == 0.0:
        den.pop(0)
    if den[0] == 0.0:
        raise MechError('den: leading coefficient must be non-zero')
    return num, den


def _phase_sweep(num, den, w):
    """Continuous (unwrapped) phase of G(jω) = N(jω)/D(jω): sum of zero angles minus pole angles.

    Each term atan2(ω − Im c, −Re c) is the angle of the vector from a pole/zero c to the point jω on
    the imaginary axis; summing them (np.roots carries multiplicity) yields a continuous phase that keeps
    decreasing through −180° instead of wrapping, unlike the principal angle np.angle. The sign of the
    (real) leading coefficient contributes ±π.
    """
    phase = np.zeros_like(w, dtype=float)
    nz = list(num)
    while len(nz) > 1 and nz[0] == 0.0:
        nz.pop(0)
    if any(nz):
        if nz[0] < 0.0:
            phase += PI
        for c in np.roots(nz):
            phase += np.arctan2(w - c.imag, -c.real)
    dz = list(den)
    while len(dz) > 1 and dz[0] == 0.0:
        dz.pop(0)
    if dz[0] < 0.0:
        phase -= PI
    for c in np.roots(dz):
        phase -= np.arctan2(w - c.imag, -c.real)
    return phase


def _lead_sign(poly, eps):
    """Sign of a polynomial p(ε) for small ε > 0: the sign of its lowest-degree non-zero coefficient."""
    if poly == 0:
        return 0
    p = sp.Poly(sp.expand(poly), eps)
    for i in range(p.degree() + 1):
        c = p.coeff_monomial(eps ** i)
        if c != 0:
            return 1 if c > 0 else -1
    return 0


def _sign_limit(expr, eps):
    """Sign of a Routh first-column entry in the limit ε → 0⁺ (sign of its leading ε term)."""
    e = sp.together(expr)
    if not e.has(eps):
        v = float(e)
        return 1 if v > 0 else (-1 if v < 0 else 0)
    num, den = sp.fraction(e)
    return _lead_sign(num, eps) * _lead_sign(den, eps)


def _as_float(expr, eps):
    """Finite float limit of a Routh entry as ε → 0⁺ (None where it diverges), for JSON display."""
    if not expr.has(eps):
        return float(expr)
    lim = sp.limit(sp.together(expr), eps, 0, dir='+')
    if lim in (sp.oo, -sp.oo, sp.zoo) or not lim.is_number:
        return None
    return float(lim)


# ---- procedures -----------------------------------------------------------------------------------
def _poles_zeros(v):
    """Poles, zeros and DC gain of G(s) = num/den via scipy.signal.tf2zpk, plus the system type."""
    from scipy import signal
    num, den = _coeffs(v['num'], 'num'), _coeffs(v['den'], 'den')
    if den[0] == 0:
        raise MechError('den: leading coefficient must be non-zero')
    z, p, k = signal.tf2zpk(num, den)
    t = 0
    while t < len(den) and den[len(den) - 1 - t] == 0.0:
        t += 1  # poles at the origin: trailing zero coefficients
    return {'zeros': _clist(z), 'poles': _clist(p), 'gain': float(k), 'type': t}


def _block(v):
    """Block-diagram reduction of two transfer functions (series, parallel, feedback)."""
    s = v['setup']
    if not isinstance(s, dict):
        raise MechError('setup: give {"num1","den1","num2","den2","conn"}')
    n1, d1 = _coeffs(s.get('num1'), 'num1'), _coeffs(s.get('den1'), 'den1')
    n2, d2 = _coeffs(s.get('num2'), 'num2'), _coeffs(s.get('den2'), 'den2')
    conn = s.get('conn')
    if conn in ('series',):
        num = np.polymul(n1, n2)
        den = np.polymul(d1, d2)
    elif conn in ('parallel',):
        num = np.polyadd(np.polymul(n1, d2), np.polymul(n2, d1))
        den = np.polymul(d1, d2)
    elif conn in ('feedback', 'feedback_neg'):
        num = np.polymul(n1, d2)
        den = np.polyadd(np.polymul(d1, d2), np.polymul(n1, n2))
    elif conn in ('feedback_pos',):
        num = np.polymul(n1, d2)
        den = np.polysub(np.polymul(d1, d2), np.polymul(n1, n2))
    else:
        raise MechError("setup.conn: use 'series', 'parallel', 'feedback' (negative) or 'feedback_pos'")
    return {'num_out': [float(x) for x in num], 'den_out': [float(x) for x in den]}


def _ss_error(v):
    """System type, static error constants and steady-state errors from the open-loop G(s) = num/den.

    Type N = number of poles at the origin (trailing zero coefficients in den). With K = N(0)/D_red(0):
    Kp = K (N=0) else ∞; Kv = 0 (N=0), K (N=1), ∞ (N≥2); Ka = 0 (N≤1), K (N=2), ∞ (N>2). Steady-state
    errors: e_step = 1/(1+Kp); e_ramp = 1/Kv; e_para = 1/Ka. An infinite error is reported as null.
    """
    num, den = _coeffs(v['num'], 'num'), _coeffs(v['den'], 'den')
    if den[0] == 0:
        raise MechError('den: leading coefficient must be non-zero')
    # Cancel common factors of s at the origin (pole-zero cancellation at s = 0) so a zero open-loop
    # error constant is reported as an infinite error instead of dividing by zero.
    while len(num) > 1 and len(den) > 1 and num[-1] == 0.0 and den[-1] == 0.0:
        num.pop()
        den.pop()
    t = 0
    while t < len(den) and den[len(den) - 1 - t] == 0.0:
        t += 1
    red = den[:len(den) - t]  # remove the s^N factor
    n0 = num[-1] if num else 0.0
    d0 = red[-1] if red else 1.0
    if d0 == 0.0:
        raise MechError('den: constant term is zero after removing the s^N factor')
    K = n0 / d0
    Kp = K if t == 0 else None
    Kv = K if t == 1 else (0.0 if t == 0 else None)
    Ka = K if t == 2 else (0.0 if t < 2 else None)
    e_step = (1.0 / (1.0 + K)) if (t == 0 and 1.0 + K != 0.0) else (0.0 if t > 0 else None)
    e_ramp = (1.0 / K) if (t == 1 and K != 0.0) else (None if t == 0 else 0.0)
    e_para = (1.0 / K) if (t == 2 and K != 0.0) else (None if t < 2 else 0.0)
    return {'result': {'type': t, 'Kp': Kp, 'Kv': Kv, 'Ka': Ka,
                       'e_step': e_step, 'e_ramp': e_ramp, 'e_para': e_para}}


def _routh(v):
    """Routh–Hurwitz stability of the characteristic polynomial (descending coefficients).

    Builds the Routh array exactly (sympy rationals). A zero first-column entry with a non-zero row is
    replaced by ε and the first column is read in the limit ε → 0⁺; an all-zero row is replaced by the
    derivative of the auxiliary polynomial and signals marginal stability (roots on the imaginary axis).
    Sign changes in the first column equal the number of right-half-plane roots.
    """
    coeffs = _coeffs(v['coeffs'], 'coeffs')
    if coeffs[0] == 0.0:
        raise MechError('coeffs: leading coefficient must be non-zero')
    a0 = coeffs[0]
    coeffs = [c / a0 for c in coeffs]  # monic: does not change stability
    n = len(coeffs) - 1
    eps = sp.Symbol('_eps', positive=True)
    r0 = [sp.Rational(c) for c in coeffs[0::2]]
    r1 = [sp.Rational(c) for c in coeffs[1::2]]
    L = len(r0)
    r0 += [sp.Integer(0)] * (L - len(r0))
    r1 += [sp.Integer(0)] * (L - len(r1))
    rows = [r0, r1]
    n_imag = 0          # roots exactly on the imaginary axis, from the first auxiliary polynomial
    while len(rows) <= n:
        prev, prev2 = rows[-1], rows[-2]
        if all(x == 0 for x in prev):
            # All-zero row: replace it with the derivative of the auxiliary polynomial built from the
            # row above (prev2); signals roots on the imaginary axis (marginal stability).
            m = n - len(rows) + 2  # s-power associated with the row above
            # An all-zero row only says the polynomial has root pairs symmetric about the origin; they may be
            # ±jω (imaginary axis), ±σ (real) or quadruples. Count the imaginary-axis ones from the auxiliary
            # polynomial's roots instead of assuming all are on the axis (review of brief 22f: s²−1, s⁴+1).
            if n_imag == 0:
                aux = [0.0] * (m + 1)  # descending powers s^m … s^0
                for j in range(len(prev2)):
                    if m - 2 * j >= 0:
                        aux[2 * j] = float(prev2[j])
                roots = np.roots(aux)
                scale = max(1.0, max((abs(r) for r in roots), default=1.0))
                n_imag = int(sum(1 for r in roots if abs(r.real) <= 1e-7 * scale))
            der = [sp.Integer(0)] * L
            for j in range(len(prev2)):
                p = m - 2 * j
                if p > 0:
                    der[j] = p * prev2[j]
            if all(x == 0 for x in der):
                raise MechError('Routh table has a degenerate auxiliary polynomial; cannot decide stability')
            rows[-1] = der
            continue  # do not grow: the next pass computes the following row from the derivative
        pivot = prev[0]
        if pivot == 0:
            prev[0] = eps  # zero first-column entry (non-zero row): epsilon substitution
            pivot = eps
        new = [sp.Integer(0)] * L
        for j in range(L - 1):
            new[j] = sp.together((prev[0] * prev2[j + 1] - prev2[0] * prev[j + 1]) / pivot)
        rows.append(new)

    first = [r[0] for r in rows]
    signs = []
    for x in first:
        s = _sign_limit(x, eps)
        if s == 0:
            raise MechError('Routh table has an unresolved zero in the first column; cannot decide stability')
        signs.append(s)
    n_rhp = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
    marginal = n_rhp == 0 and n_imag > 0
    stable = 1 if (n_rhp == 0 and n_imag == 0) else 0
    first_column = [_as_float(x, eps) for x in first]
    table = [[_as_float(x, eps) for x in row] for row in rows]
    return {'n_rhp': n_rhp, 'stable': stable, 'marginal': 1 if marginal else 0, 'n_imag': n_imag,
            'first_column': first_column, 'routh_table': table}


def _freq_response(v):
    """Magnitude (linear and dB) and continuous phase of G(jω) at one frequency."""
    from scipy import signal
    num, den = _tf(v['num'], v['den'])
    w = v['omega']
    if w <= 0:
        raise MechError('omega must be positive')
    H = signal.freqresp((num, den), w=[w])[1][0]
    mag = float(abs(H))
    phase = float(_phase_sweep(num, den, np.array([w]))[0])
    return {'mag': mag, 'mag_db': 20.0 * math.log10(mag) if mag > 0 else float('-inf'),
            'phase': phase}


def _root_locus(v):
    """Closed-loop poles of 1 + K G(s) = 0 at a given gain K (den + K·num = 0)."""
    s = v['setup']
    if not isinstance(s, dict) or 'num' not in s or 'den' not in s or 'K' not in s:
        raise MechError('setup: give {"num", "den", "K"}')
    num, den = _coeffs(s['num'], 'num'), _coeffs(s['den'], 'den')
    K = float(s['K'])
    closed = np.polyadd(den, [K * c for c in num])
    return {'closed_poles': _clist(np.roots(closed))}


def _stability_margins(v):
    """Gain margin (dB), phase margin (rad), crossover frequencies and closed-loop bandwidth.

    Gain crossover ω_gc: |G| = 1 (0 dB); phase margin = π + ∠G(jω_gc) with the continuous (unwrapped)
    phase, so an already-unstable loop gives a negative margin. Phase crossover ω_pc: Im G crosses zero
    with Re G < 0 (∠G = −180°); gain margin = −20 log10 |G(jω_pc)|. A phase pinned at −180° (e.g. a
    double integrator 1/s²) has no genuine crossing, so those margins are null. Bandwidth: first ω where
    the closed loop T = G/(1+G) = N/(D+N) falls 3.0103 dB below its DC gain (null for zero/infinite DC).
    """
    from scipy import signal, optimize
    num, den = _tf(v['num'], v['den'])
    w = np.logspace(-8, 8, 16001)
    H = signal.freqresp((num, den), w=w)[1]
    mag_db = 20.0 * np.log10(np.abs(H) + 1e-300)
    re, im = np.real(H), np.imag(H)
    phase = _phase_sweep(num, den, w)

    def H_at(x):
        return signal.freqresp((num, den), w=[x])[1][0]

    def db_at(x):
        return 20.0 * np.log10(abs(H_at(x)) + 1e-300)

    w_gc = None
    for i in range(len(w) - 1):
        if mag_db[i] >= 0.0 > mag_db[i + 1]:
            w_gc = float(optimize.brentq(db_at, w[i], w[i + 1]))
            break
    phase_margin = None
    if w_gc is not None:
        ph_gc = float(np.interp(math.log(w_gc), np.log(w), phase))
        phase_margin = PI + ph_gc

    w_pc = None
    for i in range(len(w) - 1):
        if im[i] * im[i + 1] < 0.0 and re[i] < 0.0 and re[i + 1] < 0.0:
            w_pc = float(optimize.brentq(lambda x: float(np.imag(H_at(x))), w[i], w[i + 1]))
            break
    gain_margin = -db_at(w_pc) if w_pc is not None else None

    bandwidth = None
    closed_den = np.polyadd(np.array(den, dtype=float), np.array(num, dtype=float))
    T = signal.freqresp((np.array(num, dtype=float), closed_den), w=w)[1]
    t_mag_db = 20.0 * np.log10(np.abs(T) + 1e-300)
    if closed_den[-1] != 0.0:
        dc = abs(float(num[-1]) / float(closed_den[-1]))
        if dc > 0.0:
            thr = 20.0 * math.log10(dc) - 20.0 * math.log10(math.sqrt(2.0))
            for i in range(len(w) - 1):
                if t_mag_db[i] >= thr > t_mag_db[i + 1]:
                    def Tdb_at(x):
                        return 20.0 * math.log10(abs(signal.freqresp(
                            (np.array(num, dtype=float), closed_den), w=[x])[1][0]) + 1e-300)
                    bandwidth = float(optimize.brentq(lambda x: Tdb_at(x) - thr, w[i], w[i + 1]))
                    break
    return {'gain_margin': gain_margin, 'phase_margin': phase_margin,
            'w_gc': w_gc, 'w_pc': w_pc, 'bandwidth': bandwidth}


def _pid_zn(v):
    """PID/P/PI gains by the closed-loop (ultimate-gain) Ziegler–Nichols rules from K_u and T_u.

    Variant: ultimate-sensitivity (Ziegler–Nichols, 1942). P: Kp = 0.5 K_u. PI: Kp = 0.45 K_u,
    Ti = T_u/1.2. PID: Kp = 0.6 K_u, Ti = T_u/2, Td = T_u/8 (parallel form Ki = Kp/Ti, Kd = Kp·Td).
    """
    Ku, Tu = v['K_u'], v['T_u']
    if Ku <= 0 or Tu <= 0:
        raise MechError('K_u and T_u must be positive')
    return {'Ki_PID': 1.2 * Ku / Tu, 'Kp_PID': 0.6 * Ku, 'Ti_PID': Tu / 2.0, 'Td_PID': Tu / 8.0,
            'Kd_PID': 0.6 * Ku * Tu / 8.0, 'Kp_P': 0.5 * Ku, 'Kp_PI': 0.45 * Ku, 'Ti_PI': Tu / 1.2,
            'Ki_PI': 0.45 * Ku / (Tu / 1.2)}


def _state_space(v):
    """Poles and controllability/observability of ẋ = A x + B u, y = C x.

    Controllability matrix [B, AB, …, A^(n−1)B]; observability matrix [C; CA; …; CA^(n−1)]. Full rank ⇔
    controllable/observable. The feedthrough matrix D does not affect any of these quantities.
    """
    A = np.array(v['A'], dtype=float)
    B = np.array(v['B'], dtype=float)
    C = np.array(v['C'], dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise MechError('A must be a square matrix')
    n = A.shape[0]
    if B.ndim != 2 or B.shape[0] != n:
        raise MechError(f'B must have {n} rows')
    if C.ndim != 2 or C.shape[1] != n:
        raise MechError(f'C must have {n} columns')
    ctrb = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    obsv = np.vstack([C @ np.linalg.matrix_power(A, i) for i in range(n)])
    rc, ro = int(np.linalg.matrix_rank(ctrb)), int(np.linalg.matrix_rank(obsv))
    return {'eigenvalues': _clist(np.linalg.eigvals(A)), 'ctrb_rank': rc, 'obsv_rank': ro,
            'controllable': 1 if rc == n else 0, 'observable': 1 if ro == n else 0}


# ---- chart builders --------------------------------------------------------------------------------
def _bode_chart(vals):
    from scipy import signal
    num, den = vals['num'], vals['den']
    w0 = max(vals['omega'], 1e-6)
    w = np.logspace(math.log10(w0) - 2, math.log10(w0) + 2, 120)
    H = signal.freqresp((num, den), w=w)[1]
    mag = 20.0 * np.log10(np.abs(H) + 1e-300)
    ph = np.unwrap(np.angle(H)) * 180.0 / PI
    return [charts.chart('Bode magnitude', 'ω [rad/s]', '|G| [dB]',
                         [charts.series('|G|', w, mag)], xlog=True),
            charts.chart('Bode phase', 'ω [rad/s]', '∠G [°]', [charts.series('∠G', w, ph)], xlog=True)]


def _root_locus_chart(vals):
    s = vals['setup']
    num, den = s['num'], s['den']
    K0 = max(float(s['K']), 1e-9)
    series_by_branch = []
    Ks = np.logspace(math.log10(K0) - 3, math.log10(K0) + 3, 60)
    n_poles = len(den) - 1
    branch_re, branch_im = [[] for _ in range(n_poles)], [[] for _ in range(n_poles)]
    for K in Ks:
        roots = np.roots(np.polyadd(den, [K * c for c in num]))
        for i, r in enumerate(roots):
            branch_re[i].append(float(r.real))
            branch_im[i].append(float(r.imag))
    series_list = []
    for i in range(n_poles):
        series_list.append(charts.series(f'branch {i + 1}', branch_re[i], branch_im[i]))
    return charts.chart('Root locus (K swept)', 'Re(s)', 'Im(s)', series_list, equal=True,
                        markers=[charts.marker(0, 0, 'origin')], hlines=[0], vlines=[0])


CALCS = [
    Calc('poles_zeros', 'Poles, zeros and system type of a transfer function',
         vars=[V('num', '', 'numerator coefficients, descending powers, e.g. [1, 1]', kind='json'),
               V('den', '', 'denominator coefficients, descending powers, e.g. [1, 2, 0]', kind='json'),
               V('zeros', '', 'zeros as [real, imag] pairs, sorted', kind='json'),
               V('poles', '', 'poles (characteristic-equation roots) as [real, imag] pairs, sorted', kind='json'),
               V('gain', '', 'transfer-function gain k from scipy tf2zpk (signed: k can be negative)', signed=True),
               V('type', '', 'system type (number of open-loop poles at the origin)', integer=True)],
         procedure=_poles_zeros, outputs=['zeros', 'poles', 'gain', 'type'],
         example={'num': [1, 1], 'den': [1, 2, 0]},
         keywords=['transfer function', 'poles', 'zeros', 'characteristic equation', 'type'],
         assumptions=['The characteristic equation is den(s) = 0; its roots are the poles. A pole with '
                      'positive real part means the open loop is unstable.'],
         refs=['scipy.signal.tf2zpk']),

    Calc('block_reduction', 'Block-diagram reduction (series / parallel / feedback)',
         vars=[V('setup', '', '{"num1","den1","num2","den2","conn"} with conn series | parallel | feedback (negative) | feedback_pos', kind='json'),
               V('num_out', '', 'reduced numerator coefficients (descending powers)', kind='json'),
               V('den_out', '', 'reduced denominator coefficients (descending powers)', kind='json')],
         procedure=_block, outputs=['num_out', 'den_out'],
         example={'setup': {'num1': [1], 'den1': [1, 1], 'num2': [1], 'den2': [1, 2], 'conn': 'series'}},
         keywords=['block diagram', 'series', 'parallel', 'feedback', 'transfer function'],
         assumptions=['Two-block reduction: series G1·G2, parallel G1 + G2, feedback G1/(1 ± G1·G2). Common '
                      'pole/zero factors are not cancelled (use poles_zeros to inspect).'],
         refs=['Block-diagram algebra (series, parallel, feedback)']),

    Calc('second_order_step', 'Second-order step response metrics (damping ratio and natural frequency)',
         vars=[V('tau', 's', 'time constant τ = 1/(ζ ωn)'),
               V('ts', 's', 'settling time to within 2 % (t_s = 4τ)'),
               V('tp', 's', 'peak time'),
               V('tr', 's', 'rise time (0–100 %, t_r = (π − acos ζ)/ωd)'),
               V('OS_pct', '', 'percent overshoot %OS = 100·exp(−ζπ/√(1−ζ²))'),
               V('zeta', '', 'damping ratio (0 < ζ < 1 underdamped)', hi=0.999, label='ζ'),
               V('wn', 'rad/s', 'undamped natural frequency', label='ωn')],
         equations=['tau = 1/(zeta*wn)', 'ts = 4*tau', 'tp = pi/(wn*sqrt(1 - zeta**2))',
                    'tr = (pi - acos(zeta))/(wn*sqrt(1 - zeta**2))',
                    'OS_pct = 100*exp(-zeta*pi/sqrt(1 - zeta**2))'],
         outputs=['tau', 'ts', 'tp', 'tr', 'OS_pct'],
         example={'zeta': 0.5, 'wn': 10.0},
         keywords=['overshoot', 'rise time', 'peak time', 'settling time', 'time constant', 'damping'],
         assumptions=['Standard underdamped second-order step response, 0 < ζ < 1. Settling uses the 2 % '
                      'criterion t_s = 4/(ζωn); the rise-time formula is the 0–100 % definition.'],
         refs=['Second-order step metrics: tp = π/ωd; %OS = 100 e^(−ζπ/√(1−ζ²)); ts = 4/(ζωn) (2 % criterion)']),

    Calc('steady_state_error', 'System type, error constants and steady-state errors',
         vars=[V('num', '', 'open-loop numerator coefficients (descending powers)', kind='json'),
               V('den', '', 'open-loop denominator coefficients (descending powers)', kind='json'),
               V('result', '', '{"type","Kp","Kv","Ka","e_step","e_ramp","e_para"}; null = infinite', kind='json')],
         procedure=_ss_error, outputs=['result'],
         example={'num': [10], 'den': [1, 3, 2]},
         keywords=['steady state error', 'error constant', 'system type', 'position', 'velocity'],
         assumptions=['Unity feedback, unit test inputs. e_step = 1/(1+Kp), e_ramp = 1/Kv, e_para = 1/Ka. '
                      'Infinite error/constant is reported as null (e.g. a type-0 system tracking a ramp).'],
         refs=['Static error constants Kp, Kv, Ka; steady-state errors']),

    Calc('routh_hurwitz', 'Routh–Hurwitz stability of a characteristic polynomial',
         vars=[V('coeffs', '', 'characteristic coefficients, descending powers, e.g. [1, 6, 11, 6]', kind='json'),
               V('n_rhp', '', 'number of right-half-plane (unstable) roots', integer=True),
               V('stable', '', '1 if all roots are strictly in the left half-plane, else 0', integer=True),
               V('marginal', '', '1 if no RHP roots but roots on the imaginary axis (repeated jω roots are in fact unstable)', integer=True),
               V('n_imag', '', 'number of roots on the imaginary axis (from the auxiliary polynomial)', integer=True),
               V('first_column', '', 'first column of the Routh array (ε → 0 limit; null where unbounded)', kind='json'),
               V('routh_table', '', 'Routh array rows (ε → 0 limit; padded with zeros)', kind='json')],
         procedure=_routh, outputs=['n_rhp', 'stable', 'marginal', 'n_imag', 'first_column', 'routh_table'],
         example={'coeffs': [1, 6, 11, 6]},
         keywords=['routh', 'hurwitz', 'stability', 'characteristic equation'],
         assumptions=['Sign changes in the first column equal the number of RHP roots (Routh–Hurwitz). A '
                      'zero first-column entry is handled by the ε → 0 limit; an all-zero row is replaced '
                      'by the derivative of the auxiliary polynomial and signals marginal stability.'],
         refs=['Routh–Hurwitz stability criterion']),

    Calc('freq_response', 'Frequency response magnitude and phase at one frequency (Bode)',
         vars=[V('num', '', 'numerator coefficients (descending powers)', kind='json'),
               V('den', '', 'denominator coefficients (descending powers)', kind='json'),
               V('omega', 'rad/s', 'frequency (enter rad/s or rpm)', label='ω'),
               V('mag', '', 'magnitude |G(jω)|'),
               V('mag_db', '', 'magnitude in dB (negative for attenuation)', signed=True),
               V('phase', 'rad', 'continuous phase of G(jω) (sum of zero angles minus pole angles)', signed=True)],
         procedure=_freq_response, outputs=['mag', 'mag_db', 'phase'],
         example={'num': [1], 'den': [1, 1], 'omega': 1.0},
         plots=[{'id': 'bode', 'title': 'Bode plot', 'fn': _bode_chart}],
         keywords=['bode', 'frequency response', 'magnitude', 'phase'],
         assumptions=['Single frequency point; use the bode plot for a sweep. Phase is the continuous '
                      '(unwrapped) angle built as the sum of zero angles minus pole angles, so it keeps '
                      'decreasing through −180° instead of wrapping to +180°.'],
         refs=['G(jω) = N(jω)/D(jω) via scipy.signal.freqresp']),

    Calc('root_locus', 'Closed-loop poles at a gain (root locus)',
         vars=[V('setup', '', '{"num","den","K"} for the loop gain K·G(s)', kind='json'),
               V('closed_poles', '', 'roots of den + K·num as [real, imag] pairs, sorted', kind='json')],
         procedure=_root_locus, outputs=['closed_poles'],
         example={'setup': {'num': [1], 'den': [1, 2, 0], 'K': 2.0}},
         plots=[{'id': 'locus', 'title': 'Root locus', 'fn': _root_locus_chart}],
         keywords=['root locus', 'closed loop', 'poles', 'gain'],
         assumptions=['Negative unity feedback: closed-loop characteristic equation den + K·num = 0. The '
                      'locus chart sweeps K about the given value; branches are drawn as pole trajectories.'],
         refs=['1 + K G(s) = 0 (closed-loop characteristic equation)']),

    Calc('stability_margins', 'Gain margin, phase margin, crossover frequencies and bandwidth',
         vars=[V('num', '', 'open-loop numerator coefficients (descending powers)', kind='json'),
               V('den', '', 'open-loop denominator coefficients (descending powers)', kind='json'),
               V('gain_margin', '', 'gain margin in dB (null if undefined; negative for an unstable system)', signed=True),
               V('phase_margin', 'rad', 'phase margin (null if undefined; negative for an unstable system)', signed=True),
               V('w_gc', 'rad/s', 'gain-crossover frequency (|G| = 1)'),
               V('w_pc', 'rad/s', 'phase-crossover frequency (∠G = −180°; null if the phase never crosses −180°)'),
               V('bandwidth', 'rad/s', 'closed-loop −3 dB bandwidth of T = G/(1+G) (null if DC gain is zero or infinite)')],
         procedure=_stability_margins, outputs=['gain_margin', 'phase_margin', 'w_gc', 'w_pc', 'bandwidth'],
         example={'num': [1], 'den': [1, 1, 0]},
         keywords=['gain margin', 'phase margin', 'bandwidth', 'crossover', 'bode'],
         assumptions=['Open-loop frequency response with a continuous (unwrapped) phase. Margins are taken '
                      'from genuine crossings: a phase pinned at −180° (e.g. a double integrator) has no '
                      'phase crossover, so gain margin and ω_pc are null. Bandwidth is the closed-loop '
                      'T = G/(1+G) −3 dB frequency.'],
         refs=['Gain/phase margin definitions from the Bode plot']),

    Calc('pid_ziegler_nichols', 'Ziegler–Nichols PID gains from ultimate gain and period',
         vars=[V('Ki_PID', '1/s', 'PID integral gain Ki = Kp/Ti'),
               V('Kp_PID', '', 'PID proportional gain'),
               V('Ti_PID', 's', 'PID integral time'),
               V('Td_PID', 's', 'PID derivative time'),
               V('Kd_PID', 's', 'PID derivative gain Kd = Kp·Td'),
               V('Kp_P', '', 'P-only proportional gain'),
               V('Kp_PI', '', 'PI proportional gain'),
               V('Ti_PI', 's', 'PI integral time'),
               V('Ki_PI', '1/s', 'PI integral gain Ki = Kp/Ti'),
               V('K_u', '', 'ultimate gain (from the sustained-oscillation test)', label='Ku'),
               V('T_u', 's', 'ultimate period (of the sustained oscillation)', label='Tu')],
         procedure=_pid_zn, outputs=['Ki_PID', 'Kp_PID', 'Ti_PID', 'Td_PID', 'Kd_PID',
                                     'Kp_P', 'Kp_PI', 'Ti_PI', 'Ki_PI'],
         example={'K_u': 4.0, 'T_u': 2.0},
         keywords=['pid', 'ziegler nichols', 'tuning', 'ultimate gain'],
         assumptions=['Closed-loop Ziegler–Nichols (ultimate-sensitivity) method: P Kp = 0.5 K_u; PI Kp = '
                      '0.45 K_u, Ti = T_u/1.2; PID Kp = 0.6 K_u, Ti = T_u/2, Td = T_u/8. These give a '
                      'quarter-decay response; retune for your process.'],
         refs=['Ziegler–Nichols closed-loop (ultimate-gain) tuning rules']),

    Calc('state_space', 'State-space eigenvalues, controllability and observability',
         vars=[V('A', '', 'state matrix A (n×n)', kind='json'),
               V('B', '', 'input matrix B (n×m)', kind='json'),
               V('C', '', 'output matrix C (p×n)', kind='json'),
               V('eigenvalues', '', 'eigenvalues (system poles) as [real, imag] pairs, sorted', kind='json'),
               V('ctrb_rank', '', 'rank of the controllability matrix', integer=True),
               V('obsv_rank', '', 'rank of the observability matrix', integer=True),
               V('controllable', '', '1 if ctrb_rank = n, else 0', integer=True),
               V('observable', '', '1 if obsv_rank = n, else 0', integer=True)],
         procedure=_state_space, outputs=['eigenvalues', 'ctrb_rank', 'obsv_rank', 'controllable', 'observable'],
         example={'A': [[0, 1], [-2, -3]], 'B': [[0], [1]], 'C': [[1, 0]]},
         keywords=['state space', 'controllability', 'observability', 'kalman'],
         assumptions=['ẋ = A x + B u, y = C x. The feedthrough matrix D does not affect the eigenvalues, '
                      'controllability or observability.'],
         refs=['Controllability/observability matrices [B, AB, …] and [C; CA; …]']),
]

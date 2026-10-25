# DESCRIPTION: Vibrations section: SDOF natural frequency and damping, logarithmic decrement, free and
#              forced response, transmissibility and isolation, rotating-unbalance and base excitation,
#              multi-DOF natural frequencies and modes, vibration absorbers, torsional and shaft critical
#              speeds, continuous systems (string, bar, beam), quality factor and the Duhamel integral.
"""Vibrations calculators.

Conventions (read before using these relations):
- omega_n is the undamped natural (angular) frequency in rad/s; f_n = omega_n/(2 pi) is the cyclic
  frequency in Hz (declared as separate variables so the angular/cyclic conversion is explicit, not a
  hidden 2 pi unit factor). The period is T = 1/f_n.
- Damping ratio zeta = c/c_c with critical damping c_c = 2 sqrt(k m); underdamped means 0 <= zeta < 1.
  The damped natural frequency is omega_d = omega_n sqrt(1 - zeta^2).
- Forced response uses the frequency ratio r = omega/omega_n. The phase angle phi lags the force.
- All masses/stiffnesses are SI; enter rpm for angular speeds and it converts to rad/s.

Reference relations are the standard ones in vibration texts (e.g. Thomson, *Theory of Vibration*;
Rao, *Mechanical Vibrations*; Den Hartog, *Mechanical Vibrations*). They are cited by relation name,
recalled rather than re-read, so no page numbers are given.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts

SECTION = section('vibrations', 'Vibrations', order=80,
                  summary='SDOF/MDOF natural frequencies, damping, forced and shock response, '
                          'isolation, absorbers, torsional/critical speeds and continuous systems.')

PI = math.pi
TWOPI = 2.0 * PI

_STEP_TIMES = [0.05 * i for i in range(61)]
_STEP_FORCES = [10.0] * 61


# ---------------------------------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------------------------------
def _mdof(values):
    """Natural frequencies (rad/s, Hz) and mass-normalized mode shapes of an undamped MDOF system.

    Solves the generalized eigenproblem K phi = omega^2 M phi with scipy.linalg.eigh; M and K are
    symmetric positive-definite matrices (json, kg and N/m respectively)."""
    import numpy as np
    from scipy.linalg import eigh
    M = np.asarray(values['M'], float)
    K = np.asarray(values['K'], float)
    if M.ndim != 2 or M.shape != K.shape or M.shape[0] != M.shape[1]:
        raise MechError('M and K must be square matrices of the same size')
    if M.shape[0] < 1:
        raise MechError('M and K must be non-empty')
    if not np.allclose(M, M.T, atol=1e-9) or not np.allclose(K, K.T, atol=1e-9):
        raise MechError('M and K must be symmetric')
    try:
        w2, V = eigh(K, M)
    except Exception as exc:
        raise MechError('generalized eigenproblem failed: %s' % exc)
    if np.any(w2 <= 0.0):
        raise MechError('M and K must be positive definite (a natural frequency <= 0 was found)')
    omega = np.sqrt(w2)
    freqs = [float(x) for x in omega]
    f_n = [float(x) / TWOPI for x in omega]
    modes = [V[:, j].tolist() for j in range(M.shape[0])]
    return {'freqs': freqs, 'f_n': f_n, 'modes': modes}


def _absorber_frequencies(values):
    """Natural frequencies of a main mass m (spring k) with a tuned/attached absorber m_a, k_a.

    Roots of m m_a omega^4 - [k m_a + k_a (m + m_a)] omega^2 + k k_a = 0 (det of [K - omega^2 M])."""
    m, k, ma, ka = values['m'], values['k'], values['m_a'], values['k_a']
    A = m * ma
    Bc = k * ma + ka * (m + ma)
    C = k * ka
    disc = Bc * Bc - 4.0 * A * C
    if disc < 0.0:
        raise MechError('negative discriminant (check the input units)')
    Om1 = (Bc - math.sqrt(disc)) / (2.0 * A)
    Om2 = (Bc + math.sqrt(disc)) / (2.0 * A)
    return {'omega1': math.sqrt(Om1), 'omega2': math.sqrt(Om2)}


def _rayleigh(values):
    """First critical speed by Rayleigh's method from measured static deflections.

    masses = [[W (N), y (m)], ...]: load and its static deflection. omega_cr = sqrt(g sum(W y)/sum(W y^2))."""
    masses = values['masses']
    if not isinstance(masses, list) or not masses:
        raise MechError('masses: give [[W (N), y (m)], ...]')
    num = den = 0.0
    for row in masses:
        if len(row) != 2:
            raise MechError('each row must be [W_N, y_m]')
        W, y = float(row[0]), float(row[1])
        num += W * y
        den += W * y * y
    if den <= 0.0:
        raise MechError('deflection data must give a positive denominator')
    w = math.sqrt(9.80665 * num / den)
    return {'omega_cr': w, 'f_cr': w / TWOPI}


def _dunkerley(values):
    """First critical speed by Dunkerley's method from the individual critical speeds (rad/s)."""
    speeds = values['speeds']
    if not isinstance(speeds, list) or not speeds:
        raise MechError('speeds: give the individual critical speeds in rad/s')
    s = 0.0
    for x in speeds:
        s += 1.0 / (float(x) ** 2)
    w = 1.0 / math.sqrt(s)
    return {'omega_cr': w, 'f_cr': w / TWOPI}


def _beam_frequency(values):
    """Natural frequency of a uniform beam from its (numerically found) betaL root.

    bc: 'pinned-pinned', 'fixed-free' (cantilever), 'fixed-fixed' or 'free-free'. The characteristic
    equation cos(betaL) cosh(betaL) = 1 (fixed-fixed/free-free) or cos(betaL) cosh(betaL) = -1
    (fixed-free) is root-found per mode; pinned-pinned is closed form n pi. omega_n = beta^2 sqrt(EI/(rho A))."""
    E, I, rho, A, L = values['E'], values['I'], values['rho'], values['A'], values['L']
    n = int(round(values['n']))
    bc = values['bc']
    if bc not in ('pinned-pinned', 'fixed-free', 'fixed-fixed', 'free-free'):
        raise MechError('bc must be pinned-pinned, fixed-free, fixed-fixed or free-free')
    if n < 1:
        raise MechError('n must be a positive mode number')
    if bc == 'pinned-pinned':
        betaL = n * PI
    else:
        import scipy.optimize as opt

        def g(x):
            if bc == 'fixed-free':
                return math.cos(x) + 1.0 / math.cosh(x)
            return math.cos(x) - 1.0 / math.cosh(x)

        roots = []
        x = 0.01
        prev = g(x)
        while len(roots) < n and x < n * PI + 3.0:
            x += 0.02
            y = g(x)
            if y * prev < 0.0:
                roots.append(opt.brentq(g, x - 0.02, x))
            prev = y
        if len(roots) < n:
            raise MechError('could not find the mode-%d root' % n)
        betaL = roots[n - 1]
    beta = betaL / L
    omega = beta * beta * math.sqrt(E * I / (rho * A))
    return {'omega_n': omega, 'f_n': omega / TWOPI, 'betaL': betaL}


def _duhamel(values):
    """Response of an underdamped SDOF system to a force history via the Duhamel (convolution) integral.

    times and forces are equal-length json lists (s and N). The force history is treated as
    piecewise-constant between the given time points (a zero-order hold) and the response is built by
    superposition of the exact unit-step response h(t) = (1/k)[1 - e^{-zeta wn t}(cos(wd t) +
    (zeta wn/wd) sin(wd t))], i.e. x(t) = sum_j (F_j - F_{j-1}) h(t - t_j). This is exact for a
    zero-order-hold force history (and for a step input) and accurate for smooth histories sampled finely."""
    import numpy as np
    m, k, zeta = values['m'], values['k'], values['zeta']
    t = np.asarray(values['times'], float)
    F = np.asarray(values['forces'], float)
    if t.ndim != 1 or F.ndim != 1 or len(t) != len(F):
        raise MechError('times and forces must be equal-length lists')
    if len(t) < 2:
        raise MechError('need at least two time points')
    if zeta < 0.0 or zeta >= 1.0:
        raise MechError('Duhamel integral here covers underdamped systems (0 <= zeta < 1)')
    if np.any(np.diff(t) <= 0.0):
        raise MechError('times must be strictly increasing')
    wn = math.sqrt(k / m)
    wd = wn * math.sqrt(1.0 - zeta * zeta)
    coef = zeta * wn / wd

    def step_resp(tau):
        if tau < 0.0:
            return 0.0
        e = math.exp(-zeta * wn * tau)
        return (1.0 - e * (math.cos(wd * tau) + coef * math.sin(wd * tau))) / k

    resp = []
    for i in range(len(t)):
        ti = t[i]
        x = 0.0
        prevF = 0.0
        for j in range(i + 1):
            dF = F[j] - prevF
            if dF != 0.0:
                x += dF * step_resp(ti - t[j])
            prevF = F[j]
        resp.append(float(x))
    # Extreme response = the sample with the largest |x|, reported with its sign and time.
    i_max = max(range(len(resp)), key=lambda i: abs(resp[i]))
    x_max = resp[i_max]
    t_max = float(t[i_max])
    return {'response': resp, 'x_max': x_max, 't_max': t_max}


def _frequency_response_chart(values):
    """Magnification factor and phase vs frequency ratio r (0.05..3) for the given damping ratio."""
    zeta = values['zeta']
    rs = [0.05 + 2.95 * i / 240.0 for i in range(241)]
    Ms = [1.0 / math.sqrt((1.0 - r * r) ** 2 + (2.0 * zeta * r) ** 2) for r in rs]
    phis = [math.degrees(math.atan2(2.0 * zeta * r, 1.0 - r * r)) for r in rs]
    return charts.chart('Frequency response', 'frequency ratio r = omega/omega_n', 'M and phase [deg]',
                        [charts.series('magnification M', rs, Ms),
                         charts.series('phase angle', rs, phis)])


CALCS = [
    # ---- free SDOF -------------------------------------------------------------------------------
    Calc('natural_frequency', 'Undamped natural frequency, cyclic frequency and period',
         vars=[V('omega_n', 'rad/s', 'natural (angular) frequency', label='ωn'),
               V('f_n', 'Hz', 'natural (cyclic) frequency'), V('T', 's', 'period'),
               V('k', 'N/m', 'stiffness'), V('m', 'kg', 'mass')],
         equations=['omega_n = sqrt(k/m)', 'f_n = omega_n/(2*pi)', 'T = 1/f_n'],
         outputs=['omega_n', 'f_n', 'T'], example={'k': 4000.0, 'm': 10.0},
         keywords=['natural frequency', 'period', 'sqrt k/m'],
         assumptions=['Undamped single degree of freedom. omega_n is in rad/s and f_n = omega_n/(2 pi) in Hz '
                      '(the 2 pi is the physical angular/cyclic relation, not a unit conversion).'],
         refs=['omega_n = sqrt(k/m); f_n = omega_n/(2 pi); T = 1/f_n']),

    Calc('equiv_stiffness', 'Equivalent stiffness of two springs in series or parallel',
         vars=[V('k_series', 'N/m', 'equivalent stiffness in series (1/k = sum 1/k_i)'),
               V('k_parallel', 'N/m', 'equivalent stiffness in parallel (k = sum k_i)'),
               V('k1', 'N/m', 'first spring stiffness'), V('k2', 'N/m', 'second spring stiffness')],
         equations=['k_series = k1*k2/(k1 + k2)', 'k_parallel = k1 + k2'],
         outputs=['k_series', 'k_parallel'], example={'k1': 2000.0, 'k2': 3000.0},
         keywords=['springs', 'series', 'parallel', 'equivalent stiffness'],
         refs=['Series: 1/k_eq = 1/k1 + 1/k2; parallel: k_eq = k1 + k2']),

    Calc('equiv_mass', 'Equivalent mass of a spring-mass system (spring mass correction)',
         vars=[V('m_eq', 'kg', 'equivalent mass'), V('m', 'kg', 'attached mass'),
               V('m_s', 'kg', 'spring mass')],
         equations=['m_eq = m + m_s/3'], outputs=['m_eq'], example={'m': 10.0, 'm_s': 0.3},
         keywords=['equivalent mass', 'spring mass', 'rayleigh'],
         assumptions=['Approximate Rayleigh correction for a spring of uniform mass: one third of the '
                      'spring mass acts at the free end.'],
         refs=['m_eq = m + m_spring/3 (Rayleigh spring-mass correction, recalled)']),

    # ---- damping ---------------------------------------------------------------------------------
    Calc('damping', 'Damping ratio and critical damping coefficient',
         vars=[V('zeta', '', 'damping ratio c/c_c', label='ζ'), V('c_c', 'N*s/m', 'critical damping coefficient'),
               V('c', 'N*s/m', 'damping coefficient'), V('k', 'N/m', 'stiffness'), V('m', 'kg', 'mass')],
         equations=['c_c = 2*sqrt(k*m)', 'zeta = c/c_c'], outputs=['c_c', 'zeta'],
         example={'c': 50.0, 'k': 4000.0, 'm': 10.0},
         keywords=['damping ratio', 'critical damping', 'dashpot'],
         assumptions=['zeta < 1 underdamped, = 1 critically damped, > 1 overdamped.'],
         refs=['c_c = 2 sqrt(k m) = 2 m omega_n; zeta = c/c_c']),

    Calc('log_decrement', 'Logarithmic decrement and damped natural frequency',
         vars=[V('delta', '', 'logarithmic decrement', label='δ'), V('omega_d', 'rad/s', 'damped natural frequency', label='ωd'),
               V('zeta', '', 'damping ratio (< 1)', hi=0.99, label='ζ'), V('omega_n', 'rad/s', 'undamped natural frequency', label='ωn')],
         equations=['delta = 2*pi*zeta/sqrt(1 - zeta**2)', 'omega_d = omega_n*sqrt(1 - zeta**2)'],
         outputs=['delta', 'omega_d'], example={'zeta': 0.125, 'omega_n': 20.0},
         keywords=['log decrement', 'damped frequency'],
         assumptions=['Underdamped only (zeta < 1). delta is the natural log of the ratio of two successive '
                      'same-side peak amplitudes.'],
         refs=['delta = 2 pi zeta / sqrt(1 - zeta^2); omega_d = omega_n sqrt(1 - zeta^2)']),

    # ---- free and forced response ----------------------------------------------------------------
    Calc('free_response', 'Free-vibration amplitude and phase from initial conditions',
         vars=[V('A', 'm', 'response amplitude'), V('phi', 'rad', 'phase angle (rad)', signed=True, label='φ'),
               V('omega_d', 'rad/s', 'damped natural frequency', label='ωd'),
               V('x0', 'm', 'initial displacement', signed=True), V('v0', 'm/s', 'initial velocity', signed=True),
               V('zeta', '', 'damping ratio (< 1)', hi=0.99, label='ζ'), V('omega_n', 'rad/s', 'undamped natural frequency', label='ωn')],
         equations=['omega_d = omega_n*sqrt(1 - zeta**2)',
                    'A = sqrt(x0**2 + ((v0 + zeta*omega_n*x0)/omega_d)**2)',
                    'phi = atan2(v0 + zeta*omega_n*x0, omega_d*x0)'],
         outputs=['A', 'phi', 'omega_d'], example={'x0': 0.01, 'v0': 0.2, 'zeta': 0.1, 'omega_n': 20.0},
         keywords=['initial conditions', 'amplitude', 'phase'],
         assumptions=['Underdamped free response x(t) = A e^{-zeta omega_n t} cos(omega_d t - phi) with '
                      'omega_d = omega_n sqrt(1 - zeta^2) derived from zeta and omega_n.'],
         refs=['A and phi from the initial displacement and velocity of an underdamped oscillator']),

    Calc('forced_response', 'Forced steady-state response: magnification factor, amplitude and phase',
         vars=[V('r', '', 'frequency ratio omega/omega_n'), V('M', '', 'magnification factor X/(F0/k)'),
               V('X', 'm', 'steady-state amplitude'), V('phi', 'rad', 'phase angle (rad, lags the force)', signed=True, label='φ'),
               V('F0', 'N', 'forcing amplitude'), V('k', 'N/m', 'stiffness'), V('omega', 'rad/s', 'forcing frequency', label='ω'),
               V('omega_n', 'rad/s', 'natural frequency', label='ωn'), V('zeta', '', 'damping ratio', label='ζ')],
         equations=['r = omega/omega_n', 'M = 1/sqrt((1 - r**2)**2 + (2*zeta*r)**2)',
                    'X = F0/k*M', 'phi = atan2(2*zeta*r, 1 - r**2)'],
         outputs=['r', 'M', 'X', 'phi'], example={'F0': 100.0, 'k': 4000.0, 'omega': 15.0, 'omega_n': 20.0, 'zeta': 0.1},
         plots=[{'id': 'freq_resp', 'title': 'Frequency response', 'fn': _frequency_response_chart}],
         keywords=['forced vibration', 'resonance', 'magnification', 'phase angle'],
         assumptions=['Harmonic force F0 sin(omega t); steady-state response at the forcing frequency. M peaks '
                      'near r = 1 (resonance). The phase lag phi is atan2(2 zeta r, 1 - r^2).'],
         refs=['Steady-state forced response of an SDOF oscillator']),

    # ---- isolation and excitation ----------------------------------------------------------------
    Calc('transmissibility', 'Force transmissibility and isolation efficiency',
         vars=[V('TR', '', 'transmissibility (transmitted/applied force)'),
               V('eta', '', 'isolation efficiency 1 - TR', signed=True, label='η'),
               V('zeta', '', 'damping ratio', label='ζ'), V('r', '', 'frequency ratio omega/omega_n')],
         equations=['TR = sqrt((1 + (2*zeta*r)**2)/((1 - r**2)**2 + (2*zeta*r)**2))', 'eta = 1 - TR'],
         outputs=['TR', 'eta'], example={'zeta': 0.05, 'r': 2.0},
         keywords=['isolation', 'transmissibility', 'vibration isolation'],
         assumptions=['Force transmitted through the mount to the foundation. Isolation (TR < 1) requires '
                      'r > sqrt(2); eta < 0 below that means amplification.'],
         refs=['TR = sqrt((1 + (2 zeta r)^2)/((1 - r^2)^2 + (2 zeta r)^2))']),

    Calc('rotating_unbalance_response', 'Steady-state response to a rotating unbalance',
         vars=[V('X', 'm', 'response amplitude'), V('phi', 'rad', 'phase angle (rad)', signed=True, label='φ'),
               V('m_e', 'kg', 'eccentric (unbalance) mass'), V('e', 'm', 'eccentricity'),
               V('M', 'kg', 'total mass of the machine'), V('r', '', 'frequency ratio omega/omega_n'),
               V('zeta', '', 'damping ratio', label='ζ')],
         equations=['X = m_e*e/M*r**2/sqrt((1 - r**2)**2 + (2*zeta*r)**2)',
                    'phi = atan2(2*zeta*r, 1 - r**2)'],
         outputs=['X', 'phi'], example={'m_e': 0.5, 'e': 0.01, 'M': 20.0, 'r': 0.8, 'zeta': 0.05},
         keywords=['unbalance', 'rotor', 'eccentric mass', 'forced response'],
         assumptions=['Rotating unbalance m_e at radius e. X -> m_e e/M as r -> infinity (independent of '
                      'damping there).'],
         refs=['X = (m_e e/M) r^2 / sqrt((1 - r^2)^2 + (2 zeta r)^2) (recalled)']),

    Calc('base_excitation', 'Response to harmonic base (support) excitation: absolute and relative displacement',
         vars=[V('X', 'm', 'absolute displacement amplitude of the mass'),
               V('Z', 'm', 'relative displacement amplitude between mass and base'),
               V('Y', 'm', 'base excitation amplitude'), V('zeta', '', 'damping ratio', label='ζ'),
               V('r', '', 'frequency ratio omega/omega_n')],
         equations=['X = Y*sqrt((1 + (2*zeta*r)**2)/((1 - r**2)**2 + (2*zeta*r)**2))',
                    'Z = Y*r**2/sqrt((1 - r**2)**2 + (2*zeta*r)**2)'],
         outputs=['X', 'Z'], example={'Y': 0.002, 'zeta': 0.05, 'r': 2.0},
         keywords=['base excitation', 'support motion', 'seismic', 'relative displacement'],
         assumptions=['X is the ABSOLUTE displacement amplitude of the mass; Z is the relative displacement '
                      'amplitude between the mass and the base. |X|/Y equals the force transmissibility; '
                      '|Z|/Y = r^2/sqrt((1 - r^2)^2 + (2 zeta r)^2).'],
         refs=['Base-excitation absolute response X/Y = sqrt((1 + (2 zeta r)^2)/((1 - r^2)^2 + (2 zeta r)^2)) '
               'and relative response Z/Y = r^2/sqrt((1 - r^2)^2 + (2 zeta r)^2)']),

    # ---- MDOF and absorbers ----------------------------------------------------------------------
    Calc('mdof', 'Multi-degree-of-freedom natural frequencies and mode shapes',
         vars=[V('M', '', 'mass matrix (json, symmetric, kg)', kind='json'),
               V('K', '', 'stiffness matrix (json, symmetric, N/m)', kind='json'),
               V('freqs', '', 'natural frequencies omega_n (rad/s), ascending', kind='json'),
               V('f_n', '', 'natural frequencies f_n (Hz), ascending', kind='json'),
               V('modes', '', 'mode shape vectors: modes[j] is the j-th mode vector (mass-normalized)', kind='json')],
         procedure=_mdof, outputs=['freqs', 'f_n', 'modes'],
         example={'M': [[2.0, 0.0], [0.0, 1.0]], 'K': [[300.0, -100.0], [-100.0, 100.0]]},
         keywords=['eigenvalue', 'mode shape', 'modal', '2 dof', 'natural frequency'],
         assumptions=['Undamped, symmetric, positive-definite M and K. Solves the generalized eigenproblem '
                      'K phi = omega^2 M phi; modes are mass-normalized (phi^T M phi = 1), signs arbitrary.'],
         refs=['Generalized eigenproblem K phi = omega^2 M phi (scipy.linalg.eigh)']),

    Calc('absorber_tuning', 'Tuned dynamic vibration absorber stiffness',
         vars=[V('k_a', 'N/m', 'absorber stiffness'), V('m_a', 'kg', 'absorber mass'),
               V('omega', 'rad/s', 'disturbing frequency to cancel (enter rad/s or rpm)', label='ω')],
         equations=['k_a = m_a*omega**2'], outputs=['k_a'], example={'m_a': 0.5, 'omega': 20.0},
         keywords=['vibration absorber', 'tuned mass damper', 'frahm'],
         assumptions=['The absorber is tuned so its own natural frequency sqrt(k_a/m_a) equals the disturbing '
                      'frequency omega; at that frequency the main mass stays still.'],
         refs=['Absorber tuning k_a = m_a omega^2 (Frahm absorber)']),

    Calc('absorber_frequencies', 'Natural frequencies of a main mass with an attached absorber',
         vars=[V('omega1', 'rad/s', 'lower natural frequency of the combined system', label='ω1'),
               V('omega2', 'rad/s', 'upper natural frequency of the combined system', label='ω2'),
               V('m', 'kg', 'main mass'), V('k', 'N/m', 'main stiffness'),
               V('m_a', 'kg', 'absorber mass'), V('k_a', 'N/m', 'absorber stiffness')],
         procedure=_absorber_frequencies, outputs=['omega1', 'omega2'],
         example={'m': 10.0, 'k': 4000.0, 'm_a': 0.5, 'k_a': 200.0},
         keywords=['vibration absorber', 'split frequencies', '2 dof'],
         assumptions=['Undamped two-degree-of-freedom system; the two frequencies bracket the tuned absorber '
                      'frequency (the disturbing frequency should be kept off these peaks).'],
         refs=['Characteristic equation det([K - omega^2 M]) = 0 for main + absorber masses']),

    # ---- torsional, shafts, continuous systems ---------------------------------------------------
    Calc('torsional_vibration', 'Torsional natural frequency of a shaft-disk system',
         vars=[V('k_t', 'N*m/rad', 'torsional stiffness'), V('omega_n', 'rad/s', 'torsional natural frequency', label='ωn'),
               V('G', 'Pa', 'shear modulus'), V('J', 'm^4', 'polar second moment of area'),
               V('L', 'm', 'shaft length'), V('I', 'kg*m^2', 'mass moment of inertia of the disk')],
         equations=['k_t = G*J/L', 'omega_n = sqrt(k_t/I)'], outputs=['k_t', 'omega_n'],
         example={'G': 79.3e9, 'J': 2.5e-7, 'L': 1.0, 'I': 0.02},
         keywords=['torsional vibration', 'shaft', 'flywheel'],
         assumptions=['Massless shaft, single rigid disk; free (undamped) torsional vibration.'],
         refs=['k_t = GJ/L; omega_n = sqrt(k_t/I)']),

    Calc('critical_speed_shaft', 'Shaft critical speed from static deflection (single disk)',
         vars=[V('omega_cr', 'rad/s', 'critical (whirl) speed', label='ωcr'),
               V('f_cr', 'Hz', 'critical speed in Hz'), V('delta_st', 'm', 'static deflection under the disk weight'),
               V('g', 'm/s^2', 'gravitational acceleration (default 9.80665)')],
         equations=['omega_cr = sqrt(g/delta_st)', 'f_cr = omega_cr/(2*pi)'],
         outputs=['omega_cr', 'f_cr'], example={'delta_st': 0.001, 'g': 9.80665},
         keywords=['critical speed', 'whirl', 'lateral vibration'],
         assumptions=['Single disk (any location along the shaft), weight-loaded; omega_cr = sqrt(g/delta_st) '
                      'with delta_st the static deflection under the disk. To get rpm, convert '
                      'the rad/s result. For several disks use rayleigh or dunkerley.'],
         refs=['omega_cr = sqrt(g/delta_st)']),

    Calc('rayleigh', "Rayleigh's method: first critical speed from static deflections",
         vars=[V('masses', '', 'rows [W (N), y (m)]: each load and its static deflection', kind='json'),
               V('omega_cr', 'rad/s', 'first critical speed (rad/s)'), V('f_cr', 'Hz', 'first critical speed (Hz)')],
         procedure=_rayleigh, outputs=['omega_cr', 'f_cr'],
         example={'masses': [[200.0, 0.0005], [150.0, 0.0004]]},
         keywords=['rayleigh', 'critical speed', 'multi disk'],
         assumptions=['Rayleigh energy method with the static-deflection curve as the assumed mode; gives an '
                      'upper bound on the fundamental critical speed.'],
         refs=["Rayleigh's method: omega_cr^2 = g sum(W y)/sum(W y^2)"]),

    Calc('dunkerley', "Dunkerley's method: first critical speed from individual speeds",
         vars=[V('speeds', '', 'individual critical speeds (rad/s), one per load acting alone', kind='json'),
               V('omega_cr', 'rad/s', 'combined first critical speed (rad/s)'), V('f_cr', 'Hz', 'combined first critical speed (Hz)')],
         procedure=_dunkerley, outputs=['omega_cr', 'f_cr'],
         example={'speeds': [100.0, 150.0]},
         keywords=['dunkerley', 'critical speed'],
         assumptions=["Dunkerley's method: 1/omega_cr^2 = sum 1/omega_i^2; gives a lower bound on the "
                      'fundamental critical speed.'],
         refs=["Dunkerley's reciprocal formula 1/omega_cr^2 = sum 1/omega_i^2"]),

    Calc('string_frequency', 'Natural frequencies of a taut string',
         vars=[V('omega_n', 'rad/s', 'natural (angular) frequency', label='ωn'),
               V('f_n', 'Hz', 'natural (cyclic) frequency'),
               V('n', '', 'mode (harmonic) number', lo=1.0, hi=20.0, integer=True),
               V('L', 'm', 'string length'), V('T', 'N', 'tension'), V('mu', 'kg/m', 'mass per unit length', label='μ')],
         equations=['omega_n = n*pi/L*sqrt(T/mu)', 'f_n = omega_n/(2*pi)'],
         outputs=['omega_n', 'f_n'], example={'n': 1, 'L': 1.0, 'T': 100.0, 'mu': 0.01},
         keywords=['string', 'wave', 'guitar', 'natural frequency'],
         assumptions=['Flexible string, small amplitude, both ends fixed; n is the harmonic number.'],
         refs=['omega_n = (n pi/L) sqrt(T/mu)']),

    Calc('bar_longitudinal', 'Longitudinal natural frequencies of a bar (fixed-fixed)',
         vars=[V('omega_n', 'rad/s', 'natural (angular) frequency', label='ωn'),
               V('f_n', 'Hz', 'natural (cyclic) frequency'),
               V('n', '', 'mode number', lo=1.0, hi=20.0, integer=True),
               V('L', 'm', 'bar length'), V('E', 'Pa', "Young's modulus"), V('rho', 'kg/m^3', 'density', label='ρ')],
         equations=['omega_n = n*pi/L*sqrt(E/rho)', 'f_n = omega_n/(2*pi)'],
         outputs=['omega_n', 'f_n'], example={'n': 1, 'L': 1.0, 'E': 200e9, 'rho': 7850.0},
         keywords=['bar', 'rod', 'longitudinal', 'axial vibration'],
         assumptions=['Longitudinal (axial) vibration, fixed-fixed or free-free (same frequencies). A fixed-free '
                      'bar gives f_n = (2n - 1)/(4L) sqrt(E/rho) instead.'],
         refs=['omega_n = (n pi/L) sqrt(E/rho) for a bar in longitudinal vibration']),

    Calc('beam_frequency', 'Beam natural frequency from numerically found betaL roots',
         vars=[V('E', 'Pa', "Young's modulus"), V('I', 'm^4', 'second moment of area'),
               V('rho', 'kg/m^3', 'density', label='ρ'), V('A', 'm^2', 'cross-sectional area'),
               V('L', 'm', 'beam length'), V('n', '', 'mode number', lo=1.0, hi=20.0, integer=True),
               V('bc', '', 'boundary condition: pinned-pinned, fixed-free, fixed-fixed or free-free', kind='text',
                 choices=['pinned-pinned', 'fixed-free', 'fixed-fixed', 'free-free']),
               V('omega_n', 'rad/s', 'natural (angular) frequency', label='ωn'),
               V('f_n', 'Hz', 'natural (cyclic) frequency'),
               V('betaL', '', 'beta_n L root of the characteristic equation (dimensionless)')],
         procedure=_beam_frequency, outputs=['omega_n', 'f_n', 'betaL'],
         example={'E': 200e9, 'I': 1e-6, 'rho': 7850.0, 'A': 0.001, 'L': 1.0, 'n': 1, 'bc': 'pinned-pinned'},
         keywords=['beam', 'euler-bernoulli', 'flexural vibration', 'cantilever'],
         assumptions=['Uniform Euler-Bernoulli beam; betaL roots are found by root-finding the characteristic '
                      'equation (not tabulated). Pinned-pinned betaL = n pi is closed form; fixed-free '
                      'cos(betaL) cosh(betaL) = -1; fixed-fixed/free-free cos(betaL) cosh(betaL) = 1. '
                      'omega_n = beta^2 sqrt(EI/(rho A)).'],
         refs=['Euler-Bernoulli beam frequency equation; omega_n = beta_n^2 sqrt(EI/(rho A)) with beta_n L '
               'the roots of the boundary-condition characteristic equation']),

    # ---- quality factor and shock ----------------------------------------------------------------
    Calc('quality_factor', 'Quality factor and half-power bandwidth',
         vars=[V('Q', '', 'quality factor'), V('delta_omega', 'rad/s', 'half-power bandwidth', label='Δω'),
               V('zeta', '', 'damping ratio', label='ζ'), V('omega_n', 'rad/s', 'natural frequency', label='ωn')],
         equations=['Q = 1/(2*zeta)', 'delta_omega = omega_n/Q'], outputs=['Q', 'delta_omega'],
         example={'zeta': 0.05, 'omega_n': 100.0},
         keywords=['quality factor', 'half power', 'bandwidth', 'q factor'],
         assumptions=['Light damping: Q = 1/(2 zeta) relates damping to the resonance sharpness; the half-power '
                      'bandwidth is omega_n/Q = 2 zeta omega_n.'],
         refs=['Q = 1/(2 zeta); half-power bandwidth = omega_n/Q']),

    Calc('duhamel', 'Shock/transient response by the Duhamel (convolution) integral',
         vars=[V('m', 'kg', 'mass'), V('k', 'N/m', 'stiffness'), V('zeta', '', 'damping ratio (< 1)', hi=0.99, label='ζ'),
               V('times', '', 'time points (s, increasing)', kind='json'),
               V('forces', '', 'force values (N) at each time point', kind='json'),
               V('response', '', 'displacement response x(t) at each time point (m)', kind='json'),
               V('x_max', 'm', 'extreme (peak) response: the sample with the largest |x|, signed', signed=True),
               V('t_max', 's', 'time of the extreme response')],
         procedure=_duhamel, outputs=['response', 'x_max', 't_max'],
         example={'m': 1.0, 'k': 100.0, 'zeta': 0.0, 'times': _STEP_TIMES, 'forces': _STEP_FORCES},
         keywords=['duhamel', 'convolution', 'shock', 'transient', 'impulse'],
         assumptions=['Underdamped SDOF at rest initially; the force history is a zero-order hold (piecewise '
                      'constant between time points) superposed with the exact unit-step response. For a '
                      'suddenly applied constant force the peak response is twice the static deflection '
                      '(2 F0/k) — a useful check.'],
         refs=['Duhamel integral: x(t) = (1/(m omega_d)) integral_0^t F(tau) e^{-zeta omega_n (t-tau)} '
               'sin(omega_d (t-tau)) dtau']),
]

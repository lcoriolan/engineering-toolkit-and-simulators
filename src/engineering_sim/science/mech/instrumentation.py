# DESCRIPTION: Measurements and instrumentation section: sensor specifications, strain gauges and the
#              Wheatstone bridge, thermocouples and RTDs, LVDT and accelerometer sensitivity, SNR and
#              decibels, sampling/Nyquist/aliasing, uncertainty propagation, and first/second-order
#              sensor dynamics.
"""Measurements and instrumentation calculators.

Generic ratio quantities (sensitivity, span, resolution, SNR, standard error) are declared dimensionless
(unit '') because their unit depends on the measurand; the physical unit is stated in each description and
the magnitude is the ratio in the chosen [output]/[input] units. The RTD relation takes an absolute
temperature in kelvin and converts to °C internally; thermocouple temperature differences use kelvin
(enter 'delta_degC' for a Celsius difference).

Relations are the standard ones in measurement and instrumentation texts (e.g. Doebelin, *Measurement
Systems*; Figliola & Beasley, *Theory and Design for Mechanical Measurements*; Beckwith et al., *Mechanical
Measurements*). They are cited by relation name, recalled rather than re-read, so no page numbers are given.
Thermocouple and RTD coefficients are user inputs — no NIST ITS-90 or IEC 60751 table is embedded.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section, safe_expr

SECTION = section('instrumentation', 'Measurements and instrumentation', order=370,
                  summary='Sensors, strain gauges and bridges, thermocouples and RTDs, LVDT/accelerometer, '
                          'SNR and dB, sampling and aliasing, uncertainty propagation and sensor dynamics.')

PI = math.pi


# ---- procedures -----------------------------------------------------------------------------------
def _thermocouple_poly(v):
    """Thermocouple emf from a user-supplied polynomial in the junction−reference temperature difference.

    V = Σ c_i (T_junction − T_ref)^i, with coefficients c_i in V/K^i (c0 in V, c1 in V/K, …). The
    coefficients are user inputs from a calibration or a published table (e.g. NIST ITS-90); none are built
    in because the standard coefficients are temperature-range specific and must not be recalled.
    """
    coeffs = v['coeffs']
    if not isinstance(coeffs, (list, tuple)) or not coeffs:
        raise MechError('coeffs: give coefficients [c0, c1, …] of V = Σ c_i (T−Tref)^i (V/K^i)')
    dT = v['T_junction'] - v['T_ref']
    volts = 0.0
    for i, c in enumerate(coeffs):
        volts += float(c) * dT ** i
    return {'V': volts}


def _rtd_cvd(v):
    """Platinum RTD resistance by the Callendar–Van Dusen equation.

    T is an absolute temperature (K); it is converted internally to T_C = T − 273.15 (°C). Then
    R = R0 (1 + A·T_C + B·T_C² + C·(T_C − 100)·T_C³) for T_C < 0 °C and R = R0 (1 + A·T_C + B·T_C²) for
    T_C ≥ 0 °C. A, B, C are user inputs; IEC 60751 nominals (A = 3.9083e-3, B = −5.775e-7,
    C = −4.183e-12) are commonly quoted but not built in.
    """
    R0, A, B, C, T = v['R0'], v['A'], v['B'], v['C'], v['T']
    T_C = T - 273.15
    if T_C < 0.0:
        R = R0 * (1.0 + A * T_C + B * T_C ** 2 + C * (T_C - 100.0) * T_C ** 3)
    else:
        R = R0 * (1.0 + A * T_C + B * T_C ** 2)
    return {'R': R}


def _alias(v):
    """Aliased frequency when a signal at f_sig is sampled at f_s (fold-back / undersampling).

    f_alias = |f_sig − f_s · round(f_sig/f_s)|: the distance from f_sig to the nearest multiple of the
    sample rate. For f_sig ≤ f_s/2 (satisfying Nyquist) this returns f_sig unchanged.
    """
    fs, f = v['f_s'], v['f_sig']
    if fs <= 0:
        raise MechError('f_s must be positive')
    m = round(f / fs)
    return {'f_alias': abs(f - m * fs)}


def _uncertainty(v):
    """Combined standard uncertainty of a function of measured variables by first-order propagation.

    setup = {"expr": "a*b + c**2", "variables": {"a": [value, uncertainty], …}}. Partial derivatives are
    taken with sympy; for uncorrelated inputs u_y = sqrt(Σ (∂f/∂x_i · u_i)²). All values are plain numbers
    (consistent units, stated by the user); sensitivities are returned in the same output/input units.
    """
    import sympy as sp
    setup = v['setup']
    if not isinstance(setup, dict) or 'expr' not in setup or 'variables' not in setup:
        raise MechError('setup: give {"expr": "a*b + c**2", "variables": {"a": [value, uncertainty], …}}')
    expr = setup['expr']
    variables = setup['variables']
    if not isinstance(variables, dict) or not variables:
        raise MechError('setup.variables: give {"name": [value, uncertainty], …}')
    for name, vv in variables.items():
        if not isinstance(vv, (list, tuple)) or len(vv) != 2:
            raise MechError(f'setup.variables.{name}: give [value, uncertainty]')
    # Parse the user expression with the AST whitelist (no eval); rejects attribute access, lambdas, etc.
    try:
        f, syms = safe_expr(str(expr), list(variables.keys()))
    except MechError as exc:
        raise MechError(f'setup.expr: cannot parse {expr!r} ({exc})')
    subs = {syms[name]: float(variables[name][0]) for name in variables}
    value = float(f.subs(subs))
    unc2 = 0.0
    sens = {}
    for name in variables:
        dfv = float(sp.diff(f, syms[name]).subs(subs))
        sens[name] = dfv
        unc2 += (dfv * float(variables[name][1])) ** 2
    return {'value': value, 'uncertainty': math.sqrt(unc2), 'sensitivities': sens}


CALCS = [
    Calc('sensor_sensitivity', 'Sensor sensitivity (Δoutput / Δinput)',
         vars=[V('S', '', 'sensitivity = output change per input change [output]/[input]'),
               V('dy', '', 'output span Δoutput (in the sensor output units)'),
               V('dx', '', 'input span Δinput (in the measurand units)')],
         equations=['S = dy/dx'], outputs=['S'], example={'dy': 5.0, 'dx': 10.0},
         keywords=['sensitivity', 'transducer', 'calibration', 'slope'],
         assumptions=['Static sensitivity over the linear range: S = (y_max − y_min)/(x_max − x_min). '
                      'Enter Δoutput and Δinput in the sensor’s own units; S carries [output]/[input].'],
         refs=['S = Δoutput/Δinput (slope of the calibration curve)']),

    Calc('sensor_span', 'Sensor range and span',
         vars=[V('span', '', 'span = x_max − x_min (in measurand units)'),
               V('x_max', '', 'upper range limit (full-scale input)'),
               V('x_min', '', 'lower range limit')],
         equations=['span = x_max - x_min'], outputs=['span'], example={'x_max': 100.0, 'x_min': 20.0},
         keywords=['range', 'span', 'full scale'],
         assumptions=['The measurement range is the interval [x_min, x_max]; its width is the span.'],
         refs=['Span = upper range limit − lower range limit']),

    Calc('adc_resolution', 'Quantisation resolution of an n-bit ADC',
         vars=[V('res', '', 'resolution = one least-significant bit (LSB), in the input units'),
               V('span', '', 'full-scale input range (span)'),
               V('n', '', 'number of bits', integer=True, lo=1.0, hi=32.0)],
         equations=['res = span/(2**n)'], outputs=['res'],
         example={'span': 10.0, 'n': 8},
         keywords=['resolution', 'adc', 'quantisation', 'lsb', 'bits'],
         assumptions=['Uniform quantisation: Q = FS/2ⁿ. Some texts use FS/(2ⁿ − 1) when full scale maps to the '
                      'top code; the difference is negligible for n ≳ 8.'],
         refs=['Quantisation step Q = FS/2ⁿ for a uniform n-bit converter']),

    Calc('strain_gauge', 'Strain gauge factor (GF = (ΔR/R)/ε)',
         vars=[V('GF', '', 'gauge factor'), V('eps', '', 'strain (ΔL/L)', signed=True, label='ε'),
               V('dR', 'ohm', 'change in gauge resistance', signed=True, label='ΔR'),
               V('R', 'ohm', 'unstrained gauge resistance')],
         equations=['GF = (dR/R)/eps'], outputs=['GF'],
         example={'dR': 0.24, 'R': 120.0, 'eps': 1000e-6},
         keywords=['strain gauge', 'gauge factor', 'resistance'],
         assumptions=['Linear gauge response; typical metal-foil gauge factors are near 2 (user input). ε is '
                      'the engineering strain along the gauge axis.'],
         refs=['GF = (ΔR/R)/ε']),

    Calc('wheatstone_bridge', 'Wheatstone bridge output voltage (strain gauges)',
         vars=[V('V_out', 'V', 'bridge output voltage'),
               V('V_ex', 'V', 'bridge excitation voltage'),
               V('GF', '', 'gauge factor'), V('eps', '', 'strain magnitude at each active gauge', label='ε'),
               V('k', '', 'number of active arms (1, 2 or 4)', integer=True, lo=1.0, hi=4.0)],
         equations=['V_out = V_ex*GF*eps*k/4'], outputs=['V_out'],
         example={'V_ex': 10.0, 'GF': 2.0, 'eps': 500e-6, 'k': 1},
         keywords=['wheatstone bridge', 'strain', 'quarter bridge', 'half bridge', 'full bridge'],
         assumptions=['Linearised bridge (ΔR/R ≪ 1). Active gauges must be arranged to add: adjacent arms of '
                      'opposite sign (bending pair) or all four arms with the right signs. Quarter (k = 1), '
                      'half (k = 2), full (k = 4) bridge.'],
         refs=['V_o = V_ex/4 · (ΔR1/R1 − ΔR2/R2 + ΔR3/R3 − ΔR4/R4)']),

    Calc('thermocouple_seebeck', 'Thermocouple voltage (linear Seebeck approximation)',
         vars=[V('V', 'V', 'thermocouple emf (negative when the junction is colder than the reference)', signed=True),
               V('S', 'V/K', 'Seebeck coefficient (enter e.g. 41e-6 V/K)', label='S'),
               V('T_junction', 'K', 'measuring (hot) junction temperature', label='Tj'),
               V('T_ref', 'K', 'reference (cold) junction temperature', label='Tref')],
         equations=['V = S*(T_junction - T_ref)'], outputs=['V'],
         example={'S': 41e-6, 'T_junction': 373.15, 'T_ref': 273.15},
         keywords=['thermocouple', 'seebeck', 'cold junction'],
         assumptions=['Linear (constant-Seebeck) approximation, valid over a small temperature span. The '
                      'Seebeck coefficient is a user input (a single value, not the NIST ITS-90 table).'],
         refs=['V = S·(T_j − T_ref) (linear thermocouple law)']),

    Calc('thermocouple_polynomial', 'Thermocouple voltage (user polynomial coefficients)',
         vars=[V('coeffs', '', 'polynomial coefficients [c0, c1, …] in V/K^i (V = Σ c_i (T−Tref)^i)', kind='json'),
               V('T_junction', 'K', 'measuring (hot) junction temperature', label='Tj'),
               V('T_ref', 'K', 'reference (cold) junction temperature', label='Tref'),
               V('V', 'V', 'thermocouple emf (negative when the junction is colder than the reference)', signed=True)],
         procedure=_thermocouple_poly, outputs=['V'],
         example={'coeffs': [0.0, 4.0e-5, 1.0e-8], 'T_junction': 373.15, 'T_ref': 273.15},
         keywords=['thermocouple', 'its-90', 'polynomial', 'cold junction'],
         assumptions=['Coefficients are user inputs from a calibration or a published table (e.g. NIST ITS-90 '
                      'sub-range polynomials, whose coefficients are temperature-range specific and are not '
                      'built in here). V = Σ c_i (T_j − T_ref)^i.'],
         refs=['Thermocouple reference-function polynomial (user coefficients)']),

    Calc('rtd_cvd', 'RTD resistance (Callendar–Van Dusen)',
         vars=[V('R', 'ohm', 'resistance at temperature T'),
               V('R0', 'ohm', 'resistance at 0 °C (e.g. 100 Ω for a Pt100)'),
               V('A', '1/K', 'Callendar–Van Dusen coefficient A (IEC 60751 nominally 3.9083e-3 1/K)'),
               V('B', '1/K^2', 'Callendar–Van Dusen coefficient B (IEC 60751 nominally -5.775e-7 1/K^2)', signed=True),
               V('C', '1/K^4', 'Callendar–Van Dusen coefficient C, used below 0 °C (IEC 60751 nominally -4.183e-12 1/K^4)', signed=True),
               V('T', 'K', 'absolute temperature (enter e.g. -50 degC or 223.15 K; converted internally to °C)', label='T')],
         procedure=_rtd_cvd, outputs=['R'],
         example={'R0': 100.0, 'A': 3.9083e-3, 'B': -5.775e-7, 'C': -4.183e-12, 'T': 223.15},
         keywords=['rtd', 'pt100', 'platinum resistance', 'callendar', 'van dusen'],
         assumptions=['Callendar–Van Dusen with T_C = T − 273.15 (°C): R = R0(1 + A·T_C + B·T_C²) for '
                      'T_C ≥ 0 °C, and R = R0(1 + A·T_C + B·T_C² + C(T_C − 100)T_C³) for T_C < 0 °C. A, B, '
                      'C are user inputs — the IEC 60751 nominal values are quoted in the descriptions only.'],
         refs=['Callendar–Van Dusen equation for platinum RTDs']),

    Calc('lvdt', 'LVDT output voltage (V_out = S·x)',
         vars=[V('V_out', 'V', 'output voltage (about the null, sign follows displacement)', signed=True),
               V('S', 'V/m', 'LVDT sensitivity (volts per metre of core displacement)'),
               V('x', 'm', 'core displacement from null', signed=True)],
         equations=['V_out = S*x'], outputs=['V_out'],
         example={'S': 2000.0, 'x': 0.001},
         keywords=['lvdt', 'displacement', 'transformer', 'position'],
         assumptions=['Linear region of the LVDT, displacement measured from the null (zero) position.'],
         refs=['V_out = S·x (linear differential transformer)']),

    Calc('accelerometer', 'Accelerometer output voltage (V_out = S·a)',
         vars=[V('V_out', 'V', 'output voltage'),
               V('S', 'V/(m/s^2)', 'accelerometer sensitivity (e.g. 10 mV/g ≈ 1.02e-3 V/(m/s²))'),
               V('a', 'm/s^2', 'acceleration (enter m/s^2 or g0)')],
         equations=['V_out = S*a'], outputs=['V_out'],
         example={'S': 0.00102, 'a': 9.80665},
         keywords=['accelerometer', 'piezoelectric', 'mems', 'sensitivity', 'vibration'],
         assumptions=['Charge/voltage sensitivity is the ratio of output to applied acceleration; the '
                      'relation is linear within the sensor bandwidth.'],
         refs=['S = V_out/a (accelerometer sensitivity)']),

    Calc('snr', 'Signal-to-noise ratio (power) and decibels',
         vars=[V('SNR', '', 'signal-to-noise ratio (power ratio, linear)'),
               V('SNR_dB', '', 'signal-to-noise ratio in dB (negative when the signal is below the noise)', signed=True),
               V('P_sig', 'W', 'signal power'), V('P_noise', 'W', 'noise power')],
         equations=['SNR = P_sig/P_noise', 'SNR_dB = 10*log(SNR)/log(10)'],
         outputs=['SNR', 'SNR_dB'], example={'P_sig': 1e-3, 'P_noise': 1e-6},
         keywords=['snr', 'decibel', 'noise'],
         assumptions=['Power ratio: SNR_dB = 10 log10(P_sig/P_noise). For amplitude (voltage) ratios use 20 '
                      'log10 (see the decibel calculator).'],
         refs=['SNR = P_sig/P_noise; SNR_dB = 10 log10(SNR)']),

    Calc('decibel', 'Decibels from a voltage (amplitude) ratio',
         vars=[V('dB', '', 'level in decibels (negative for attenuation)', signed=True),
               V('V2', 'V', 'voltage of interest'), V('V1', 'V', 'reference voltage')],
         equations=['dB = 20*log(V2/V1)/log(10)'], outputs=['dB'],
         example={'V2': 10.0, 'V1': 1.0},
         keywords=['decibel', 'db', 'amplitude', 'voltage'],
         assumptions=['Amplitude ratio: dB = 20 log10(V2/V1). For power ratios use 10 log10 (see the SNR '
                      'calculator).'],
         refs=['dB = 20 log10(V2/V1)']),

    Calc('nyquist', 'Nyquist frequency from the sampling rate',
         vars=[V('f_N', 'Hz', 'Nyquist frequency (half the sampling rate)', label='fN'),
               V('f_s', 'Hz', 'sampling frequency', label='fs')],
         equations=['f_N = f_s/2'], outputs=['f_N'],
         example={'f_s': 2000.0},
         keywords=['nyquist', 'sampling', 'shannon'],
         assumptions=['To represent a signal without aliasing the highest signal frequency must be below '
                      'f_s/2 (Nyquist–Shannon).'],
         refs=['f_N = f_s/2']),

    Calc('aliased_frequency', 'Aliased (folded) frequency of an undersampled signal',
         vars=[V('f_alias', 'Hz', 'apparent (aliased) frequency in the sampled record'),
               V('f_s', 'Hz', 'sampling frequency', label='fs'),
               V('f_sig', 'Hz', 'true signal frequency', label='fsig')],
         procedure=_alias, outputs=['f_alias'],
         example={'f_s': 100.0, 'f_sig': 130.0},
         keywords=['aliasing', 'fold', 'undersampling', 'nyquist'],
         assumptions=['Closed form f_alias = |f_sig − f_s·round(f_sig/f_s)|. The mapping is many-to-one '
                      '(folding), so the true frequency is not recoverable from the alias alone.'],
         refs=['f_alias = |f_sig − f_s·round(f_sig/f_s)|']),

    Calc('uncertainty_propagation', 'Uncertainty propagation (sympy partial derivatives)',
         vars=[V('setup', '', '{"expr": "a*b + c**2", "variables": {"a": [value, uncertainty], …}}', kind='json'),
               V('value', '', 'computed value of the expression'),
               V('uncertainty', '', 'combined standard uncertainty (quadrature)'),
               V('sensitivities', '', '{"name": ∂f/∂name} at the given values', kind='json')],
         procedure=_uncertainty, outputs=['value', 'uncertainty', 'sensitivities'],
         example={'setup': {'expr': 'a*b + c**2', 'variables': {'a': [2.0, 0.1], 'b': [3.0, 0.2], 'c': [4.0, 0.05]}}},
         keywords=['uncertainty', 'error propagation', 'partial derivatives', 'gum'],
         assumptions=['First-order (linearised) propagation for uncorrelated inputs: u_y = sqrt(Σ (∂f/∂x_i · '
                      'u_i)²). Values and uncertainties are plain numbers in consistent units.'],
         refs=['Law of propagation of uncertainty (GUM, uncorrelated case)']),

    Calc('standard_error', 'Standard error of the mean (SE = σ/√n)',
         vars=[V('SE', '', 'standard error of the mean'),
               V('sigma', '', 'sample standard deviation (in the data units)', label='σ'),
               V('n', '', 'number of measurements', integer=True, lo=1.0)],
         equations=['SE = sigma/sqrt(n)'], outputs=['SE'],
         example={'sigma': 0.5, 'n': 16},
         keywords=['standard error', 'statistics', 'repeated measurements'],
         assumptions=['For n independent measurements of the same quantity; the standard error of the mean '
                      'shrinks as 1/√n.'],
         refs=['SE = σ/√n']),

    Calc('first_order_response', 'First-order sensor step response and time constant',
         vars=[V('y', '', 'output at time t (in the output units)'),
               V('frac', '', 'fraction of the final value reached, y/y_ss', hi=1.0),
               V('y_ss', '', 'steady-state (final) output'),
               V('t', 's', 'elapsed time'), V('tau', 's', 'time constant (63.2 % rise time)', label='τ')],
         equations=['y = y_ss*(1 - exp(-t/tau))', 'frac = y/y_ss'],
         outputs=['y', 'frac'], example={'y_ss': 10.0, 't': 2.0, 'tau': 2.0},
         keywords=['first order', 'time constant', 'step response', 'lag'],
         assumptions=['First-order (single-lag) system: at t = τ the output reaches 1 − e⁻¹ ≈ 63.2 % of the '
                      'final value. The output y carries the same units as y_ss.'],
         refs=['y(t) = y_ss (1 − exp(−t/τ))']),

    Calc('second_order_params', 'Second-order sensor natural frequency and damping ratio (m, c, k)',
         vars=[V('wn', 'rad/s', 'undamped natural frequency', label='ωn'),
               V('zeta', '', 'damping ratio (ζ < 1 underdamped, ζ = 1 critical, ζ > 1 overdamped)', label='ζ'),
               V('m', 'kg', 'seismic mass'), V('c', 'N*s/m', 'damping coefficient'),
               V('k', 'N/m', 'spring stiffness')],
         equations=['wn = sqrt(k/m)', 'zeta = c/(2*sqrt(k*m))'], outputs=['wn', 'zeta'],
         example={'m': 1.0, 'k': 100.0, 'c': 2.0},
         keywords=['second order', 'seismic sensor', 'natural frequency', 'damping', 'mass spring damper'],
         assumptions=['Single-degree-of-freedom mass–spring–damper model of a seismic sensor (accelerometer, '
                      'geophone). ζ < 1 underdamped (the usual design); ζ = 1 critically damped; ζ > 1 '
                      'overdamped.'],
         refs=['ωn = √(k/m); ζ = c/(2√(km))']),

    Calc('second_order_freq_response', 'Second-order sensor frequency response (amplitude ratio and phase)',
         vars=[V('AR', '', 'amplitude ratio |y|/|x_ss| = 1/√((1−r²)² + (2ζr)²)'),
               V('phase', 'rad', 'phase lag of the output relative to the input (−atan2(2ζr, 1−r²))', signed=True),
               V('zeta', '', 'damping ratio (any ζ ≥ 0)', label='ζ'),
               V('r', '', 'frequency ratio r = ω/ωn')],
         equations=['AR = 1/sqrt((1 - r**2)**2 + (2*zeta*r)**2)',
                    'phase = -atan2(2*zeta*r, 1 - r**2)'],
         outputs=['AR', 'phase'],
         example={'zeta': 0.1, 'r': 0.5},
         keywords=['second order', 'frequency response', 'amplitude ratio', 'phase', 'resonance'],
         assumptions=['Steady-state sinusoidal response of a second-order sensor; r = ω/ωn is the excitation '
                      'frequency normalised by the undamped natural frequency. At resonance r = 1 the '
                      'amplitude ratio is 1/(2ζ) and the phase is −90°.'],
         refs=['Second-order frequency response AR = 1/√((1−r²)² + (2ζr)²); φ = −atan2(2ζr, 1−r²)']),
]

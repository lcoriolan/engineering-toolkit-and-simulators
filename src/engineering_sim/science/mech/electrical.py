# DESCRIPTION: Electrical / mechatronics section: Ohm's law, power and energy, series-parallel
#              passive elements, RC/RL transients, DC and AC power, stepper motors, gear motors,
#              encoders and PWM. Generic textbook relations only (no project motor models).
"""Mechatronics and electrical basics.

Conventions: voltages, currents and powers are magnitudes (DC, and RMS for AC). The DC-motor model is the
steady-state permanent-magnet / separately-excited equivalent circuit V = I·R + ke·ω with torque
constant kt (N·m/A) and back-EMF constant ke (V·s/rad); in coherent SI units kt = ke for an ideal
lossless machine. Angular speed ω is in rad/s; cyclic rates (frequency, pump/rev rates) are in Hz.

Relations are the standard ones found in introductory electrical-engineering and mechatronics texts
(e.g. Hughes, *Electric Motors and Drives*; Nise, *Control Systems Engineering*). They are cited by
relation name, recalled rather than re-read, so no page numbers are given.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('electrical', 'Mechatronics and electrical basics', order=390,
                  summary='Ohm\'s law, power and energy, series-parallel R/C/L, RC/RL transients, DC and AC '
                          'power, stepper motors, gear motors, encoders and PWM.')

PI = math.pi


# ---- procedures ------------------------------------------------------------------------------------
def _dc_motor(v):
    """Steady-state permanent-magnet / separately-excited DC model: V = I·R + ke·ω, T = kt·I.

    Adds the back EMF E = ke·ω, the no-load speed ω_nl = V/ke (T = 0) and the stall torque
    T_stall = kt·V/R (ω = 0). T above the stall torque would need ω < 0 and raises MechError; a
    mismatch between kt and ke is surfaced as a warning (they are equal in coherent SI for an ideal
    machine).
    """
    V, R, ke, kt, T = v['V'], v['R'], v['ke'], v['kt'], v['T']
    if R <= 0 or ke <= 0 or kt <= 0:
        raise MechError('R, ke and kt must be positive')
    T_stall = kt * V / R
    if T > T_stall:
        raise MechError(f'T = {T:g} N·m exceeds the stall torque kt·V/R = {T_stall:g} N·m at this supply '
                        f'voltage — the motor cannot develop this torque (it would need ω < 0)')
    I = T / kt
    omega = (V - I * R) / ke
    E = ke * omega
    P_out = T * omega
    P_in = V * I
    eta = (P_out / P_in) if P_in != 0.0 else 0.0  # efficiency → 0 as the load vanishes
    omega_nl = V / ke
    warnings = []
    if abs(kt - ke) > 1e-12 * max(1.0, abs(ke)):
        warnings.append(f'kt = {kt:g} N·m/A and ke = {ke:g} V·s/rad differ; in coherent SI units they are '
                        f'equal for an ideal lossless machine — check the units or gearing')
    return {'omega': omega, 'I': I, 'E': E, 'P_out': P_out, 'P_in': P_in, 'eta': eta,
            'omega_nl': omega_nl, 'T_stall': T_stall, '_warnings': warnings}


def _encoder(v):
    """Quadrature encoder resolution; the decoding multiplier m is restricted to 1, 2 or 4."""
    N = v['N_lines']
    m = v['m']
    if N <= 0:
        raise MechError('N_lines must be positive')
    if m not in (1.0, 2.0, 4.0):
        raise MechError(f'm: decoding multiplier must be 1, 2 or 4 (×1/×2/×4 quadrature), got {m:g}')
    cpr = m * N
    return {'cpr': cpr, 'theta_res': 2.0 * PI / cpr}


CALCS = [
    # ---- fundamentals -----------------------------------------------------------------------------
    Calc('ohms_law', "Ohm's law (V = I·R)",
         vars=[V('V', 'V', 'voltage'), V('I', 'A', 'current'), V('R', 'ohm', 'resistance')],
         equations=['V = I*R'], outputs=['V'], example={'I': 2.0, 'R': 50.0},
         keywords=['voltage drop', 'resistor'],
         assumptions=['Ohmic (linear) conductor at constant temperature.'],
         refs=["Ohm's law, V = I R"]),

    Calc('electrical_power', 'Electrical power and energy (P = V·I, E = P·t)',
         vars=[V('P', 'W', 'power'), V('V', 'V', 'voltage'), V('I', 'A', 'current'),
               V('E', 'J', 'energy'), V('t', 's', 'time')],
         equations=['P = V*I', 'E = P*t'], outputs=['P', 'E'],
         example={'V': 230.0, 'I': 5.0, 't': 3600.0},
         keywords=['watt', 'joule', 'kWh'],
         assumptions=['DC or unity-power-factor AC (for reactive loads use the AC power calculator).'],
         refs=['P = V I; E = P t']),

    # ---- series and parallel passive elements ------------------------------------------------------
    Calc('resistance_series_parallel', 'Series and parallel resistance (two resistors)',
         vars=[V('R1', 'ohm', 'resistor 1'), V('R2', 'ohm', 'resistor 2'),
               V('R_s', 'ohm', 'series equivalent'), V('R_p', 'ohm', 'parallel equivalent')],
         equations=['R_s = R1 + R2', 'R_p = R1*R2/(R1 + R2)'], outputs=['R_s', 'R_p'],
         example={'R1': 100.0, 'R2': 300.0},
         keywords=['equivalent resistance'],
         assumptions=['Two resistors; for more, chain the rule (1/Rp = Σ 1/Ri).'],
         refs=['Series and parallel resistance combinations']),

    Calc('capacitance_series_parallel', 'Series and parallel capacitance (two capacitors)',
         vars=[V('C1', 'F', 'capacitor 1'), V('C2', 'F', 'capacitor 2'),
               V('C_p', 'F', 'parallel equivalent'), V('C_s', 'F', 'series equivalent')],
         equations=['C_p = C1 + C2', 'C_s = C1*C2/(C1 + C2)'], outputs=['C_p', 'C_s'],
         example={'C1': 1e-6, 'C2': 2e-6},
         keywords=['equivalent capacitance'],
         assumptions=['Two capacitors; note capacitance adds in parallel, the opposite of resistance.'],
         refs=['Series and parallel capacitance combinations']),

    Calc('inductance_series_parallel', 'Series and parallel inductance (two inductors, no coupling)',
         vars=[V('L1', 'H', 'inductor 1'), V('L2', 'H', 'inductor 2'),
               V('L_s', 'H', 'series equivalent'), V('L_p', 'H', 'parallel equivalent')],
         equations=['L_s = L1 + L2', 'L_p = L1*L2/(L1 + L2)'], outputs=['L_s', 'L_p'],
         example={'L1': 0.1, 'L2': 0.3},
         keywords=['equivalent inductance'],
         assumptions=['Magnetically uncoupled inductors; mutual inductance changes both combinations.'],
         refs=['Series and parallel inductance combinations']),

    # ---- transients --------------------------------------------------------------------------------
    Calc('rc_transient', 'RC time constant and charging voltage',
         vars=[V('R', 'ohm', 'resistance'), V('C', 'F', 'capacitance'),
               V('tau', 's', 'time constant', label='τ'),
               V('V0', 'V', 'source (final) voltage'), V('t', 's', 'time after the step'),
               V('Vc', 'V', 'capacitor voltage at time t')],
         equations=['tau = R*C', 't = -tau*log(1 - Vc/V0)'], outputs=['tau', 'Vc'],
         example={'R': 1000.0, 'C': 1e-6, 'V0': 10.0, 't': 1e-3},
         keywords=['charging', 'first order', 'time constant'],
         assumptions=['Series RC, capacitor charging from 0 toward V0 after a step; reaches ≈63% at t = τ '
                      'and ≈95% at t = 3τ. Discharge replaces (1 − e) with e. Solving for t with Vc ≥ V0 has '
                      'no finite solution (t → ∞) and raises an error.'],
         refs=['τ = RC; t = −τ ln(1 − Vc/V0) (charging from 0)']),

    Calc('rl_transient', 'RL time constant and current rise',
         vars=[V('R', 'ohm', 'series resistance'), V('L', 'H', 'inductance'),
               V('tau', 's', 'time constant', label='τ'),
               V('V0', 'V', 'source voltage'), V('t', 's', 'time after the step'),
               V('i', 'A', 'inductor current at time t')],
         equations=['tau = L/R', 'i = (V0/R)*(1 - exp(-t/tau))'], outputs=['tau', 'i'],
         example={'R': 10.0, 'L': 0.05, 'V0': 12.0, 't': 5e-3},
         keywords=['first order', 'time constant'],
         assumptions=['Series RL energised by a step; current rises toward V0/R with the same 63%/95% '
                      'milestones. Interruption causes an inductive kick — use a freewheel diode.'],
         refs=['τ = L/R; i(t) = (V0/R)(1 − e^(−Rt/L))']),

    # ---- DC motor ----------------------------------------------------------------------------------
    Calc('dc_motor', 'DC motor: torque, back EMF, speed, no-load and stall points',
         vars=[V('V', 'V', 'terminal (supply) voltage'), V('R', 'ohm', 'armature resistance'),
               V('ke', 'V*s/rad', 'back-EMF constant (V per rad/s)'),
               V('kt', 'N*m/A', 'torque constant'),
               V('T', 'N*m', 'load torque at the shaft'),
               V('omega', 'rad/s', 'shaft angular speed', label='ω'),
               V('I', 'A', 'armature current'),
               V('E', 'V', 'back EMF at the operating speed'),
               V('P_out', 'W', 'mechanical output power'), V('P_in', 'W', 'electrical input power'),
               V('eta', '', 'efficiency P_out/P_in', hi=1.0, label='η'),
               V('omega_nl', 'rad/s', 'no-load speed V/ke (at T = 0)', label='ω_nl'),
               V('T_stall', 'N*m', 'stall torque kt·V/R (at ω = 0)')],
         procedure=_dc_motor,
         outputs=['omega', 'I', 'E', 'P_out', 'P_in', 'eta', 'omega_nl', 'T_stall'],
         example={'V': 12.0, 'R': 0.5, 'ke': 0.05, 'kt': 0.05, 'T': 0.1},
         keywords=['back emf', 'armature', 'torque constant', 'motor speed', 'PMDC', 'no-load', 'stall'],
         assumptions=['Steady-state permanent-magnet / separately-excited model; brush and iron losses '
                      'neglected except as captured by the efficiency. In coherent SI units kt = ke for an '
                      'ideal machine (kt in N·m/A equals ke in V·s/rad); a warning is raised when they '
                      'differ. A load torque above the stall torque kt·V/R (which would need ω < 0) raises '
                      'an error.'],
         refs=['DC motor equivalent circuit V = I R + ke·ω; T = kt·I; η = Tω/(V I); stall torque kt·V/R; '
               'no-load speed V/ke']),

    # ---- AC power ----------------------------------------------------------------------------------
    Calc('ac_power', 'AC power: apparent, real, reactive and power factor',
         vars=[V('V_rms', 'V', 'RMS voltage'), V('I_rms', 'A', 'RMS current'),
               V('phi', 'rad', 'phase angle of current relative to voltage (negative = leading)',
                 lo=-PI / 2, hi=PI / 2, label='φ'),
               V('S', 'W', 'apparent power (VA)'), V('P', 'W', 'real (active) power'),
               V('Q', 'W', 'reactive power (var; negative = leading/capacitive)', signed=True),
               V('pf', '', 'power factor cos φ', hi=1.0)],
         equations=['S = V_rms*I_rms', 'P = V_rms*I_rms*cos(phi)',
                    'Q = V_rms*I_rms*sin(phi)', 'pf = cos(phi)'],
         outputs=['S', 'P', 'Q', 'pf'],
         example={'V_rms': 230.0, 'I_rms': 10.0, 'phi': math.acos(0.8)},
         keywords=['power factor', 'reactive', 'apparent power', 'real power', 'VA', 'var'],
         assumptions=['Sinusoidal single-phase AC; S, P, Q are dimensionally watts but reported as VA, W, var. '
                      'φ > 0 is a lagging (inductive) load, φ < 0 leading (capacitive). pf = cos φ ignores '
                      'harmonic distortion.'],
         refs=['S = V I; P = V I cos φ; Q = V I sin φ; pf = cos φ']),

    # ---- stepper, gear motor, encoder, PWM ---------------------------------------------------------
    Calc('stepper_motor', 'Stepper motor step angle and resolution',
         vars=[V('m', '', 'number of phases', lo=1.0, integer=True),
               V('Z_r', '', 'number of rotor teeth', lo=1.0, integer=True, label='Z_r'),
               V('mu', '', 'microstepping factor (1 = full step)', lo=1.0, integer=True, label='μ'),
               V('theta_step', 'rad', 'full-step angle', label='θ_step'),
               V('theta_micro', 'rad', 'microstep angle', label='θ_micro'),
               V('res', '', 'effective resolution (steps per revolution)', lo=1.0, integer=True)],
         equations=['theta_step = 2*pi/(m*Z_r)', 'theta_micro = theta_step/mu', 'res = m*Z_r*mu'],
         outputs=['theta_step', 'theta_micro', 'res'],
         example={'m': 4, 'Z_r': 50, 'mu': 16},
         keywords=['microstepping', 'resolution', 'steps per rev', 'rotor teeth', 'phases'],
         assumptions=['Variable-reluctance stepper relation θ = 2π/(m·Z_r): a 4-phase motor with 50 rotor '
                      'teeth steps 1.8° (200 steps/rev). A hybrid stepper with a two-pole rotor halves this '
                      '(θ = 2π/(2·m·Z_r)) — enter the equivalent Z_r. Microstepping divides each full step '
                      'by μ, raising resolution but not guaranteed positional accuracy.'],
         refs=['Step angle = 2π/(m·Z_r) from the number of phases and rotor teeth; resolution = m·Z_r·μ '
               'steps/rev']),

    Calc('gearmotor_output', 'Gear-motor output torque and speed (gearbox with efficiency)',
         vars=[V('omega_in', 'rad/s', 'motor input speed', label='ω_in'),
               V('i', '', 'gear reduction ratio (i > 1 reduces speed)', lo=0.0),
               V('T_in', 'N*m', 'motor input torque'), V('eta', '', 'gearbox efficiency', hi=1.0, label='η'),
               V('omega_out', 'rad/s', 'output speed', label='ω_out'),
               V('T_out', 'N*m', 'output torque'), V('P_out', 'W', 'output power')],
         equations=['omega_out = omega_in/i', 'T_out = T_in*i*eta', 'P_out = T_out*omega_out'],
         outputs=['omega_out', 'T_out', 'P_out'],
         example={'omega_in': 3000 * 2 * PI / 60, 'i': 50.0, 'T_in': 0.05, 'eta': 0.75},
         keywords=['gearbox', 'reduction', 'gearhead', 'speed reducer'],
         assumptions=['Single-stage equivalent; efficiency multiplies torque, not speed. Ideal (η = 1) '
                      'speed/torque ratio alone is gears.gear_ratio.'],
         refs=['ω_out = ω_in/i; T_out = i·η·T_in; power conservation with loss']),

    Calc('encoder_resolution', 'Encoder resolution (counts and angle per count)',
         vars=[V('N_lines', '', 'encoder lines (pulses) per revolution', lo=1.0, integer=True),
               V('m', '', 'decoding multiplier (1, 2 or 4)', lo=1.0, integer=True),
               V('cpr', '', 'counts per revolution', lo=1.0, integer=True),
               V('theta_res', 'rad', 'angular resolution (rad per count)', label='θ_res')],
         procedure=_encoder, outputs=['cpr', 'theta_res'],
         example={'N_lines': 1000, 'm': 4},
         keywords=['quadrature', 'x4 decoding', 'counts per revolution', 'CPR'],
         assumptions=['Quadrature encoders decode both edges of both channels, so ×4 gives 4 counts per line. '
                      'm = 1 (×1) or 2 (×2) if only one channel/edge is used; other multipliers are rejected.'],
         refs=['CPR = m·N_lines; angular resolution = 2π/CPR']),

    Calc('pwm_duty_cycle', 'PWM duty cycle, frequency and average voltage',
         vars=[V('t_on', 's', 'on-time per period'), V('T', 's', 'period'),
               V('D', '', 'duty cycle t_on/T', hi=1.0),
               V('f', 'Hz', 'switching frequency'), V('V_supply', 'V', 'supply voltage'),
               V('V_avg', 'V', 'average output voltage')],
         equations=['D = t_on/T', 'f = 1/T', 'V_avg = D*V_supply'], outputs=['D', 'f', 'V_avg'],
         example={'t_on': 0.25e-3, 'T': 1e-3, 'V_supply': 12.0},
         keywords=['pulse width modulation', 'motor speed control', 'duty cycle'],
         assumptions=['Averaging model: the load (e.g. a DC motor) responds to V_avg when the switching '
                      'frequency is far above the load time constant; current ripple is neglected.'],
         refs=['D = t_on/T; V_avg = D·V_supply']),
]

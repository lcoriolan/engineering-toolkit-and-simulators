# DESCRIPTION: Clutches, brakes and flywheels section: disk/cone clutches, band and drum brakes, disk
#              (caliper) brakes, braking energy/time/temperature, and flywheel energy, sizing and
#              turning-moment diagrams.
"""Clutches, brakes and flywheels calculators.

Sign conventions: torques and angular speeds are treated as magnitudes (friction always opposes the
stated motion). For the drum-brake procedures the shoe 'sense' is explicit ('leading' self-energizing
vs 'trailing'), and the geometric lever arms (a, b, c) are defined as perpendicular distances from the
pivot to the respective lines of action, so the moment equations hold by definition of a moment.

Relations are the standard machine-design results (e.g. Budynas & Nisbett, *Shigley's Mechanical
Engineering Design*; Juvinall & Marshek, *Fundamentals of Machine Component Design*). They are cited by
relation name, recalled rather than re-read, so no page numbers are given.
"""
import math

import numpy as np

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts

SECTION = section('clutches_brakes', 'Clutches, brakes, and flywheels', order=190,
                  summary='Disk and cone clutches, band/drum/disk brakes, braking energy and heating, '
                          'and flywheel energy fluctuation, sizing and turning-moment diagrams.')

PI = math.pi


# ---- procedures ---------------------------------------------------------------------------------
def _drum_short(v):
    """Single internal-expanding short shoe (point-force model).

    sense 'leading' is self-energizing (F = N(a - mu b)/c); 'trailing' adds the friction moment.
    T = mu N r. a, b, c are perpendicular lever arms from the pivot to the lines of action of N, fN
    and F respectively (see module docstring).
    """
    F, mu, r, a, b, c = v['F'], v['mu'], v['r'], v['a'], v['b'], v['c']
    sense = v['sense']
    denom = (a - mu * b) if sense == 'leading' else (a + mu * b)
    if denom <= 0:
        raise MechError('self-locking: a <= mu*b for a leading shoe, so no finite actuating force exists')
    N = F * c / denom
    return {'T': mu * N * r, 'N': N}


def _drum_long(v):
    """Single internal-expanding long shoe with sinusoidal pressure p = p_a sin(theta)/sin(theta_a).

    The angle of maximum pressure is theta_a = min(theta2, 90 deg): the pressure peaks at the heel when
    the shoe does not wrap past 90 deg, otherwise at 90 deg (so it must also satisfy theta1 <= theta_a).
    Standard internal-shoe integrals (Shigley): moment of normal forces M_N, of friction M_f about the
    hinge pin, braking torque T about the drum centre, and actuating force F = (M_N - M_f)/c for a
    self-energizing (leading) shoe or (M_N + M_f)/c for a trailing shoe. A leading shoe self-locks when
    M_N - M_f <= 0.
    """
    mu, r, b, p_a, a, c = v['mu'], v['r'], v['b'], v['p_a'], v['a'], v['c']
    th1, th2 = v['theta1'], v['theta2']
    sense = v['sense']
    if th2 <= th1:
        raise MechError('theta2 (shoe heel angle) must be greater than theta1 (toe angle)')
    th_a = min(th2, PI / 2)
    if th1 > th_a:
        raise MechError('theta1 (toe angle) must not exceed theta_a (angle of maximum pressure): '
                        'a shoe that wraps past 90 deg must start before 90 deg')
    s = math.sin(th_a)
    if abs(s) < 1e-12:
        raise MechError('theta_a (angle of maximum pressure) must not be 0')
    I1 = 0.5 * (th2 - th1) - 0.25 * (math.sin(2 * th2) - math.sin(2 * th1))
    I2 = r * (math.cos(th1) - math.cos(th2)) - (a / 2.0) * (math.sin(th2) ** 2 - math.sin(th1) ** 2)
    I3 = math.cos(th1) - math.cos(th2)
    M_N = p_a * b * r * a / s * I1
    M_f = mu * p_a * b * r / s * I2
    T = mu * p_a * b * r ** 2 / s * I3
    if sense == 'leading':
        if M_N - M_f <= 0:
            raise MechError('self-locking: for a leading shoe M_N - M_f <= 0, so no finite actuating '
                            'force exists (increase a, reduce mu, or use a trailing shoe)')
        F = (M_N - M_f) / c
    else:
        F = (M_N + M_f) / c
    return {'F': F, 'T': T, 'M_N': M_N, 'M_f': M_f, 'theta_a': th_a}


def _turning_moment(v):
    """Flywheel energy fluctuation from a turning-moment diagram.

    torque_curve: [[angle (deg), torque (N·m)], ...] over one full cycle with strictly increasing
    angles. The mean torque is the trapezoidal average; the cumulative energy W(theta) = integral of
    (T - T_mean) d(theta) gives the fluctuation delta_E = W_max - W_min (largest excess of energy,
    which the flywheel must store and return).
    """
    curve = v['torque_curve']
    if not isinstance(curve, list) or len(curve) < 2:
        raise MechError('torque_curve: [[angle (deg), torque (N·m)], ...] over one full cycle')
    ang = np.array([math.radians(float(row[0])) for row in curve])
    Tq = np.array([float(row[1]) for row in curve])
    if ang[-1] <= ang[0]:
        raise MechError('torque_curve must span a positive angle range (ang_last > ang_first)')
    dth = np.diff(ang)
    if np.any(dth < 0):
        raise MechError('torque_curve angles must be non-decreasing (repeat an angle for a vertical step)')
    Tmean = float(np.trapezoid(Tq, ang) / (ang[-1] - ang[0]))
    W = np.zeros(len(curve))
    for i in range(1, len(curve)):
        W[i] = W[i - 1] + 0.5 * (Tq[i - 1] + Tq[i] - 2 * Tmean) * (ang[i] - ang[i - 1])
    energy_curve = [[float(math.degrees(ang[i])), float(W[i])] for i in range(len(curve))]
    return {'T_mean': Tmean, 'W_max': float(W.max()), 'W_min': float(W.min()),
            'delta_E': float(W.max() - W.min()), 'energy_curve': energy_curve}


def _turning_moment_chart(v):
    curve = v['torque_curve']
    ang = [float(row[0]) for row in curve]
    Tq = [float(row[1]) for row in curve]
    mean = float(v.get('T_mean', 0.0))
    ec = v.get('energy_curve') or []
    ea = [row[0] for row in ec]
    eW = [row[1] for row in ec]
    return [
        charts.chart('Turning-moment diagram', 'crank angle [deg]', 'torque [N·m]',
                     [charts.series('torque', ang, Tq)], hlines=[mean]),
        charts.chart('Flywheel energy fluctuation', 'crank angle [deg]', 'cumulative energy [J]',
                     [charts.series('W(theta)', ea, eW)], hlines=[0.0]),
    ]


CALCS = [
    Calc('disk_clutch_uniform_wear', 'Disk clutch torque (uniform wear / worn lining)',
         vars=[V('T', 'N*m', 'clutch torque capacity'), V('n', '', 'number of friction surfaces (2 for one plate)', integer=True),
               V('mu', '', 'friction coefficient', label='μ'), V('F', 'N', 'axial actuating (clamping) force'),
               V('r_o', 'm', 'outer radius of the friction lining'), V('r_i', 'm', 'inner radius of the friction lining')],
         equations=['T = n*mu*F*(r_o + r_i)/2'], outputs=['T'],
         example={'n': 2, 'mu': 0.3, 'F': 4000.0, 'r_o': 0.1, 'r_i': 0.06},
         keywords=['clutch', 'worn lining', 'plate clutch'],
         assumptions=['Uniform-wear model (worn lining): pressure p = C/r, so the effective radius is the mean '
                      'radius (r_o + r_i)/2. Peak pressure p_max = F/(2 pi r_i (r_o - r_i)) occurs at the inner edge.'],
         refs=['Disk-clutch torque, uniform wear (Shigley; recalled)']),

    Calc('disk_clutch_uniform_pressure', 'Disk clutch torque (uniform pressure / new lining)',
         vars=[V('T', 'N*m', 'clutch torque capacity'), V('n', '', 'number of friction surfaces (2 for one plate)', integer=True),
               V('mu', '', 'friction coefficient', label='μ'), V('F', 'N', 'axial actuating (clamping) force'),
               V('r_o', 'm', 'outer radius of the friction lining'), V('r_i', 'm', 'inner radius of the friction lining')],
         equations=['T = n*mu*F*(2/3)*(r_o**3 - r_i**3)/(r_o**2 - r_i**2)'], outputs=['T'],
         example={'n': 2, 'mu': 0.3, 'F': 4000.0, 'r_o': 0.1, 'r_i': 0.06},
         keywords=['clutch', 'new lining', 'plate clutch'],
         assumptions=['Uniform-pressure model (new, unworn lining): p = F/(pi(r_o^2 - r_i^2)) uniform, effective radius '
                      '(2/3)(r_o^3 - r_i^3)/(r_o^2 - r_i^2).'],
         refs=['Disk-clutch torque, uniform pressure (Shigley; recalled)']),

    Calc('cone_clutch', 'Cone clutch torque and normal/axial force',
         vars=[V('T', 'N*m', 'clutch torque'), V('F', 'N', 'axial actuating force'),
               V('N', 'N', 'normal force on the conical friction surface'),
               V('mu', '', 'friction coefficient', label='μ'), V('alpha', 'rad', 'cone semi-apex angle', hi=PI / 2, label='α'),
               V('r_m', 'm', 'mean radius of the cone')],
         equations=['N = F/sin(alpha)', 'T = mu*N*r_m'], outputs=['T', 'N'],
         example={'mu': 0.3, 'F': 2000.0, 'alpha': math.radians(12.5), 'r_m': 0.08},
         keywords=['clutch', 'cone', 'wedging'],
         assumptions=['Wedge action of the cone raises the normal force to F/sin(alpha); the torque uses the mean '
                      'radius. A smaller alpha increases torque but risks sticking (alpha should exceed the friction '
                      'angle to allow disengagement).'],
         refs=['Cone-clutch torque, T = mu F r_m / sin(alpha) (Shigley; recalled)']),

    Calc('band_brake', 'Band brake tensions, braking torque and peak band pressure',
         vars=[V('T1', 'N', 'tight-side band tension'), V('T2', 'N', 'slack-side band tension'),
               V('mu', '', 'friction coefficient between band and drum', label='μ'),
               V('theta', 'rad', 'wrap angle of the band', label='θ'), V('r', 'm', 'drum radius'),
               V('b', 'm', 'band width (axial)'),
               V('T_brake', 'N*m', 'braking torque'), V('p_max', 'Pa', 'peak band pressure (at the tight end)')],
         equations=['T1 = T2*exp(mu*theta)', 'T_brake = (T1 - T2)*r', 'p_max = T1/(b*r)'],
         outputs=['T1', 'T_brake', 'p_max'],
         example={'T2': 100.0, 'mu': 0.3, 'theta': math.radians(225), 'r': 0.15, 'b': 0.05},
         keywords=['band', 'capstan', 'winch'],
         assumptions=['Tension relation is the capstan (Euler–Eytelwein) equation, identical to statics.belt_friction. '
                      'The braking torque is the net band tension times the drum radius. The peak band pressure at the '
                      'tight end is p = 2 P1/(b D) with drum diameter D = 2 r, i.e. p = P1/(b r).'],
         refs=['Capstan equation; band-brake torque and peak pressure (recalled)']),

    Calc('drum_brake_short_shoe', 'Short-shoe drum brake: normal force and braking torque',
         vars=[V('F', 'N', 'actuating force applied to the shoe'),
               V('mu', '', 'lining friction coefficient', label='μ'), V('r', 'm', 'drum radius'),
               V('a', 'm', 'perpendicular distance from pivot to the normal-force line of action'),
               V('b', 'm', 'perpendicular distance from pivot to the friction-force line of action'),
               V('c', 'm', 'perpendicular distance from pivot to the actuating-force line of action'),
               V('sense', '', 'shoe sense: leading (self-energizing) or trailing', kind='text',
                 choices=['leading', 'trailing']),
               V('T', 'N*m', 'braking torque from this shoe'), V('N', 'N', 'resultant normal (lining) force')],
         procedure=_drum_short, outputs=['T', 'N'],
         example={'F': 500.0, 'mu': 0.35, 'r': 0.12, 'a': 0.1, 'b': 0.05, 'c': 0.2, 'sense': 'leading'},
         keywords=['drum brake', 'shoe', 'self-energizing', 'point force'],
         assumptions=['Short shoe: the lining subtends a small angle so the pressure is uniform and the total normal '
                      'force acts at the shoe midpoint. Leading shoe: F = N(a - mu b)/c (self-energizing, a > mu b '
                      'required or the shoe self-locks); trailing shoe: F = N(a + mu b)/c. A two-shoe brake is the sum '
                      'of one leading and one trailing shoe.'],
         refs=['Short-shoe drum brake moment balance (Juvinall; recalled)']),

    Calc('drum_brake_long_shoe', 'Long-shoe drum brake: Shigley internal-shoe integrals',
         vars=[V('mu', '', 'lining friction coefficient', label='μ'), V('r', 'm', 'drum radius'),
               V('b', 'm', 'face width of the lining'), V('p_a', 'Pa', 'maximum lining pressure'),
               V('a', 'm', 'distance from the hinge pin to the drum centre'), V('c', 'm', 'lever arm of the actuating force about the pin'),
               V('theta1', 'rad', 'shoe toe angle from the hinge pin', hi=PI, label='θ1'),
               V('theta2', 'rad', 'shoe heel angle from the hinge pin (θ2 > θ1)', hi=PI, label='θ2'),
               V('sense', '', 'shoe sense: leading (self-energizing) or trailing', kind='text',
                 choices=['leading', 'trailing']),
               V('F', 'N', 'actuating force'), V('T', 'N*m', 'braking torque from this shoe'),
               V('M_N', 'N*m', 'moment of the normal forces about the hinge pin'),
               V('M_f', 'N*m', 'moment of the friction forces about the hinge pin'),
               V('theta_a', 'rad', 'angle of maximum pressure (θa = min(θ2, 90°))', hi=PI, label='θa')],
         procedure=_drum_long, outputs=['F', 'T', 'M_N', 'M_f', 'theta_a'],
         example={'mu': 0.35, 'r': 0.15, 'b': 0.05, 'p_a': 1e6, 'a': 0.13, 'c': 0.18,
                  'theta1': math.radians(10), 'theta2': math.radians(90), 'sense': 'leading'},
         keywords=['drum brake', 'internal expanding', 'sinusoidal pressure', 'self-energizing'],
         assumptions=['Long shoe with sinusoidal pressure p = p_a sin(theta)/sin(theta_a); the angle of maximum '
                      'pressure is theta_a = min(theta2, 90°) (at the heel when the shoe does not wrap past 90°, else '
                      'at 90°). F = (M_N - M_f)/c for a leading (self-energizing) shoe and (M_N + M_f)/c for a trailing '
                      'shoe; a two-shoe brake pairs one of each. A leading shoe self-locks (M_N - M_f <= 0) and raises '
                      'an error. Angles measured from the hinge-pin line.'],
         refs=['Internal expanding shoe: Shigley eqs. for M_N, M_f, T (recalled)']),

    Calc('disk_brake_caliper', 'Disk (caliper) brake torque and effective radius',
         vars=[V('T', 'N*m', 'braking torque'), V('F', 'N', 'caliper clamping force on each pad (two pads)'),
               V('mu', '', 'pad friction coefficient', label='μ'), V('r_m', 'm', 'mean (effective) pad radius'),
               V('r_o', 'm', 'outer pad radius'), V('r_i', 'm', 'inner pad radius')],
         equations=['T = 2*mu*F*r_m', 'r_m = (r_o + r_i)/2'], outputs=['T', 'r_m'],
         example={'mu': 0.4, 'F': 5000.0, 'r_o': 0.14, 'r_i': 0.09},
         keywords=['caliper', 'disk brake', 'rotor', 'pad'],
         assumptions=['Two pads acting on a disk (rotor); uniform-wear effective radius (r_o + r_i)/2 is standard for '
                      'disk brakes. For a new (unworn) lining use disk_brake_uniform_pressure.'],
         refs=['Disk-brake torque, T = 2 mu F r_m (Shigley; recalled)']),

    Calc('disk_brake_uniform_pressure', 'Disk (caliper) brake torque (uniform pressure / new lining)',
         vars=[V('T', 'N*m', 'braking torque'), V('F', 'N', 'caliper clamping force on each pad (two pads)'),
               V('mu', '', 'pad friction coefficient', label='μ'), V('r_o', 'm', 'outer pad radius'),
               V('r_i', 'm', 'inner pad radius'), V('r_m', 'm', 'effective (mean) pad radius')],
         equations=['T = 2*mu*F*(2/3)*(r_o**3 - r_i**3)/(r_o**2 - r_i**2)',
                    'r_m = (2/3)*(r_o**3 - r_i**3)/(r_o**2 - r_i**2)'],
         outputs=['T', 'r_m'],
         example={'mu': 0.4, 'F': 5000.0, 'r_o': 0.14, 'r_i': 0.09},
         keywords=['caliper', 'disk brake', 'rotor', 'new lining', 'uniform pressure'],
         assumptions=['Uniform-pressure (new, unworn) pad: p = F/(pi(r_o^2 - r_i^2)) uniform, effective radius '
                      '(2/3)(r_o^3 - r_i^3)/(r_o^2 - r_i^2). Two pads acting on a disk.'],
         refs=['Disk-brake torque, uniform pressure (Shigley; recalled)']),

    Calc('disk_brake_button', 'Button (circular) pad brake torque',
         vars=[V('T', 'N*m', 'braking torque'), V('F', 'N', 'clamping force on each pad (two pads)'),
               V('mu', '', 'pad friction coefficient', label='μ'), V('r_e', 'm', 'effective radius (pad centre distance from the rotation axis)')],
         equations=['T = 2*mu*F*r_e'], outputs=['T'],
         example={'mu': 0.4, 'F': 5000.0, 'r_e': 0.115},
         keywords=['button pad', 'circular pad', 'disk brake', 'caliper'],
         assumptions=['A small circular button pad carries nearly uniform pressure, so the resultant friction force acts '
                      'at the pad centre and the effective radius is the centre distance r_e. Two pads, T = 2 mu F r_e.'],
         refs=['Button-pad brake torque, T = 2 mu F r_e (Shigley; recalled)']),

    Calc('brake_energy', 'Kinetic energy absorbed by a brake (speed change)',
         vars=[V('E', 'J', 'kinetic energy to be absorbed (translational + rotational)'),
               V('m', 'kg', 'translating mass'), V('v1', 'm/s', 'initial translating speed', label='v₁'),
               V('v2', 'm/s', 'final translating speed', label='v₂'),
               V('I', 'kg*m^2', 'rotating inertia'), V('omega1', 'rad/s', 'initial angular speed', label='ω₁'),
               V('omega2', 'rad/s', 'final angular speed', label='ω₂')],
         equations=['E = 0.5*m*(v1**2 - v2**2) + 0.5*I*(omega1**2 - omega2**2)'], outputs=['E'],
         example={'m': 1000.0, 'v1': 10.0, 'v2': 0.0, 'I': 1.5, 'omega1': 20.0, 'omega2': 0.0},
         keywords=['kinetic energy', 'stop', 'vehicle', 'retardation'],
         assumptions=['Energy removed when speed drops from v1, omega1 to v2, omega2 (set v2 = omega2 = 0 for a stop; '
                      'keep v2 <= v1 and omega2 <= omega1 so the absorbed energy is non-negative). For a pure flywheel '
                      'set m = 0 (or v1 = v2); for a pure vehicle set I = 0 (or omega1 = omega2).'],
         refs=['Kinetic energy change, E = 1/2 m (v1^2 - v2^2) + 1/2 I (omega1^2 - omega2^2)']),

    Calc('brake_stop_time', 'Braking time to a stop (constant torque)',
         vars=[V('t', 's', 'time to bring the rotor to rest'), V('I', 'kg*m^2', 'rotating inertia'),
               V('omega', 'rad/s', 'initial angular speed', label='ω'), V('T', 'N*m', 'constant braking torque')],
         equations=['t = I*omega/T'], outputs=['t'],
         example={'I': 2.0, 'omega': 100.0, 'T': 50.0},
         keywords=['stopping time', 'deceleration'],
         assumptions=['Constant braking torque gives uniform angular deceleration alpha = T/I, so t = omega/alpha. '
                      'For a translating vehicle the analogous time is t = m v / F_brake.'],
         refs=['Uniform-deceleration stopping time']),

    Calc('brake_temp_rise', 'Brake temperature rise (adiabatic)',
         vars=[V('dT', 'K', 'temperature rise (K = °C difference)', label='ΔT'),
               V('E', 'J', 'energy absorbed by the brake'), V('m_brake', 'kg', 'mass of the brake drum/disk'),
               V('cp', 'J/(kg*K)', 'specific heat capacity of the brake material')],
         equations=['dT = E/(m_brake*cp)'], outputs=['dT'],
         example={'E': 50000.0, 'm_brake': 5.0, 'cp': 500.0},
         keywords=['heating', 'fade', 'thermal'],
         assumptions=['Adiabatic single-stop estimate (no cooling, no conduction to other parts): all absorbed energy '
                      'heats the brake mass. Enter dT in K or delta_degC.'],
         refs=['Heat balance, delta_T = E/(m c_p)']),

    Calc('flywheel_energy', 'Flywheel energy fluctuation and coefficient of speed fluctuation',
         vars=[V('delta_E', 'J', 'energy fluctuation (largest energy the flywheel stores/returns)', label='ΔE'),
               V('C_s', '', 'coefficient of speed fluctuation (omega_max - omega_min)/omega_avg', label='Cs'),
               V('omega_avg', 'rad/s', 'mean angular speed', label='ωavg'),
               V('omega_max', 'rad/s', 'maximum angular speed', label='ωmax'),
               V('omega_min', 'rad/s', 'minimum angular speed', label='ωmin'),
               V('I', 'kg*m^2', 'flywheel inertia')],
         equations=['omega_avg = (omega_max + omega_min)/2', 'C_s = (omega_max - omega_min)/omega_avg',
                    'delta_E = 0.5*I*(omega_max**2 - omega_min**2)'],
         outputs=['omega_avg', 'C_s', 'delta_E'],
         example={'omega_max': 105.0, 'omega_min': 95.0, 'I': 3.0},
         keywords=['flywheel', 'speed fluctuation', 'coefficient of fluctuation'],
         assumptions=['delta_E = 1/2 I (omega_max^2 - omega_min^2) is the exact fluctuation; it also equals '
                      'I omega_avg^2 C_s. The coefficient C_s is the standard definition.'],
         refs=['Flywheel energy fluctuation; coefficient of speed fluctuation (Shigley; recalled)']),

    Calc('flywheel_inertia_required', 'Required flywheel inertia for a given speed fluctuation',
         vars=[V('I', 'kg*m^2', 'required flywheel inertia'), V('delta_E', 'J', 'energy fluctuation to be absorbed', label='ΔE'),
               V('C_s', '', 'allowable coefficient of speed fluctuation', label='Cs'),
               V('omega', 'rad/s', 'mean angular speed', label='ω')],
         equations=['I = delta_E/(C_s*omega**2)'], outputs=['I'],
         example={'delta_E': 3000.0, 'C_s': 0.1, 'omega': 100.0},
         keywords=['flywheel', 'sizing', 'coefficient of fluctuation'],
         assumptions=['Inverts delta_E = I omega^2 C_s. Combine with flywheel_rim to turn the required inertia into '
                      'rim dimensions.'],
         refs=['Flywheel sizing, I = delta_E/(C_s omega^2)']),

    Calc('flywheel_rim', 'Thin-rim flywheel: mass, volume and inertia',
         vars=[V('I', 'kg*m^2', 'rim moment of inertia about the spin axis'), V('m', 'kg', 'rim mass'),
               V('V', 'm^3', 'rim volume'), V('rho', 'kg/m^3', 'rim material density', label='ρ'),
               V('R', 'm', 'mean rim radius'), V('b', 'm', 'rim width (axial thickness)'),
               V('t', 'm', 'rim thickness (radial)')],
         equations=['V = 2*pi*R*b*t', 'm = rho*V', 'I = m*R**2'], outputs=['V', 'm', 'I'],
         example={'rho': 7850.0, 'R': 0.3, 'b': 0.05, 't': 0.02},
         keywords=['flywheel', 'rim', 'ring', 'inertia'],
         assumptions=['Thin rim (t << R) so the whole mass is treated at radius R and I = m R^2; volume is the mean '
                      'circumference times cross-section. Spokes/hub add a little mass and inertia.'],
         refs=['Thin-ring inertia, I = m R^2']),

    Calc('flywheel_torque_angle', 'Flywheel energy fluctuation from a turning-moment diagram',
         vars=[V('torque_curve', '', '[[angle (deg), torque (N·m)], ...] over one full cycle, non-decreasing angles '
                                     '(repeat an angle for a vertical step)', kind='json'),
               V('T_mean', 'N*m', 'mean (resisting) torque over the cycle'),
               V('W_max', 'J', 'maximum cumulative energy above the mean line'),
               V('W_min', 'J', 'minimum cumulative energy above the mean line (typically 0 or negative)'),
               V('delta_E', 'J', 'energy fluctuation W_max - W_min', label='ΔE'),
               V('energy_curve', '', '[[angle (deg), cumulative energy (J)], ...]', kind='json')],
         procedure=_turning_moment, outputs=['T_mean', 'W_max', 'W_min', 'delta_E', 'energy_curve'],
         example={'torque_curve': [[0, 150], [180, 150], [180, 50], [360, 50]]},
         plots=[{'id': 'turning_moment', 'title': 'Turning-moment and energy-fluctuation diagrams',
                 'fn': _turning_moment_chart}],
         keywords=['turning moment', 'energy fluctuation', 'crank effort'],
         assumptions=['The torque curve is one complete cycle (a repeating periodic input); the mean torque is the '
                      'trapezoidal average, and the energy fluctuation is the swing of the cumulative (T - T_mean) '
                      'integral. delta_E is the energy the flywheel must store and return.'],
         refs=['Turning-moment diagram; energy fluctuation = area between torque curve and mean-torque line']),
]

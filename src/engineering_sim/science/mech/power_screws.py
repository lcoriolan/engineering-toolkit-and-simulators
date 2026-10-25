# DESCRIPTION: Power screws section: raising/lowering torque for square and Acme/trapezoidal threads,
#              collar friction torque, efficiency and self-locking, and thread bearing/bending stress.
"""Power-screw calculators.

Relation to ``statics.screw_jack``: that pilot covers the square thread with collar friction folded out;
this section adds the thread-angle (secant) term for Acme/trapezoidal threads, a separate collar
torque, multi-start lead l = starts × pitch, efficiency/self-locking and thread stresses. For a square
thread (α = 0) with no collar, ``power_screw_torque`` reduces to ``screw_jack``'s T_raise/T_lower.

α is the thread flank *half*-angle measured from the plane normal to the screw axis: 0 for square,
14.5° for a 29° Acme thread, 15° for an ISO metric trapezoidal thread. The secant (1/cos α) term is the
wedge action that raises the normal force on the flank.

References are the standard power-screw relations in machine-design texts (Shigley, *Mechanical
Engineering Design*, ch. 8), cited by relation name and recalled rather than re-read.
"""
import math

from engineering_sim.science.mech.core import Calc, V, section

SECTION = section('power_screws', 'Power screws', order=140,
                  summary='Raising/lowering torque (square and Acme), collar friction, efficiency and '
                          'self-locking, thread bearing and bending stress.')

PI = math.pi


CALCS = [
    Calc('power_screw_torque', 'Power screw raising and lowering torque (square or Acme)',
         vars=[V('T_R', 'N*m', 'torque to raise the load (thread only, collar excluded)'),
               V('T_L', 'N*m', 'torque to lower (thread only; negative = overhauling)', signed=True),
               V('l', 'm', 'lead (advance per revolution = starts × pitch)'),
               V('lam', 'rad', 'lead angle atan(l/(π·d_m))', hi=PI / 2, label='λ'),
               V('F', 'N', 'axial load'),
               V('d_m', 'm', 'mean (pitch) diameter'),
               V('f', '', 'thread friction coefficient', label='f'),
               V('alpha', 'rad', 'thread flank half-angle (0 square, 14.5° Acme, 15° ISO trapezoidal)',
                 hi=PI / 2, label='α'),
               V('n_s', '', 'number of thread starts', integer=True, lo=1),
               V('p', 'm', 'pitch')],
         equations=['T_R = F*d_m/2*(l + pi*f*d_m/cos(alpha))/(pi*d_m - f*l/cos(alpha))',
                    'T_L = F*d_m/2*(pi*f*d_m/cos(alpha) - l)/(pi*d_m + f*l/cos(alpha))',
                    'l = n_s*p',
                    'lam = atan(l/(pi*d_m))'],
         outputs=['T_R', 'T_L', 'l', 'lam'],
         example={'F': 5000.0, 'd_m': 0.032, 'f': 0.15, 'alpha': 0.0, 'n_s': 1, 'p': 0.006},
         keywords=['acme', 'trapezoidal', 'multi-start', 'lead', 'jack'],
         assumptions=['1/cos(α) is the secant wedge term (α the flank half-angle; sec α = 1 for square '
                      'threads). Collar friction is separate — add the collar torque T_c to T_R and T_L '
                      'for the total.', 'T_L > 0 means self-locking against back-drive. Multi-start lead '
                      'l = n_s·p.'],
         refs=['Power screw torque with thread angle (Shigley eqs. 8-1/8-2 form, recalled)']),

    Calc('power_screw_collar', 'Collar friction torque',
         vars=[V('T_c', 'N*m', 'collar friction torque'),
               V('F', 'N', 'axial load'),
               V('f_c', '', 'collar friction coefficient', label='fc'),
               V('d_c', 'm', 'mean collar diameter')],
         equations=['T_c = F*f_c*d_c/2'], outputs=['T_c'],
         example={'F': 5000.0, 'f_c': 0.15, 'd_c': 0.05},
         keywords=['thrust collar', 'flat collar', 'mean radius'],
         assumptions=['Uniform pressure on a flat annular collar with friction at the mean radius; add '
                      'this torque to the thread torque for the total input.'],
         refs=['T_c = F·f_c·d_c/2']),

    Calc('power_screw_efficiency', 'Power screw efficiency and self-locking condition',
         vars=[V('e', '', 'thread efficiency (collar excluded)', label='η'),
               V('n_lock', '', 'self-locking factor π·f·d_m/(l·cos α); > 1 self-locking'),
               V('d_m', 'm', 'mean (pitch) diameter'),
               V('l', 'm', 'lead'),
               V('f', '', 'thread friction coefficient', label='f'),
               V('alpha', 'rad', 'thread flank half-angle', hi=PI / 2, label='α')],
         equations=['e = l*(pi*d_m - f*l/cos(alpha))/(pi*d_m*(l + pi*f*d_m/cos(alpha)))',
                    'n_lock = pi*f*d_m/(l*cos(alpha))'],
         outputs=['e', 'n_lock'],
         example={'d_m': 0.032, 'l': 0.006, 'f': 0.15, 'alpha': 0.0},
         keywords=['self-locking', 'back-drive', 'efficiency'],
         assumptions=['Thread-only efficiency e = F·l/(2π·T_R); collar friction lowers the overall '
                      'efficiency below this. Self-locking (no back-drive) requires π·f·d_m > l·cos α, '
                      'i.e. n_lock > 1, equivalently T_L > 0 in power_screw_torque.'],
         refs=['Power screw efficiency; self-locking condition (recalled)']),

    Calc('power_screw_thread_stress', 'Power screw thread bearing and bending stress',
         vars=[V('sigma_B', 'Pa', 'thread bearing stress (compressive; magnitude)'),
               V('sigma_b', 'Pa', 'thread root bending stress (magnitude)'),
               V('c_arm', 'm', 'thread depth (d_m − d_r)/2'),
               V('F', 'N', 'axial load'),
               V('d_m', 'm', 'mean (pitch) diameter'),
               V('d_r', 'm', 'root (minor) diameter'),
               V('p', 'm', 'pitch'),
               V('n_t', '', 'number of engaged threads', integer=True, lo=1),
               V('t', 'm', 'thread thickness at the root (square p/2; 29° Acme ≈ 0.63p — not the 0.37p flat width)')],
         equations=['sigma_B = 2*F/(pi*d_m*n_t*p)',
                    'c_arm = (d_m - d_r)/2',
                    'sigma_b = 6*F*c_arm/(pi*d_r*n_t*t**2)'],
         outputs=['sigma_B', 'c_arm', 'sigma_b'],
         example={'F': 5000.0, 'd_m': 0.032, 'd_r': 0.029, 'p': 0.006, 'n_t': 8, 't': 0.003},
         keywords=['bearing pressure', 'nut', 'engaged threads', 'root'],
         assumptions=['Bearing stress uses the projected pitch-line area π·d_m·(p/2) per engaged thread, '
                      'so σ_B = 2F/(π·d_m·n_t·p). Bending models each thread as a cantilever of depth '
                      'c_arm = (d_m − d_r)/2 and root section π·d_r·t carrying F/n_t at the pitch line '
                      '(σ_b = M·(t/2)/I with M = (F/n_t)·c_arm and I = π·d_r·t³/12). Threads are assumed to share '
                      'the load equally — real nut threads load unevenly (first engaged thread ≈ 0.38F).',
                      'For a square thread (depth p/2, so c_arm = p/4 and t = p/2) this reduces exactly to the '
                      'standard σ_b = 6F/(π·d_r·n_t·p). The root thickness t is a geometry input (square: p/2; '
                      '29° Acme ≈ p/2 + 2·(0.25p)·tan14.5° ≈ 0.63p), not a built-in table. '
                      '[Reviewer fix 2026-09-15: Acme guidance previously quoted the 0.37p flat width; example '
                      'geometry made a consistent square thread.]'],
         refs=['σ_B = 2F/(π·d_m·n_t·p); cantilever thread-bending model (Shigley sec. 8-2 form, recalled)']),
]

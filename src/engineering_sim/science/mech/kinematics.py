# DESCRIPTION: Kinematics section: rectilinear and curvilinear motion, normal/tangential and polar
#              components, projectile motion, relative motion, rotation about a fixed axis, the
#              instantaneous centre, Coriolis acceleration and rotating reference frames.
"""Kinematics calculators.

Conventions: displacement/velocity/acceleration are signed along their axis; angles are radians with
CCW positive; ω and α are angular (enter rad/s, rpm or deg/s — a Hz value is refused). Only the
kinematics of a point or a rigid body is treated here; the forces that cause motion are in kinetics.

Relations are standard particle/rigid-body kinematics (Hibbeler, *Engineering Mechanics: Dynamics*;
Meriam & Kraige; Beer & Johnston). Cited by relation name, recalled rather than re-read.
"""
import math

from engineering_sim.science.mech.core import Calc, V, section

SECTION = section('kinematics', 'Kinematics', order=60,
                  summary='Rectilinear/curvilinear motion, normal-tangential and polar components, '
                          'projectiles, relative motion, rotation, instantaneous centre, Coriolis.')

PI = math.pi


CALCS = [
    Calc('rectilinear_constant_a', 'Rectilinear motion with constant acceleration',
         vars=[V('v', 'm/s', 'final velocity', signed=True), V('v0', 'm/s', 'initial velocity', signed=True),
               V('a', 'm/s^2', 'constant acceleration', signed=True), V('t', 's', 'time interval'),
               V('s', 'm', 'displacement', signed=True)],
         equations=['v = v0 + a*t', 's = v0*t + 0.5*a*t**2'], outputs=['v', 's'],
         example={'v0': 10.0, 'a': 2.0, 't': 5.0},
         keywords=['suvat', 'uniform acceleration', 'linear motion'],
         assumptions=['Straight-line motion, constant acceleration; signs are along the chosen axis '
                      '(negative a decelerates).'],
         refs=['Constant-acceleration kinematics v = v0 + at; s = v0t + ½at²']),

    Calc('rectilinear_vsq', 'Velocity from displacement without time (v² = v0² + 2as)',
         vars=[V('v', 'm/s', 'final speed (magnitude)'), V('v0', 'm/s', 'initial velocity', signed=True),
               V('a', 'm/s^2', 'constant acceleration', signed=True), V('s', 'm', 'displacement', signed=True)],
         equations=['v**2 = v0**2 + 2*a*s'], outputs=['v'],
         example={'v0': 10.0, 'a': 2.0, 's': 75.0},
         keywords=['no time', 'speed'],
         assumptions=['Constant acceleration; the equation gives v² so the recovered speed is a '
                      'magnitude — assign the direction from the motion sense.'],
         refs=['v² = v0² + 2as (eliminate t from the constant-acceleration equations)']),

    Calc('normal_tangential', 'Normal and tangential acceleration components (curvilinear motion)',
         vars=[V('a_n', 'm/s^2', 'normal (centripetal) acceleration', label='an'),
               V('a_t', 'm/s^2', 'tangential acceleration (rate of change of speed)', signed=True, label='at'),
               V('a', 'm/s^2', 'total acceleration magnitude'), V('v', 'm/s', 'speed'),
               V('rho', 'm', 'radius of curvature', label='ρ')],
         equations=['a_n = v**2/rho', 'a = sqrt(a_n**2 + a_t**2)'], outputs=['a_n', 'a'],
         example={'v': 20.0, 'rho': 50.0, 'a_t': 6.0},
         keywords=['centripetal', 'curvilinear', 'radius of curvature', 'path'],
         assumptions=['a_n points toward the centre of curvature; a_t is along the path (negative for '
                      'deceleration). Solving for a_t from the total magnitude returns its absolute value.'],
         refs=['a_n = v²/ρ; a = √(a_n² + a_t²)']),

    Calc('polar_motion', 'Polar-coordinate velocity and acceleration components',
         vars=[V('r', 'm', 'radial distance'), V('rdot', 'm/s', 'radial velocity ṙ', signed=True, label='ṙ'),
               V('theta_dot', 'rad/s', 'angular velocity θ̇', signed=True, label='θ̇'),
               V('rddot', 'm/s^2', 'radial acceleration r̈', signed=True, label='r̈'),
               V('theta_ddot', 'rad/s^2', 'angular acceleration θ̈', signed=True, label='θ̈'),
               V('v_r', 'm/s', 'radial velocity component', signed=True, label='vr'),
               V('v_theta', 'm/s', 'transverse velocity component', signed=True, label='vθ'),
               V('a_r', 'm/s^2', 'radial acceleration component', signed=True, label='ar'),
               V('a_theta', 'm/s^2', 'transverse acceleration component', signed=True, label='aθ')],
         equations=['v_r = rdot', 'v_theta = r*theta_dot', 'a_r = rddot - r*theta_dot**2',
                    'a_theta = r*theta_ddot + 2*rdot*theta_dot'],
         outputs=['v_r', 'v_theta', 'a_r', 'a_theta'],
         example={'r': 2.0, 'rdot': 3.0, 'theta_dot': 0.5, 'rddot': 1.0, 'theta_ddot': 0.2},
         keywords=['polar coordinates', 'radial', 'transverse', 'r theta'],
         assumptions=['θ̇ and θ̈ are angular (rad/s, rad/s²). The −rθ̇² term is the centripetal part and '
                      '2ṙθ̇ the Coriolis part of the radial/transverse split.'],
         refs=['Polar components: v_r = ṙ, v_θ = rθ̇; a_r = r̈ − rθ̇², a_θ = rθ̈ + 2ṙθ̇']),

    Calc('projectile', 'Projectile range, maximum height and time of flight',
         vars=[V('R', 'm', 'horizontal range'), V('H', 'm', 'maximum height'), V('t_flight', 's', 'time of flight'),
               V('v0', 'm/s', 'launch speed'), V('theta', 'rad', 'launch angle above horizontal', hi=PI / 2, label='θ'),
               V('g', 'm/s^2', 'gravitational acceleration')],
         equations=['R = v0**2*sin(2*theta)/g', 'H = v0**2*sin(theta)**2/(2*g)',
                    't_flight = 2*v0*sin(theta)/g'],
         outputs=['R', 'H', 't_flight'],
         example={'v0': 30.0, 'theta': math.radians(45), 'g': 9.80665},
         keywords=['projectile', 'range', 'trajectory', 'flight time'],
         assumptions=['Flat ground, launch and landing at the same height, no air resistance. Range is '
                      'maximum at θ = 45°.'],
         refs=['Projectile range R = v0² sin(2θ)/g; height H = v0² sin²θ/(2g); flight time 2v0 sinθ/g']),

    Calc('relative_velocity', 'Relative velocity and acceleration of two particles (1-D)',
         vars=[V('v_AB', 'm/s', 'velocity of A relative to B', signed=True, label='vA/B'),
               V('v_A', 'm/s', 'velocity of A', signed=True, label='vA'),
               V('v_B', 'm/s', 'velocity of B', signed=True, label='vB'),
               V('a_AB', 'm/s^2', 'acceleration of A relative to B', signed=True, label='aA/B'),
               V('a_A', 'm/s^2', 'acceleration of A', signed=True, label='aA'),
               V('a_B', 'm/s^2', 'acceleration of B', signed=True, label='aB')],
         equations=['v_AB = v_A - v_B', 'a_AB = a_A - a_B'], outputs=['v_AB', 'a_AB'],
         example={'v_A': 30.0, 'v_B': 20.0, 'a_A': 2.0, 'a_B': 1.0},
         keywords=['relative motion', 'velocity difference', 'overtaking'],
         assumptions=['One-dimensional (collinear) relative motion: v_A/B = v_A − v_B.'],
         refs=['Relative velocity v_A/B = v_A − v_B; a_A/B = a_A − a_B']),

    Calc('rotation_fixed_axis', 'Rotation about a fixed axis: velocity and acceleration of a point',
         vars=[V('v', 'm/s', 'tangential speed'), V('a_t', 'm/s^2', 'tangential acceleration', signed=True, label='at'),
               V('a_n', 'm/s^2', 'normal (centripetal) acceleration', label='an'),
               V('omega', 'rad/s', 'angular velocity', label='ω'), V('alpha', 'rad/s^2', 'angular acceleration', signed=True, label='α'),
               V('r', 'm', 'distance of the point from the axis')],
         equations=['v = omega*r', 'a_t = alpha*r', 'a_n = omega**2*r'], outputs=['v', 'a_t', 'a_n'],
         example={'omega': 10.0, 'alpha': 2.0, 'r': 0.5},
         keywords=['angular velocity', 'angular acceleration', 'rigid body', 'centripetal'],
         assumptions=['Point attached to a body rotating about a fixed axis; a_n points toward the axis, '
                      'a_t is tangent (sign follows α). Enter ω/α as angular units (rpm, rad/s).'],
         refs=['v = ωr; a_t = αr; a_n = ω²r']),

    Calc('instantaneous_center', 'Instantaneous centre of rotation (r = v/ω)',
         vars=[V('r_ic', 'm', 'distance from the point to the instantaneous centre', label='rIC'),
               V('v', 'm/s', 'speed of the point'), V('omega', 'rad/s', 'angular velocity of the body', label='ω')],
         equations=['r_ic = v/omega'], outputs=['r_ic'],
         example={'v': 6.0, 'omega': 2.0},
         keywords=['instantaneous center', 'ic', 'rolling', 'zero velocity point'],
         assumptions=['The IC is the point with zero velocity, on the line through the point perpendicular '
                      'to v, at distance r = v/ω. For pure rolling it is the contact point. ω = 0 (pure '
                      'translation) has no finite IC.'],
         refs=['Instantaneous centre: r = v/ω']),

    Calc('coriolis_accel', 'Coriolis acceleration of a point moving in a rotating frame',
         vars=[V('a_cor', 'm/s^2', 'Coriolis acceleration magnitude', label='acor'),
               V('omega', 'rad/s', 'angular velocity of the rotating frame', label='ω'),
               V('v_rel', 'm/s', 'relative velocity (component perpendicular to ω)', label='vrel')],
         equations=['a_cor = 2*omega*v_rel'], outputs=['a_cor'],
         example={'omega': 5.0, 'v_rel': 3.0},
         keywords=['coriolis', 'rotating frame', 'earth rotation'],
         assumptions=['Magnitude for v_rel perpendicular to ω; direction from the right-hand rule (ω × v_rel). '
                      'Only the velocity component perpendicular to the axis contributes.'],
         refs=['Coriolis acceleration a_cor = 2ω × v_rel (magnitude 2ωv_rel)']),

    Calc('rotating_frame_accel', 'Acceleration of a particle sliding along a rod rotating at constant ω',
         vars=[V('a_rad', 'm/s^2', 'radial acceleration r̈ − ω²r (negative = toward the axis)', signed=True, label='arad'),
               V('a_cor', 'm/s^2', 'Coriolis (transverse) acceleration 2ωv_r', label='acor'),
               V('a_total', 'm/s^2', 'total acceleration magnitude', label='atot'),
               V('omega', 'rad/s', 'angular velocity (constant)', label='ω'),
               V('v_r', 'm/s', 'radial relative velocity along the rod'), V('r', 'm', 'radial distance'),
               V('rddot', 'm/s^2', 'radial relative acceleration along the rod (0 = constant sliding speed)',
                 signed=True, label='r̈')],
         equations=['a_rad = rddot - omega**2*r', 'a_cor = 2*omega*v_r', 'a_total = sqrt(a_rad**2 + a_cor**2)'],
         outputs=['a_rad', 'a_cor', 'a_total'],
         example={'omega': 5.0, 'v_r': 2.0, 'r': 0.4, 'rddot': 0.0},
         keywords=['rotating reference frame', 'bead on rod', 'coriolis', 'centripetal'],
         assumptions=['Rod rotating at constant ω (ω̇ = 0, so the transverse term is only Coriolis 2ωv_r); '
                      'radial term includes the relative acceleration r̈. For ω̇ ≠ 0 use kinematics.polar_motion '
                      '(a_θ = rθ̈ + 2ṙθ̇). Coriolis direction by right-hand rule. '
                      '[Generalised with r̈ after review 2026-09-15; previously implicit r̈ = 0.]'],
         refs=['Polar-coordinate acceleration a_r = r̈ − rθ̇², a_θ = rθ̈ + 2ṙθ̇ (recalled)']),
]

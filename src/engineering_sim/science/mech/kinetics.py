# DESCRIPTION: Kinetics section: Newton's second law, centripetal force, work/energy, power and
#              efficiency, impulse-momentum, angular momentum, impact and restitution, mass moment of
#              inertia of standard solids, planar rigid-body motion, gyroscopic effects and rockets.
"""Kinetics calculators.

Conventions: forces and velocities are signed along their axis; ω and α are angular (rad/s, rad/s²);
standard gravity is entered explicitly as g (default example value from common.g0()). Only particle and
rigid-body dynamics are treated; the resulting motion is the kinematics section's concern.

Relations are standard particle/rigid-body dynamics (Hibbeler, *Engineering Mechanics: Dynamics*;
Meriam & Kraige; Beer & Johnston). Cited by relation name, recalled rather than re-read.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import common

SECTION = section('kinetics', 'Kinetics', order=65,
                  summary='Newton/F=ma, centripetal force, work and energy, power, impulse-momentum, '
                          'impact, mass moment of inertia, rigid-body motion, gyroscopes, rockets.')

PI = math.pi


def _mass_inertia(values):
    """Mass moment of inertia of a standard solid about a centroidal axis, plus parallel-axis shift.

    ``solid`` is a dict: {'shape': 'rod'|'disk'|'cylinder'|'sphere'|'shell'|'block', 'm': mass kg,
    and the needed dimension(s)}: rod → 'L' (length); disk/cylinder (central axis) / sphere / shell (thin SPHERICAL shell, about a diameter) → 'r'; block → 'a','b'. Solid-cylinder transverse axis m(3r²+h²)/12 is not included
    (the two in-plane side lengths). ``d`` is the parallel-axis offset from that centroidal axis.
    Returns I_c (about the centroidal axis) and I = I_c + m·d².
    """
    solid = values['solid']
    if not isinstance(solid, dict):
        raise MechError('solid: give {"shape": ..., "m": ..., and the dimension(s)} in SI')
    shape = str(solid.get('shape', '')).lower()
    m = float(solid.get('m'))
    d = float(values['d'])
    if m <= 0:
        raise MechError('solid.m must be positive')

    def dim(name):
        if solid.get(name) is None:
            raise MechError(f"shape '{shape}' needs dimension {name!r}")
        return float(solid[name])

    if shape == 'rod':
        I = m * dim('L') ** 2 / 12          # thin rod, about perpendicular centroidal axis
    elif shape in ('disk', 'cylinder'):
        I = m * dim('r') ** 2 / 2           # solid disk/cylinder, about the central axis
    elif shape == 'sphere':
        I = 2 * m * dim('r') ** 2 / 5       # solid sphere, about a diameter
    elif shape == 'shell':
        I = 2 * m * dim('r') ** 2 / 3       # thin spherical shell, about a diameter
    elif shape == 'block':
        I = m * (dim('a') ** 2 + dim('b') ** 2) / 12   # rectangular block, axis ⊥ to the a×b face
    else:
        raise MechError(f"unknown shape {shape!r}: use rod, disk, cylinder, sphere, shell or block")
    return {'I_c': I, 'I': I + m * d ** 2}


CALCS = [
    Calc('newton_second_law', "Newton's second law (F = ma)",
         vars=[V('F', 'N', 'net force', signed=True), V('m', 'kg', 'mass'),
               V('a', 'm/s^2', 'acceleration', signed=True)],
         equations=['F = m*a'], outputs=['F'],
         example={'m': 5.0, 'a': 3.0},
         keywords=['force', 'acceleration', 'newton', 'f=ma'],
         assumptions=['Net force on a particle (or the mass-centre of a body), in an inertial frame.'],
         refs=["Newton's second law F = ma"]),

    Calc('centripetal_force', 'Centripetal force (F_c = mv²/r = mω²r)',
         vars=[V('F_c', 'N', 'centripetal (radially inward) force', label='Fc'),
               V('m', 'kg', 'mass'), V('v', 'm/s', 'tangential speed'), V('r', 'm', 'path radius'),
               V('omega', 'rad/s', 'angular velocity (v = ωr)', label='ω')],
         equations=['F_c = m*v**2/r', 'v = omega*r'], outputs=['F_c', 'v'],
         example={'m': 2.0, 'r': 0.5, 'omega': 4.0},
         keywords=['centripetal', 'circular motion', 'normal force'],
         assumptions=['Uniform circular motion; the force points toward the centre. Enter either v or '
                      'ω (with r) — the two are tied by v = ωr.'],
         refs=['F_c = mv²/r = mω²r']),

    Calc('work_constant_force', 'Work done by a constant force (W = F·d·cosθ)',
         vars=[V('W', 'J', 'work done', signed=True), V('F', 'N', 'force magnitude'),
               V('d', 'm', 'displacement magnitude'), V('theta', 'rad', 'angle between force and displacement', hi=PI, label='θ')],
         equations=['W = F*d*cos(theta)'], outputs=['W'],
         example={'F': 100.0, 'd': 2.0, 'theta': math.radians(60)},
         keywords=['work', 'dot product', 'energy'],
         assumptions=['Constant force along a straight displacement; W < 0 when the force opposes the motion.'],
         refs=['Work W = F·d·cosθ']),

    Calc('kinetic_energy', 'Translational kinetic energy (KE = ½mv²)',
         vars=[V('KE', 'J', 'kinetic energy'), V('m', 'kg', 'mass'), V('v', 'm/s', 'speed')],
         equations=['KE = 0.5*m*v**2'], outputs=['KE'],
         example={'m': 2.0, 'v': 5.0},
         keywords=['energy', 'translation'],
         refs=['Kinetic energy KE = ½mv²']),

    Calc('rotational_kinetic_energy', 'Rotational kinetic energy (KE = ½Iω²)',
         vars=[V('KE', 'J', 'rotational kinetic energy'), V('I', 'kg*m^2', 'mass moment of inertia'),
               V('omega', 'rad/s', 'angular velocity', label='ω')],
         equations=['KE = 0.5*I*omega**2'], outputs=['KE'],
         example={'I': 0.5, 'omega': 4.0},
         keywords=['energy', 'rotation', 'flywheel'],
         refs=['Rotational kinetic energy KE = ½Iω²']),

    Calc('potential_energy', 'Gravitational and spring potential energy',
         vars=[V('PE_g', 'J', 'gravitational potential energy mgh', signed=True), V('PE_s', 'J', 'spring potential energy ½kx²'),
               V('m', 'kg', 'mass'), V('g', 'm/s^2', 'gravitational acceleration (standard g = 9.80665 m/s²)'),
               V('h', 'm', 'height above the datum', signed=True),
               V('k', 'N/m', 'spring stiffness'), V('x', 'm', 'spring displacement from free length', signed=True)],
         equations=['PE_g = m*g*h', 'PE_s = 0.5*k*x**2'], outputs=['PE_g', 'PE_s'],
         example={'m': 3.0, 'g': common.g0(), 'h': 10.0, 'k': 200.0, 'x': 0.1},
         keywords=['energy', 'gravity', 'spring', 'datum'],
         assumptions=['h measured from an arbitrary datum (PE is relative); x is the spring stretch or '
                      'compression from its free length.'],
         refs=['PE_g = mgh; PE_s = ½kx²']),

    Calc('power', 'Mechanical power (P = F·v)',
         vars=[V('P', 'W', 'power'), V('F', 'N', 'force (component along the velocity)'),
               V('v', 'm/s', 'velocity')],
         equations=['P = F*v'], outputs=['P'],
         example={'F': 200.0, 'v': 3.0},
         keywords=['power', 'work rate', 'horsepower'],
         assumptions=['Force and velocity in the same direction. Rotational power P = Tω is the shaft-power '
                      'calculator in the mechanics-of-materials section.'],
         refs=['Power P = F·v']),

    Calc('efficiency', 'Mechanical efficiency (η = P_out/P_in)',
         vars=[V('eta', '', 'efficiency', hi=1.0, label='η'), V('P_out', 'W', 'output (useful) power'),
               V('P_in', 'W', 'input power')],
         equations=['eta = P_out/P_in'], outputs=['eta'],
         example={'P_out': 600.0, 'P_in': 750.0},
         keywords=['efficiency', 'losses', 'output/input'],
         assumptions=['Power losses are P_in − P_out = P_in(1 − η).'],
         refs=['Efficiency η = P_out/P_in']),

    Calc('linear_impulse', 'Linear impulse and momentum (J = F·Δt = mΔv)',
         vars=[V('J', 'N*s', 'impulse', signed=True), V('F', 'N', 'average force', signed=True),
               V('dt', 's', 'time interval', label='Δt'), V('m', 'kg', 'mass'),
               V('v_i', 'm/s', 'initial velocity', signed=True, label='vi'),
               V('v_f', 'm/s', 'final velocity', signed=True, label='vf')],
         equations=['J = F*dt', 'v_f = v_i + F*dt/m'], outputs=['J', 'v_f'],
         example={'F': 50.0, 'dt': 0.1, 'm': 2.0, 'v_i': 5.0},
         keywords=['impulse', 'momentum', 'collision', 'impact force'],
         assumptions=['Impulse of a constant average force over Δt equals the change in linear momentum.'],
         refs=['Impulse–momentum: J = FΔt = m(v_f − v_i)']),

    Calc('angular_momentum', 'Angular momentum of a rigid body (H = Iω)',
         vars=[V('H', 'kg*m^2/s', 'angular momentum'), V('I', 'kg*m^2', 'mass moment of inertia'),
               V('omega', 'rad/s', 'angular velocity', label='ω')],
         equations=['H = I*omega'], outputs=['H'],
         example={'I': 0.5, 'omega': 4.0},
         keywords=['angular momentum', 'spin', 'moment of momentum'],
         assumptions=['About the axis of rotation (a principal axis). For a particle, I = mr² about the axis.'],
         refs=['Angular momentum H = Iω']),

    Calc('restitution', 'Coefficient of restitution (e = (v2′ − v1′)/(v1 − v2))',
         vars=[V('e', '', 'coefficient of restitution', hi=1.0),
               V('v1', 'm/s', 'velocity of body 1 before impact', signed=True),
               V('v2', 'm/s', 'velocity of body 2 before impact', signed=True),
               V('v1p', 'm/s', 'velocity of body 1 after impact', signed=True),
               V('v2p', 'm/s', 'velocity of body 2 after impact', signed=True)],
         equations=['e = (v2p - v1p)/(v1 - v2)'], outputs=['e'],
         example={'v1': 10.0, 'v2': 0.0, 'v1p': 2.0, 'v2p': 8.0},
         keywords=['restitution', 'bounce', 'impact', 'elastic'],
         assumptions=['Direct central impact; e = 1 perfectly elastic, e = 0 perfectly plastic. Requires '
                      'v1 ≠ v2 (bodies approaching).'],
         refs=['Coefficient of restitution e = (v2′ − v1′)/(v1 − v2)']),

    Calc('impact_velocities', 'Direct central impact: velocities after impact (given e and masses)',
         vars=[V('v1p', 'm/s', 'velocity of body 1 after impact', signed=True, label='v1′'),
               V('v2p', 'm/s', 'velocity of body 2 after impact', signed=True, label='v2′'),
               V('m1', 'kg', 'mass of body 1'), V('m2', 'kg', 'mass of body 2'),
               V('v1', 'm/s', 'velocity of body 1 before impact', signed=True),
               V('v2', 'm/s', 'velocity of body 2 before impact', signed=True),
               V('e', '', 'coefficient of restitution', hi=1.0)],
         equations=['v1p = ((m1 - e*m2)*v1 + m2*(1 + e)*v2)/(m1 + m2)',
                    'v2p = (m1*(1 + e)*v1 + (m2 - e*m1)*v2)/(m1 + m2)'],
         outputs=['v1p', 'v2p'],
         example={'m1': 2.0, 'm2': 3.0, 'v1': 10.0, 'v2': 0.0, 'e': 0.6},
         keywords=['collision', 'momentum conservation', 'central impact'],
         assumptions=['Direct central impact (motion along the line of centres) with coefficient of '
                      'restitution e; momentum and the restitution relation are solved together.'],
         refs=['Direct central impact solution from momentum conservation and e = (v2′ − v1′)/(v1 − v2)']),

    Calc('mass_moment_inertia', 'Mass moment of inertia of a standard solid (plus parallel-axis)',
         vars=[V('solid', '', '{"shape": "rod"|"disk"|"cylinder"|"sphere"|"shell"|"block", "m": kg, '
                              'dims}: rod → "L"; disk/cylinder/sphere/shell → "r"; block → "a","b"', kind='json'),
               V('d', 'm', 'parallel-axis offset distance (0 for the centroidal axis)'),
               V('I_c', 'kg*m^2', 'moment of inertia about the centroidal axis', label='Ic'),
               V('I', 'kg*m^2', 'moment of inertia about the parallel axis (I_c + m·d²)')],
         procedure=_mass_inertia, outputs=['I_c', 'I'],
         example={'solid': {'shape': 'sphere', 'm': 5.0, 'r': 0.2}, 'd': 0.3},
         keywords=['mass moment of inertia', 'parallel axis', 'rod', 'disk', 'sphere', 'block'],
         assumptions=['Standard solids about a centroidal principal axis: rod m L²/12; disk/solid cylinder '
                      'm r²/2; sphere 2mr²/5; thin spherical shell 2mr²/3; block m(a²+b²)/12 (axis ⊥ to the '
                      'a×b face). Parallel-axis theorem I = I_c + m·d².'],
         refs=['Standard mass moments of inertia; parallel-axis theorem']),

    Calc('radius_of_gyration', 'Radius of gyration (k = √(I/m))',
         vars=[V('k', 'm', 'radius of gyration'), V('I', 'kg*m^2', 'mass moment of inertia'), V('m', 'kg', 'mass')],
         equations=['k = sqrt(I/m)'], outputs=['k'],
         example={'I': 0.5, 'm': 2.0},
         keywords=['radius of gyration', 'k'],
         assumptions=['The distance k at which the whole mass could be concentrated to give the same I.'],
         refs=['Radius of gyration k = √(I/m)']),

    Calc('rigid_body_planar', 'Rigid-body planar motion: translation and rotation equations',
         vars=[V('a', 'm/s^2', 'acceleration of the mass centre', signed=True),
               V('F', 'N', 'net force on the body', signed=True), V('m', 'kg', 'mass'),
               V('alpha', 'rad/s^2', 'angular acceleration', signed=True, label='α'),
               V('M', 'N*m', 'net moment about the mass centre', signed=True),
               V('I_G', 'kg*m^2', 'mass moment of inertia about the mass centre', label='IG')],
         equations=['a = F/m', 'alpha = M/I_G'], outputs=['a', 'alpha'],
         example={'F': 30.0, 'm': 5.0, 'M': 6.0, 'I_G': 2.0},
         keywords=['newton-euler', 'planar motion', 'rolling', 'translation rotation'],
         assumptions=['Newton–Euler equations for a rigid body in plane motion: ΣF = m·a_G (translation of '
                      'the mass centre) and ΣM_G = I_G·α (rotation about the mass centre). For rolling '
                      'without slipping add a = αr; for a fixed-axis pivot use ΣM_O = I_O·α.'],
         refs=['Plane rigid-body equations of motion: ΣF = m a_G, ΣM_G = I_G α']),

    Calc('gyroscopic_torque', 'Gyroscopic torque and precession (M = IωΩ)',
         vars=[V('M', 'N*m', 'gyroscopic torque (moment required to precess)', label='M'),
               V('I', 'kg*m^2', 'spin moment of inertia about the spin axis'),
               V('omega', 'rad/s', 'spin angular velocity', label='ω'),
               V('Omega', 'rad/s', 'precession angular velocity', label='Ω')],
         equations=['M = I*omega*Omega'], outputs=['M'],
         example={'I': 0.5, 'omega': 100.0, 'Omega': 2.0},
         keywords=['gyroscope', 'precession', 'spin', 'bicycle wheel'],
         assumptions=['Spin axis perpendicular to the precession axis; the torque is perpendicular to both. '
                      'The reaction the spinning body exerts on its supports has the same magnitude.'],
         refs=['Gyroscopic torque M = I·ω·Ω']),

    Calc('rocket_thrust', 'Rocket thrust (T = ṁ·v_e)',
         vars=[V('T', 'N', 'thrust'), V('mdot', 'kg/s', 'propellant mass flow rate', label='ṁ'),
               V('v_e', 'm/s', 'effective exhaust velocity', label='ve')],
         equations=['T = mdot*v_e'], outputs=['T'],
         example={'mdot': 2.0, 'v_e': 3000.0},
         keywords=['rocket', 'thrust', 'exhaust velocity', 'variable mass'],
         assumptions=['Steady propellant mass flow; v_e is the effective exhaust velocity (specific impulse '
                      'I_sp = v_e/g0). With a nozzle pressure mismatch T = ṁ·v_exit + (p_e − p_a)·A_e, i.e. '
                      'v_e = v_exit + (p_e − p_a)·A_e/ṁ.'],
         refs=['Rocket thrust T = ṁ·v_e']),

    Calc('tsiolkovsky', 'Tsiolkovsky rocket equation (Δv = v_e·ln(m0/mf))',
         vars=[V('dv', 'm/s', 'velocity change (delta-v)', label='Δv'),
               V('v_e', 'm/s', 'effective exhaust velocity', label='ve'),
               V('m0', 'kg', 'initial mass (with propellant)'), V('mf', 'kg', 'final mass (propellant spent)')],
         equations=['dv = v_e*log(m0/mf)'], outputs=['dv'],
         example={'v_e': 3000.0, 'm0': 1000.0, 'mf': 200.0},
         keywords=['rocket equation', 'delta-v', 'variable mass', 'tsiolkovsky'],
         assumptions=['Ideal rocket, no drag or gravity, constant v_e; log is the natural logarithm.'],
         refs=['Tsiolkovsky rocket equation Δv = v_e·ln(m0/mf)']),
]

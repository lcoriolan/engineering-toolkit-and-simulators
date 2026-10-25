# DESCRIPTION: Aerospace basics section: level-flight lift coefficient, wing loading, aspect ratio,
#              induced drag, stall speed, glide ratio and distance, thrust-to-weight, specific impulse,
#              mass ratio and propellant fraction, orbital velocity/period, escape velocity, geometric-
#              to-geopotential altitude conversion and the ISA standard atmosphere.
"""Aerospace-basics calculators.

Conventions:
- Airspeed V is m/s; aerodynamic force L = ½ρV²SC_L with dynamic pressure q = ½ρV² (the 'C_L'/'C_D'
  here are dimensionless coefficients). Generic drag/lift forces live in fluid_dynamics.drag_lift and
  are not duplicated here; level_flight_cl gives the lift coefficient that balances weight in level
  flight.
- The standard atmosphere is the ISA / US Standard Atmosphere 1976 up to 86 km geometric altitude: the
  INPUT is geometric altitude z and is converted to geopotential altitude H = r0·z/(r0 + z) with
  r0 = 6356766 m before the layer equations (lapse-rate and isothermal) are applied. geopotential_altitude
  exposes that conversion both ways.
- Orbital quantities use the standard gravitational parameter GM (Earth 3.986004418e14 m³/s²) as a
  USER INPUT (common value stated in the description); specific impulse uses standard gravity g0.
- The Tsiolkovsky delta-v itself lives in kinetics.tsiolkovsky; rocket_equation here gives only the mass
  ratio and propellant fraction that feed it.

Relations follow Anderson, *Introduction to Flight*; Vallado, *Fundamentals of Astrodynamics*; and the
US Standard Atmosphere 1976 — cited by relation name, recalled.
"""
import math

from engineering_sim import correlations
from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import common

SECTION = section('aerospace', 'Aerospace basics', order=430,
                  summary='Level-flight lift coefficient, wing loading, stall speed, glide, thrust-to-weight, '
                          'specific impulse, mass ratio, orbits and the standard atmosphere.')

PI = math.pi

# ISA / US Standard Atmosphere 1976 layers (geopotential altitude):
# (H_base [m], T_base [K], lapse_rate [K/m])
_ISA_LAYERS = [
    (0.0, 288.15, -6.5e-3),
    (11000.0, 216.65, 0.0),
    (20000.0, 216.65, 1.0e-3),
    (32000.0, 228.65, 2.8e-3),
    (47000.0, 270.65, 0.0),
    (51000.0, 270.65, -2.8e-3),
    (71000.0, 214.65, -2.0e-3),
    (84852.0, 186.946, 0.0),   # top of the tabulated region
]
_G0 = 9.80665          # standard gravity, m/s²
_R_AIR = correlations.R_AIR_USSA1976   # dry-air gas constant from USSA 1976 R* and M0, J/(kg·K)
_GAMMA = 1.4           # ratio of specific heats (speed of sound)
_R_E = 6356766.0       # mean Earth radius used for the geometric -> geopotential conversion, m


def _atmosphere(values):
    """ISA / US Standard Atmosphere 1976 temperature, pressure, density and sound speed vs altitude.

    The input is GEOMETRIC altitude z; it is converted to geopotential altitude H = r0·z/(r0 + z) (the
    US 1976 independent variable) before the layer equations are applied.
    """
    z = values['z']
    r0 = values['r0']
    if r0 <= 0:
        raise MechError('r0 (mean Earth radius) must be positive')
    if z < 0:
        raise MechError('z (geometric altitude) must be >= 0')
    H = r0 * z / (r0 + z)                     # geopotential altitude
    if H > 84852.0 + 0.1:  # 86 km geometric converts to 84852.05 m; allow that boundary (verify24f)
        raise MechError('geometric altitude z = %.1f m is above the tabulated US 1976 range: geopotential '
                        'altitude H = r0·z/(r0+z) = %.1f m exceeds 84852 m (≈ 86 km geometric)' % (z, H))
    # Base pressure/temperature at each layer's lower boundary (march up from sea level).
    Pb = [101325.0]
    Tb = [288.15]
    for i in range(len(_ISA_LAYERS) - 1):
        H0, T0, L0 = _ISA_LAYERS[i]
        H1, T1, _ = _ISA_LAYERS[i + 1]
        if L0 == 0.0:
            P1 = Pb[-1] * math.exp(-_G0 * (H1 - H0) / (_R_AIR * T0))
        else:
            P1 = Pb[-1] * (T1 / T0) ** (-_G0 / (_R_AIR * L0))
        Pb.append(P1)
        Tb.append(T1)
    idx = 0
    for i in range(len(_ISA_LAYERS)):
        if H >= _ISA_LAYERS[i][0]:
            idx = i
        else:
            break
    H0, T0, L0 = _ISA_LAYERS[idx]
    if L0 == 0.0:                           # isothermal layer
        T = T0
        P = Pb[idx] * math.exp(-_G0 * (H - H0) / (_R_AIR * T0))
    else:                                   # constant-lapse-rate layer
        T = T0 + L0 * (H - H0)
        P = Pb[idx] * (T / T0) ** (-_G0 / (_R_AIR * L0))
    rho = P / (_R_AIR * T)
    a = math.sqrt(_GAMMA * _R_AIR * T)
    out = {'H': H, 'T': T, 'P': P, 'rho': rho, 'a': a}
    if H >= _ISA_LAYERS[-1][0] - 200.0:
        out['_warnings'] = ['altitude is near the top of the tabulated US 1976 layers '
                            '(H = 84852 m geopotential ≈ 86 km geometric)']
    return out


CALCS = [
    Calc('level_flight_cl', 'Lift coefficient required for level flight (C_L = 2W/(ρV²S))',
         vars=[V('C_L', '', 'lift coefficient required in level flight', label='CL'),
               V('W', 'N', 'aircraft weight'), V('rho', 'kg/m^3', 'air density', label='ρ'),
               V('V', 'm/s', 'airspeed'), V('S', 'm^2', 'wing reference area')],
         equations=['C_L = 2*W/(rho*V**2*S)'], outputs=['C_L'],
         example={'W': 12000.0, 'rho': 1.225, 'V': 60.0, 'S': 20.0},
         keywords=['lift', 'coefficient', 'level flight', 'trim'],
         assumptions=['Level (unaccelerated) flight: lift balances weight, so C_L = 2W/(ρV²S). For the '
                      'force itself see fluid_dynamics.drag_lift.'],
         refs=['L = W at level flight: C_L = 2W/(ρV²S)']),

    Calc('wing_loading', 'Wing loading (W/S)',
         vars=[V('W_S', 'N/m^2', 'wing loading'), V('W', 'N', 'aircraft weight'), V('S', 'm^2', 'wing reference area')],
         equations=['W_S = W/S'], outputs=['W_S'],
         example={'W': 12000.0, 'S': 20.0},
         keywords=['wing loading', 'W/S'],
         assumptions=['Gross-weight wing loading; drives stall speed and manoeuvrability.'],
         refs=['Wing loading = W/S']),

    Calc('stall_speed', 'Stall speed from maximum lift coefficient',
         vars=[V('v_stall', 'm/s', 'stall speed'), V('W', 'N', 'aircraft weight'), V('rho', 'kg/m^3', 'air density', label='ρ'),
               V('S', 'm^2', 'wing reference area'), V('C_Lmax', '', 'maximum lift coefficient', label='CLmax')],
         equations=['v_stall = sqrt(2*W/(rho*S*C_Lmax))'], outputs=['v_stall'],
         example={'W': 12000.0, 'rho': 1.225, 'S': 20.0, 'C_Lmax': 1.6},
         keywords=['stall', 'CLmax', 'landing speed'],
         assumptions=['Level flight: L = W at C_Lmax; steady, flaps as configured.'],
         refs=['L = W at stall: v_stall = sqrt(2W/(ρSC_Lmax))']),

    Calc('aspect_ratio', 'Wing aspect ratio',
         vars=[V('AR', '', 'aspect ratio'), V('b', 'm', 'wing span'), V('S', 'm^2', 'wing reference area')],
         equations=['AR = b**2/S'], outputs=['AR'],
         example={'b': 12.0, 'S': 20.0},
         keywords=['aspect ratio', 'span'],
         assumptions=['Planform aspect ratio; higher AR lowers induced drag.'],
         refs=['AR = b²/S']),

    Calc('induced_drag', 'Induced drag coefficient and force',
         vars=[V('C_Di', '', 'induced drag coefficient', label='CDi'), V('D_i', 'N', 'induced drag force'),
               V('C_L', '', 'lift coefficient', label='CL'), V('e', '', 'Oswald (span) efficiency factor', hi=1.0),
               V('AR', '', 'aspect ratio'), V('q', 'Pa', 'dynamic pressure ½ρv²'), V('S', 'm^2', 'wing reference area')],
         equations=['C_Di = C_L**2/(pi*e*AR)', 'D_i = C_Di*q*S'], outputs=['C_Di', 'D_i'],
         example={'C_L': 0.4, 'e': 0.8, 'AR': 7.2, 'q': 2205.0, 'S': 20.0},
         keywords=['induced drag', 'oswald', 'vortex drag'],
         assumptions=['Elliptical-loading correction via the Oswald factor e (user input ≈ 0.7–0.9); '
                      'C_Di adds to the parasite drag coefficient.'],
         refs=['C_Di = C_L²/(π e AR)']),

    Calc('glide_ratio', 'Glide (lift-to-drag) ratio, glide angle and glide distance',
         vars=[V('L_D', '', 'lift-to-drag ratio (glide ratio)', label='L/D'),
               V('gamma', 'rad', 'glide angle below horizontal', hi=PI / 2, label='γ'),
               V('d', 'm', 'still-air glide distance from a given height'),
               V('C_L', '', 'lift coefficient', label='CL'), V('C_D', '', 'drag coefficient', label='CD'),
               V('h', 'm', 'height (altitude) available for the glide')],
         equations=['L_D = C_L/C_D', 'gamma = atan(C_D/C_L)', 'd = h*L_D'], outputs=['L_D', 'gamma', 'd'],
         example={'C_L': 0.5, 'C_D': 0.025, 'h': 1000.0},
         keywords=['glide', 'L/D', 'lift to drag', 'glide distance'],
         assumptions=['Unpowered steady glide at the best (or chosen) L/D; the still-air glide distance is '
                      'h·(L/D). L/D itself is the glide distance per unit height.'],
         refs=['L/D = C_L/C_D; tan γ = D/L = 1/(L/D); glide distance d = h·(L/D)']),

    Calc('thrust_to_weight', 'Thrust-to-weight ratio',
         vars=[V('T_W', '', 'thrust-to-weight ratio', label='T/W'), V('T', 'N', 'thrust'), V('W', 'N', 'weight')],
         equations=['T_W = T/W'], outputs=['T_W'],
         example={'T': 40000.0, 'W': 60000.0},
         keywords=['thrust to weight', 'T/W'],
         assumptions=['Instantaneous value; > 1 enables a vertical climb.'],
         refs=['T/W ratio']),

    Calc('specific_impulse', 'Specific impulse and effective exhaust velocity',
         vars=[V('I_sp', 's', 'specific impulse'), V('v_e', 'm/s', 'effective exhaust velocity', label='ve'),
               V('F', 'N', 'thrust'), V('mdot', 'kg/s', 'propellant mass flow rate'),
               V('g0', 'm/s^2', 'standard gravity')],
         equations=['v_e = F/mdot', 'I_sp = v_e/g0'], outputs=['v_e', 'I_sp'],
         example={'F': 500000.0, 'mdot': 200.0, 'g0': common.g0()},
         keywords=['Isp', 'specific impulse', 'exhaust velocity', 'rocket'],
         assumptions=['Steady thrust; g0 = 9.80665 m/s² so I_sp is the thrust per unit weight flow.'],
         refs=['I_sp = F/(ṁ g0) = v_e/g0']),

    Calc('rocket_equation', 'Rocket mass ratio and propellant fraction',
         vars=[V('R', '', 'mass ratio (m0/mf)'), V('m_p_frac', '', 'propellant mass fraction (m_p/m0)', hi=1.0),
               V('m0', 'kg', 'initial mass (with propellant)'), V('mf', 'kg', 'final (dry) mass')],
         equations=['R = m0/mf', 'm_p_frac = (m0 - mf)/m0'], outputs=['R', 'm_p_frac'],
         example={'m0': 500000.0, 'mf': 60000.0},
         keywords=['tsiolkovsky', 'mass ratio', 'propellant fraction', 'rocket equation'],
         assumptions=['Mass bookkeeping for the ideal rocket: R = m0/mf and propellant fraction '
                      'm_p/m0 = (m0 − mf)/m0 = 1 − 1/R. The resulting velocity change '
                      'Δv = v_e·ln(R) is kinetics.tsiolkovsky.'],
         refs=['Mass ratio R = m0/mf; propellant fraction = 1 − 1/R (Δv in kinetics.tsiolkovsky)']),

    Calc('orbital_velocity', 'Circular orbital velocity',
         vars=[V('v', 'm/s', 'orbital velocity'), V('GM', 'm^3/s^2', 'standard gravitational parameter of the central body'),
               V('r', 'm', 'orbital radius from the centre of the body')],
         equations=['v = sqrt(GM/r)'], outputs=['v'],
         example={'GM': 3.986004418e14, 'r': 6.678e6},
         keywords=['orbit', 'circular velocity', 'LEO'],
         assumptions=['Circular orbit; r is measured from the body centre (add the body radius to the altitude). '
                      'GM Earth = 3.986004418e14 m³/s².'],
         refs=['v_circ = sqrt(GM/r)']),

    Calc('escape_velocity', 'Escape velocity',
         vars=[V('v_esc', 'm/s', 'escape velocity'), V('GM', 'm^3/s^2', 'standard gravitational parameter of the central body'),
               V('r', 'm', 'radius from the centre of the body')],
         equations=['v_esc = sqrt(2*GM/r)'], outputs=['v_esc'],
         example={'GM': 3.986004418e14, 'r': 6.378e6},
         keywords=['escape', 'parabolic', 'C3=0'],
         assumptions=['Minimum speed for a parabolic escape trajectory from radius r.'],
         refs=['v_esc = sqrt(2GM/r)']),

    Calc('orbital_period', 'Circular orbital period',
         vars=[V('T', 's', 'orbital period'), V('GM', 'm^3/s^2', 'standard gravitational parameter of the central body'),
               V('r', 'm', 'orbital radius from the centre of the body')],
         equations=['T = 2*pi*sqrt(r**3/GM)'], outputs=['T'],
         example={'GM': 3.986004418e14, 'r': 6.678e6},
         keywords=['orbit', 'period', 'kepler'],
         assumptions=["Kepler's third law for a circular orbit (r = semi-major axis a)."],
         refs=['T = 2π√(r³/GM)']),

    Calc('geopotential_altitude', 'Geometric altitude ↔ geopotential altitude (H = r0·z/(r0+z))',
         vars=[V('H', 'm', 'geopotential altitude'), V('z', 'm', 'geometric altitude'),
               V('r0', 'm', 'mean Earth radius (6356766 m)')],
         equations=['H = r0*z/(r0 + z)'], outputs=['H'],
         example={'z': 11000.0, 'r0': 6356766.0},
         keywords=['geopotential', 'geometric', 'altitude', 'ISA', 'atmosphere'],
         assumptions=['Converts geometric altitude z (the geometric distance above mean sea level) to '
                      'geopotential altitude H (the US Standard Atmosphere 1976 independent variable, which '
                      'accounts for the fall-off of gravity with height). H = r0·z/(r0+z) with '
                      'r0 = 6356766 m; the two differ by < 0.3 % below 86 km.'],
         refs=['Geopotential altitude H = r0·z/(r0 + z) (US Standard Atmosphere 1976 convention)']),

    Calc('standard_atmosphere', 'Standard atmosphere (ISA / US 1976) properties vs geometric altitude',
         vars=[V('z', 'm', 'geometric altitude above mean sea level'),
               V('r0', 'm', 'mean Earth radius (6356766 m)'),
               V('H', 'm', 'geopotential altitude (= r0·z/(r0+z))'),
               V('T', 'K', 'temperature'), V('P', 'Pa', 'pressure'), V('rho', 'kg/m^3', 'density', label='ρ'),
               V('a', 'm/s', 'speed of sound')],
         procedure=_atmosphere, outputs=['H', 'T', 'P', 'rho', 'a'],
         example={'z': 5000.0, 'r0': 6356766.0},
         keywords=['ISA', 'atmosphere', 'altitude', 'air density', 'geometric'],
         assumptions=['US Standard Atmosphere 1976 layers, valid to 84852 m geopotential (= 86 km geometric): '
                      'the input is GEOMETRIC altitude z, converted internally to geopotential '
                      'H = r0·z/(r0+z) (r0 = 6356766 m) before the lapse-rate and isothermal layer equations '
                      'give T, P; ρ = P/(R·T) with R = R*/M0 = 287.0531 J/(kg·K) (USSA 1976). Speed of sound a = sqrt(1.4·R·T). '
                      'Values are the ISA mid-latitude mean, not a weather forecast.'],
         refs=['US Standard Atmosphere 1976 layer equations (recalled base temperatures, lapse rates and '
               'geopotential-altitude convention)']),
]

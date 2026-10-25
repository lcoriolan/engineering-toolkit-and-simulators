# DESCRIPTION: Automotive engineering section: road load (rolling resistance + aerodynamic drag + grade),
#              gradeability, acceleration, top speed from an engine torque curve, gear ratios, tractive
#              effort from engine torque, engine power required, braking (with reaction time) and brake
#              force distribution from vehicle geometry, longitudinal weight transfer, tyre lateral force,
#              slip angles and understeer gradient, rollover threshold, ride frequency (spring + tyre in
#              series) and fuel consumption.
"""Automotive-engineering calculators.

Conventions:
- Longitudinal vehicle relations follow the standard road-load model: tractive effort at the wheels
  overcomes rolling resistance C_rr·N, aerodynamic drag ½ρ·Cd·A·v² and grade W·sinθ. Speed is m/s;
  engine / wheel angular speeds are rad/s (enter rpm as "3000 rpm"); ride frequency is cyclic (Hz).
- Tyre cornering stiffness (N/rad) and rolling-resistance coefficients are USER INPUTS (tyre maker
  data); no tyre or engine maps are embedded — the top-speed engine torque curve is a JSON input.
- Driveline efficiency η multiplies engine power; gear ratios are overall (transmission × final drive).
- Longitudinal acceleration is signed: positive = forward acceleration, negative = braking deceleration.
  Grade angle θ is signed (positive = uphill, negative = downhill).
- Aerodynamic drag/lift as generic forces live in fluid_dynamics.drag_lift; centripetal force in
  kinetics.centripetal_force — they are not duplicated here.

Relations follow Gillespie, *Fundamentals of Vehicle Dynamics*; Wong, *Theory of Ground Vehicles*;
Milliken & Milliken, *Race Car Vehicle Dynamics* — cited by relation name, recalled.
"""
import math

import numpy as np

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts, common

SECTION = section('automotive', 'Automotive engineering', order=420,
                  summary='Road load, gradeability, acceleration, top speed, gearing, braking, weight '
                          'transfer, cornering, understeer, rollover, ride frequency and fuel economy.')

PI = math.pi


# ---- procedures ------------------------------------------------------------------------------------
def _top_speed(values):
    """Top speed: largest speed where the best tractive force (over all gears whose engine speed is inside
    the torque curve) meets the road load, or the top-gear redline speed if force still exceeds it there.

    At each speed only gears with w_min <= v·i/r_w <= w_max contribute; the best gear is the one giving the
    largest tractive force. Below the first gear's idle speed no gear can pull (clutch-slip region), so the
    crossing is sought above that speed. The top speed is the first speed where the best force falls below
    road load; if the force is still above road load at the top-gear redline, the top speed is redline-limited.
    """
    curve = values['torque_curve']
    ratios = values['gear_ratios']
    if not isinstance(curve, list) or len(curve) < 2:
        raise MechError('torque_curve: give [[rpm, N*m], ...] with at least 2 points')
    if not isinstance(ratios, list) or not ratios:
        raise MechError('gear_ratios: give a non-empty list of overall (transmission x final drive) ratios')
    rows = []
    for r in curve:
        if not isinstance(r, (list, tuple)) or len(r) != 2:
            raise MechError('torque_curve rows must be [rpm, N*m]')
        rows.append((float(r[0]), float(r[1])))
    rows.sort(key=lambda t: t[0])
    rpm = np.array([t[0] for t in rows], dtype=float)
    tor = np.array([t[1] for t in rows], dtype=float)
    if rpm[0] <= 0 or np.any(np.diff(rpm) <= 0):
        raise MechError('torque_curve engine speeds must be positive and strictly increasing')
    ratios = [float(i) for i in ratios]
    if any(i <= 0 for i in ratios):
        raise MechError('gear_ratios must be positive')
    w_curve = rpm * (2 * PI / 60)          # rad/s; redline = top of the torque curve
    w_min, w_max = w_curve[0], w_curve[-1]
    rw, m, Cd, A = values['r_wheel'], values['m'], values['Cd'], values['A']
    Crr, rho, eta, g = values['C_rr'], values['rho'], values['eta_driveline'], values['g']
    F_roll = Crr * m * g
    i_top = min(ratios)                     # top gear = smallest overall ratio
    v_red_top = w_max * rw / i_top

    def road(v):
        return F_roll + 0.5 * rho * Cd * A * v * v

    def gear_force(v, ig):
        """Tractive force of one gear, or -inf when the engine speed is outside the torque curve."""
        we = v * ig / rw
        if we < w_min or we > w_max:
            return float('-inf')
        return eta * float(np.interp(we, w_curve, tor)) * ig / rw

    def best_force(v):
        return max(gear_force(v, ig) for ig in ratios)

    v = np.linspace(0.0, v_red_top, 1200)
    ex = np.array([best_force(x) - road(x) for x in v])
    moving = [k for k in range(len(v)) if ex[k] >= 0.0]
    if not moving:
        raise MechError('no solution: available tractive effort never exceeds the road load')
    start = moving[0]
    cross = None
    for k in range(start, len(v) - 1):
        if ex[k] >= 0.0 and ex[k + 1] < 0.0:
            cross = k
            break
    if cross is None:
        vmax, redline = v_red_top, True
    else:
        a, b = v[cross], v[cross + 1]
        for _ in range(80):                 # bisect the force balance at the crossing
            mid = 0.5 * (a + b)
            if best_force(mid) - road(mid) >= 0.0:
                a = mid
            else:
                b = mid
        vmax, redline = 0.5 * (a + b), False
    kbest = max(range(len(ratios)), key=lambda k: gear_force(vmax, ratios[k]))
    i_best = ratios[kbest]
    F_top = gear_force(vmax, i_best)
    out = {'v_max': vmax,
           'P_max_wheel': F_top * vmax,
           'gear_at_top': float(kbest + 1),    # 1-based gear number
           'omega_e_top': min(vmax * i_best / rw, w_max)}
    if redline:
        out['_warnings'] = ['top speed is at the engine redline in the top (longest) gear '
                            '(speed-limited rather than drag-limited)']
    return out


def _top_speed_chart(values):
    """Power available vs power required in the gear that actually reaches the top speed."""
    res = _top_speed(values)
    curve = sorted((float(r[0]), float(r[1])) for r in values['torque_curve'])
    ratios = [float(i) for i in values['gear_ratios']]
    k = int(round(res['gear_at_top'])) - 1
    k = max(0, min(k, len(ratios) - 1))
    i = ratios[k]
    rpm = np.array([t[0] for t in curve])
    tor = np.array([t[1] for t in curve])
    w_curve = rpm * (2 * PI / 60)
    w_red = w_curve[-1]
    rw, m, Cd, A = values['r_wheel'], values['m'], values['Cd'], values['A']
    Crr, rho, eta, g = values['C_rr'], values['rho'], values['eta_driveline'], values['g']
    v = np.linspace(0.0, w_red * rw / i, 200)
    we = np.minimum(v * i / rw, w_red)
    Pa = eta * np.interp(we, w_curve, tor) * we
    Pr = (Crr * m * g + 0.5 * rho * Cd * A * v * v) * v
    vmax = float(res['v_max'])
    we_top = min(vmax * i / rw, w_red)
    Pa_top = eta * float(np.interp(we_top, w_curve, tor)) * we_top
    return charts.chart('Top speed: power vs speed (gear %d)' % (k + 1), 'Speed [m/s]', 'Power [W]',
                        [charts.series('power available at wheels', v, Pa),
                         charts.series('power required (road load)', v, Pr)],
                        markers=[charts.marker(vmax, Pa_top, 'v_max')], vlines=[vmax])


CALCS = [
    Calc('rolling_resistance', 'Rolling resistance force (F = C_rr·N)',
         vars=[V('F_rr', 'N', 'rolling resistance force'), V('C_rr', '', 'rolling resistance coefficient'),
               V('N', 'N', 'normal load on the driven tyres')],
         equations=['F_rr = C_rr*N'], outputs=['F_rr'], example={'C_rr': 0.015, 'N': 15000.0},
         keywords=['tyre', 'road load', 'coefficient of rolling resistance'],
         assumptions=['C_rr is a tyre-road input (tyre maker or road-test data); it rises slightly with '
                      'speed and inflation pressure, treated constant here.'],
         refs=['Rolling resistance F = C_rr·N']),

    Calc('road_load', 'Tractive effort and road-load power (rolling + aero + grade)',
         vars=[V('F_t', 'N', 'tractive effort required at the wheels (negative downhill)', signed=True),
               V('P_req', 'W', 'road-load power required (F_t·v; negative when descending)', signed=True),
               V('C_rr', '', 'rolling resistance coefficient'), V('W', 'N', 'vehicle weight'),
               V('theta', 'rad', 'grade angle (0 = level; negative = downhill)', hi=PI / 2, signed=True, label='θ'),
               V('rho', 'kg/m^3', 'air density', label='ρ'), V('Cd', '', 'drag coefficient', hi=2.0),
               V('A', 'm^2', 'frontal area'), V('v', 'm/s', 'vehicle speed')],
         equations=['F_t = C_rr*W*cos(theta) + 0.5*rho*Cd*A*v**2 + W*sin(theta)', 'P_req = F_t*v'],
         outputs=['F_t', 'P_req'],
         example={'C_rr': 0.015, 'W': 15000.0, 'theta': math.radians(4), 'rho': 1.225, 'Cd': 0.32,
                  'A': 2.2, 'v': 25.0},
         keywords=['tractive effort', 'road load', 'grade climbing', 'downhill'],
         assumptions=['Steady speed on a constant grade; no headwind. θ is signed: positive = uphill, '
                      'negative = downhill (on a steep enough descent gravity overcomes drag + rolling '
                      'resistance, so F_t and P_req go negative — the vehicle needs braking, not drive). '
                      'F_t is the traction demand, not the tyre limit (that needs the available tractive '
                      'force, see tractive_effort / gradeability).'],
         refs=['Road-load equation F = C_rr·W·cosθ + ½ρCdAv² + W·sinθ']),

    Calc('gradeability', 'Maximum grade from available tractive effort',
         vars=[V('theta', 'rad', 'maximum grade angle', hi=PI / 2, label='θ'),
               V('grade_pct', '', 'grade as a percentage (100·tanθ)'),
               V('F_avail', 'N', 'available tractive effort (tyre/adhesion-limited)'),
               V('W', 'N', 'vehicle weight'), V('C_rr', '', 'rolling resistance coefficient')],
         equations=['theta = asin((F_avail/W)/sqrt(1 + C_rr**2)) - atan(C_rr)', 'grade_pct = 100*tan(theta)'],
         outputs=['theta', 'grade_pct'],
         example={'F_avail': 3000.0, 'W': 15000.0, 'C_rr': 0.015},
         keywords=['gradeability', 'hill climb', 'climb'],
         assumptions=['Low-speed climb (aerodynamic drag neglected). The equilibrium on a grade is '
                      'W·sinθ + C_rr·W·cosθ = F_avail, solved exactly as '
                      'θ = asin((F_avail/W)/√(1 + C_rr²)) − atan(C_rr) — consistent with the cosθ rolling '
                      'term in road_load. F_avail is the lesser of engine-traction and tyre-adhesion limit; '
                      'valid for F_avail/W ≤ √(1 + C_rr²).'],
         refs=['Equilibrium on a grade: W·sinθ + C_rr·W·cosθ = F_avail']),

    Calc('acceleration', 'Longitudinal acceleration from excess tractive force',
         vars=[V('a', 'm/s^2', 'acceleration', signed=True),
               V('F_t', 'N', 'tractive effort at the wheels'), V('F_rr', 'N', 'rolling resistance force'),
               V('F_d', 'N', 'aerodynamic drag force'), V('k_m', '', 'rotational mass factor (≥ 1)', lo=1.0),
               V('m', 'kg', 'vehicle mass')],
         equations=['a = (F_t - F_rr - F_d)/(k_m*m)'], outputs=['a'],
         example={'F_t': 4000.0, 'F_rr': 225.0, 'F_d': 388.0, 'k_m': 1.08, 'm': 1500.0},
         keywords=['acceleration', 'launch', 'performance'],
         assumptions=['Flat road, one value of resistance at the instant considered (get F_rr from '
                      'rolling_resistance, F_d from fluid_dynamics.drag_lift, or F_t directly from '
                      'road_load). k_m accounts for rotating inertia (≈ 1.04–1.10 in low gears).'],
         refs=["Newton's second law with effective (rotating) mass"]),

    Calc('top_speed', 'Top speed from an engine torque curve and gear ratios',
         vars=[V('torque_curve', '', 'full-throttle engine torque [[rpm, N*m], ...] ascending', kind='json'),
               V('gear_ratios', '', 'overall gear ratios (transmission x final drive), one per gear', kind='json'),
               V('r_wheel', 'm', 'driven-wheel rolling radius'), V('m', 'kg', 'vehicle mass'),
               V('Cd', '', 'drag coefficient', hi=2.0), V('A', 'm^2', 'frontal area'),
               V('C_rr', '', 'rolling resistance coefficient'), V('rho', 'kg/m^3', 'air density', label='ρ'),
               V('eta_driveline', '', 'driveline efficiency (≤ 1)', hi=1.0, label='η'),
               V('g', 'm/s^2', 'standard gravity'),
               V('v_max', 'm/s', 'top speed'), V('P_max_wheel', 'W', 'wheel power at top speed'),
               V('gear_at_top', '', 'gear (1-based) that reaches the top speed', integer=True),
               V('omega_e_top', 'rad/s', 'engine speed at top speed', label='ωe')],
         procedure=_top_speed, outputs=['v_max', 'P_max_wheel', 'gear_at_top', 'omega_e_top'],
         example={'torque_curve': [[1000, 200], [2500, 240], [4000, 250], [5500, 220], [6500, 190]],
                  'gear_ratios': [13.5, 8.2, 5.6, 4.1, 3.2], 'r_wheel': 0.31, 'm': 1500.0, 'Cd': 0.32,
                  'A': 2.2, 'C_rr': 0.015, 'rho': 1.225, 'eta_driveline': 0.88, 'g': common.g0()},
         plots=[{'id': 'power', 'title': 'Power vs speed (gear at top speed)', 'fn': _top_speed_chart}],
         keywords=['vmax', 'top speed', 'engine torque', 'redline'],
         assumptions=['At each speed the best tractive force is taken over every gear whose engine speed is '
                      'inside the torque curve; the top speed is the first speed where that best force falls '
                      'below the road load, or the top-gear (smallest-ratio) redline speed if force still '
                      'exceeds it there. The torque curve is interpolated linearly between points; below the '
                      'first gear\'s idle speed no gear can pull (the clutch slips), so the crossing is sought '
                      'above it. A redline-limited result is reported with a warning.'],
         refs=['Power available (η·T·ω) = power required ((C_rr·m·g + ½ρCdAv²)·v) at the top speed']),

    Calc('gear_ratio', 'Gear pair: torque and speed through a reduction',
         vars=[V('T_out', 'N*m', 'output torque'), V('omega_out', 'rad/s', 'output angular speed', label='ωout'),
               V('T_in', 'N*m', 'input torque'), V('omega_in', 'rad/s', 'input angular speed', label='ωin'),
               V('i', '', 'gear ratio (N_in/N_out)', lo=0.0), V('eta', '', 'gear efficiency (≤ 1)', hi=1.0, label='η')],
         equations=['T_out = T_in*i*eta', 'omega_out = omega_in/i'], outputs=['T_out', 'omega_out'],
         example={'T_in': 250.0, 'omega_in': 4000 * 2 * PI / 60, 'i': 3.5, 'eta': 0.96},
         keywords=['gear ratio', 'transmission', 'reduction', 'final drive'],
         assumptions=['Single mesh with efficiency η; i > 1 reduces speed and multiplies torque.'],
         refs=['Ideal gear power balance: T_out·ω_out = η·T_in·ω_in']),

    Calc('vehicle_speed', 'Vehicle speed from engine speed and overall gear ratio',
         vars=[V('v', 'm/s', 'vehicle speed'), V('omega_e', 'rad/s', 'engine angular speed', label='ωe'),
               V('r_wheel', 'm', 'driven-wheel rolling radius'), V('i_total', '', 'overall gear ratio (trans x final)')],
         equations=['v = omega_e*r_wheel/i_total'], outputs=['v'],
         example={'omega_e': 3000 * 2 * PI / 60, 'r_wheel': 0.31, 'i_total': 5.6},
         keywords=['speed in gear', 'rpm to speed'],
         assumptions=['No tyre slip (drive wheels roll without slipping).'],
         refs=['Rolling contact: v = ω_e·r_wheel/i_total']),

    Calc('tractive_effort', 'Tractive effort at the wheels from engine torque',
         vars=[V('F_t', 'N', 'tractive effort at the driven wheels'),
               V('T_e', 'N*m', 'engine torque'), V('i_g', '', 'transmission gear ratio', lo=0.0),
               V('i_d', '', 'final drive (axle) ratio', lo=0.0), V('eta', '', 'driveline efficiency (≤ 1)', hi=1.0, label='η'),
               V('r_wheel', 'm', 'driven-wheel rolling radius')],
         equations=['F_t = T_e*i_g*i_d*eta/r_wheel'], outputs=['F_t'],
         example={'T_e': 250.0, 'i_g': 3.5, 'i_d': 4.0, 'eta': 0.9, 'r_wheel': 0.31},
         keywords=['tractive effort', 'engine torque', 'drive force', 'transmission'],
         assumptions=['Overall ratio i_g·i_d from the engine to the wheels, with driveline efficiency η; '
                      'no tyre slip. This is the engine-side tractive effort — compare with road_load to '
                      'see whether the vehicle can accelerate at a given speed.'],
         refs=['Tractive effort F = T·i_g·i_d·η/r']),

    Calc('engine_power', 'Engine power required from wheel power',
         vars=[V('P_eng', 'W', 'engine (brake) power required'), V('P_wheel', 'W', 'power required at the wheels'),
               V('eta', '', 'driveline efficiency (≤ 1)', hi=1.0, label='η')],
         equations=['P_eng = P_wheel/eta'], outputs=['P_eng'],
         example={'P_wheel': 30000.0, 'eta': 0.88},
         keywords=['engine power', 'driveline efficiency', 'road load power'],
         assumptions=['The engine must deliver P_wheel/η to overcome a road-load power P_wheel at the '
                      'wheels (the driveline dissipates (1 − η)·P_eng). Get P_wheel from road_load.'],
         refs=['Driveline power balance: P_eng = P_wheel/η']),

    Calc('braking_distance', 'Braking distance with reaction time (flat road)',
         vars=[V('a_dec', 'm/s^2', 'braking deceleration magnitude'), V('d', 'm', 'total stopping distance'),
               V('mu', '', 'tyre-road friction coefficient (all wheels at the limit)', label='μ'),
               V('g', 'm/s^2', 'standard gravity'), V('v', 'm/s', 'initial speed'),
               V('t_r', 's', 'driver reaction time (distance travelled before braking)')],
         equations=['a_dec = mu*g', 'd = v*t_r + v**2/(2*a_dec)'], outputs=['a_dec', 'd'],
         example={'mu': 0.85, 'g': common.g0(), 'v': 30.0, 't_r': 1.0},
         keywords=['stopping distance', 'braking', 'deceleration', 'reaction time'],
         assumptions=['All wheels braked at the adhesion limit (μ); flat, dry-road single value. d = v·t_r '
                      '(reaction) + v²/(2μg) (braking). On a grade, replace μ·g by g·(μ·cosθ + sinθ) for '
                      'braking uphill.'],
         refs=['Constant-deceleration kinematics: d = v·t_r + v²/(2a) with a = μg']),

    Calc('weight_transfer', 'Longitudinal weight transfer and dynamic axle loads',
         vars=[V('delta_W', 'N', 'longitudinal load transfer (rearward + for acceleration)', signed=True, label='ΔW'),
               V('W_f_dyn', 'N', 'dynamic front-axle load', signed=True), V('W_r_dyn', 'N', 'dynamic rear-axle load', signed=True),
               V('W', 'N', 'vehicle weight'), V('l_f', 'm', 'front axle to centre of gravity'),
               V('l_r', 'm', 'rear axle to centre of gravity'), V('h', 'm', 'centre-of-gravity height'),
               V('a', 'm/s^2', 'longitudinal acceleration (positive forward, negative braking)', signed=True),
               V('g', 'm/s^2', 'standard gravity')],
         equations=['delta_W = W*a*h/(g*(l_f + l_r))',
                    'W_f_dyn = W*l_r/(l_f + l_r) - delta_W',
                    'W_r_dyn = W*l_f/(l_f + l_r) + delta_W'],
         outputs=['delta_W', 'W_f_dyn', 'W_r_dyn'],
         example={'W': 15000.0, 'l_f': 1.3, 'l_r': 1.5, 'h': 0.55, 'a': 3.0, 'g': common.g0()},
         keywords=['weight transfer', 'load transfer', 'cog height', 'pitch', 'wheel lift'],
         assumptions=['Rigid vehicle, quasi-static transfer, small pitch (CG height taken constant). A negative '
                      'dynamic axle load means that axle has lifted off — the vehicle is about to tip.'],
         refs=['Moment equilibrium about the axles: ΔW = m·a·h/L']),

    Calc('brake_distribution', 'Brake force distribution from vehicle geometry (ideal split)',
         vars=[V('a_g', '', 'braking deceleration in g (= μ at the adhesion limit)'),
               V('delta_W', 'N', 'longitudinal load transfer to the front axle', signed=True, label='ΔW'),
               V('W_f_dyn', 'N', 'dynamic front-axle load (braking)', signed=True),
               V('W_r_dyn', 'N', 'dynamic rear-axle load (braking)', signed=True),
               V('F_bf', 'N', 'front brake force'), V('F_br', 'N', 'rear brake force'),
               V('front_frac', '', 'front fraction of total brake force', hi=1.0),
               V('W', 'N', 'vehicle weight'), V('l_f', 'm', 'front axle to centre of gravity'),
               V('l_r', 'm', 'rear axle to centre of gravity'), V('h', 'm', 'centre-of-gravity height'),
               V('mu', '', 'tyre-road friction coefficient (deceleration in g at the limit)', label='μ')],
         equations=['a_g = mu',
                    'delta_W = W*a_g*h/(l_f + l_r)',
                    'W_f_dyn = W*l_r/(l_f + l_r) + delta_W',
                    'W_r_dyn = W*l_f/(l_f + l_r) - delta_W',
                    'F_bf = mu*W_f_dyn',
                    'F_br = mu*W_r_dyn',
                    'front_frac = (l_r + mu*h)/(l_f + l_r)'],
         outputs=['a_g', 'delta_W', 'W_f_dyn', 'W_r_dyn', 'F_bf', 'F_br', 'front_frac'],
         example={'W': 12000.0, 'l_f': 1.2, 'l_r': 1.4, 'h': 0.5, 'mu': 0.85},
         keywords=['brake balance', 'brake bias', 'front rear split'],
         assumptions=['Ideal (both axles at lockup simultaneously): the braking deceleration in g equals the '
                      'friction coefficient (a_g = μ), the front axle gains ΔW = W·a_g·h/L, and each axle '
                      'carries μ × its dynamic load. The front fraction (l_r + μ·h)/L is the brake bias that '
                      'locks both axles together. Real proportioning valves bias toward the front and fall '
                      'below this ideal line.'],
         refs=['Brake force proportional to dynamic normal load at each axle; front fraction (l_r + μ·h)/L']),

    Calc('tyre_lateral_force', 'Tyre lateral force from cornering stiffness (linear model)',
         vars=[V('F_y', 'N', 'tyre lateral (cornering) force', signed=True),
               V('C_alpha', 'N/rad', 'cornering stiffness', label='Cα'),
               V('alpha', 'rad', 'slip angle', signed=True, label='α')],
         equations=['F_y = C_alpha*alpha'], outputs=['F_y'],
         example={'C_alpha': 60000.0, 'alpha': 0.05},
         keywords=['cornering', 'lateral force', 'slip angle', 'cornering stiffness', 'tyre'],
         assumptions=['Linear (small-slip-angle) tyre model: F_y = C_α·α. C_α is a tyre input (N/rad); the '
                      'model saturates at large slip (α ≳ 5–8°, tyre-specific) where F_y ≈ μ·F_z instead. For '
                      'the pure centripetal force of a vehicle in a turn see kinetics.centripetal_force.'],
         refs=['Linear tyre model F_y = C_α·α']),

    Calc('understeer_gradient', 'Slip angles and understeer gradient (bicycle model)',
         vars=[V('alpha_f', 'rad', 'front axle slip angle', label='αf'),
               V('alpha_r', 'rad', 'rear axle slip angle', label='αr'),
               V('K_us', 'rad', 'understeer gradient (positive = understeer)', signed=True, label='Kus'),
               V('W_f', 'N', 'front axle static load'), V('W_r', 'N', 'rear axle static load'),
               V('C_af', 'N/rad', 'front axle cornering stiffness', label='Cαf'),
               V('C_ar', 'N/rad', 'rear axle cornering stiffness', label='Cαr'),
               V('a_y', 'm/s^2', 'lateral acceleration'), V('g', 'm/s^2', 'standard gravity')],
         equations=['alpha_f = W_f*a_y/(g*C_af)', 'alpha_r = W_r*a_y/(g*C_ar)', 'K_us = W_f/C_af - W_r/C_ar'],
         outputs=['alpha_f', 'alpha_r', 'K_us'],
         example={'W_f': 7800.0, 'W_r': 7200.0, 'C_af': 60000.0, 'C_ar': 60000.0, 'a_y': 4.0, 'g': common.g0()},
         keywords=['understeer', 'oversteer', 'slip angle', 'cornering stiffness', 'steering'],
         assumptions=['Linear (small-slip-angle) bicycle model; cornering stiffness C_af, C_ar are axle '
                      'sums and tyre inputs (N/rad). K_us is the slip-angle difference at a_y = g, i.e. '
                      'radians per g: positive = understeer, negative = oversteer.'],
         refs=['Bicycle-model slip angles α = W·a_y/(g·C_α); understeer gradient K_us = W_f/C_αf − W_r/C_αr']),

    Calc('rollover_threshold', 'Static rollover (tip-over) threshold',
         vars=[V('a_y_roll', 'm/s^2', 'lateral acceleration at the rollover threshold'),
               V('a_y_g', '', 'rollover threshold in g'),
               V('track', 'm', 'track width'), V('h', 'm', 'centre-of-gravity height'),
               V('g', 'm/s^2', 'standard gravity')],
         equations=['a_y_roll = g*track/(2*h)', 'a_y_g = track/(2*h)'], outputs=['a_y_roll', 'a_y_g'],
         example={'track': 1.5, 'h': 0.55, 'g': common.g0()},
         keywords=['rollover', 'tip over', 'stability', 'ssf'],
         assumptions=['Rigid vehicle, no suspension compliance (the quasi-static tip-over limit). The static '
                      'stability factor is track/(2h); suspension roll and tyre forces lower the real threshold.'],
         refs=['Moment balance about the outer tyre contact: a_y/g = track/(2h)']),

    Calc('suspension_frequency', 'Ride (bounce) natural frequency with spring and tyre in series',
         vars=[V('f_n', 'Hz', 'ride natural frequency', label='fn'),
               V('k_ride', 'N/m', 'ride rate (spring and tyre in series)'),
               V('k_s', 'N/m', 'spring rate at the wheel (spring rate x motion-ratio²)'),
               V('k_t', 'N/m', 'tyre vertical (radial) stiffness'),
               V('m', 'kg', 'sprung mass carried by the axle')],
         equations=['k_ride = k_s*k_t/(k_s + k_t)', 'f_n = sqrt(k_ride/m)/(2*pi)'],
         outputs=['k_ride', 'f_n'],
         example={'k_s': 25000.0, 'k_t': 200000.0, 'm': 350.0},
         keywords=['ride', 'bounce', 'natural frequency', 'suspension', 'spring', 'tyre stiffness'],
         assumptions=['Single corner quarter-car, undamped. Spring and tyre are springs in series, '
                      'k_ride = k_s·k_t/(k_s + k_t); k_s is the wheel rate (spring rate × motion-ratio² for a '
                      'spring not mounted on the wheel centreline).'],
         refs=['Series springs 1/k = 1/k_s + 1/k_t; f_n = (1/2π)·√(k/m)']),

    Calc('fuel_consumption', 'Fuel consumption rate and fuel economy',
         vars=[V('m_dot_f', 'kg/s', 'fuel mass flow rate'), V('V_dot', 'm^3/s', 'fuel volume flow rate'),
               V('economy', 'm/m^3', 'fuel economy (distance per fuel volume)'),
               V('bsfc', 'kg/J', 'brake-specific fuel consumption (engine)'),
               V('P', 'W', 'engine brake power'), V('rho_f', 'kg/m^3', 'fuel density', label='ρf'),
               V('v', 'm/s', 'vehicle speed')],
         equations=['m_dot_f = bsfc*P', 'V_dot = m_dot_f/rho_f', 'economy = v/V_dot'],
         outputs=['m_dot_f', 'V_dot', 'economy'],
         example={'bsfc': 250.0 / 3.6e9, 'P': 40e3, 'rho_f': 745.0, 'v': 25.0},
         keywords=['fuel economy', 'bsfc', 'consumption', 'mpg'],
         assumptions=['Engine bsfc is a user input (≈ 0.25–0.28 kg/kWh typical SI range); the example uses '
                      '250 g/kWh. Steady speed; economy = v/V_dot in m per m³ (multiply by 1e-3 for km/L).'],
         refs=['Fuel mass rate = bsfc·P; economy = distance / fuel volume']),
]

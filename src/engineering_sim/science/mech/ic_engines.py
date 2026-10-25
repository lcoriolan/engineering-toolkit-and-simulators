# DESCRIPTION: Internal combustion engines section: displacement and compression ratio, indicated/brake/
#              friction power, mean effective pressures, thermal/mechanical/volumetric efficiencies, BSFC,
#              piston speed and crank kinematics, air-fuel/equivalence ratio and heat balance.
"""Internal-combustion engine calculators.

Conventions: engine speed N is a *cyclic* rate in rev/s (declared Hz — enter '1/s', '1/min' or 'Hz';
'rpm' is refused because pint treats it as angular). The input ``n_rev`` is the engine cycle as a text
choice ('two-stroke' -> 1 crankshaft revolution per power stroke, 'four-stroke' -> 2). Brake power from
torque and speed is provided by materials_mech.shaft_power; slider-crank kinematics by
mechanisms.slider_crank (both referenced rather than duplicated here).

Reference relations are the standard ones in internal-combustion-engine texts (Heywood, *Internal
Combustion Engine Fundamentals*; Pulkrabek, *Engineering Fundamentals of the Internal Combustion
Engine*). Cited by relation name, recalled rather than re-read, so no page or equation numbers are given.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('ic_engines', 'Internal combustion engines', order=290,
                  summary='Displacement, compression ratio, indicated/brake/friction power, MEP, '
                          'thermal/mechanical/volumetric efficiency, BSFC, piston speed, crank '
                          'kinematics, air-fuel ratio and heat balance.')

PI = math.pi


def _speed(desc="engine speed in rev/s"):
    return V('N', 'Hz', desc)


def _n_rev(values):
    """Revolutions per power stroke from the text choice: 1 (two-stroke) or 2 (four-stroke)."""
    n_rev = values['n_rev']
    if n_rev == 'two-stroke':
        return 1.0
    if n_rev == 'four-stroke':
        return 2.0
    raise MechError('n_rev must be "two-stroke" or "four-stroke"')


def _indicated_power(values):
    """Indicated power IP = MEP · V_d · N / n_rev (V_d = total engine displacement)."""
    return {'IP': values['MEP'] * values['V_d'] * values['N'] / _n_rev(values)}


def _brake_mep(values):
    """Brake mean effective pressure BMEP = 2π n_rev T / V_d (V_d = total engine displacement)."""
    return {'BMEP': 2.0 * PI * _n_rev(values) * values['T'] / values['V_d']}


def _volumetric_efficiency(values):
    """Volumetric efficiency η_v = ṁ_a / (ρ_a V_d N / n_rev) (V_d = total engine displacement)."""
    n = _n_rev(values)
    return {'eta_v': values['m_dot_a'] / (values['rho_a'] * values['V_d'] * values['N'] / n)}


CALCS = [
    # ---- geometry ---------------------------------------------------------------------------------
    Calc('displacement_volume', 'Engine displacement (swept) volume',
         vars=[V('V_d', 'm^3', 'swept (displacement) volume per cylinder'),
               V('B', 'm', 'bore (cylinder) diameter'), V('S', 'm', 'stroke'),
               V('N_cyl', '', 'number of cylinders (positive integer)', integer=True, lo=1.0),
               V('V_total', 'm^3', 'total engine displacement')],
         equations=['V_d = pi/4*B**2*S', 'V_total = V_d*N_cyl'], outputs=['V_d', 'V_total'],
         example={'B': 0.086, 'S': 0.086, 'N_cyl': 4.0},
         keywords=['displacement', 'bore', 'stroke', 'swept volume', 'engine size'],
         assumptions=['V_d = (π/4) B² S is the volume swept by one piston; total = V_d × N_cyl.'],
         refs=['Cylinder swept-volume definition']),

    Calc('compression_ratio', 'Engine compression ratio',
         vars=[V('r', '', 'compression ratio', lo=1.0),
               V('V_d', 'm^3', 'swept (displacement) volume'), V('V_c', 'm^3', 'clearance volume (at TDC)')],
         equations=['r = 1 + V_d/V_c'], outputs=['r'],
         example={'V_d': 5.0e-4, 'V_c': 6.25e-5},
         keywords=['compression ratio', 'clearance volume', 'tdc', 'bdc'],
         assumptions=['r = (V_d + V_c)/V_c = 1 + V_d/V_c, with V_c the combustion-chamber volume at TDC.'],
         refs=['Compression-ratio definition']),

    # ---- power and mean effective pressure ---------------------------------------------------------
    Calc('engine_power', 'Indicated, brake and friction power',
         vars=[V('IP', 'W', 'indicated power'), V('BP', 'W', 'brake power'), V('FP', 'W', 'friction power')],
         equations=['IP = BP + FP'], outputs=['IP'],
         example={'BP': 75e3, 'FP': 15e3},
         keywords=['indicated power', 'brake power', 'friction power', 'friction loss'],
         assumptions=['Friction (and pumping) power is the difference between the cylinder indicated power '
                      'and the brake power delivered at the shaft: IP = BP + FP.'],
         refs=['IP = BP + FP power balance']),

    # Brake power from torque and speed is provided by materials_mech.shaft_power (not duplicated here).

    Calc('mep', 'Mean effective pressure (work per cycle / displacement)',
         vars=[V('mep', 'Pa', 'mean effective pressure'),
               V('W_cycle', 'J', 'net work per cycle'), V('V_d', 'm^3', 'swept (displacement) volume')],
         equations=['mep = W_cycle/V_d'], outputs=['mep'],
         example={'W_cycle': 1.2e3, 'V_d': 2.0e-3},
         keywords=['mep', 'mean effective pressure', 'work per cycle'],
         assumptions=['MEP is the constant pressure that would do the same net work over one cycle: '
                      'MEP = W_cycle/V_d.'],
         refs=['Mean effective pressure definition']),

    Calc('indicated_power', 'Indicated power from MEP and engine speed',
         vars=[V('IP', 'W', 'indicated power'), V('MEP', 'Pa', 'indicated mean effective pressure'),
               V('V_d', 'm^3', 'total engine displacement (all cylinders)'), _speed(),
               V('n_rev', '', 'engine cycle (two-stroke = 1 rev/power stroke, four-stroke = 2)',
                 kind='text', choices=['two-stroke', 'four-stroke'])],
         procedure=_indicated_power, outputs=['IP'],
         example={'MEP': 800e3, 'V_d': 2.0e-3, 'N': 50.0, 'n_rev': 'four-stroke'},
         keywords=['indicated power', 'mep', 'engine speed', 'four stroke', 'two stroke'],
         assumptions=['N in rev/s (3000 rpm = 50 rev/s); cycles per second are N/n_rev. V_d is the TOTAL '
                      'displacement of all cylinders (multiply a per-cylinder swept volume by the cylinder count).'],
         refs=['IP = MEP · V_d · N / n_rev']),

    Calc('brake_mep', 'Brake mean effective pressure from torque',
         vars=[V('BMEP', 'Pa', 'brake mean effective pressure'), V('T', 'N*m', 'brake torque'),
               V('V_d', 'm^3', 'total engine displacement (all cylinders)'),
               V('n_rev', '', 'engine cycle (two-stroke = 1 rev/power stroke, four-stroke = 2)',
                 kind='text', choices=['two-stroke', 'four-stroke'])],
         procedure=_brake_mep, outputs=['BMEP'],
         example={'T': 250.0, 'V_d': 2.0e-3, 'n_rev': 'four-stroke'},
         keywords=['bmep', 'brake mean effective pressure', 'torque'],
         assumptions=['BMEP = BP/(V_d N/n_rev) = 2π n_rev T / V_d, so a four-stroke gives 4π T/V_d and a '
                      'two-stroke 2π T/V_d. V_d is the TOTAL displacement of all cylinders.'],
         refs=['BMEP from torque (recalled)']),

    # ---- efficiencies and consumption --------------------------------------------------------------
    Calc('brake_thermal_efficiency', 'Brake thermal efficiency',
         vars=[V('eta_bth', '', 'brake thermal efficiency', hi=1.0, label='ηbth'),
               V('BP', 'W', 'brake power'), V('m_dot_f', 'kg/s', 'fuel mass flow rate'),
               V('LHV', 'J/kg', 'fuel lower heating value')],
         equations=['eta_bth = BP/(m_dot_f*LHV)'], outputs=['eta_bth'],
         example={'BP': 75e3, 'm_dot_f': 0.005, 'LHV': 44e6},
         keywords=['brake thermal efficiency', 'fuel energy', 'lhv'],
         assumptions=['η_bth = brake power / fuel-energy rate. LHV is the lower heating value (water vapour '
                      'in the exhaust); use HHV for a gross figure.'],
         refs=['η_bth = BP/(ṁ_f LHV)']),

    Calc('mechanical_efficiency', 'Mechanical efficiency',
         vars=[V('eta_m', '', 'mechanical efficiency', hi=1.0, label='ηm'),
               V('BP', 'W', 'brake power'), V('IP', 'W', 'indicated power')],
         equations=['eta_m = BP/IP'], outputs=['eta_m'],
         example={'BP': 75e3, 'IP': 90e3},
         keywords=['mechanical efficiency', 'friction'],
         assumptions=['η_m = BP/IP = 1 − FP/IP; the share of indicated power delivered at the shaft.'],
         refs=['η_m = BP/IP']),

    Calc('volumetric_efficiency', 'Volumetric efficiency',
         vars=[V('eta_v', '', 'volumetric efficiency', hi=1.0, label='ηv'),
               V('m_dot_a', 'kg/s', 'actual air mass flow rate'), V('rho_a', 'kg/m^3', 'intake air density'),
               V('V_d', 'm^3', 'total engine displacement (all cylinders)'), _speed(),
               V('n_rev', '', 'engine cycle (two-stroke = 1 rev/power stroke, four-stroke = 2)',
                 kind='text', choices=['two-stroke', 'four-stroke'])],
         procedure=_volumetric_efficiency, outputs=['eta_v'],
         example={'m_dot_a': 0.05, 'rho_a': 1.18, 'V_d': 2.0e-3, 'N': 50.0, 'n_rev': 'four-stroke'},
         keywords=['volumetric efficiency', 'air flow', 'induction'],
         assumptions=['η_v compares the actual inducted air to the ideal charge ρ_a V_d N/n_rev the swept '
                      'volume could draw at intake conditions. V_d is the TOTAL displacement of all cylinders.'],
         refs=['η_v = ṁ_a / (ρ_a V_d N / n_rev)']),

    Calc('bsfc', 'Brake specific fuel consumption',
         vars=[V('bsfc', 'kg/J', 'brake specific fuel consumption'),
               V('m_dot_f', 'kg/s', 'fuel mass flow rate'), V('BP', 'W', 'brake power')],
         equations=['bsfc = m_dot_f/BP'], outputs=['bsfc'],
         example={'m_dot_f': 0.005, 'BP': 75e3},
         keywords=['bsfc', 'specific fuel consumption', 'fuel economy'],
         assumptions=['BSFC = ṁ_f/BP in kg per joule of shaft work (multiply by 3.6e9 for g/kWh).'],
         refs=['BSFC = ṁ_f/BP']),

    # ---- piston / crank kinematics -------------------------------------------------------------------
    Calc('piston_speed', 'Mean piston speed',
         vars=[V('S_p', 'm/s', 'mean piston speed'), V('S', 'm', 'stroke'), _speed()],
         equations=['S_p = 2*S*N'], outputs=['S_p'],
         example={'S': 0.086, 'N': 50.0},
         keywords=['piston speed', 'mean piston speed', 'stroke'],
         assumptions=['Mean piston speed = 2 × stroke × revolutions per second (N in rev/s).'],
         refs=['Mean piston speed S_p = 2 S N']),

    # Piston position/velocity/acceleration (slider-crank) is provided by mechanisms.slider_crank.

    # ---- mixture and heat balance --------------------------------------------------------------------
    Calc('air_fuel_ratio', 'Air-fuel ratio',
         vars=[V('AF', '', 'air-fuel ratio (mass)'), V('m_air', 'kg', 'air mass'), V('m_fuel', 'kg', 'fuel mass')],
         equations=['AF = m_air/m_fuel'], outputs=['AF'],
         example={'m_air': 14.7, 'm_fuel': 1.0},
         keywords=['air fuel ratio', 'afr', 'mixture'],
         assumptions=['Mass basis. The fuel-air ratio is the reciprocal, FA = 1/AF.'],
         refs=['AF = m_air/m_fuel']),

    Calc('equivalence_ratio', 'Equivalence ratio and excess-air ratio',
         vars=[V('phi', '', 'equivalence ratio AF_stoich/AF_actual', label='φ'),
               V('lam', '', 'excess-air (lambda) ratio AF_actual/AF_stoich', label='λ'),
               V('AF_stoich', '', 'stoichiometric air-fuel ratio'), V('AF_actual', '', 'actual air-fuel ratio')],
         equations=['phi = AF_stoich/AF_actual', 'lam = AF_actual/AF_stoich'],
         outputs=['phi', 'lam'], example={'AF_stoich': 14.7, 'AF_actual': 14.7},
         keywords=['equivalence ratio', 'lambda', 'stoichiometric', 'rich', 'lean'],
         assumptions=['φ = 1 (λ = 1) is stoichiometric; φ > 1 (λ < 1) is fuel-rich; φ < 1 (λ > 1) is fuel-lean. '
                      'λ = 1/φ.'],
         refs=['Equivalence ratio φ = AF_st/AF; excess-air ratio λ = AF/AF_st']),

    Calc('heat_balance', 'Engine heat (energy) balance',
         vars=[V('Q_fuel', 'W', 'fuel-energy input rate'),
               V('BP', 'W', 'brake power'), V('Q_coolant', 'W', 'heat rejected to the coolant'),
               V('Q_exhaust', 'W', 'heat rejected in the exhaust'), V('Q_misc', 'W', 'unaccounted and other losses')],
         equations=['Q_fuel = BP + Q_coolant + Q_exhaust + Q_misc'], outputs=['Q_fuel'],
         example={'BP': 75e3, 'Q_coolant': 90e3, 'Q_exhaust': 80e3, 'Q_misc': 15e3},
         keywords=['heat balance', 'energy balance', 'coolant', 'exhaust', 'losses'],
         assumptions=['First-law split of the fuel energy rate into brake power, coolant and exhaust heat, '
                      'and the residual (radiation, oil, unaccounted) losses.'],
         refs=['Engine energy balance']),
]

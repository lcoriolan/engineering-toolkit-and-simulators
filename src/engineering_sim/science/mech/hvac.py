# DESCRIPTION: HVAC section: real-gas moist-air psychrometrics (humidity ratio, relative humidity, dew
#              point, dry/wet-bulb, enthalpy, specific volume), moist-air processes (sensible, adiabatic
#              humidification, cooling/dehumidification) and mixing, cooling-tower range/approach/makeup,
#              space heating/cooling loads, sensible heat ratio and apparatus dew point, ventilation and
#              air changes, duct sizing and fan power, and hydronic heating/cooling capacity. Duct friction/
#              fitting losses are fluid_dynamics.head_loss (Δp = ρg·h_L); COP is thermo.cop; airflow is
#              fluid_dynamics.continuity.
"""HVAC (psychrometrics and air conditioning) calculators.

Conventions: moist-air properties are evaluated with CoolProp HAPropsSI (the real-gas ASHRAE RP-1485
model) through ``common.humid_air`` — NOT ideal-gas psychrometric approximations. Temperatures are
absolute (K); humidity ratio W is kg water vapour per kg dry air; enthalpy H and specific volume V are
per kg dry air. Mass flows of dry air are m_da (kg/s). Enthalpy is referenced to 0 at 0 °C (dry air and
liquid water), so condensate enthalpy is c_pw·(T − 273.15 K).

Relations are standard HVAC/psychrometrics ones (ASHRAE Handbook — Fundamentals; Incropera; McQuiston,
Parker & Spitler, *Heating, Ventilating and Air Conditioning*). Cited by relation name, recalled rather
than re-read. Chart coefficients (SHR slopes, solar heat gain factors, CLTD, duct K values) are user
inputs whose ``desc`` names the source table.

Adjacent sections cover the duplicated relations: airflow/continuity and duct friction/minor losses live
in ``fluid_dynamics`` (head_loss, continuity), and coefficient of performance in ``thermo`` (cop).
"""
from scipy import optimize

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts
from engineering_sim.science.mech import common

SECTION = section('hvac', 'Psychrometrics and HVAC', order=310,
                  summary='Moist-air psychrometrics, air-conditioning processes and mixing, cooling '
                          'towers, space loads, ventilation, ducts, fans, and chiller/boiler capacity.')

# Standard constants used by the psychrometric process procedures (recalled).
CP_AIR = 1005.0        # dry-air specific heat, J/(kg·K)
HFG_WATER = 2.501e6     # latent heat of vaporisation of water, J/kg (used for SHR/ADP slope)
CP_WATER = 4186.0       # liquid-water specific heat, J/(kg·K) (condensate enthalpy)


def _humid_air_state(values):
    """Moist-air state from dry-bulb T, relative humidity and total pressure (HAPropsSI)."""
    T, RH, P = values['T'], values['RH'], values['P']
    if RH <= 0.0:
        raise MechError('relative humidity must be positive: the dew point is undefined at RH = 0 '
                        '(dry air has no condensation temperature)')
    return {'W': common.humid_air('W', T=T, R=RH, P=P),
            'H': common.humid_air('H', T=T, R=RH, P=P),
            'V': common.humid_air('V', T=T, R=RH, P=P),
            'T_wb': common.humid_air('B', T=T, R=RH, P=P),
            'T_dp': common.humid_air('D', T=T, R=RH, P=P)}


def _psych_chart(values):
    """Psychrometric chart spec: T vs W with constant-RH lines at the calculator's total pressure."""
    P = values['P']
    T0 = values['T']
    series_list = []
    for rh in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
        Ts = [T0 - 20.0 + 40.0 * i / 40 for i in range(41)]
        Ws = []
        for T in Ts:
            try:
                Ws.append(common.humid_air('W', T=T, R=rh, P=P))
            except MechError:
                Ws.append(float('nan'))
        series_list.append(charts.series(f'RH = {rh:.0%}', Ts, Ws))
    return charts.chart('Psychrometric chart (T vs W)', 'T [K]', 'W [kg/kg dry air]', series_list,
                        markers=[charts.marker(values['T'], values['W'], 'state')])


def _moist_air_mixing(values):
    """Adiabatic mixing of two moist-air streams by humidity-ratio and enthalpy balance."""
    m1, T1, RH1 = values['m1'], values['T1'], values['RH1']
    m2, T2, RH2 = values['m2'], values['T2'], values['RH2']
    P = values['P']
    if m1 <= 0 or m2 <= 0:
        raise MechError('stream mass flows must be positive')
    W1 = common.humid_air('W', T=T1, R=RH1, P=P)
    H1 = common.humid_air('H', T=T1, R=RH1, P=P)
    W2 = common.humid_air('W', T=T2, R=RH2, P=P)
    H2 = common.humid_air('H', T=T2, R=RH2, P=P)
    W3 = (m1 * W1 + m2 * W2) / (m1 + m2)
    H3 = (m1 * H1 + m2 * H2) / (m1 + m2)
    T3 = common.humid_air('T', W=W3, H=H3, P=P)
    RH3 = common.humid_air('R', W=W3, H=H3, P=P)
    return {'W3': W3, 'H3': H3, 'T3': T3, 'RH3': RH3}


def _sensible_process(values):
    """Sensible heating/cooling at constant humidity ratio: Q = m_da (H2 − H1)."""
    m_da, T1, RH1 = values['m_da'], values['T1'], values['RH1']
    T2, P = values['T2'], values['P']
    W = common.humid_air('W', T=T1, R=RH1, P=P)
    H1 = common.humid_air('H', T=T1, R=RH1, P=P)
    H2 = common.humid_air('H', T=T2, W=W, P=P)
    try:
        RH2 = common.humid_air('R', T=T2, W=W, P=P)
    except MechError:
        raise MechError(f'outlet state at T2 = {T2:.1f} K is supersaturated (cooled below the dew point): '
                        'moisture would condense — use cooling_dehumidification to remove it')
    if RH2 > 1.0:
        raise MechError(f'outlet state is supersaturated (RH = {RH2:.3f} at T2 = {T2:.1f} K): cooling to '
                        'below the dew point condenses moisture — use cooling_dehumidification instead')
    return {'Q_dot': m_da * (H2 - H1), 'W': W, 'RH2': RH2, 'H1': H1, 'H2': H2}


def _adiabatic_humidification(values):
    """Adiabatic (evaporative) humidification: constant enthalpy, water sprayed in.

    Outlet humidity ratio W2 = W1 + m_w/m_da at H2 = H1; the outlet temperature follows the wet-bulb
    (constant-enthalpy) line toward saturation.
    """
    m_da, T1, RH1 = values['m_da'], values['T1'], values['RH1']
    m_w, P = values['m_w'], values['P']
    if m_da <= 0 or m_w <= 0:
        raise MechError('m_da and m_w must be positive')
    W1 = common.humid_air('W', T=T1, R=RH1, P=P)
    H = common.humid_air('H', T=T1, R=RH1, P=P)
    W2 = W1 + m_w / m_da
    try:
        T2 = common.humid_air('T', W=W2, H=H, P=P)
        RH2 = common.humid_air('R', W=W2, H=H, P=P)
    except MechError:
        raise MechError(f'outlet humidity ratio W2 = {W2:.5f} exceeds saturation at the constant-enthalpy '
                        'state: the sprayed water cannot all be absorbed; reduce m_w or use '
                        'cooling_dehumidification for moisture removal')
    if RH2 > 1.0:
        raise MechError(f'outlet state is supersaturated (RH = {RH2:.3f}): reduce the water spray rate m_w '
                        'so W2 stays below saturation')
    return {'T2': T2, 'W1': W1, 'W2': W2, 'RH2': RH2, 'H': H}


def _cooling_dehumidification(values):
    """Cooling and dehumidification: air leaves saturated at T2 (below the inlet dew point).

    Condensate m_cond = m_da (W1 − W2) leaves at T2; coil load
    Q = m_da (H1 − H2) − m_cond c_pw (T2 − 273.15 K).
    """
    m_da, T1, RH1 = values['m_da'], values['T1'], values['RH1']
    T2, P = values['T2'], values['P']
    W1 = common.humid_air('W', T=T1, R=RH1, P=P)
    H1 = common.humid_air('H', T=T1, R=RH1, P=P)
    W2 = common.humid_air('W', T=T2, R=1.0, P=P)
    if W2 >= W1:
        raise MechError('T2 is not below the inlet dew point (no condensation)')
    H2 = common.humid_air('H', T=T2, R=1.0, P=P)
    m_cond = m_da * (W1 - W2)
    Q = m_da * (H1 - H2) - m_cond * CP_WATER * (T2 - 273.15)
    return {'Q_dot': Q, 'm_cond': m_cond, 'W1': W1, 'W2': W2, 'RH2': 1.0, 'H1': H1, 'H2': H2}


def _apparatus_dew_point(values):
    """Apparatus dew point (ADP) of a cooling coil from the room condition and sensible heat ratio.

    The room condition line through (T_room, W_room) has slope dW/dT = c_p (1 − SHR)/(SHR h_fg). The ADP
    is where this line meets the saturation curve (RH = 1), found by scanning down from the room dew point.
    """
    T_room, RH_room = values['T_room'], values['RH_room']
    P, SHR = values['P'], values['SHR']
    if not (0.0 < SHR < 1.0):
        raise MechError('SHR must be strictly between 0 and 1')
    W_room = common.humid_air('W', T=T_room, R=RH_room, P=P)
    T_dp = common.humid_air('D', T=T_room, R=RH_room, P=P)
    slope = CP_AIR * (1.0 - SHR) / (SHR * HFG_WATER)

    def f(T):
        try:
            W_sat = common.humid_air('W', T=T, R=1.0, P=P)
        except MechError:
            return None
        return (W_room + slope * (T - T_room)) - W_sat

    fhi = f(T_dp)
    if fhi is None:
        raise MechError('cannot evaluate saturation at the room dew point')
    if abs(fhi) < 1e-9:                    # SHR = 1 limit: ADP equals the dew point
        T_adp = T_dp
    else:
        lo, found = T_dp, False
        for _ in range(400):
            lo -= 0.25
            flo = f(lo)
            if flo is None:
                continue
            if flo > 0.0 and fhi < 0.0:
                found = True
                break
        if not found:
            raise MechError('apparatus dew point not found (check SHR and room state)')
        T_adp = optimize.brentq(lambda T: f(T), lo, T_dp)
    W_adp = common.humid_air('W', T=T_adp, R=1.0, P=P)
    return {'T_adp': T_adp, 'W_adp': W_adp}


CALCS = [
    # ---- psychrometric state and chart --------------------------------------------------------------
    Calc('humid_air_state', 'Moist-air properties at a state (real-gas HAPropsSI)',
         vars=[V('T', 'K', 'dry-bulb temperature', lo=240.0, hi=330.0),
               V('RH', '', 'relative humidity (0..1)', lo=0.0, hi=1.0),
               V('P', 'Pa', 'total (barometric) pressure', lo=50e3, hi=120e3),
               V('W', '', 'humidity ratio (kg vapour / kg dry air)'),
               V('H', 'J/kg', 'specific enthalpy per kg dry air'),
               V('V', 'm^3/kg', 'specific volume per kg dry air'),
               V('T_wb', 'K', 'wet-bulb temperature'),
               V('T_dp', 'K', 'dew-point temperature')],
         procedure=_humid_air_state, outputs=['W', 'H', 'V', 'T_wb', 'T_dp'],
         example={'T': 300.0, 'RH': 0.5, 'P': 101325.0},
         plots=[{'id': 'psych', 'title': 'Psychrometric chart', 'fn': _psych_chart}],
         keywords=['psychrometrics', 'humidity ratio', 'dew point', 'wet bulb', 'relative humidity',
                   'enthalpy', 'specific volume'],
         assumptions=['Real-gas ASHRAE RP-1485 model via CoolProp HAPropsSI (not ideal-gas psychrometrics). '
                      'The wet-bulb temperature B equals the adiabatic-saturation temperature for moist air.'],
         refs=['ASHRAE RP-1485 real-gas moist-air properties via common.humid_air (HAPropsSI)']),

    Calc('moist_air_mixing', 'Adiabatic mixing of two moist-air streams',
         vars=[V('m1', 'kg/s', 'dry-air mass flow of stream 1', lo=0.01, hi=100.0),
               V('T1', 'K', 'stream 1 dry-bulb temperature', lo=240.0, hi=330.0),
               V('RH1', '', 'stream 1 relative humidity (0..1)', lo=0.0, hi=1.0),
               V('m2', 'kg/s', 'dry-air mass flow of stream 2', lo=0.01, hi=100.0),
               V('T2', 'K', 'stream 2 dry-bulb temperature', lo=240.0, hi=330.0),
               V('RH2', '', 'stream 2 relative humidity (0..1)', lo=0.0, hi=1.0),
               V('P', 'Pa', 'total pressure', lo=50e3, hi=120e3),
               V('W3', '', 'mixed humidity ratio'), V('H3', 'J/kg', 'mixed enthalpy per kg dry air'),
               V('T3', 'K', 'mixed dry-bulb temperature'), V('RH3', '', 'mixed relative humidity (0..1)')],
         procedure=_moist_air_mixing, outputs=['W3', 'H3', 'T3', 'RH3'],
         example={'m1': 1.0, 'T1': 298.15, 'RH1': 0.3, 'm2': 2.0, 'T2': 308.15, 'RH2': 0.7, 'P': 101325.0},
         keywords=['mixing', 'air streams', 'psychrometrics', 'adiabatic mixing'],
         assumptions=['Adiabatic mixing; the mixed state lies on the straight line between the two states '
                      'in the H–W plane (mass-weighted humidity-ratio and enthalpy balances).'],
         refs=['Mixing: W3 = Σ(m W)/Σm; H3 = Σ(m H)/Σm']),

    # ---- processes ----------------------------------------------------------------------------------
    Calc('sensible_process', 'Sensible heating / cooling of moist air (constant W)',
         vars=[V('m_da', 'kg/s', 'dry-air mass flow', lo=0.01, hi=100.0),
               V('T1', 'K', 'inlet dry-bulb temperature', lo=240.0, hi=330.0),
               V('RH1', '', 'inlet relative humidity (0..1)', lo=0.0, hi=1.0),
               V('T2', 'K', 'outlet dry-bulb temperature', lo=240.0, hi=330.0),
               V('P', 'Pa', 'total pressure', lo=50e3, hi=120e3),
               V('Q_dot', 'W', 'sensible heat rate (heating +, cooling −)', signed=True),
               V('W', '', 'humidity ratio (constant through the process)'),
               V('RH2', '', 'outlet relative humidity (0..1)'),
               V('H1', 'J/kg', 'inlet enthalpy per kg dry air'), V('H2', 'J/kg', 'outlet enthalpy per kg dry air')],
         procedure=_sensible_process, outputs=['Q_dot', 'W', 'RH2', 'H1', 'H2'],
         example={'m_da': 2.0, 'T1': 285.15, 'RH1': 0.6, 'T2': 300.15, 'P': 101325.0},
         keywords=['sensible heating', 'sensible cooling', 'heating coil', 'cooling coil'],
         assumptions=['Constant humidity ratio W; covers both heating (T2 > T1, Q > 0) and sensible cooling '
                      'above the dew point (T2 > T_dp).'],
         refs=['Q = m_da (H2 − H1) at constant W']),

    Calc('adiabatic_humidification', 'Adiabatic humidification / evaporative cooling (constant enthalpy)',
         vars=[V('m_da', 'kg/s', 'dry-air mass flow', lo=0.01, hi=100.0),
               V('T1', 'K', 'inlet dry-bulb temperature', lo=240.0, hi=330.0),
               V('RH1', '', 'inlet relative humidity (0..1)', lo=0.0, hi=1.0),
               V('m_w', 'kg/s', 'water spray rate', lo=1e-5, hi=1.0),
               V('P', 'Pa', 'total pressure', lo=50e3, hi=120e3),
               V('T2', 'K', 'outlet dry-bulb temperature (lower, toward wet-bulb)'),
               V('W1', '', 'inlet humidity ratio'), V('W2', '', 'outlet humidity ratio'),
               V('RH2', '', 'outlet relative humidity (0..1)'),
               V('H', 'J/kg', 'constant enthalpy per kg dry air')],
         procedure=_adiabatic_humidification, outputs=['T2', 'W1', 'W2', 'RH2', 'H'],
         example={'m_da': 1.0, 'T1': 310.0, 'RH1': 0.2, 'm_w': 0.002, 'P': 101325.0},
         keywords=['evaporative cooling', 'adiabatic humidification', 'water spray', 'swamp cooler'],
         assumptions=['No external heat: the outlet state follows the constant-enthalpy line toward '
                      'saturation (limits at the thermodynamic wet-bulb temperature). W2 must stay below '
                      'saturation.'],
         refs=['Adiabatic humidification: H = const, W2 = W1 + m_w/m_da']),

    Calc('cooling_dehumidification', 'Cooling and dehumidification of moist air',
         vars=[V('m_da', 'kg/s', 'dry-air mass flow', lo=0.01, hi=100.0),
               V('T1', 'K', 'inlet dry-bulb temperature', lo=240.0, hi=330.0),
               V('RH1', '', 'inlet relative humidity (0..1)', lo=0.0, hi=1.0),
               V('T2', 'K', 'coil (outlet) dry-bulb temperature', lo=240.0, hi=330.0),
               V('P', 'Pa', 'total pressure', lo=50e3, hi=120e3),
               V('Q_dot', 'W', 'total coil cooling load (negative of the air-side heat loss)', signed=True),
               V('m_cond', 'kg/s', 'condensate (moisture removal) rate', signed=True),
               V('W1', '', 'inlet humidity ratio'), V('W2', '', 'outlet humidity ratio (saturated)'),
               V('RH2', '', 'outlet relative humidity (= 1)'),
               V('H1', 'J/kg', 'inlet enthalpy per kg dry air'), V('H2', 'J/kg', 'outlet enthalpy per kg dry air')],
         procedure=_cooling_dehumidification, outputs=['Q_dot', 'm_cond', 'W1', 'W2', 'RH2', 'H1', 'H2'],
         example={'m_da': 1.0, 'T1': 305.0, 'RH1': 0.7, 'T2': 285.0, 'P': 101325.0},
         keywords=['dehumidification', 'cooling coil', 'condensate', 'latent load'],
         assumptions=['Air leaves saturated at T2, which must be below the inlet dew point; condensate '
                      'leaves at T2 (enthalpy c_pw (T2 − 273.15 K)). Reheat to the supply condition is a '
                      'separate sensible_process.'],
         refs=['Q = m_da(H1 − H2) − m_cond c_pw (T2 − 0 °C); m_cond = m_da (W1 − W2)']),

    # ---- cooling towers -----------------------------------------------------------------------------
    Calc('cooling_tower_range_approach', 'Cooling tower range and approach',
         vars=[V('TR', 'K', 'range = hot-water in − cold-water out'),
               V('approach', 'K', 'approach = cold-water out − ambient wet-bulb'),
               V('T_in', 'K', 'hot-water inlet temperature'), V('T_out', 'K', 'cold-water outlet temperature'),
               V('T_wb', 'K', 'ambient wet-bulb temperature')],
         equations=['TR = T_in - T_out', 'approach = T_out - T_wb'], outputs=['TR', 'approach'],
         example={'T_in': 313.15, 'T_out': 303.15, 'T_wb': 298.15},
         keywords=['cooling tower', 'range', 'approach', 'wet bulb'],
         assumptions=['Approach is the practical lower limit set by the ambient wet-bulb temperature.'],
         refs=['Cooling-tower range and approach definitions']),

    Calc('cooling_tower_makeup', 'Cooling tower makeup water (evaporation, blowdown, drift)',
         vars=[V('E', 'kg/s', 'evaporation loss'), V('B', 'kg/s', 'blowdown loss'),
               V('D', 'kg/s', 'drift loss'), V('M', 'kg/s', 'makeup water required'),
               V('m_w', 'kg/s', 'circulating water flow'), V('c_p', 'J/(kg*K)', 'water specific heat'),
               V('TR', 'K', 'range (hot in − cold out)'), V('h_fg', 'J/kg', 'latent heat of vaporisation'),
               V('n_cycles', '', 'cycles of concentration (blowdown cycles)', lo=1.5),
               V('drift_frac', '', 'drift fraction of circulating flow (≈0.002)', lo=0.0, hi=0.05)],
         equations=['E = m_w*c_p*TR/h_fg', 'D = m_w*drift_frac', 'B = E/(n_cycles - 1) - D',
                    'M = E + B + D'],
         outputs=['E', 'B', 'D', 'M'],
         example={'m_w': 50.0, 'c_p': 4186.0, 'TR': 10.0, 'h_fg': 2.45e6, 'n_cycles': 3.0, 'drift_frac': 0.002},
         keywords=['cooling tower', 'makeup water', 'blowdown', 'evaporation', 'cycles of concentration'],
         assumptions=['Evaporation removes the heat (E = Q/h_fg) and carries no dissolved solids; blowdown '
                      'and drift together carry the solids away, so B + D = E/(n − 1) and M = E + B + D = '
                      'E·n/(n − 1). Drift fraction is a tower-design input.'],
         refs=['Cooling-tower water balance M = E + B + D; solids balance B + D = E/(n − 1)']),

    # ---- loads --------------------------------------------------------------------------------------
    Calc('conduction_load', 'Conduction (envelope) heating/cooling load (Q = U A ΔT)',
         vars=[V('Q', 'W', 'conduction heat load', signed=True), V('U', 'W/(m^2*K)', 'overall heat-transfer coefficient'),
               V('A', 'm^2', 'surface area'), V('dT', 'K', 'indoor–outdoor temperature difference', signed=True)],
         equations=['Q = U*A*dT'], outputs=['Q'],
         example={'U': 0.5, 'A': 20.0, 'dT': 25.0},
         keywords=['conduction load', 'envelope', 'u-value', 'heat loss'],
         assumptions=['Use U from the overall-U calculator; dT = T_indoor − T_outdoor (heating positive, '
                      'cooling negative).'],
         refs=['Q = U A ΔT']),

    Calc('air_infiltration', 'Infiltration / ventilation sensible and latent loads',
         vars=[V('Q_sens', 'W', 'sensible load', signed=True), V('Q_lat', 'W', 'latent load', signed=True),
               V('rho', 'kg/m^3', 'air density', label='ρ'), V('c_p', 'J/(kg*K)', 'air specific heat'),
               V('h_fg', 'J/kg', 'water latent heat of vaporisation'), V('Q_air', 'm^3/s', 'volumetric air flow'),
               V('dT', 'K', 'indoor–outdoor temperature difference', signed=True),
               V('dW', '', 'indoor–outdoor humidity-ratio difference', signed=True)],
         equations=['Q_sens = rho*c_p*Q_air*dT', 'Q_lat = rho*h_fg*Q_air*dW'], outputs=['Q_sens', 'Q_lat'],
         example={'rho': 1.2, 'c_p': 1005.0, 'h_fg': 2.5e6, 'Q_air': 0.1, 'dT': 20.0, 'dW': 0.005},
         keywords=['infiltration', 'ventilation load', 'sensible', 'latent'],
         assumptions=['Standard air ρ ≈ 1.2 kg/m³, c_p ≈ 1005 J/(kg·K); latent heat uses the water value '
                      'h_fg ≈ 2.5 MJ/kg. dT and dW are signed (humidification/dehumidification change the '
                      'sign of Q_lat).'],
         refs=['Q_sens = ρ c_p Q dT; Q_lat = ρ h_fg Q dW']),

    Calc('air_changes_per_hour', 'Air changes per hour and the corresponding volumetric flow',
         vars=[V('ACH', '1/h', 'air changes per hour', label='ACH'),
               V('Q', 'm^3/s', 'volumetric air flow'), V('V_room', 'm^3', 'room volume')],
         equations=['Q = ACH*V_room/3600'], outputs=['Q'],
         example={'ACH': 6.0, 'V_room': 200.0},
         keywords=['air changes', 'ach', 'ventilation', 'room volume'],
         assumptions=['Q is the supply/outdoor flow giving the stated air-change rate.'],
         refs=['ACH = 3600 Q/V']),

    Calc('ventilation_rate', 'Required outdoor-air (ventilation) rate (ASHRAE 62.1 form)',
         vars=[V('Q_oa', 'm^3/s', 'required outdoor-air flow'), V('Rp', 'm^3/s', 'per-person outdoor-air rate'),
               V('P_z', '', 'number of people (zone population)', integer=True),
               V('Ra', 'm^3/(s*m^2)', 'per-area outdoor-air rate'), V('A_z', 'm^2', 'zone floor area')],
         equations=['Q_oa = Rp*P_z + Ra*A_z'], outputs=['Q_oa'],
         example={'Rp': 0.0025, 'P_z': 20.0, 'Ra': 0.0003, 'A_z': 150.0},
         keywords=['ventilation', 'outdoor air', 'ashrae 62.1', 'iaq'],
         assumptions=['Rp and Ra are the ASHRAE 62.1 breathing-zone rates for the occupancy category '
                      '(user inputs, standard table).'],
         refs=['ASHRAE 62.1 ventilation-rate procedure form (recalled)']),

    Calc('sensible_heat_ratio', 'Sensible heat ratio (SHR = Q_sens/(Q_sens + Q_lat))',
         vars=[V('SHR', '', 'sensible heat ratio', hi=1.0), V('Q_sens', 'W', 'sensible heat gain'),
               V('Q_lat', 'W', 'latent heat gain')],
         equations=['SHR = Q_sens/(Q_sens + Q_lat)'], outputs=['SHR'],
         example={'Q_sens': 7000.0, 'Q_lat': 3000.0},
         keywords=['sensible heat ratio', 'shr', 'room condition line'],
         assumptions=['Defines the slope of the room condition line on the psychrometric chart.'],
         refs=['SHR = Q_sens/(Q_sens + Q_lat)']),

    Calc('apparatus_dew_point', 'Apparatus dew point (coil ADP) from room condition and SHR',
         vars=[V('T_room', 'K', 'room dry-bulb temperature', lo=273.0, hi=320.0),
               V('RH_room', '', 'room relative humidity (0..1)', lo=0.1, hi=0.95),
               V('P', 'Pa', 'total pressure', lo=50e3, hi=120e3),
               V('SHR', '', 'room sensible heat ratio (0..1)', lo=0.05, hi=0.99),
               V('T_adp', 'K', 'apparatus dew-point temperature'),
               V('W_adp', '', 'humidity ratio at the apparatus dew point')],
         procedure=_apparatus_dew_point, outputs=['T_adp', 'W_adp'],
         example={'T_room': 297.15, 'RH_room': 0.5, 'P': 101325.0, 'SHR': 0.7},
         keywords=['apparatus dew point', 'adp', 'cooling coil', 'shr line'],
         assumptions=['ADP is where the room condition line (slope from SHR) meets saturation; it always '
                      'lies below the room dew point. Coil sizing uses the ADP with the bypass factor.'],
         refs=['Room-condition-line / apparatus-dew-point construction (psychrometric chart)']),

    Calc('total_space_load', 'Total space cooling/heating load (sum of components)',
         vars=[V('Q_total', 'W', 'total load', signed=True), V('Q_cond', 'W', 'conduction load'),
               V('Q_infil', 'W', 'infiltration/ventilation load'), V('Q_solar', 'W', 'solar heat gain'),
               V('Q_internal', 'W', 'internal gains (people, lights, equipment)')],
         equations=['Q_total = Q_cond + Q_infil + Q_solar + Q_internal'], outputs=['Q_total'],
         example={'Q_cond': 2000.0, 'Q_infil': 800.0, 'Q_solar': 1500.0, 'Q_internal': 2500.0},
         keywords=['space load', 'cooling load', 'heating load', 'internal gains', 'solar gain'],
         assumptions=['Solar heat gain factors and CLTD values are user inputs (standard tables), folded '
                      'into Q_solar/Q_cond.'],
         refs=['Space load = Σ (conduction + infiltration + solar + internal gains)']),

    # ---- ducts, fans --------------------------------------------------------------------------------
    Calc('rectangular_duct_equivalent', 'Circular equivalent diameter of a rectangular duct (Huebscher)',
         vars=[V('D_e', 'm', 'equivalent circular diameter', label='De'),
               V('a', 'm', 'duct width'), V('b', 'm', 'duct height')],
         equations=['D_e = 1.3*(a*b)**0.625/(a + b)**0.25'], outputs=['D_e'],
         example={'a': 0.4, 'b': 0.2},
         keywords=['rectangular duct', 'equivalent diameter', 'huebscher'],
         assumptions=['Huebscher equivalent diameter for equal friction at the same flow (aspect ratios '
                      'below ~8).'],
         refs=['Huebscher circular-equivalent-diameter relation (recalled)']),

    Calc('fan_power', 'Fan air power and shaft power',
         vars=[V('P_air', 'W', 'air (fluid) power'), V('P_shaft', 'W', 'shaft power'),
               V('Q', 'm^3/s', 'volumetric air flow'), V('dp', 'Pa', 'fan total pressure rise'),
               V('eta', '', 'fan efficiency', hi=1.0, label='η')],
         equations=['P_air = Q*dp', 'P_shaft = P_air/eta'], outputs=['P_air', 'P_shaft'],
         example={'Q': 0.5, 'dp': 250.0, 'eta': 0.7},
         keywords=['fan power', 'static pressure', 'fan efficiency', 'blower'],
         assumptions=['dp is the fan TOTAL pressure rise (static + velocity pressure across the fan); '
                      'decompose it with turbomachinery.fan_total_pressure. η = fan total efficiency.'],
         refs=['P_air = Q Δp; P_shaft = P_air/η']),

    # ---- hydronic ----------------------------------------------------------------------------------
    Calc('hydronic_capacity', 'Hydronic heating/cooling capacity (Q = ṁ c_p ΔT)',
         vars=[V('Q', 'W', 'heat rate delivered to/from the water (heating +, cooling −)', signed=True),
               V('m_dot', 'kg/s', 'water mass flow'),
               V('c_p', 'J/(kg*K)', 'water specific heat'),
               V('dT', 'K', 'supply–return temperature difference (signed)', signed=True)],
         equations=['Q = m_dot*c_p*dT'], outputs=['Q'],
         example={'m_dot': 2.0, 'c_p': 4186.0, 'dT': 5.5},
         keywords=['chiller', 'boiler', 'chilled water', 'hot water', 'hydronic', 'cooling capacity',
                   'heating capacity'],
         assumptions=['One relation for both chilled-water (cooling, Q < 0 in the heating sign convention) '
                      'and hot-water (heating) loops; dT ≈ 5–7 K for chilled water, 10–20 K for hot water; '
                      'c_p ≈ 4186 J/(kg·K).'],
         refs=['Q = m_dot c_p ΔT']),
]

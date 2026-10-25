# DESCRIPTION: Renewables and power-systems section: solar geometry (declination, hour angle, incidence
#              angle), tilted-plane irradiance (isotropic-sky Liu–Jordan), collector efficiency,
#              photovoltaic power and temperature derating, wind power density, geothermal efficiency,
#              boiler efficiency, steam generation rate, power-plant heat rate and capacity factor.
#              Hydropower is the committed turbomachinery.turbine_power calculator, not duplicated here.
"""Renewable and power-system calculators.

Conventions:
- Solar geometry follows Duffie & Beckman: latitude φ positive north, declination δ positive north,
  hour angle ω negative before solar noon and positive after, surface slope β from horizontal (0..π/2),
  surface azimuth γ zero when the surface faces the equator (south in the northern hemisphere) and
  positive toward the west. The incidence angle θ is the angle between the beam and the surface normal.
- Wind power density here is the kinetic-energy flux ½ρv³; the Betz turbine limit lives in the
  turbomachinery section (as does hydraulic turbine power — see turbomachinery.turbine_power).
- Steam/feed-water enthalpies are read from CoolProp (IAPWS-IF97) through engineering_sim.science.mech.common;
  steam may be specified as saturated vapour (P, quality Q = 1) or superheated (T, P).
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('renewables', 'Renewable and power systems', order=330,
                  summary='Solar geometry and irradiance, collectors, PV (power and temperature derating), '
                          'wind power density, geothermal, boiler efficiency, steam generation, heat rate, '
                          'capacity factor. (Hydropower: see turbomachinery.turbine_power.)')

PI = math.pi


def _tilted_irradiance(values):
    """Total irradiance on a tilted plane: isotropic-sky Liu–Jordan model.

    G_T = G_bn·cos θ (beam on the tilted plane) + G_d·(1+cos β)/2 (sky-diffuse view factor)
        + (G_bn·cos θ_z + G_d)·ρ_g·(1−cos β)/2 (ground-reflected, isotropic).
    """
    G_bn, G_d = values['G_bn'], values['G_d']
    rho_g, beta = values['rho_g'], values['beta']
    gamma, phi, delta, omega = values['gamma'], values['phi'], values['delta'], values['omega']
    ctz = math.sin(delta) * math.sin(phi) + math.cos(delta) * math.cos(phi) * math.cos(omega)
    if ctz <= 0:
        raise MechError('sun is below the horizon at this hour angle/latitude/declination '
                        '(cos of the zenith angle <= 0)')
    ct = (math.sin(delta) * math.sin(phi) * math.cos(beta)
          - math.sin(delta) * math.cos(phi) * math.sin(beta) * math.cos(gamma)
          + math.cos(delta) * math.cos(phi) * math.cos(beta) * math.cos(omega)
          + math.cos(delta) * math.sin(phi) * math.sin(beta) * math.cos(gamma) * math.cos(omega)
          + math.cos(delta) * math.sin(beta) * math.sin(gamma) * math.sin(omega))
    warnings = []
    beam_cos = ct
    if ct <= 0:
        beam_cos = 0.0
        warnings.append('beam irradiance set to zero: the tilted surface faces away from the sun '
                        '(cos of the incidence angle <= 0)')
    Rb = beam_cos / ctz
    G_bT = G_bn * beam_cos
    G_dT = G_d * (1.0 + math.cos(beta)) / 2.0
    G_horiz = G_bn * ctz + G_d  # global irradiance on the horizontal
    G_rT = G_horiz * rho_g * (1.0 - math.cos(beta)) / 2.0
    out = {'G_T': G_bT + G_dT + G_rT, 'cos_theta_z': ctz, 'cos_theta': ct, 'R_b': Rb,
           'G_bT': G_bT, 'G_dT': G_dT, 'G_rT': G_rT}
    if warnings:
        out['_warnings'] = warnings
    return out


def _steam_enthalpy(values):
    """Enthalpy of the generated steam.

    'saturated': saturated vapour at P_steam (CoolProp state with quality Q = 1), so a temperature just
    below T_sat can never silently resolve to liquid water. 'superheated': the T_steam/P_steam state,
    rejected with a clear error when that state is a liquid (T_steam below T_sat).
    """
    from engineering_sim.science.mech import common
    if values['steam_condition'] == 'saturated':
        return common.fluid_state('Water', P=values['P_steam'], Q=1.0)['h']
    st = common.fluid_state('Water', T=values['T_steam'], P=values['P_steam'])
    if st['phase'] == 'liquid':
        raise MechError('steam temperature is below the saturation temperature at this pressure '
                        '(the state is liquid, not steam): raise T_steam or use '
                        'steam_condition="saturated"')
    return st['h']


def _feed_enthalpy(values):
    from engineering_sim.science.mech import common
    return common.fluid_state('Water', T=values['T_feed'], P=values['P_feed'])['h']


def _solar_collector(values):
    """Hottel–Whillier–Bliss flat-plate collector: useful heat and instantaneous efficiency.

    Q_u and η are both signed — a collector running at night or at high ΔT loses heat (negative Q_u, η).
    Zero irradiance is rejected (efficiency is undefined, and the heat gain is a pure loss there).
    """
    G = values['G']
    if G <= 0:
        raise MechError('incident irradiance G must be positive (efficiency is undefined at zero irradiance)')
    Q_u = values['A_c'] * (values['Fr_ta'] * G - values['Fr_UL'] * (values['T_in'] - values['T_amb']))
    return {'Q_u': Q_u, 'eta': Q_u / (values['A_c'] * G)}


def _boiler_efficiency(values):
    """Direct-method boiler efficiency: η = ṁ_s(h_steam − h_feed) / (ṁ_f·HHV)."""
    h_s = _steam_enthalpy(values)
    h_f = _feed_enthalpy(values)
    if h_s <= h_f:
        raise MechError('steam enthalpy must exceed the feed-water enthalpy (no net heat can be added)')
    Q_steam = values['mdot_s'] * (h_s - h_f)
    Q_fuel = values['mdot_f'] * values['HHV']
    if Q_fuel <= 0:
        raise MechError('fuel heat input ṁ_f·HHV must be positive')
    eta = Q_steam / Q_fuel
    if eta > 1.0:
        raise MechError('computed efficiency exceeds 1 — check the steam/feed states and fuel flow')
    return {'eta': eta, 'Q_steam': Q_steam, 'Q_fuel': Q_fuel, 'h_steam': h_s, 'h_feed': h_f}


def _steam_generation(values):
    """Steam generation rate from a heat input: ṁ_s = Q_in / (h_steam − h_feed)."""
    h_s = _steam_enthalpy(values)
    h_f = _feed_enthalpy(values)
    if h_s <= h_f:
        raise MechError('steam enthalpy must exceed the feed-water enthalpy (no net heat can be added)')
    if values['Q_in'] <= 0:
        raise MechError('heat input Q_in must be positive')
    return {'mdot_s': values['Q_in'] / (h_s - h_f), 'h_steam': h_s, 'h_feed': h_f}


def _geothermal_efficiency(values):
    """Geothermal thermal and utilization efficiency, with clear errors on non-physical inputs."""
    W_net, Q_in = values['W_net'], values['Q_in']
    mdot, h_in, h_0 = values['mdot'], values['h_in'], values['h_0']
    if Q_in <= 0:
        raise MechError('heat transfer rate Q_in must be positive')
    if mdot <= 0:
        raise MechError('geofluid mass flow ṁ must be positive')
    if h_in <= h_0:
        raise MechError('wellhead enthalpy h_in must exceed the dead-state enthalpy h_0')
    return {'eta_th': W_net / Q_in, 'eta_util': W_net / (mdot * (h_in - h_0))}


CALCS = [
    # ---- solar geometry --------------------------------------------------------------------------
    Calc('solar_declination', 'Solar declination angle (Cooper)',
         vars=[V('delta', 'rad', 'declination angle (north positive)', signed=True, lo=-0.41, hi=0.41, label='δ'),
               V('n', '', 'day of year (1 = Jan 1, 365 = Dec 31)', lo=1.0, hi=365.0, integer=True)],
         equations=['delta = 23.45*pi/180*sin(2*pi*(284 + n)/365)'],
         outputs=['delta'], example={'n': 172.0},
         keywords=['solar geometry', 'season', 'summer solstice'],
         assumptions=['Cooper approximation; declination ranges ±23.45° (solstices), 0 at the equinoxes. '
                      'Inverting δ → n is two-valued: a given declination occurs on two days of the year, and '
                      'the solver returns the root closest in log-distance to the example n (for δ = 0 with the '
                      'example n = 172 that is n ≈ 263.5, the September equinox).'],
         refs=['Cooper (1969) declination, δ = 23.45° sin(2π(284+n)/365) (recalled)']),

    Calc('solar_hour_angle', 'Solar hour angle from clock (solar) time',
         vars=[V('omega', 'rad', 'hour angle (morning negative, afternoon positive)', signed=True, lo=-PI, hi=PI, label='ω'),
               V('h', '', 'solar time of day in hours as a number (noon = 12)', lo=0.0, hi=24.0)],
         equations=['omega = pi/12*(h - 12)'],
         outputs=['omega'], example={'h': 10.0},
         keywords=['solar geometry', 'solar noon'],
         assumptions=['Solar time, not clock time: 15° of hour angle per hour (ω = 15°(h − 12)).'],
         refs=['Hour angle ω = (π/12)(solar hour − 12)']),

    Calc('solar_incidence', 'Solar incidence angle on a tilted surface',
         vars=[V('theta', 'rad', 'incidence angle between beam and surface normal', hi=PI, label='θ'),
               V('cos_theta', '', 'cosine of the incidence angle', lo=-1.0, hi=1.0, signed=True),
               V('delta', 'rad', 'declination (north positive)', signed=True, lo=-0.41, hi=0.41, label='δ'),
               V('phi', 'rad', 'latitude (north positive)', signed=True, lo=-PI / 2, hi=PI / 2, label='φ'),
               V('beta', 'rad', 'surface slope from horizontal (tilt)', hi=PI / 2, label='β'),
               V('gamma', 'rad', 'surface azimuth from due SOUTH, west positive (γ = 0 faces the equator only in the northern hemisphere; use γ = 180° for equator-facing south of the equator)', signed=True, lo=-PI, hi=PI, label='γ'),
               V('omega', 'rad', 'hour angle (morning negative)', signed=True, lo=-PI, hi=PI, label='ω')],
         equations=['cos_theta = sin(delta)*sin(phi)*cos(beta) - sin(delta)*cos(phi)*sin(beta)*cos(gamma) '
                    '+ cos(delta)*cos(phi)*cos(beta)*cos(omega) + cos(delta)*sin(phi)*sin(beta)*cos(gamma)*cos(omega) '
                    '+ cos(delta)*sin(beta)*sin(gamma)*sin(omega)',
                    'theta = acos(cos_theta)'],
         outputs=['theta', 'cos_theta'],
         example={'delta': math.radians(20), 'phi': math.radians(40), 'beta': math.radians(30),
                  'gamma': math.radians(15), 'omega': math.radians(-30)},
         keywords=['solar geometry', 'beam', 'surface normal', 'duffie beckman'],
         assumptions=['Duffie–Beckman convention: γ = 0 when the surface faces the equator (south in the '
                      'northern hemisphere) and is positive toward the west. For a horizontal surface (β = 0) '
                      'this reduces to the zenith angle cos θ_z = sin δ sin φ + cos δ cos φ cos ω.'],
         refs=['Duffie & Beckman, Solar Engineering of Thermal Processes, incidence-angle relation (recalled)']),

    # ---- irradiance ------------------------------------------------------------------------------
    Calc('tilted_irradiance', 'Irradiance on a tilted surface (isotropic-sky Liu–Jordan)',
         vars=[V('G_bn', 'W/m^2', 'beam (direct normal) irradiance'),
               V('G_d', 'W/m^2', 'diffuse irradiance on the horizontal'),
               V('rho_g', '', 'ground albedo (reflectance)', hi=1.0, label='ρg'),
               V('beta', 'rad', 'surface slope from horizontal (tilt)', hi=PI / 2, label='β'),
               V('gamma', 'rad', 'surface azimuth from due SOUTH, west positive (γ = 0 faces the equator only in the northern hemisphere; use γ = 180° for equator-facing south of the equator)', signed=True, lo=-PI, hi=PI, label='γ'),
               V('phi', 'rad', 'latitude (north positive)', signed=True, lo=-PI / 2, hi=PI / 2, label='φ'),
               V('delta', 'rad', 'declination (north positive)', signed=True, lo=-0.41, hi=0.41, label='δ'),
               V('omega', 'rad', 'hour angle (morning negative)', signed=True, lo=-PI, hi=PI, label='ω'),
               V('G_T', 'W/m^2', 'total irradiance on the tilted plane'),
               V('cos_theta_z', '', 'cosine of the zenith angle', lo=0.0, hi=1.0),
               V('cos_theta', '', 'cosine of the incidence angle', lo=-1.0, hi=1.0, signed=True),
               V('R_b', '', 'beam geometric ratio cos θ / cos θ_z', lo=0.0),
               V('G_bT', 'W/m^2', 'beam irradiance on the tilted plane'),
               V('G_dT', 'W/m^2', 'sky-diffuse irradiance on the tilted plane'),
               V('G_rT', 'W/m^2', 'ground-reflected irradiance on the tilted plane')],
         procedure=_tilted_irradiance,
         outputs=['G_T', 'cos_theta_z', 'cos_theta', 'R_b', 'G_bT', 'G_dT', 'G_rT'],
         example={'G_bn': 800.0, 'G_d': 150.0, 'rho_g': 0.2, 'beta': math.radians(30),
                  'gamma': math.radians(15), 'phi': math.radians(40), 'delta': math.radians(20),
                  'omega': math.radians(-30)},
         keywords=['liu jordan', 'isotropic sky', 'diffuse', 'albedo', 'poa'],
         assumptions=['Isotropic sky and ground; beam on the horizontal is G_bn·cos θ_z. '
                      'Raises an error when the sun is below the horizon (cos θ_z ≤ 0) and zeroes the beam '
                      'term (with a warning) when the surface faces away from the sun.'],
         refs=['Liu & Jordan isotropic-sky model (recalled)']),

    # ---- collectors and PV ------------------------------------------------------------------------
    Calc('solar_collector', 'Flat-plate collector efficiency (Hottel–Whillier–Bliss)',
         vars=[V('eta', '', 'instantaneous collector efficiency (can be negative at night/high ΔT)', signed=True, hi=1.0, label='η'),
               V('Q_u', 'W', 'useful heat gain (negative = net loss)', signed=True),
               V('Fr_ta', '', 'heat-removal factor × transmittance–absorptance product, F_R(τα)', hi=1.0),
               V('Fr_UL', 'W/(m^2*K)', 'heat-removal factor × overall loss coefficient, F_R U_L'),
               V('T_in', 'K', 'collector inlet fluid temperature'),
               V('T_amb', 'K', 'ambient temperature'),
               V('G', 'W/m^2', 'incident solar irradiance on the collector plane'),
               V('A_c', 'm^2', 'collector aperture area')],
         procedure=_solar_collector,
         outputs=['Q_u', 'eta'],
         example={'Fr_ta': 0.75, 'Fr_UL': 4.0, 'T_in': 333.15,
                  'T_amb': 293.15, 'G': 800.0, 'A_c': 2.0},
         keywords=['flat plate', 'hottel whillier', 'collector', 'solar thermal'],
         assumptions=['Steady state, F_R(τα) and F_R U_L from collector test data (inputs), no incidence-angle '
                      'modifier. Uses T_in rather than a mean plate temperature. Q_u and η are signed (night-time '
                      'or high-ΔT operation loses heat); zero irradiance is rejected because efficiency is '
                      'undefined there.'],
         refs=['Hottel–Whillier–Bliss collector model, η = F_R(τα) − F_R U_L (T_in − T_amb)/G (recalled)']),

    Calc('pv_power', 'Photovoltaic array power output',
         vars=[V('P', 'W', 'electrical power output'), V('G', 'W/m^2', 'irradiance in the module plane'),
               V('A', 'm^2', 'module (array) area'), V('eta', '', 'overall conversion efficiency', hi=1.0, label='η')],
         equations=['P = G*A*eta'], outputs=['P'], example={'G': 1000.0, 'A': 1.6, 'eta': 0.18},
         keywords=['solar panel', 'pv', 'module', 'stc'],
         assumptions=['η is the overall system efficiency at the operating conditions (cell, inverter, wiring, '
                      'temperature and mismatch losses already folded in).'],
         refs=['P = G·A·η (photovoltaic energy balance)']),

    Calc('pv_derate', 'Photovoltaic efficiency temperature derating',
         vars=[V('eta', '', 'module efficiency at the cell temperature', hi=1.0, label='η'),
               V('eta_ref', '', 'module efficiency at the reference temperature', hi=1.0, label='ηref'),
               V('beta_T', '1/K', 'temperature coefficient of maximum power (magnitude, e.g. 0.004 /K)'),
               V('T_c', 'K', 'cell temperature'),
               V('T_ref', 'K', 'reference temperature (typically 25 °C)')],
         equations=['eta = eta_ref*(1 - beta_T*(T_c - T_ref))'],
         outputs=['eta'], example={'eta_ref': 0.20, 'beta_T': 0.004, 'T_c': 318.15, 'T_ref': 298.15},
         keywords=['pv', 'temperature coefficient', 'derating', 'noct', 'cell temperature'],
         assumptions=['Linear derating η = η_ref[1 − β_T(T_c − T_ref)]; β_T is the (positive) magnitude of the '
                      'power temperature coefficient (enter e.g. 0.004 /K, not −0.004 /K). A cell cooler than '
                      'T_ref gives η above η_ref.'],
         refs=['Photovoltaic temperature derating, η = η_ref[1 − β_T(T_c − T_ref)] (recalled)']),

    # ---- wind ------------------------------------------------------------------------------------
    Calc('wind_power_density', 'Wind power density (kinetic-energy flux)',
         vars=[V('p_density', 'W/m^2', 'wind power per unit swept area'), V('rho', 'kg/m^3', 'air density', label='ρ'),
               V('v', 'm/s', 'wind speed')],
         equations=['p_density = rho*v**3/2'], outputs=['p_density'],
         example={'rho': 1.225, 'v': 10.0},
         keywords=['wind energy', 'wind power class'],
         assumptions=['Power in the wind per unit area, ½ρv³. The Betz limit (16/27 of this, for a turbine) is in '
                      'the turbomachinery section.'],
         refs=['Kinetic-energy flux of a wind stream, P/A = ½ρv³']),

    # ---- geothermal -------------------------------------------------------------------------------
    Calc('geothermal_efficiency', 'Geothermal plant thermal and utilization efficiency',
         vars=[V('eta_th', '', 'thermal efficiency W_net / Q_in', hi=1.0, label='ηth'),
               V('eta_util', '', 'utilization efficiency W_net / ṁ(h_in − h_0)', hi=1.0, label='ηu'),
               V('W_net', 'W', 'net electrical power output'),
               V('Q_in', 'W', 'heat transfer rate from the geofluid'),
               V('mdot', 'kg/s', 'geofluid mass flow rate'),
               V('h_in', 'J/kg', 'geofluid enthalpy at the wellhead'),
               V('h_0', 'J/kg', 'dead-state (discharge) enthalpy of the geofluid')],
         procedure=_geothermal_efficiency,
         outputs=['eta_util', 'eta_th'],
         example={'W_net': 10e6, 'Q_in': 120e6, 'mdot': 100.0, 'h_in': 1.5e6, 'h_0': 0.15e6},
         no_invert={'Q_in': 'Q_in appears only in the thermal efficiency η_th, not in the utilization '
                            'efficiency η_util used for inversion'},
         keywords=['geothermal', 'binary cycle', 'flash', 'utilization'],
         assumptions=['Utilization efficiency is referenced to the dead-state temperature of the surroundings; '
                      'enthalpies are inputs (or take them from the steam calculators). Requires h_in > h_0 and '
                      'positive Q_in and ṁ.'],
         refs=['Geothermal thermal and utilization efficiency definitions (recalled)']),

    # ---- boiler / steam ----------------------------------------------------------------------------
    Calc('boiler_efficiency', 'Boiler efficiency by the direct method (steam/feed enthalpies from CoolProp)',
         vars=[V('mdot_s', 'kg/s', 'steam mass flow rate'),
               V('mdot_f', 'kg/s', 'fuel mass flow rate'),
               V('HHV', 'J/kg', 'fuel higher heating value'),
               V('steam_condition', '', 'steam state', kind='text',
                 choices=['saturated', 'superheated']),
               V('T_steam', 'K', 'steam outlet temperature (used only when superheated)', lo=273.16, hi=1000.0),
               V('P_steam', 'Pa', 'steam outlet (absolute) pressure', lo=1e4, hi=2e7),
               V('T_feed', 'K', 'feed-water temperature', lo=273.16, hi=600.0),
               V('P_feed', 'Pa', 'feed-water pressure', lo=1e5, hi=1e7),
               V('eta', '', 'boiler efficiency Q_steam / Q_fuel', hi=1.0, label='η'),
               V('Q_steam', 'W', 'heat absorbed by the steam'), V('Q_fuel', 'W', 'heat released by the fuel'),
               V('h_steam', 'J/kg', 'steam enthalpy'), V('h_feed', 'J/kg', 'feed-water enthalpy')],
         procedure=_boiler_efficiency,
         outputs=['eta', 'Q_steam', 'Q_fuel', 'h_steam', 'h_feed'],
         example={'mdot_s': 10.0, 'mdot_f': 0.8, 'HHV': 45e6, 'steam_condition': 'saturated',
                  'T_steam': 453.03, 'P_steam': 1e6, 'T_feed': 333.15, 'P_feed': 1.2e6},
         no_invert={'T_steam': 'used only for superheated steam; not recoverable when '
                               'steam_condition="saturated"'},
         keywords=['steam boiler', 'direct method', 'fuel', 'hfg', 'iaspws'],
         assumptions=['Direct method on the fuel HHV. Steam and feed enthalpies come from CoolProp (IAPWS-IF97); '
                      'steam_condition="saturated" uses saturated vapour at P_steam (quality Q = 1) and '
                      '"superheated" uses the T_steam/P_steam state (rejected if that state is liquid). Errors '
                      'when h_steam ≤ h_feed or the efficiency would exceed 1.'],
         refs=['Boiler efficiency by the direct method, η = ṁ_s(h_s − h_f)/(ṁ_f·HHV) (recalled)']),

    Calc('steam_generation_rate', 'Steam generation rate from a heat input (enthalpies from CoolProp)',
         vars=[V('Q_in', 'W', 'heat transfer rate into the water/steam'),
               V('steam_condition', '', 'steam state', kind='text',
                 choices=['saturated', 'superheated']),
               V('T_steam', 'K', 'steam outlet temperature (used only when superheated)', lo=273.16, hi=1000.0),
               V('P_steam', 'Pa', 'steam outlet (absolute) pressure', lo=1e4, hi=2e7),
               V('T_feed', 'K', 'feed-water temperature', lo=273.16, hi=600.0),
               V('P_feed', 'Pa', 'feed-water pressure', lo=1e5, hi=1e7),
               V('mdot_s', 'kg/s', 'steam mass flow rate'),
               V('h_steam', 'J/kg', 'steam enthalpy'), V('h_feed', 'J/kg', 'feed-water enthalpy')],
         procedure=_steam_generation,
         outputs=['mdot_s', 'h_steam', 'h_feed'],
         example={'Q_in': 20.25e6, 'steam_condition': 'saturated', 'T_steam': 453.03, 'P_steam': 1e6,
                  'T_feed': 333.15, 'P_feed': 1.2e6},
         no_invert={'T_steam': 'used only for superheated steam; not recoverable when '
                               'steam_condition="saturated"'},
         keywords=['steam rate', 'boiler', 'evaporation', 'iaspws'],
         assumptions=['Enthalpies from CoolProp (IAPWS-IF97); ṁ_s = Q_in/(h_s − h_f). steam_condition="saturated" '
                      'uses saturated vapour at P_steam (quality Q = 1), "superheated" the T_steam/P_steam state. '
                      'Errors when h_steam ≤ h_feed. No blowdown or reheater terms.'],
         refs=['Steam generation rate ṁ = Q/(h_steam − h_feed) (recalled)']),

    # ---- plant metrics -----------------------------------------------------------------------------
    Calc('heat_rate', 'Power plant heat rate (energy input per unit electricity out)',
         vars=[V('HR', '', 'heat rate as energy in per unit electricity out (1/η; ×3600 → kJ/kWh, ×3412 → Btu/kWh)',
                 lo=1.0),
               V('eta', '', 'plant net efficiency (electricity out / heat in)', lo=1e-6, hi=1.0, label='η')],
         equations=['HR = 1/eta'], outputs=['HR'], example={'eta': 0.4},
         keywords=['heat rate', 'gross', 'net', 'kj/kwh', 'btu/kwh'],
         assumptions=['HR = 1/η is the dimensionless heat rate (J of heat in per J of electricity out). The '
                      'conventional numbers follow by re-expressing the output unit: 3600/η kJ/kWh or 3412/η '
                      'Btu/kWh — the unit conversion carries the 3600/3412, not the equation.'],
         refs=['Heat rate = 1/η (recalled)']),

    Calc('capacity_factor', 'Power plant capacity factor',
         vars=[V('CF', '', 'capacity factor E/(P·t)', hi=1.0),
               V('E', 'J', 'actual energy generated over the period'),
               V('P', 'W', 'rated (nameplate) power'), V('t', 's', 'period length (e.g. 8760 h = 1 year)')],
         equations=['CF = E/(P*t)'], outputs=['CF'],
         example={'E': 1.8e15, 'P': 1e8, 't': 3.1536e7},
         keywords=['capacity factor', 'availability', 'utilization', 'annual'],
         assumptions=['Fraction of the rated output actually produced; distinct from availability (which ignores '
                      'derating and curtailment).'],
         refs=['Capacity factor = energy delivered / (rated power × time) (recalled)']),
]

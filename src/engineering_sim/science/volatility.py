# DESCRIPTION: Additive volatility during compounding: Antoine / Clausius-Clapeyron vapour pressure and an evaporation-limited vent-loss estimate with flash-point and TGA-onset flags. Standard library only.
"""Volatility screening for compounding.

Two tools:
  vapor_pressure(record, T_K) -- saturated vapour pressure of one additive at T_K,
      from NIST-form Antoine coefficients (log10 P[bar] = A - B/(T + C), T in K,
      range-checked) or from a normal boiling point + heat of vaporisation via the
      Clausius-Clapeyron equation.
  screen(args) -- additive loss through a compounding vent, an evaporation-limited
      (upper-bound) estimate: ideal Raoult dilution at the additive mole fraction,
      a film mass-transfer coefficient as input (default + sensitivity range), and
      flags comparing flash point and TGA onset against the melt temperature.

Everything returned is a cite value record. These are screening estimates, not
measurements: the film coefficient and surface area are the dominant uncertainties.
"""
import math

from . import cite

R = 8.314462618  # J/(mol K), CODATA 2018

# NIST Chemistry WebBook, water, Antoine Equation Parameters (log10 P/bar, T in K):
#   https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=4&Type=ANTOINE&Plot=on
# Bridgeman & Aldrich, 1964 (344 to 373 K): A = 5.08354, B = 1663.125, C = -45.622.
WATER_ANTOINE = {'A': 5.08354, 'B': 1663.125, 'C': -45.622, 'tmin_K': 344.0, 'tmax_K': 373.15,
                 'source': 'https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=4&Type=ANTOINE&Plot=on',
                 'reference': 'Bridgeman and Aldrich, 1964 (NIST Antoine fit, 344-373 K)'}

# Film mass-transfer coefficient k_m [m/s] for an additive evaporating from a melt
# surface into a swept vent. Literature film coefficients for vapour into flowing
# gas span ~1e-3 (stagnant) to ~5e-2 (strongly mixed) m/s; 1e-2 is the screening
# default and the sensitivity range below is reported with every screen() result.
DEFAULT_K_M_S = 0.01
K_M_SENSITIVITY = (0.001, 0.05)

# Specific vent geometry defaults: a twin-screw atmospheric vent port of ~5 cm x
# 10 cm exposed melt surface, on a 1 kg shot. A/m_melt is the controlling geometry.
DEFAULT_SURFACE_AREA_M2 = 0.005
DEFAULT_MELT_MASS_KG = 1.0

# Trouton's rule: entropy of vaporisation at the normal boiling point is ~85-88
# J/(mol K) for most non-associating liquids. Used only when no Antoine data and
# no measured heat of vaporisation are available; labelled 'estimated'.
TROUTON_DSVAP = 88.0  # J/(mol K)


def vapor_pressure(record, T_K):
    """Saturated vapour pressure of one additive at T_K (Kelvin), as a cite record.

    record may carry:
      antoine = {A, B, C, tmin_K, tmax_K, source}  -> log10(P/bar) = A - B/(T + C)
      normal_boiling_point_K + dhvap_kj_mol        -> Clausius-Clapeyron from P0 = 1 atm
      (dhvap_kj_mol omitted -> estimated by Trouton's rule from the boiling point)

    Returns cite.value(p_bar, 'bar', 'estimated', method=..., source=...).
    Raises ValueError when an Antoine temperature is outside its stated range.
    """
    if not isinstance(T_K, (int, float)) or not math.isfinite(T_K) or T_K <= 0.0:
        raise ValueError('T_K must be a positive finite Kelvin temperature.')
    if 'antoine' in record and record['antoine']:
        a = record['antoine']
        tmin, tmax = a.get('tmin_K'), a.get('tmax_K')
        if (tmin is not None and T_K < tmin) or (tmax is not None and T_K > tmax):
            raise ValueError('Antoine range [%s, %s] K does not cover %g K'
                             % (tmin, tmax, T_K))
        p_bar = 10.0 ** (a['A'] - a['B'] / (T_K + a['C']))
        return cite.value(p_bar, 'bar', 'estimated',
                          method='Antoine log10(P/bar) = A - B/(T+C), T in K',
                          source=a.get('source'), note=a.get('reference'))
    tbp = record.get('normal_boiling_point_K')
    if tbp is None:
        tbp = _bp_k_from_record(record)
    if tbp is None:
        raise ValueError('record needs antoine or normal_boiling_point_K / boiling_point_c')
    dhvap = record.get('dhvap_kj_mol')
    if dhvap is None:
        # Trouton's rule estimate: labelled, not a measured value.
        dhvap = TROUTON_DSVAP * tbp / 1000.0  # kJ/mol
        method = 'Clausius-Clapeyron from normal boiling point; dHvap estimated by Trouton rule'
    else:
        method = 'Clausius-Clapeyron from normal boiling point + dHvap'
    p_bar = 1.01325 * math.exp(-dhvap * 1000.0 / R * (1.0 / T_K - 1.0 / tbp))
    return cite.value(p_bar, 'bar', 'estimated', method=method,
                      source=record.get('source'))


def _bp_k_from_record(record):
    """Normal boiling point in Kelvin from a library record's boiling_point_c cite field."""
    bp = record.get('boiling_point_c')
    if isinstance(bp, dict) and bp.get('value') is not None:
        return float(bp['value']) + 273.15
    return None


def loss_fraction(k_m_s, resin_mn_g_mol, p_sat_bar, specific_surface_m2_kg, residence_s, T_K):
    """Evaporation-limited upper-bound mass fraction of one additive lost through the vent.

    Model and assumptions (see screen()): the additive is a dilute ideal solution in
    the resin (Raoult), the vent sweeps the gas phase so the bulk additive partial
    pressure is ~0, and the only resistance is the gas-side film (coefficient k_m).
    In that limit the additive's own loading and molar mass cancel and

        f = k_m * M_resin * P_sat * (A/m_melt) * t / (R * T)

    with M_resin in kg/mol, P_sat in Pa. The result is an upper bound: a partially
    loaded vent or a thicker film raises the resistance and lowers the loss.
    """
    m_resin_kg_mol = resin_mn_g_mol / 1000.0
    p_sat_pa = p_sat_bar * 1.0e5
    f = k_m_s * m_resin_kg_mol * p_sat_pa * specific_surface_m2_kg * residence_s / (R * T_K)
    return min(1.0, f)  # a mass fraction cannot exceed 1


def screen(args):
    """Estimate additive vent loss and volatility flags for one compounding run.

    args keys (all optional except melt_temp_c, residence_s, vent_pressure_mbar,
    additives):
      melt_temp_c, residence_s, vent_pressure_mbar  -- process
      resin_mn_g_mol (default 20000)                -- melt Mn for the Raoult dilution
      surface_area_m2 (default 0.005), melt_mass_kg (default 1.0)  -- vent geometry
      film_mass_transfer_m_s (default 0.01)         -- gas-side film coefficient
      additives -- list of library records (each may carry boiling_point_c, antoine,
                   tga_1pct_loss_c, tga_5pct_loss_c, flash_point_c)

    Returns a dict: 'model' (assumption text), 'sensitivity' (loss range at the
    low/high film coefficients), and 'additives' with one entry per additive giving
    vapour pressure, estimated loss fraction and the flash/TGA flags.
    """
    melt_c = args['melt_temp_c']
    T_K = melt_c + 273.15
    t_s = args.get('residence_s', 180.0)
    vent_mbar = args.get('vent_pressure_mbar', 100.0)
    vent_bar = vent_mbar / 1000.0
    mn = args.get('resin_mn_g_mol', 20000.0)
    k_m = args.get('film_mass_transfer_m_s', DEFAULT_K_M_S)
    a_over_m = args.get('surface_area_m2', DEFAULT_SURFACE_AREA_M2) / args.get('melt_mass_kg', DEFAULT_MELT_MASS_KG)

    model = ('Evaporation-limited upper bound: ideal Raoult dilution of each additive at its mole '
             'fraction, gas-side film mass transfer k_m = %g m/s, swept vent (bulk partial pressure ~0), '
             'specific surface A/m = %g m^2/kg, resin Mn %g g/mol. The additive loading and molar mass '
             'cancel in the dilute limit, so the loss depends on P_sat, k_m, A/m, t and the resin Mn. '
             'A partially loaded vent lowers the loss below this estimate.'
             % (k_m, a_over_m, mn))

    out = []
    for rec in args.get('additives', []):
        entry = {'id': rec.get('id'), 'name': rec.get('name')}
        flags = []
        # Volatile data first: vapour pressure at the melt temperature, if the record
        # has any volatility input (Antoine or boiling point).
        try:
            p_sat = vapor_pressure(rec, T_K)
            entry['vapor_pressure_bar'] = p_sat
            pv = p_sat['value']
        except ValueError:
            # Polymeric / ionic additives have no boiling point: treat as non-volatile.
            pv = None
            entry['vapor_pressure_bar'] = cite.value(None, 'bar', 'estimated',
                                                     method='no volatility data (polymeric/ionic): treated as non-volatile',
                                                     source=rec.get('boiling_point_c', {}).get('source') if isinstance(rec.get('boiling_point_c'), dict) else None)
            entry['loss_fraction'] = cite.value(0.0, '-', 'estimated',
                                                method='non-volatile additive: negligible vent loss')
        if pv is not None:
            f = loss_fraction(k_m, mn, pv, a_over_m, t_s, T_K)
            entry['loss_fraction'] = cite.value(f, '-', 'estimated',
                                                method='k_m * Mn * P_sat * (A/m) * t / (R*T)')
            if pv > vent_bar:
                flags.append('vapour pressure %.3g bar exceeds vent pressure %.3g bar: additive boils at the melt temperature' % (pv, vent_bar))
            elif pv > 0.01 * vent_bar:
                flags.append('vapour pressure %.3g bar is a measurable fraction of the vent pressure: expect slow evaporative loss' % pv)
        # Flash point and TGA onset are compare-against-melt flags, not loss estimates.
        fp = rec.get('flash_point_c')
        if isinstance(fp, dict) and fp.get('value') is not None and float(fp['value']) < melt_c:
            flags.append('flash point %g C is below the melt temperature %g C (flammability risk at the vent)' % (float(fp['value']), melt_c))
        for key, label in (('tga_1pct_loss_c', 'TGA 1 % loss'), ('tga_5pct_loss_c', 'TGA 5 % loss')):
            tg = rec.get(key)
            if isinstance(tg, dict) and tg.get('value') is not None:
                tgv = float(tg['value'])
                if tgv < melt_c:
                    flags.append('%s temperature %g C is below the melt temperature %g C (degradation/volatilisation risk)' % (label, tgv, melt_c))
        entry['flags'] = flags
        out.append(entry)

    # Sensitivity: re-run the volatile ones at the low and high film coefficients.
    sens = {}
    if out:
        for name, kk in (('low_k_m', K_M_SENSITIVITY[0]), ('high_k_m', K_M_SENSITIVITY[1])):
            total = 0.0
            for e in out:
                if e.get('vapor_pressure_bar', {}).get('value') is not None:
                    total += loss_fraction(kk, mn, e['vapor_pressure_bar']['value'], a_over_m, t_s, T_K)
            sens[name] = total
    return {'model': model, 'sensitivity': sens, 'additives': out}

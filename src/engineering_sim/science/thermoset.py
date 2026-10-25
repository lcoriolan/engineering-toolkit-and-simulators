# DESCRIPTION: Thermoset cure screening: stoichiometric mix ratio (phr or NCO index), Flory-Stockmayer gel conversion, DiBenedetto Tg-vs-conversion, and exotherm/pot-life/hazard flags. Standard library only.
"""Thermoset system screening.

Closed-form cure models only:
  epoxy-amine/anhydride -- phr hardener = 100 * AHEW / EEW
  polyurethane        -- phr polyol from NCO % and OH number at a given NCO index
  Flory-Stockmayer    -- gel conversion of the A groups for real functionalities fA, fB
                          and group ratio r = A/B:  alpha^2 * r * (fA-1) * (fB-1) = 1
  DiBenedetto         -- Tg(x) between Tg0 (x=0) and Tg_inf (x=1)

The model reports thermodynamics of the network (gel point, limiting Tg), never cure
kinetics: pot life, exotherm and cure schedule come from the cited library record.
"""
import math

from . import cite

# DiBenedetto parameter lambda: the ratio of segmental mobilities of the partially
# cured and fully cured network. Reported values for epoxy-amine systems cluster
# around 0.6-0.8; 0.6 is the screening default (DiBenedetto, J. Polym. Sci. B 25
# (1987) 1949). Recalled typical, not measured here.
DEFAULT_LAMBDA = 0.6
DI_BENEDETTO_REF = 'DiBenedetto, J. Polym. Sci. B 25 (1987) 1949; lambda ~0.6-0.8 typical for epoxy-amine (recalled)'


def epoxy_amine_phr(eew, ahew):
    """Parts hardener per 100 parts resin: phr = 100 * AHEW / EEW (both g/eq)."""
    return 100.0 * ahew / eew


def polyol_phr(nco_pct, oh_number, nco_index=1.0):
    """Parts polyol per 100 parts isocyanate at a given NCO index.

    NCO equivalent weight = 4202 / NCO %; OH equivalent weight = 56100 / OH number
    (mg KOH/g). phr_polyol = 100 * eq_OH / eq_NCO / NCO_index, so an index above 1
    (excess isocyanate) reduces the polyol charge.
    """
    eq_nco = 4202.0 / nco_pct
    eq_oh = 56100.0 / oh_number
    return 100.0 * eq_oh / eq_nco / nco_index


def flory_stockmayer_gel(fa, fb, r):
    """Critical conversion of the A groups at the Flory-Stockmayer gel point.

    For an A_fa + B_fb step polymerisation at group ratio r = A/B, gelation occurs
    when alpha_A * alpha_B * (fa - 1) * (fb - 1) = 1 with alpha_B = r * alpha_A, i.e.
        alpha_A = 1 / sqrt(r * (fa - 1) * (fb - 1)).
    Returns None when the system cannot gel before a reactant is exhausted.
    """
    if fa <= 1.0 or fb <= 1.0 or r <= 0.0:
        return None
    alpha = 1.0 / math.sqrt(r * (fa - 1.0) * (fb - 1.0))
    if alpha > 1.0 or r * alpha > 1.0:
        return None
    return alpha


def dibenedetto_tg(x, tg0_c, tg_inf_c, lam=DEFAULT_LAMBDA):
    """Glass transition vs extent of cure x in [0, 1] by DiBenedetto's equation.

        Tg(x) = lam * x * (Tg_inf - Tg0) / (1 - (1 - lam) * x) + Tg0

    Limits: x=0 -> Tg0, x=1 -> Tg_inf. Temperatures in Celsius.
    """
    return lam * x * (tg_inf_c - tg0_c) / (1.0 - (1.0 - lam) * x) + tg0_c


def _rec_of(record, key):
    """Numeric value of a cite-record field, or None when the field is absent/null."""
    v = record.get(key)
    if isinstance(v, dict) and v.get('value') is not None:
        return float(v['value'])
    return None


def screen(system):
    """Screen one thermoset system.

    system keys:
      system    -- id of a library record in data/thermosets.json (or pass the record
                   directly under 'record')
      ratio     -- group ratio A/B override (default: the library stoichiometric ratio)
      conversion, lam, tg0_c  -- optional DiBenedetto inputs (defaults 0.9 / 0.6 / Tg0
                                 = the resin's unmixed-glass value, 25 C when unknown)

    Returns a dict of cite records: mix ratio (phr), gel conversion, Tg at the chosen
    conversion, plus exotherm/pot-life/hazard flags from the library record.
    """
    rec = system.get('record')
    if rec is None:
        rec = _lookup(system['system'])
    kind = rec.get('kind', '')
    out = {'system': rec.get('id'), 'name': rec.get('name'), 'kind': kind}

    # --- stoichiometric mix ratio -------------------------------------------------
    if kind in ('epoxy-amine', 'epoxy-anhydride'):
        eew = _rec_of(rec, 'resin_eew_g_eq')
        ahew = _rec_of(rec, 'hardener_ahew_g_eq')
        if eew and ahew:
            phr = epoxy_amine_phr(eew, ahew)
            out['mix_ratio_phr'] = cite.value(phr, 'phr hardener', 'derived',
                                              method='phr = 100 * AHEW / EEW')
        fa, fb = _rec_of(rec, 'functionality_resin'), _rec_of(rec, 'functionality_hardener')
    elif kind == 'polyurethane':
        nco = _rec_of(rec, 'isocyanate_nco_pct')
        oh = _rec_of(rec, 'polyol_oh_number')
        idx = system.get('nco_index', _rec_of(rec, 'stoichiometric_ratio') or 1.0)
        if nco and oh:
            phr = polyol_phr(nco, oh, idx)
            out['mix_ratio_phr'] = cite.value(phr, 'phr polyol', 'derived',
                                              method='phr = 100 * (56100/OH) / (4202/NCO%) / index')
        fa, fb = _rec_of(rec, 'functionality_resin'), _rec_of(rec, 'functionality_hardener')
        out['nco_index'] = cite.value(idx, '-', 'derived', method='NCO index = NCO equivalents / OH equivalents')
    else:
        fa, fb = _rec_of(rec, 'functionality_resin'), _rec_of(rec, 'functionality_hardener')

    # --- gel point ----------------------------------------------------------------
    ratio = system.get('ratio', _rec_of(rec, 'stoichiometric_ratio') or 1.0)
    if fa and fb:
        gel = flory_stockmayer_gel(fa, fb, ratio)
        out['gel_conversion'] = (cite.value(gel, '-', 'derived',
                                            method='Flory-Stockmayer alpha = 1/sqrt(r*(fA-1)*(fB-1))')
                                 if gel is not None else
                                 cite.value(None, '-', 'derived',
                                            method='Flory-Stockmayer: cannot gel before a reactant is exhausted'))

    # --- Tg vs conversion (DiBenedetto) -------------------------------------------
    tg_inf = _rec_of(rec, 'tg_inf_c')
    if tg_inf is not None:
        tg0 = system.get('tg0_c', 25.0)
        x = system.get('conversion', 0.9)
        lam = system.get('lam', DEFAULT_LAMBDA)
        tg = dibenedetto_tg(x, tg0, tg_inf, lam)
        out['tg_at_conversion'] = cite.value(tg, 'degC', 'estimated',
                                             method='DiBenedetto Tg(x) = lam*x*(Tg_inf-Tg0)/(1-(1-lam)*x) + Tg0',
                                             source=DI_BENEDETTO_REF)
        out['tg0_c'] = cite.value(tg0, 'degC', 'user', note='unreacted mixture glass transition (user/default)')
        out['tg_inf_c'] = cite.value(tg_inf, 'degC', 'unverified', source=rec.get('tg_inf_c', {}).get('source', 'library (recalled)'))

    # --- flags and hazards from the library ---------------------------------------
    flags = []
    pot = rec.get('pot_life')
    if isinstance(pot, dict) and pot.get('value'):
        flags.append('pot life: ' + str(pot['value']))
    exo = rec.get('exotherm')
    if isinstance(exo, dict) and exo.get('value'):
        flags.append('exotherm: ' + str(exo['value']))
    out['flags'] = flags
    out['hazards'] = rec.get('hazards', [])
    return out


def _lookup(system_id):
    """Load one thermoset record from the data library by id (stdlib json)."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent / 'data' / 'thermosets.json'
    records = json.loads(path.read_text())['records']
    for r in records:
        if r['id'] == system_id:
            return r
    raise KeyError('unknown thermoset system: ' + str(system_id))

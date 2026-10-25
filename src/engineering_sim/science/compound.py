# DESCRIPTION: General compounding engine: cited resin/additive/filler libraries, mass balance, reactive stoichiometry and mean-field gel point, polyester/polyamide hydrolysis, Halpin-Tsai modulus and Kelly-Tyson strength, HDT mechanism screen, conflicts and antagonisms, and additive volatility loss. Standard library only.
"""General compounding screen.

Generalises engineering_sim/science/rpet.py (which this package now owns) from recycled PET to
any library resin. For PET the engine reuses rpet's exact primitives (hydrolysis,
gel_critical_conversion, composite_strength) with the same constants, so
compound.screen() reproduces rpet.screen() for matching caller-supplied PET inputs.

Screening models only: closed-form mass balance, stoichiometry, mean-field gel point,
Kelly-Tyson short-fibre strength and Halpin-Tsai modulus. Nothing here is a
measurement; DSC, IV/COOH titration, rheometry, ISO 527 and ISO 75 are the evidence.
Every number returned is a cite value record (engineering_sim.science.cite).
"""
import json
import math
from pathlib import Path

from . import cite
from . import rpet
from . import volatility

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'

# rpet engine constants shared verbatim (see engineering_sim/science/rpet.py CONSTANTS).
M_WATER = 18.015          # g/mol
ETA0_MW_EXPONENT = 3.4    # -


def library(kind):
    """Records for one library: 'resins' | 'additives' | 'fillers' | 'thermosets'."""
    if kind not in ('resins', 'additives', 'fillers', 'thermosets'):
        raise ValueError('unknown library kind: ' + str(kind))
    path = DATA / (kind + '.json')
    return json.loads(path.read_text())['records']


def _by_id(kind, rid):
    for rec in library(kind):
        if rec['id'] == rid:
            return rec
    raise KeyError('unknown %s id: %s' % (kind, rid))


def _num(x):
    """Plain number from a cite record or bare number."""
    return cite.num(x)


# ---------------------------------------------------------------- hydrolysis (generalised)

def hydrolysis(iv0, moisture_ppm, reacted_fraction, cooh0, K, a):
    """One scission per reacted water for a condensation resin with Mark-Houwink K, a.

    Mirrors rpet.hydrolysis so that with the PET constants (K=2.1e-4, a=0.82,
    M_water 18.015) the result is bit-identical. 1/Mn = 1/Mn0 + scissions/M_water,
    scissions = moisture_ppm * reacted_fraction / M_water in mmol/kg.
    """
    mn0 = (iv0 / K) ** (1.0 / a)
    scissions = moisture_ppm * reacted_fraction / M_WATER
    mn = 1.0 / (1.0 / mn0 + scissions * 1e-6)
    iv = K * mn ** a
    out = dict(mn0=mn0, mn=mn, iv=iv, iv_loss=iv0 - iv, scissions_mmol_kg=scissions,
               eta0_ratio=(mn / mn0) ** ETA0_MW_EXPONENT)
    if cooh0 is not None:
        out['cooh_mmol_kg'] = cooh0 + scissions
    return out


def halpin_tsai_longitudinal(em_gpa, ef_gpa, vf, aspect):
    """Halpin-Tsai longitudinal modulus for aligned short fibres, E in GPa.

    E_c = E_m (1 + xi*eta*vf) / (1 - eta*vf),  xi = 2*(l/d),  eta = (Ef/Em - 1)/(Ef/Em + xi).
    For in-plane random fibres the laminate average (3/8 E_L + 5/8 E_T) is lower;
    this returns the aligned upper reference.
    """
    xi = 2.0 * aspect
    eta = (ef_gpa / em_gpa - 1.0) / (ef_gpa / em_gpa + xi)
    return em_gpa * (1.0 + xi * eta * vf) / (1.0 - eta * vf)


# ---------------------------------------------------------------- clarity / morphology

def _is_transparent(resin):
    """Amorphous-transparent capability from the clarity text (crude but deterministic)."""
    txt = str(_num(resin['clarity'])).lower()
    return 'transparent' in txt and 'opaque' not in txt


def _is_semicrystalline(resin):
    return _num(resin['tm_c']) is not None


def _hdt_ceiling_c(resin, reinforced):
    """HDT-A (1.8 MPa) ceiling: Tg-limited for amorphous, ~Tm-20 C for reinforced semicrystalline.

    Unfilled semicrystalline HDT-A is the datasheet value; 30 % glass pushes it toward
    but not past Tm (empirical ceiling Tm - 20 C)."""
    if not _is_semicrystalline(resin):
        return _num(resin['tg_c'])  # amorphous: HDT-A tracks Tg
    if reinforced:
        return _num(resin['tm_c']) - 20.0
    return _num(resin['hdt_1p8_mpa_c'])


# ---------------------------------------------------------------- the screen

def screen(formulation):
    """Screen one formulation; every number returned is a cite value record.

    formulation keys:
      resin        -- resins.json id
      components   -- [{"id", "wt"} ...] additive loadings in wt%
      fillers      -- [{"id", "wt", "length_mm", "ifss_mpa"} ...]
      feedstock    -- {moisture_ppm, cooh_mmol_kg, iv_dl_g, water_reacted_fraction,
                       ipa_mol_pct, deg_mol_pct, ...}
      process      -- {melt_temp_c, residence_s, mold_temp_c, vent_mbar, epoxy_conversion}
      targets      -- {tensile_mpa, hdt_c, clarity, food_contact, process}
      composite    -- optional {orientation_factor, matrix_stress_mpa} (defaults 0.4, 55)
    """
    resin = _by_id('resins', formulation['resin'])
    components = [dict(record=_by_id('additives', c['id']), wt=float(c['wt'])) for c in formulation.get('components', [])]
    fillers = [dict(record=_by_id('fillers', f['id']), wt=float(f['wt']),
                    length_mm=float(f.get('length_mm', 0.3)), ifss_mpa=float(f.get('ifss_mpa', 25.0)))
               for f in formulation.get('fillers', [])]
    feed = formulation.get('feedstock', {})
    proc = formulation.get('process', {})
    targets = formulation.get('targets', {})
    comp = formulation.get('composite', {})
    orientation = float(comp.get('orientation_factor', 0.4))
    matrix_stress = float(comp.get('matrix_stress_mpa', 55.0))

    out = {'resin': dict(id=resin['id'], name=resin['name'], class_=resin['class']),
           'components': [dict(id=c['record']['id'], name=c['record']['name'], wt=c['wt']) for c in components],
           'fillers': [dict(id=f['record']['id'], name=f['record']['name'], wt=f['wt'],
                            length_mm=f['length_mm'], ifss_mpa=f['ifss_mpa']) for f in fillers],
           'process': dict(proc)}

    # ---- mass balance ------------------------------------------------------------
    add_wt = sum(c['wt'] for c in components)
    fill_wt = sum(f['wt'] for f in fillers)
    resin_wt = 100.0 - add_wt - fill_wt
    out['mass_balance'] = {
        'resin_wt_pct': cite.value(resin_wt, 'wt%', 'derived', method='100 - additives - fillers'),
        'additives_wt_pct': cite.value(add_wt, 'wt%', 'derived', method='sum of component loadings'),
        'fillers_wt_pct': cite.value(fill_wt, 'wt%', 'derived', method='sum of filler loadings'),
        'total_wt_pct': cite.value(resin_wt + add_wt + fill_wt, 'wt%', 'derived', method='resin + additives + fillers'),
    }

    findings = []

    # ---- hydrolysis (condensation polyesters/polyamides) --------------------------
    melt = None
    K, a = _num(resin['mark_houwink_K']), _num(resin['mark_houwink_a'])
    if K is not None and a is not None and feed.get('iv_dl_g') is not None:
        h = hydrolysis(float(feed['iv_dl_g']), float(feed.get('moisture_ppm', 0.0)),
                       float(feed.get('water_reacted_fraction', 1.0)), feed.get('cooh_mmol_kg'), K, a)
        melt = h
        out['melt'] = {
            'iv_dl_g': cite.value(h['iv'], 'dL/g', 'derived', method='Mark-Houwink after hydrolysis'),
            'mn_g_mol': cite.value(h['mn'], 'g/mol', 'derived', method='1/Mn = 1/Mn0 + scissions/M_water'),
            'iv_loss_dl_g': cite.value(h['iv_loss'], 'dL/g', 'derived', method='IV0 - IV'),
            'cooh_mmol_kg': cite.value(h.get('cooh_mmol_kg', 0.0), 'mmol/kg', 'derived', method='COOH0 + water scissions') if h.get('cooh_mmol_kg') is not None else None,
        }

    # ---- reactive stoichiometry and mean-field gel point --------------------------
    chain_ends = 2.0e6 / melt['mn'] if melt else None
    epoxy_total = 0.0
    epoxy_fw_weighted = 0.0
    stoich = []
    for c in components:
        rec = c['record']
        rgroup = str(_num(rec['reactive_group'])).lower()
        eew = _num(rec['equivalent_weight_g_mol'])
        func = _num(rec['functionality'])
        if eew is None:
            continue  # non-reactive additive
        # Reactive group millimoles per kg of compound: wt% -> g/kg, / (g/mol) -> mol/kg -> mmol/kg.
        group_mmol_kg = c['wt'] * 10.0 / eew * 1000.0
        if rgroup == 'epoxy':
            epoxy_total += group_mmol_kg
            if func is not None:
                epoxy_fw_weighted += group_mmol_kg * func
        entry = {'component': rec['id'], 'name': rec['name'], 'reactive_group': rec['reactive_group'],
                 'group_mmol_kg': cite.value(group_mmol_kg, 'mmol/kg', 'derived', method='wt% * 10 / EEW * 1000'),
                 'functionality': rec['functionality']}
        if func is not None and func > 2.0 and melt is not None:
            # Multifunctional additive vs the resin end groups: mean-field gel point.
            cooh = melt['cooh_mmol_kg'] if melt.get('cooh_mmol_kg') is not None else float(feed.get('cooh_mmol_kg', 0.0))
            p = rpet.cooh_end_fraction(cooh, melt['mn'])
            alpha_c = rpet.gel_critical_conversion(group_mmol_kg, cooh, func, p)
            conv = float(proc.get('epoxy_conversion', 0.5))
            entry['gel_critical_conversion'] = (cite.value(alpha_c, '-', 'derived',
                                                           method='mean-field gel point (Macosko-Miller)') if alpha_c is not None
                                                else cite.value(None, '-', 'derived', method='mean-field gel point: cannot gel'))
            entry['gel_predicted'] = cite.value(bool(alpha_c is not None and conv >= alpha_c), '-', 'derived',
                                                method='expected conversion >= critical conversion')
        stoich.append(entry)

    # epoxy-weighted average functionality for the combined epoxy-COOH gel criterion.
    fw_eff = (epoxy_fw_weighted / epoxy_total) if epoxy_total > 0 else None
    if melt is not None and epoxy_total > 0 and fw_eff and fw_eff > 1:
        cooh = melt['cooh_mmol_kg'] if melt.get('cooh_mmol_kg') is not None else float(feed.get('cooh_mmol_kg', 0.0))
        p = rpet.cooh_end_fraction(cooh, melt['mn'])
        alpha_c = rpet.gel_critical_conversion(epoxy_total, cooh, fw_eff, p)
        conv = float(proc.get('epoxy_conversion', 0.5))
        out['gel'] = {
            'epoxy_mmol_kg': cite.value(epoxy_total, 'mmol/kg', 'derived', method='sum of epoxy additives'),
            'cooh_mmol_kg': cite.value(cooh, 'mmol/kg', 'derived', method='COOH after melt hydrolysis'),
            'epoxy_to_cooh': cite.value(epoxy_total / cooh if cooh else math.inf, '-', 'derived', method='epoxy / COOH'),
            'functionality_weight_avg': cite.value(fw_eff, '-', 'derived', method='epoxy-loading-weighted average functionality'),
            'gel_critical_conversion': (cite.value(alpha_c, '-', 'derived', method='mean-field gel point (Macosko-Miller)') if alpha_c is not None
                                        else cite.value(None, '-', 'derived', method='mean-field gel point: cannot gel')),
            'gel_predicted': cite.value(bool(alpha_c is not None and conv >= alpha_c), '-', 'derived',
                                        method='expected conversion >= critical conversion'),
        }
        if alpha_c is not None and conv >= alpha_c:
            findings.append('Gel predicted: expected conversion %.2f >= mean-field critical %.2f.' % (conv, alpha_c))
    out['stoichiometry'] = stoich

    # ---- composite (Kelly-Tyson strength, Halpin-Tsai modulus) --------------------
    rho_m = float(_num(resin['density_g_cm3']))
    em = float(_num(resin['modulus_gpa']))
    per_filler = []
    fibre_term_sum = 0.0
    vf_total = 0.0
    mod_num = 0.0
    mod_den = 0.0
    for f in fillers:
        rec = f['record']
        rho_f = float(_num(rec['density_g_cm3']))
        ef = float(_num(rec['modulus_gpa']))
        strength, diameter = _num(rec.get('strength_mpa')), _num(rec.get('diameter_um'))
        if strength is None or diameter is None:
            # Filler geometry: platelets, acicular minerals and beads (talc, mica,
            # wollastonite, CaCO3, glass beads) carry no fibre strength/diameter, and float(None) aborted the
            # whole screen. They stiffen but do not transfer fibre load: strength keeps only the (1 − Vf) matrix
            # term (conservative), and modulus uses Halpin–Tsai with the record's aspect ratio (1 when absent).
            vf = (f['wt'] / rho_f) / (f['wt'] / rho_f + (100.0 - f['wt']) / rho_m)
            aspect = float(_num(rec.get('aspect_ratio')) or 1.0)
            ht = halpin_tsai_longitudinal(em, ef, vf, aspect)
            per_filler.append({
                'id': rec['id'], 'name': rec['name'],
                'fibre_volume_fraction': cite.value(vf, '-', 'derived', method='wt%/rho_f over the sum'),
                'critical_length_mm': None, 'length_factor': None,
                'strength_mpa': cite.value((1.0 - vf) * matrix_stress, 'MPa', 'estimated',
                                           method='particulate/platelet filler: matrix term only (no fibre load transfer)'),
                'halpin_tsai_modulus_gpa': cite.value(ht, 'GPa', 'estimated', method='Halpin-Tsai, xi = 2 x record aspect ratio'),
            })
            vf_total += vf
            mod_num += vf * ef
            mod_den += vf
            continue
        sigma_f, d_um = float(strength), float(diameter)
        kt = rpet.composite_strength(f['wt'], sigma_f, d_um, f['ifss_mpa'], f['length_mm'], orientation, matrix_stress,
                                     rho_f=rho_f, rho_m=rho_m)
        aspect = f['length_mm'] * 1000.0 / d_um  # mm -> um / um
        ht = halpin_tsai_longitudinal(em, ef, kt['vf'], aspect)
        per_filler.append({
            'id': rec['id'], 'name': rec['name'],
            'fibre_volume_fraction': cite.value(kt['vf'], '-', 'derived', method='wt%/rho_f over the sum'),
            'critical_length_mm': cite.value(kt['lc_mm'], 'mm', 'derived', method='Kelly-Tyson lc = sigma_f d / (2 tau)'),
            'length_factor': cite.value(kt['eta_l'], '-', 'derived', method='Kelly-Tyson length efficiency'),
            'strength_mpa': cite.value(kt['strength'], 'MPa', 'estimated', method='modified rule of mixtures (Kelly-Tyson)'),
            'halpin_tsai_modulus_gpa': cite.value(ht, 'GPa', 'estimated', method='Halpin-Tsai longitudinal, xi = 2(l/d)'),
        })
        fibre_term_sum += kt['fibre_term']
        vf_total += kt['vf']
        mod_num += kt['vf'] * ef
        mod_den += kt['vf']
    combined_strength = (fibre_term_sum + (1.0 - vf_total) * matrix_stress) if fillers else matrix_stress
    combined_modulus = halpin_tsai_longitudinal(em, mod_num / mod_den, vf_total, 30.0) if fillers else em
    out['composite'] = {
        'fillers': per_filler,
        'combined_strength_mpa': cite.value(combined_strength, 'MPa', 'estimated', method='sum of fibre terms + (1-Vf) matrix stress'),
        'combined_modulus_gpa': cite.value(combined_modulus, 'GPa', 'estimated', method='Halpin-Tsai, volume-weighted filler modulus'),
    }
    if fillers and any(f['wt'] > 0 and per_filler[i]['length_factor'] and per_filler[i]['length_factor']['value'] < 0.5 for i, f in enumerate(fillers)):
        findings.append('Fibres are near or below l_c: strength depends on retained length and sizing.')

    # ---- HDT mechanism ------------------------------------------------------------
    mold_c = float(proc.get('mold_temp_c', 25.0))
    hdt_target = targets.get('hdt_c')
    semicryst = _is_semicrystalline(resin)
    reinforced = fill_wt >= 20.0
    crystallizable = semicryst and mold_c >= float(_num(resin['tg_c'])) + 50.0
    if semicryst:
        ceiling = _hdt_ceiling_c(resin, reinforced)
        route = crystallizable and reinforced
        mech = 'crystallised semicrystalline matrix + reinforcement' if route else \
               ('semicrystalline, %s; crystallisation needs a hotter mould' % ('reinforced' if reinforced else 'unfilled'))
    else:
        ceiling = _hdt_ceiling_c(resin, False)
        route = False
        mech = 'amorphous: HDT-A tracks Tg (no crystallisation route)'
    out['hdt'] = {
        'mechanism': cite.value(mech, '-', 'derived', method='morphology + mould temperature + filler loading'),
        'ceiling_c': cite.value(ceiling, 'degC', 'derived', method='amorphous: Tg; semicrystalline reinforced: Tm - 20 C'),
        'route_available': cite.value(route, '-', 'derived', method='crystallisable mould and >= 20 wt% reinforcement'),
    }
    if hdt_target is not None and hdt_target > ceiling:
        findings.append('HDT target %.0f C exceeds the %.0f C ceiling (%s).' % (hdt_target, ceiling, mech))

    # ---- clarity / food-contact / process conflicts ------------------------------
    if targets.get('clarity'):
        if fill_wt > 0:
            findings.append('Clarity conflicts with the filler/reinforcement loading.')
        if not _is_transparent(resin):
            findings.append('Clarity conflicts with the resin morphology (%s).' % _num(resin['clarity']))
        if semicryst and crystallizable:
            findings.append('Clarity conflicts with thermal crystallisation of a semicrystalline resin.')
    if targets.get('food_contact'):
        bad = [c['record']['name'] for c in components if 'no' in str(_num(c['record']['food_contact'])).lower()
               and 'yes' not in str(_num(c['record']['food_contact'])).lower()]
        if bad:
            findings.append('Food contact: confirm regulatory status of ' + ', '.join(bad) + '.')
        if 'yes' not in str(_num(resin['food_contact'])).lower():
            findings.append('Food contact: resin grade not clearly food-contact listed.')
    melt_temp = proc.get('melt_temp_c')
    if melt_temp is not None:
        lo, hi = float(_num(resin['melt_temp_min_c'])), float(_num(resin['melt_temp_max_c']))
        if not (lo <= melt_temp <= hi):
            findings.append('Melt temperature %.0f C is outside the resin window %.0f-%.0f C.' % (melt_temp, lo, hi))

    # ---- antagonisms ---------------------------------------------------------------
    present = {c['record']['id'] for c in components}
    findings += _antagonism_findings(present)

    # ---- volatility ---------------------------------------------------------------
    vol = None
    if components and melt_temp is not None:
        vol = volatility.screen({'melt_temp_c': melt_temp,
                                 'residence_s': proc.get('residence_s', 180.0),
                                 'vent_pressure_mbar': proc.get('vent_mbar', 100.0),
                                 'resin_mn_g_mol': melt['mn'] if melt else 20000.0,
                                 'additives': [c['record'] for c in components]})
        out['volatility'] = vol

    if melt is not None and melt['iv'] < 0.55:
        findings.append('Melt IV %.2f dL/g is below the ~0.55 embrittlement screen.' % melt['iv'])

    out['findings'] = findings
    out['ceiling'] = ('HDT-A ceiling %.0f C: %s' % (ceiling, mech))
    return out


# Documented antagonism pairs mirrored from the library 'antagonisms' fields.
# id set (triggers) -> id set (victims) -> finding text. Deterministic and
# overrides nothing: the full library text remains the source of record.
_ANTAGONISM_PAIRS = (
    (frozenset(('sodium_benzoate', 'sodium_montanate')),
     frozenset(('chain_extender_joncryl_adr4468', 'ema_gma')),
     'Sodium carboxylate nucleator catalyses epoxy ring-opening (raises gel risk).'),
    (frozenset(('dopo', 'phosphite_irgafos168', 'ao_triphenyl_phosphite')),
     frozenset(('chain_extender_joncryl_adr4468', 'ema_gma')),
     'Acidic phosphorus species consume epoxy groups (chain extender partially quenched).'),
)


def _antagonism_findings(present_ids):
    """Findings for any documented antagonism whose trigger and victim are both present."""
    out = []
    for triggers, victims, text in _ANTAGONISM_PAIRS:
        if triggers & present_ids and victims & present_ids:
            out.append(text)
    return out


# ---------------------------------------------------------------- alternatives

_GF_STANDARD = {'id': 'glass_e_short', 'wt': 30.0, 'length_mm': 0.3, 'ifss_mpa': 25.0}


def alternatives(targets):
    """Rank resins (unfilled and 30 % glass) against targets using library ceilings.

    Returns a dict: 'passing' (resins meeting every target), 'results' (one entry per
    resin with per-target verdicts and a reason), and 'summary'. The reason for a
    rejected resin names the governing limit (Tg, Tm/crystallinity, clarity, tensile).
    """
    hdt_target = targets.get('hdt_c')
    tensile_target = targets.get('tensile_mpa')
    clarity_target = bool(targets.get('clarity'))
    food_target = bool(targets.get('food_contact'))

    results = []
    passing = []
    for resin in library('resins'):
        semicryst = _is_semicrystalline(resin)
        transparent = _is_transparent(resin)
        # Clarity forbids reinforcement; HDT above ~Tg of an amorphous resin forbids
        # the amorphous route, so the two ceilings are mutually exclusive in practice.
        reinforced = not clarity_target
        hdt_ceiling = _hdt_ceiling_c(resin, reinforced)
        meets_hdt = hdt_target is None or hdt_ceiling >= hdt_target
        meets_clarity = (not clarity_target) or transparent
        # Tensile ceiling: datasheet yield, or a GF30 Kelly-Tyson estimate for the
        # semicrystalline resins when reinforcement is allowed.
        sigma_m = float(_num(resin['tensile_yield_mpa']))
        if reinforced and not clarity_target:
            glass = _by_id('fillers', 'glass_e_short')
            kt = rpet.composite_strength(30.0, float(_num(glass['strength_mpa'])), float(_num(glass['diameter_um'])),
                                         25.0, 0.3, 0.4, sigma_m,
                                         rho_f=float(_num(glass['density_g_cm3'])), rho_m=float(_num(resin['density_g_cm3'])))
            tens_ceiling = kt['strength']
        else:
            tens_ceiling = sigma_m
        meets_tensile = tensile_target is None or tens_ceiling >= tensile_target
        meets_food = (not food_target) or 'yes' in str(_num(resin['food_contact'])).lower()

        reasons = []
        if not meets_hdt:
            if not semicryst:
                reasons.append('amorphous: HDT-A ceiling ~Tg %.0f C < %.0f C (Tg-limited)' % (_num(resin['tg_c']), hdt_target))
            elif reinforced:
                reasons.append('semicrystalline: reinforced HDT-A ceiling ~Tm-20 = %.0f C < %.0f C' % (hdt_ceiling, hdt_target))
            else:
                reasons.append('semicrystalline: unfilled HDT-A %.0f C < %.0f C (reinforcement forbidden by clarity)' % (hdt_ceiling, hdt_target))
        if not meets_clarity:
            reasons.append('semicrystalline/opaque: no clarity (%s)' % _num(resin['clarity']))
        if not meets_tensile:
            reasons.append('tensile ceiling %.0f MPa < %.0f MPa' % (tens_ceiling, tensile_target))
        if not meets_food:
            reasons.append('food-contact grade not confirmed')

        ok = meets_hdt and meets_clarity and meets_tensile and meets_food
        entry = dict(id=resin['id'], name=resin['name'], meets_all=ok,
                     hdt_ceiling_c=cite.value(hdt_ceiling, 'degC', 'derived', method='amorphous: Tg; semicrystalline reinforced: Tm - 20 C'),
                     clarity=resin['clarity'], semicrystalline=semicryst,
                     reason='; '.join(reasons) if reasons else 'meets all targets')
        results.append(entry)
        if ok:
            passing.append(entry)

    # Stable order: passing first (by HDT ceiling desc), then the rest by id.
    results.sort(key=lambda r: (not r['meets_all'], -r['hdt_ceiling_c']['value'], r['id']))
    summary = _alternatives_summary(results, targets)
    return dict(targets=targets, passing=[r['id'] for r in results if r['meets_all']], results=results, summary=summary)


def _alternatives_summary(results, targets):
    """One-line verdict; names Tg and crystallinity when that is the governing conflict."""
    hdt_target = targets.get('hdt_c')
    clarity = bool(targets.get('clarity'))
    n_ok = sum(1 for r in results if r['meets_all'])
    if n_ok:
        return '%d resin(s) meet every target.' % n_ok
    if clarity and hdt_target is not None:
        return ('No resin meets both clarity and HDT %g C at 1.8 MPa: clarity needs an amorphous '
                'transparent matrix, whose Tg-limited HDT stays below ~150 C, while HDT %g C needs '
                'a semicrystalline high-Tm matrix (PPS/PEEK) whose crystallinity makes it opaque.'
                % (hdt_target, hdt_target))
    return 'No resin meets every target; see the per-resin reasons.'

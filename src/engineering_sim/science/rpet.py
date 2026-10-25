# DESCRIPTION: Recycled PET chemistry: hydrolysis stoichiometry, Mark–Houwink IV, chain-extender stoichiometry and gel point, short-fibre strength, ISO 75 HDT criterion, copolymer Tm, formulation balance and claim validation. Kernel shared by compound.py.
"""Recycled PET compounding screen.

Units: molecular weight g/mol, IV dL/g, end groups mmol/kg, loadings wt%, stresses MPa,
lengths mm unless a key says otherwise. Pure standard library; deterministic.

Scope: these are closed-form screening models. They check arithmetic and internal
consistency of a formulation argument; they do not replace DSC, IV, COOH titration,
rheometry, ISO 527 or ISO 75 measurements. Datasheet-type constants below are
recalled typical values and must be confirmed against the current supplier sheet.
"""
import math
import re

# name: (value, unit, provenance). 'derived' = computed here; 'typical' = literature/datasheet
# range recalled, not verified in this workspace; 'assumed' = screening choice.
CONSTANTS = {
    'M_water': (18.015, 'g/mol', 'derived: IUPAC atomic masses'),
    'MH_K_Mn': (2.1e-4, 'dL/g', 'typical: PET in phenol/TCE 60/40, [η]=K·Mn^a; confirm the solvent basis of your IV lab'),
    'MH_a_Mn': (0.82, '-', 'typical: as MH_K_Mn'),
    'MH_K_Mw_alt': (7.44e-4, 'dL/g', 'typical: alternative PET set on Mw, [η]=K·Mw^a (sensitivity only)'),
    'MH_a_Mw_alt': (0.648, '-', 'typical: as MH_K_Mw_alt'),
    'eta0_Mw_exponent': (3.4, '-', 'typical: entangled linear melts above Mc'),
    'rho_glass': (2.55, 'g/cm3', 'typical: E-glass'),
    'rho_pet_cryst': (1.39, 'g/cm3', 'typical: PET at ~30–35 % crystallinity'),
    'joncryl_adr4468_eew': (310.0, 'g/mol', 'typical: BASF datasheet value as recalled; confirm'),
    'joncryl_adr4468_mw': (7250.0, 'g/mol', 'typical: BASF datasheet value as recalled; confirm'),
    'dHu_pet': (26.9, 'kJ/mol repeat unit', 'typical: ~140 J/g × 192.17 g/mol'),
    'Tm_homopolymer_c': (265.0, '°C', 'typical: observed DSC peak for PET homopolymer (not equilibrium Tm0)'),
    'iso75_span_mm': (64.0, 'mm', 'ISO 75-2 method A, flatwise 80×10×4 bar'),
    'iso75_strain_pct': (0.2, '%', 'ISO 75-1 standard flexural strain increase'),
    'iso75_stress_A_mpa': (1.80, 'MPa', 'ISO 75-2 method A'),
}
C = {k: v[0] for k, v in CONSTANTS.items()}

ATOMIC = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'P': 30.974, 'Mg': 24.305, 'Si': 28.085, 'Na': 22.990}
FORMULAS = {  # species used in the formulation argument
    'water': 'H2O', 'pet_repeat_unit': 'C10H8O4', 'acetaldehyde': 'C2H4O', 'anthranilamide': 'C7H8N2O',
    'npg_dibenzoate': 'C19H20O4', 'glycidyl_methacrylate': 'C7H10O3', 'pmda': 'C10H2O6',
    'irganox_1010': 'C73H108O12', 'irgafos_168': 'C42H63O3P', 'talc': 'Mg3Si4O10(OH)2', 'sodium_benzoate': 'C7H5NaO2',
}


def molar_mass(formula):
    """Molar mass from a Hill-style formula with one level of parentheses, e.g. Mg3Si4O10(OH)2."""
    def count(text):
        total = 0.0
        for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', text):
            if el not in ATOMIC:
                raise ValueError('Unknown element: ' + el)
            total += ATOMIC[el] * (int(n) if n else 1)
        return total
    total = 0.0
    for group, n in re.findall(r'\(([^()]*)\)(\d*)', formula):
        total += count(group) * (int(n) if n else 1)
    return total + count(re.sub(r'\([^()]*\)\d*', '', formula))


# ---------------------------------------------------------------- molecular weight

def iv_from_mn(mn, K=None, a=None):
    return (K or C['MH_K_Mn']) * mn ** (a or C['MH_a_Mn'])


def mn_from_iv(iv, K=None, a=None):
    return (iv / (K or C['MH_K_Mn'])) ** (1.0 / (a or C['MH_a_Mn']))


def mn_from_iv_alt(iv, pdi=2.0):
    """Sensitivity: the alternative Mw-based constants at an assumed polydispersity."""
    return (iv / C['MH_K_Mw_alt']) ** (1.0 / C['MH_a_Mw_alt']) / pdi


def viscosity_iv_exponent():
    """η0 ∝ Mw^3.4 and [η] ∝ M^a at constant PDI give η0 ∝ IV^(3.4/a)."""
    return C['eta0_Mw_exponent'] / C['MH_a_Mn']


def chain_ends_mmol_kg(mn):
    """Linear chains carry two ends: 2/Mn mol/g."""
    return 2.0e6 / mn


def hydrolysis(iv0, moisture_ppm, reacted_fraction=1.0, cooh0=None):
    """One scission, one new COOH and one new OH per reacted water. 1/Mn = 1/Mn0 + w/M_water."""
    mn0 = mn_from_iv(iv0)
    scissions = moisture_ppm * reacted_fraction / C['M_water']          # mmol/kg (ppm = mg/kg)
    mn = 1.0 / (1.0 / mn0 + scissions * 1e-6)
    iv = iv_from_mn(mn)
    out = dict(mn0=mn0, mn=mn, iv=iv, iv_loss=iv0 - iv, scissions_mmol_kg=scissions,
               eta0_ratio=(mn / mn0) ** C['eta0_Mw_exponent'])
    if cooh0 is not None:
        out['cooh_mmol_kg'] = cooh0 + scissions
    return out


# ---------------------------------------------------------------- chain extension and gel point

def epoxy_mmol_kg(loading_wt, eew):
    """Epoxy groups per kg compound for a loading in wt%."""
    return loading_wt * 10.0 / eew * 1000.0


def cooh_end_fraction(cooh, mn):
    """Fraction p of chain ends that are carboxyl (the rest hydroxyl), bounded to [0, 1]."""
    return min(1.0, cooh / chain_ends_mmol_kg(mn))


def gel_critical_conversion(epoxy, cooh, f_w, p):
    """Mean-field (Macosko–Miller) gel point for an A_f epoxy + PET chains with 0/1/2 COOH ends.

    COOH ends distributed binomially over the two chain ends with probability p give a
    weight-average COOH functionality f_B,w = 1 + p. Gelation when
        α_A · α_B · (f_A,w − 1) · (f_B,w − 1) = 1,   α_B = r · α_A,   r = epoxy / COOH.
    Returns the critical epoxy conversion α_A (None when the system cannot gel before one
    reactant is exhausted). Ignores epoxy–OH reaction (which adds gel risk), and
    ester interchange and cyclisation (which delay or dissolve a network).
    """
    if epoxy <= 0 or cooh <= 0 or f_w <= 1 or p <= 0:
        return None
    r = epoxy / cooh
    alpha = 1.0 / math.sqrt(r * (f_w - 1.0) * p)
    if alpha > 1.0 or r * alpha > 1.0:
        return None
    return alpha


def gel_benchmark_a3_b2():
    """Flory classic: stoichiometric A3 + B2 gels at α = 1/√2 (f_B,w − 1 = 1 ⇒ p = 1)."""
    return gel_critical_conversion(1.0, 1.0, 3.0, 1.0)


def max_extender_without_gel(cooh, f_w, p, eew, conversion):
    """Largest loading (wt%) whose critical conversion stays above the expected conversion."""
    if conversion <= 0:
        return math.inf
    # α_c = 1/sqrt(r (f−1) p) ≥ conversion  ⇒  r ≤ 1 / (conversion² (f−1) p)
    r_max = 1.0 / (conversion ** 2 * (f_w - 1.0) * p)
    return r_max * cooh * eew / 1e4


# ---------------------------------------------------------------- short-fibre composite

def fibre_volume_fraction(wt, rho_f=None, rho_m=None):
    rho_f, rho_m = rho_f or C['rho_glass'], rho_m or C['rho_pet_cryst']
    vf, vm = wt / 100.0 / rho_f, (1.0 - wt / 100.0) / rho_m
    return vf / (vf + vm)


def critical_length_mm(sigma_f, d_um, ifss):
    """Kelly–Tyson l_c = σ_f d / (2 τ)."""
    return sigma_f * d_um * 1e-3 / (2.0 * ifss)


def length_factor(length_mm, lc_mm):
    """Mean fibre stress / σ_f: 1 − l_c/2l above l_c, l/2l_c below (uniform length)."""
    return 1.0 - lc_mm / (2.0 * length_mm) if length_mm >= lc_mm else length_mm / (2.0 * lc_mm)


def composite_strength(wt, sigma_f, d_um, ifss, length_mm, eta_o, sigma_m, rho_f=None, rho_m=None, eta_l=None):
    """Modified rule of mixtures σ_c = η_o η_l V_f σ_f + (1 − V_f) σ_m′."""
    vf = fibre_volume_fraction(wt, rho_f, rho_m)
    lc = critical_length_mm(sigma_f, d_um, ifss)
    el = length_factor(length_mm, lc) if eta_l is None else eta_l
    return dict(vf=vf, lc_mm=lc, eta_l=el, fibre_term=eta_o * el * vf * sigma_f, matrix_term=(1 - vf) * sigma_m,
                strength=eta_o * el * vf * sigma_f + (1 - vf) * sigma_m)


# ---------------------------------------------------------------- HDT, melting, scavenging

def iso75_deflection_mm(span=None, thickness=4.0, strain_pct=None):
    """ISO 75-1 standard deflection Δs = ε_f L² / (600 h), ε_f in %."""
    span, strain_pct = span or C['iso75_span_mm'], strain_pct or C['iso75_strain_pct']
    return strain_pct * span ** 2 / (600.0 * thickness)


def iso75_modulus_criterion_mpa(stress=None, strain_pct=None):
    """Elastic estimate of the flexural modulus at which the bar reaches Δs under load."""
    return (stress or C['iso75_stress_A_mpa']) / ((strain_pct or C['iso75_strain_pct']) / 100.0)


def flory_tm_c(comonomer_mol_pct, tm_ref_c=None, dHu=None):
    """Flory ideal copolymer melting: 1/Tm − 1/Tm_ref = −(R/ΔHu) ln X. Known to under-predict depression."""
    tm_ref = (tm_ref_c or C['Tm_homopolymer_c']) + 273.15
    x = 1.0 - comonomer_mol_pct / 100.0
    inv = 1.0 / tm_ref - 8.314e-3 / (dHu or C['dHu_pet']) * math.log(x)
    return 1.0 / inv - 273.15


def aa_capacity_ppm(scavenger_ppm, formula='C7H8N2O'):
    """Stoichiometric AA bound by a 1:1 scavenger (anthranilamide by default)."""
    return scavenger_ppm * molar_mass('C2H4O') / molar_mass(formula)


# ---------------------------------------------------------------- ISO specimens (geometry owner)

DOGBONE_1A = dict(length=170.0, grip_width=20.0, narrow_width=10.0, narrow_length=80.0, radius=24.0, thickness=4.0)
HDT_BAR = dict(length=80.0, width=10.0, thickness=4.0)


def dogbone_transition_mm(d=DOGBONE_1A):
    """Axial length of the tangent R transition from the narrow half-width to the grip half-width."""
    step = (d['grip_width'] - d['narrow_width']) / 2.0
    return math.sqrt(d['radius'] ** 2 - (d['radius'] - step) ** 2)


def dogbone_area_mm2(d=DOGBONE_1A):
    """Exact planform area: grip rectangle minus narrow-section and four fillet cut-outs."""
    R, step, L = d['radius'], (d['grip_width'] - d['narrow_width']) / 2.0, dogbone_transition_mm(d)
    hw = d['grip_width'] / 2.0
    yc = d['narrow_width'] / 2.0 + R
    # ∫0^L (hw − (yc − √(R² − x²))) dx
    cut = (hw - yc) * L + (L / 2.0 * math.sqrt(R * R - L * L) + R * R / 2.0 * math.asin(L / R))
    return d['length'] * d['grip_width'] - 2.0 * step * d['narrow_length'] - 4.0 * cut


# ---------------------------------------------------------------- formulation screen

COMPONENTS = ('gf_wt', 'extender_wt', 'talc_wt', 'nucleator_wt', 'plasticizer_wt', 'antioxidant_wt', 'phosphite_wt',
              'release_wt', 'impact_wt')


def screen(p):
    """Route-level screen for an explicit caller-supplied parameter dictionary."""
    melt = hydrolysis(p['iv_flake'], p['moisture_ppm'], p['water_reacted_fraction'], p['cooh_mmol_kg'])
    cooh = melt['cooh_mmol_kg']
    pfrac = cooh_end_fraction(cooh, melt['mn'])
    epoxy = epoxy_mmol_kg(p['extender_wt'], p['extender_eew'])
    epoxy_impact = epoxy_mmol_kg(p['impact_wt'], molar_mass('C7H10O3') / (p['impact_gma_wt_pct'] / 100.0)) if p['impact_wt'] else 0.0
    epoxy_total = epoxy + epoxy_impact
    alpha_c = gel_critical_conversion(epoxy_total, cooh, p['extender_fw'], pfrac)
    gel = alpha_c is not None and p['epoxy_conversion'] >= alpha_c
    comp = composite_strength(p['gf_wt'], p['fibre_strength_mpa'], p['fibre_d_um'], p['ifss_mpa'], p['fibre_length_mm'],
                              p['orientation_factor'], p['matrix_stress_mpa'])
    tm = flory_tm_c(p['ipa_mol_pct'] + p['deg_mol_pct'])
    additives = sum(p[k] for k in COMPONENTS)
    crystallizable = p['mold_temp_c'] >= 130.0 or (p['mold_temp_c'] >= 90.0 and p['plasticizer_wt'] >= 2.0)
    hdt_route = crystallizable and p['gf_wt'] >= 20.0
    findings = []
    if gel:
        findings.append(f"Gel predicted: expected epoxy conversion {p['epoxy_conversion']:.2f} ≥ mean-field critical {alpha_c:.2f}.")
    if comp['strength'] < p['tensile_target_mpa']:
        findings.append(f"Estimated flow-direction tensile {comp['strength']:.0f} MPa is below the {p['tensile_target_mpa']:.0f} MPa target.")
    if p['gf_wt'] > 0 and comp['eta_l'] < 0.5:
        findings.append(f"Fibres are near or below l_c ({comp['lc_mm']:.2f} mm): strength depends on retained length and sizing.")
    if p['hdt_target_c'] > 100.0 and not hdt_route:
        findings.append('HDT above ~100 °C at 1.8 MPa needs a crystallised matrix (mold ≥130 °C, or ≥90 °C with plasticizer) and ≥20 wt% glass.')
    if p['hdt_target_c'] > tm - 20.0:
        findings.append(f"HDT target is within 20 °C of the copolymer Tm estimate ({tm:.0f} °C, ideal Flory upper bound).")
    if p['clarity'] and (p['gf_wt'] > 0 or p['talc_wt'] > 0 or crystallizable):
        findings.append('Clarity conflicts with glass, talc or thermal crystallisation.')
    if p['clarity'] and p['hdt_target_c'] > 100.0:
        findings.append('Clarity and HDT >100 °C at 1.8 MPa are mutually exclusive in PET.')
    if p['food_contact'] and (p['plasticizer_wt'] > 0 or p['impact_wt'] > 0):
        findings.append('Food contact: confirm regulatory status of plasticizer and impact modifier; the default route drops them.')
    if melt['iv'] < 0.55:
        findings.append(f"Melt IV after hydrolysis {melt['iv']:.2f} dL/g is below the ~0.55 embrittlement screen.")
    return dict(melt=melt, cooh_after_melt=cooh, cooh_end_fraction=pfrac, epoxy_extender_mmol_kg=epoxy,
                epoxy_impact_mmol_kg=epoxy_impact, epoxy_to_cooh=epoxy_total / cooh if cooh else math.inf,
                gel_critical_conversion=alpha_c, gel_predicted=gel,
                max_extender_wt_no_gel=max(0.0, max_extender_without_gel(cooh, p['extender_fw'], pfrac, p['extender_eew'],
                                                                        p['epoxy_conversion']) - epoxy_impact * p['extender_eew'] / 1e4),
                composite=comp, tm_flory_c=tm, rpet_balance_wt=100.0 - additives, crystallizable=crystallizable,
                hdt_route=hdt_route, aa_capacity_ppm=aa_capacity_ppm(p['scavenger_ppm']), findings=findings)


# ---------------------------------------------------------------- validation of the written formulation argument

def _check(cid, claim, computed, status, note):
    return dict(id=cid, claim=claim, computed=computed, status=status, note=note)


def _rel(a, b):
    return abs(a - b) / abs(b)


def validate_claims():
    """Re-derive each numeric claim in the rPET formulation answer (2026-09-14) and grade it.

    PASS       within 5 % (or inside the claimed range)
    CORRECTED  arithmetic did not reproduce the claim; the computed value replaces it
    REVISED    the claim used the wrong model; the corrected model changes the conclusion
    BOUNDED    only a bound can be computed; the claim needs measurement
    """
    out = []
    mn0 = 24000.0
    iv0 = iv_from_mn(mn0)
    out.append(_check('mh_baseline', 'Mn 24,000 ⇒ IV ≈ 0.80 dL/g', round(iv0, 3),
                      'PASS' if _rel(iv0, 0.80) <= 0.05 else 'CORRECTED',
                      f"Computed {iv0:.3f}; the table rows below start from IV {iv0:.3f}, not 0.80. Alternative Mw-based constants "
                      f"give Mn {mn_from_iv_alt(0.80):.0f} at IV 0.80 (PDI 2), so the baseline is constant-set consistent."))
    claimed = {30: (23000, 0.78, 10), 100: (21900, 0.76, 20), 500: (16000, 0.58, 75), 4000: (3800, 0.18, None)}
    for ppm, (mn_c, iv_c, loss_c) in claimed.items():
        h = hydrolysis(iv0, ppm)
        loss = (1 - h['eta0_ratio']) * 100
        ok = _rel(h['mn'], mn_c) <= 0.05 and abs(h['iv'] - iv_c) <= 0.02 and (loss_c is None or abs(loss - loss_c) <= 5)
        out.append(_check(f'hydrolysis_{ppm}ppm', f'{ppm} ppm ⇒ Mn ~{mn_c}, IV ~{iv_c}' + (f', η0 −{loss_c} %' if loss_c else ''),
                          dict(mn=round(h['mn']), iv=round(h['iv'], 3), eta0_loss_pct=round(loss, 1)),
                          'PASS' if ok else 'CORRECTED',
                          'Full-reaction stoichiometry (upper bound on damage; vented water does not react).'))
    ex = viscosity_iv_exponent()
    out.append(_check('eta_iv_exponent', 'η0 ∝ IV^4.1', round(ex, 3), 'PASS' if abs(ex - 4.1) <= 0.1 else 'CORRECTED',
                      'Requires constant polydispersity; reprocessing near most-probable distribution keeps PDI ≈ 2.'))
    vf = fibre_volume_fraction(30.0)
    out.append(_check('glass_vf', '30 wt% E-glass ⇒ Vf ≈ 0.19', round(vf, 4), 'PASS' if _rel(vf, 0.19) <= 0.05 else 'CORRECTED', ''))
    lc = critical_length_mm(2000.0, 10.0, 25.0)
    out.append(_check('critical_length', 'l_c ≈ 0.4 mm (σf 2000 MPa, d 10 µm, τ 25 MPa)', round(lc, 3),
                      'PASS' if _rel(lc, 0.4) <= 0.05 else 'CORRECTED', ''))
    assumed = composite_strength(30.0, 2000.0, 10.0, 25.0, 0.3, 0.4, 55.0, eta_l=0.6)
    kt = composite_strength(30.0, 2000.0, 10.0, 25.0, 0.3, 0.4, 55.0)
    kt_good = composite_strength(30.0, 2000.0, 10.0, 40.0, 0.3, 0.5, 55.0)
    low = composite_strength(30.0, 1500.0, 13.0, 15.0, 0.2, 0.3, 45.0)
    out.append(_check('tensile_rom', 'GF30 ⇒ ~135 MPa (η_o 0.4, η_l 0.6)',
                      dict(as_written=round(assumed['strength'], 1), kelly_tyson_same_inputs=round(kt['strength'], 1),
                           kelly_tyson_eta_l=round(kt['eta_l'], 3), good_sizing_tau40_eta_o05=round(kt_good['strength'], 1),
                           pessimistic=round(low['strength'], 1)),
                      'REVISED',
                      f"Arithmetic reproduces {assumed['strength']:.0f} MPa, but η_l = 0.6 is inconsistent with a 0.3 mm fibre below "
                      f"l_c = {kt['lc_mm']:.2f} mm: Kelly–Tyson gives η_l {kt['eta_l']:.3f} and {kt['strength']:.0f} MPa, under the 120 MPa target. "
                      f"τ 40 MPa (PET-matched sizing) with η_o 0.5 gives {kt_good['strength']:.0f} MPa; pessimistic inputs give "
                      f"{low['strength']:.0f} MPa. The datasheet 140–160 MPa is the evidence; the model only bounds it."))
    ds = iso75_deflection_mm()
    em = iso75_modulus_criterion_mpa()
    out.append(_check('hdt_criterion', 'HDT-A ≈ temperature where flexural modulus falls through ~0.8–1 GPa',
                      dict(standard_deflection_mm=round(ds, 4), modulus_mpa=round(em, 1)),
                      'PASS' if 800 <= em <= 1000 else 'CORRECTED',
                      'Elastic estimate at 1.80 MPa and 0.2 % strain; creep during the 120 K/h ramp lowers the effective threshold.'))
    ep = epoxy_mmol_kg(1.0, C['joncryl_adr4468_eew'])
    out.append(_check('epoxy_per_wt', '1 wt% ADR-4468 ≈ 32 mmol epoxy/kg', round(ep, 2), 'PASS' if _rel(ep, 32) <= 0.05 else 'CORRECTED',
                      'EEW 310 g/mol as recalled; confirm on the datasheet.'))
    ratios = {w: round(epoxy_mmol_kg(w, C['joncryl_adr4468_eew']) / 40.0, 3) for w in (0.2, 0.4, 1.0)}
    ok = abs(ratios[0.2] - 1 / 6) < 0.03 and abs(ratios[0.4] - 1 / 3) < 0.03 and 0.7 <= ratios[1.0] <= 1.1
    out.append(_check('epoxy_ratios', '0.2 % ≈ 1:6, 0.4 % ≈ 1:3, 1.0 % ≈ 1:1 epoxy:COOH at 40 mmol/kg', ratios,
                      'PASS' if ok else 'CORRECTED', ''))
    f_w = C['joncryl_adr4468_mw'] / C['joncryl_adr4468_eew']
    mn_flake = mn_from_iv(0.72)
    pf = cooh_end_fraction(40.0, mn_flake)
    table = {}
    for fw in (10.0, f_w):
        table[f'f_w={fw:.1f}'] = {f'{w}wt%': (None if (a := gel_critical_conversion(epoxy_mmol_kg(w, 310.0), 40.0, fw, pf)) is None else round(a, 3))
                                  for w in (0.2, 0.3, 0.4, 0.5, 1.0)}
    hi = table[f'f_w={f_w:.1f}']
    out.append(_check('gel_point', 'Gel criterion ≈ 1/(f−1) ≈ 5 %; 0.2–0.5 wt% safe, 1 wt% past gel',
                      dict(critical_epoxy_conversion=table, cooh_end_fraction=round(pf, 3), benchmark_A3_B2=round(gel_benchmark_a3_b2(), 4)),
                      'REVISED',
                      '1/(f−1) is the A_f self-condensation criterion, not A_f + PET chains. With COOH on a fraction p of ends, '
                      f"gel needs α_A·α_B·(f_w−1)·p = 1. At f_w ≈ {f_w:.1f} (Mw/EEW), IV 0.72 and 40 mmol/kg (p = {pf:.2f}), "
                      f"0.3 / 0.4 / 0.5 wt% gel at {hi['0.3wt%']:.0%} / {hi['0.4wt%']:.0%} / {hi['0.5wt%']:.0%} epoxy conversion in the "
                      'ideal model. Practical 0.2–0.5 wt% use survives only through incomplete conversion, ester interchange and a '
                      'lower number-average functionality. Set the dose by a rheometry ladder (G′ plateau), start trials at '
                      '≤0.3 wt%, and treat 0.5 wt% as gel-risk.'))
    bal = dict(no_plasticizer=round(100 - (30 + 0.35 + 0.75 + 0.3 + 0.15 + 0.15 + 0.25), 2),
               with_plasticizer_3=round(100 - (30 + 0.35 + 0.75 + 0.3 + 3 + 0.15 + 0.15 + 0.25), 2),
               ionomer_2p25_with_plasticizer=round(100 - (30 + 0.35 + 0.75 + 2.25 + 3 + 0.15 + 0.15 + 0.25), 2))
    out.append(_check('mass_balance', 'rPET balance ~64–66 wt%', bal, 'CORRECTED',
                      'Mid-range loadings give 63–68 wt%; 64–66 holds only with the plasticizer. State the balance per option.'))
    tm = flory_tm_c(4.5)
    out.append(_check('copolymer_tm', 'Bottle copolymer (≈2 % IPA + 2.5 % DEG) Tm ≈ 245–252 °C', round(tm, 1), 'BOUNDED',
                      f'Ideal Flory gives only {265 - tm:.1f} °C depression (upper bound on Tm, known to under-predict for DEG/IPA). '
                      'The 245–252 °C range is a typical DSC observation and must be measured on the incoming flake.'))
    cap = aa_capacity_ppm(50.0), aa_capacity_ppm(300.0)
    out.append(_check('aa_scavenger', 'Anthranilamide 50–300 ppm is enough for preform AA', [round(c, 1) for c in cap],
                      'PASS' if cap[0] > 8.0 else 'CORRECTED',
                      'Stoichiometric capacity 16–97 ppm AA against single-digit ppm preform AA; kinetics, not capacity, limit it.'))
    dogbone = dogbone_area_mm2()
    out.append(_check('iso527_1a_area', 'Specimen geometry owner (not a claim): ISO 527-2 1A planform area', round(dogbone, 2), 'PASS',
                      f'Transition length {dogbone_transition_mm():.3f} mm; distance between broad parallel parts '
                      f'{80 + 2 * dogbone_transition_mm():.1f} mm (standard 104–113).'))
    return out

# DESCRIPTION: Biochemistry area — buffers and biochemical thermodynamics: Henderson-Hasselbalch
#              with Davies activity correction, buffer recipes, polyprotic titration curves and
#              capacity, ionic strength, isoelectric point, and standard transformed dG'.
"""Buffer and biochemical-thermodynamics tools for the science toolkit Biochemistry area.

pKa data live in ``engineering_sim/science/data/biochem_pka.json`` (thermodynamic 25 C, I->0 values; see the
file's ``_meta`` for provenance). Calculations here apply two corrections explicitly so the model
is transparent: a temperature shift ``pKa(T) = pKa(25 C) + (dT)*(T-25)`` (where a dT is published)
and a Davies-equation activity correction at the stated ionic strength ``I``:
    log10(gamma_i) = -A z_i^2 (sqrt(I)/(1+sqrt(I)) - 0.3 I),   A = 0.509 at 25 C.
For a deprotonation HA^z -> A^(z-1) + H+ the apparent constant shifts by
    pKa' = pKa - 2 A (1 - z_HA) (sqrt(I)/(1+sqrt(I)) - 0.3 I).
Standard-library + numpy only; every reported number is a cite record.
"""
from __future__ import annotations

import json
import math
import os

from . import cite

_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "biochem_pka.json")
_DATA = None

# Davies-equation coefficient at 25 C (log10 form). The Debye-Huckel limiting slope is 0.5115;
# Davies' original used 0.50; modern texts use 0.509. Reported as unverified (not fetched).
A_DAVIES = 0.509

R = 8.314462618  # J/(mol K), CODATA 2018
R_KJ = R / 1000.0  # kJ/(mol K)

# Standard atomic weights (g/mol) used to compute molecular weights of weighing forms.
_ATOMIC = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "Na": 22.990, "P": 30.974,
           "S": 32.06, "Cl": 35.45, "K": 39.098, "B": 10.81, "Mg": 24.305, "Ca": 40.078,
           "Fe": 55.845, "Zn": 65.38, "Cu": 63.546, "I": 126.90, "Br": 79.904, "F": 18.998,
           "Mn": 54.938}

# Standard transformed free energies dG' (pH 7, 25 C, ~0.1 M ionic strength, 1 mM Mg2+ for ATP).
# Bibliographic values (Lehninger / Goldberg & Tewari 1989); the source was not fetched during
# this build, so each is unverified with its reference.
_DG0PRIME = [
    {"id": "atp_hydrolysis", "name": "ATP + H2O -> ADP + Pi",
     "dg0": -30.5,
     "source": "Lehninger Principles of Biochemistry (Nelson & Cox); Goldberg & Tewari, J. Phys. Chem. Ref. Data 18, 1657 (1989)"},
    {"id": "hexokinase", "name": "Glucose + ATP -> glucose-6-P + ADP",
     "dg0": -16.7,
     "source": "Lehninger Principles of Biochemistry (Nelson & Cox)"},
    {"id": "pfk", "name": "Fructose-6-P + ATP -> fructose-1,6-bisP + ADP",
     "dg0": -14.2,
     "source": "Lehninger Principles of Biochemistry (Nelson & Cox)"},
    {"id": "aldolase", "name": "Fructose-1,6-bisP -> DHAP + glyceraldehyde-3-P",
     "dg0": 23.8,
     "source": "Lehninger Principles of Biochemistry (Nelson & Cox)"},
    {"id": "pgk", "name": "1,3-bisphosphoglycerate + ADP -> 3-phosphoglycerate + ATP",
     "dg0": -18.5,
     "source": "Lehninger Principles of Biochemistry (Nelson & Cox)"},
    {"id": "pyruvate_kinase", "name": "Phosphoenolpyruvate + ADP -> pyruvate + ATP",
     "dg0": -31.4,
     "source": "Lehninger Principles of Biochemistry (Nelson & Cox)"},
    {"id": "pep_hydrolysis", "name": "Phosphoenolpyruvate + H2O -> pyruvate + Pi",
     "dg0": -61.9,
     "source": "Lehninger Principles of Biochemistry (Nelson & Cox)"},
    {"id": "g6p_hydrolysis", "name": "Glucose-6-P + H2O -> glucose + Pi",
     "dg0": -13.8,
     "source": "Lehninger Principles of Biochemistry (Nelson & Cox)"},
]


def _load():
    global _DATA
    if _DATA is None:
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            _DATA = json.load(f)
    return _DATA


def _pka_num(rec):
    return cite.num(rec)


def _davies_g(I):
    """sqrt(I)/(1+sqrt(I)) - 0.3 I — the Davies ionic-strength factor."""
    s = math.sqrt(I)
    return s / (1.0 + s) - 0.3 * I


def apparent_pkas(entry, I, T_C):
    """Temperature- and ionic-strength-corrected pKa ladder for one buffer entry."""
    out = []
    for step in entry["pkas"]:
        p = _pka_num(step["pka"])
        dT = step.get("dT")
        if dT is not None:
            p += _pka_num(dT) * (T_C - 25.0)
        # z_HA = charge of the acid (protonated) form at this step = charge_start - step_index.
        z_ha = entry["charge_start"] - len(out)
        p -= 2.0 * A_DAVIES * (1.0 - z_ha) * _davies_g(I)
        out.append(p)
    return out


def _species(entry, pH, I, T_C):
    """Fractional populations, average charge and per-step pKa' at a pH."""
    pkas = apparent_pkas(entry, I, T_C)
    n = len(pkas)
    z0 = entry["charge_start"]
    # alpha_i (species with i protons removed) via cumulative Ka products.
    cums = [0.0]
    prod = 1.0
    for p in pkas:
        prod *= 10.0 ** (-p)
        cums.append(prod)
    denom = 0.0
    for i in range(n + 1):
        denom += cums[i] / (10.0 ** (-i * pH))
    alphas = []
    for i in range(n + 1):
        alphas.append((cums[i] / (10.0 ** (-i * pH))) / denom)
    z = sum(alphas[i] * (z0 - i) for i in range(n + 1))
    return pkas, alphas, z


def _z(entry, pH, I, T_C):
    return _species(entry, pH, I, T_C)[2]


def _mw(formula):
    """Molecular weight (g/mol) from a chemical formula (no parentheses), standard atomic weights."""
    if not formula:
        return None
    i = 0
    counts = {}
    while i < len(formula):
        if not (formula[i].isalpha() and formula[i].isupper()):
            raise ValueError("Unsupported formula %r (use simple formulas without parentheses)." % formula)
        el = formula[i]
        i += 1
        while i < len(formula) and formula[i].isalpha() and formula[i].islower():
            el += formula[i]
            i += 1
        num = 0
        while i < len(formula) and formula[i].isdigit():
            num = num * 10 + int(formula[i])
            i += 1
        if el not in _ATOMIC:
            raise ValueError("Unknown element %r in formula %r." % (el, formula))
        counts[el] = counts.get(el, 0) + (num or 1)
    return sum(_ATOMIC[el] * n for el, n in counts.items())


def _buffer_entry(buffer):
    data = _load()
    entry = data["buffers"].get(buffer)
    if entry is None:
        raise ValueError("Unknown buffer %r; choose from %s"
                         % (buffer, sorted(data["buffers"].keys())))
    return entry


def henderson_hasselbalch(buffer, pH=None, ratio=None, T_C=25.0, I=None):
    """Henderson-Hasselbalch on the pKa step nearest the stated pH (or invert to find pH from a ratio)."""
    entry = _buffer_entry(buffer)
    pkas = [p["pka"] for p in entry["pkas"]]
    if pH is None and ratio is None:
        raise ValueError("Provide either pH or ratio.")
    if pH is not None and ratio is not None:
        raise ValueError("Provide pH or ratio, not both.")
    # Ionic strength defaults to 0.1 M unless stated (a common buffer working condition).
    if I is None:
        I = 0.1
    if I < 0:
        raise ValueError("Ionic strength must be non-negative.")

    pka_rec = None
    if pH is not None:
        pH = float(pH)
        # Choose the pKa step whose apparent value is nearest the target pH.
        apps = apparent_pkas(entry, I, T_C)
        k = min(range(len(apps)), key=lambda i: abs(apps[i] - pH))
        pka_rec = entry["pkas"][k]["pka"]
        pka_app = apps[k]
        r = 10.0 ** (pH - pka_app)
        return {
            "buffer": buffer, "name": entry["name"], "step": entry["pkas"][k]["label"],
            "pka": pka_rec,
            "pka_apparent": cite.value(pka_app, "", "derived",
                                       method="pKa(T) corrected with the Davies equation at I = %.3g M (A = %.3f)" % (I, A_DAVIES),
                                       note="apparent pKa at the stated ionic strength"),
            "ratio": cite.value(r, "A-/HA", "derived",
                                method="10^(pH - pKa_apparent)"),
            "ionic_strength": cite.value(I, "M", "user"),
            "notes": ["The nearest pKa step is used for polyprotic buffers; far from any pKa the "
                      "buffer is ineffective.", "Davies correction: apparent pKa differs from the "
                      "thermodynamic value at finite ionic strength."],
        }
    # Invert: pH = pKa' + log10(ratio). For a polyprotic buffer, pick the step nearest the
    # pH implied by the ratio around the middle pKa (a one-step fixed point).
    ratio = float(ratio)
    if ratio <= 0:
        raise ValueError("ratio must be positive.")
    apps = apparent_pkas(entry, I, T_C)
    ph_est = apps[len(apps) // 2] + math.log10(ratio)
    best = min(range(len(apps)), key=lambda i: abs(apps[i] - ph_est))
    pka_rec = entry["pkas"][best]["pka"]
    ph = apps[best] + math.log10(ratio)
    return {
        "buffer": buffer, "name": entry["name"], "step": entry["pkas"][best]["label"],
        "pka": pka_rec,
        "pka_apparent": cite.value(apps[best], "", "derived",
                                   method="Davies-corrected pKa at I = %.3g M" % I),
        "ph": cite.value(ph, "", "derived", method="pKa_apparent + log10(ratio)"),
        "ratio": cite.value(ratio, "A-/HA", "user"),
        "ionic_strength": cite.value(I, "M", "user"),
        "notes": ["pH = pKa' + log10([A-]/[HA]); the Davies-corrected apparent pKa is used."],
    }


def prepare_buffer(buffer, pH, conc_M, volume_L, form="acid+conjugate", T_C=25.0, I=None, titrant_M=1.0):
    """Masses (and titrant volumes) to prepare a buffer; Davies-corrected pKa at the stated ionic strength."""
    entry = _buffer_entry(buffer)
    if form not in ("acid+conjugate", "acid+NaOH", "base+HCl"):
        raise ValueError("form must be 'acid+conjugate', 'acid+NaOH' or 'base+HCl'.")
    pH = float(pH)
    conc_M = float(conc_M)
    volume_L = float(volume_L)
    if conc_M <= 0 or volume_L <= 0:
        raise ValueError("conc_M and volume_L must be positive.")
    if I is None:
        I = conc_M  # approximation for a 1:1 buffer; noted below
    if I < 0:
        raise ValueError("Ionic strength must be non-negative.")

    apps = apparent_pkas(entry, I, T_C)
    n = len(apps)
    z0 = entry["charge_start"]
    z = _z(entry, pH, I, T_C)
    moles = conc_M * volume_L
    _, alphas, _ = _species(entry, pH, I, T_C)

    result = {
        "buffer": buffer, "name": entry["name"], "form": form,
        "ph": cite.value(pH, "", "user"),
        "conc_M": cite.value(conc_M, "M", "user"),
        "volume_L": cite.value(volume_L, "L", "user"),
        "ionic_strength": cite.value(I, "M", "derived" if I == conc_M else "user",
                                     method="assumed equal to the buffer concentration (1:1 approximation)" if I == conc_M else None),
        "capacity": cite.value(2.303 * conc_M * _charge_variance(alphas, z0, n), "M/pH", "derived",
                               method="beta = 2.303 * C * Var(charge)"),
        "species": [{"label": entry["pkas"][i]["label"] if i < n else "fully deprotonated",
                     "charge": z0 - i, "fraction": alphas[i]} for i in range(n + 1)],
        "notes": [],
    }

    if form == "acid+conjugate":
        k = min(range(n), key=lambda i: abs(apps[i] - pH))
        step = entry["pkas"][k]
        if not step.get("acid_form") or not step.get("base_form"):
            raise ValueError("Buffer %r has no weighing forms for 'acid+conjugate'." % buffer)
        r = 10.0 ** (pH - apps[k])
        ha = moles / (1.0 + r)
        a = moles * r / (1.0 + r)
        mw_acid = _mw(step["acid_form"])
        mw_base = _mw(step["base_form"])
        result["step"] = step["label"]
        result["acid_form"] = step.get("acid_form")
        result["base_form"] = step.get("base_form")
        result["mass_acid_g"] = cite.value(ha * mw_acid, "g", "derived",
                                           method="[HA] = C/(1+r); mass = moles_HA * MW(%s)" % step["acid_form"])
        result["mass_base_g"] = cite.value(a * mw_base, "g", "derived",
                                           method="[A-] = C*r/(1+r); mass = moles_A * MW(%s)" % step["base_form"])
        result["mw_acid"] = cite.value(mw_acid, "g/mol", "derived",
                                       method="sum of standard atomic weights (IUPAC)")
        result["mw_base"] = cite.value(mw_base, "g/mol", "derived",
                                      method="sum of standard atomic weights (IUPAC)")
        result["notes"].append("Weigh the acid and conjugate-base forms, dissolve in ~80 %% of the final "
                               "volume, adjust to pH if needed, then make up to volume.")
    else:
        if form == "acid+NaOH":
            stock = entry["pkas"][0].get("acid_form")
            if not stock:
                raise ValueError("Buffer %r has no fully protonated weighing form." % buffer)
            equiv = z0 - z  # equivalents of strong base per buffer molecule
            mw = _mw(stock)
            result["stock_form"] = stock
            result["mass_stock_g"] = cite.value(moles * mw, "g", "derived",
                                                method="moles * MW(%s)" % stock)
            result["titrant_equiv"] = cite.value(equiv, "eq/mol", "derived",
                                                 method="z0 - average charge at pH (from the pKa ladder)")
            result["titrant_volume_ml"] = cite.value(moles * equiv / titrant_M * 1000.0, "mL", "derived",
                                                     method="moles * eq / titrant_M")
            result["titrant_M"] = cite.value(titrant_M, "M", "user")
        else:
            stock = entry["pkas"][-1].get("base_form")
            if not stock:
                raise ValueError("Buffer %r has no fully deprotonated weighing form." % buffer)
            z_end = z0 - n
            equiv = z - z_end  # equivalents of strong acid per buffer molecule
            mw = _mw(stock)
            result["stock_form"] = stock
            result["mass_stock_g"] = cite.value(moles * mw, "g", "derived",
                                                method="moles * MW(%s)" % stock)
            result["titrant_equiv"] = cite.value(equiv, "eq/mol", "derived",
                                                 method="average charge at pH - z_fully_deprotonated")
            result["titrant_volume_ml"] = cite.value(moles * equiv / titrant_M * 1000.0, "mL", "derived",
                                                     method="moles * eq / titrant_M")
            result["titrant_M"] = cite.value(titrant_M, "M", "user")
        result["notes"].append("Dissolve the stock form, titrate with strong base/acid to the target pH, "
                               "then make up to volume.")
    result["notes"].append("Ionic strength is assumed equal to the buffer concentration (a 1:1 approximation); "
                           "pass I explicitly for a better activity correction.")
    return result


def _charge_variance(alphas, z0, n):
    zbar = sum(alphas[i] * (z0 - i) for i in range(n + 1))
    return sum(alphas[i] * (z0 - i - zbar) ** 2 for i in range(n + 1))


def titration(buffer, conc_M=0.1, I=None, T_C=25.0, ph_range=(1.0, 13.0), points=121):
    """Polyprotic titration curve: average charge (equivalents) and buffer capacity vs pH."""
    entry = _buffer_entry(buffer)
    conc_M = float(conc_M)
    if conc_M <= 0:
        raise ValueError("conc_M must be positive.")
    if I is None:
        I = conc_M
    if I < 0:
        raise ValueError("Ionic strength must be non-negative.")
    lo, hi = ph_range
    if not (0 <= lo < hi <= 14):
        raise ValueError("ph_range must satisfy 0 <= lo < hi <= 14.")
    z0 = entry["charge_start"]
    n = len(entry["pkas"])
    curve = []
    for i in range(points):
        pH = lo + (hi - lo) * i / (points - 1)
        apps, alphas, z = _species(entry, pH, I, T_C)
        beta = 2.303 * conc_M * _charge_variance(alphas, z0, n)
        curve.append({"ph": round(pH, 3), "equivalents": round(z0 - z, 4), "capacity": round(beta, 5)})
    pkas = apparent_pkas(entry, I, T_C)
    return {
        "buffer": buffer, "name": entry["name"], "conc_M": conc_M,
        "ionic_strength": I,
        "curve": curve,
        "pka_markers": [{"label": entry["pkas"][i]["label"], "pka_apparent": pkas[i]} for i in range(n)],
        "notes": [
            "Equivalents are strong base per mole of buffer relative to the fully protonated form "
            "(charge balance over the polyprotic ladder, Davies-corrected).",
            "Capacity beta = 2.303 * C * Var(charge); the water autoionization term is omitted "
            "(negligible between pH ~2 and ~12).",
        ],
    }


def ionic_strength(solutes):
    """I = 0.5 * sum(conc * z^2) over a list of solutes {conc, charge} (or a name->dict map)."""
    items = list(solutes.values()) if isinstance(solutes, dict) else list(solutes)
    total = 0.0
    components = []
    for s in items:
        conc = float(s.get("conc", s.get("conc_M", s.get("c", 0.0))))
        z = float(s.get("charge", s.get("z", 0.0)))
        contrib = conc * z * z
        total += contrib
        components.append({"conc": conc, "charge": z, "contrib": contrib})
    return {
        "ionic_strength": cite.value(0.5 * total, "M", "derived", method="I = 0.5 * sum(c_i z_i^2)"),
        "components": components,
        "notes": ["Charges are the signed ionic charges (e.g. Na+ = +1, SO4(2-) = -2)."],
    }


def pI(groups):
    """Isoelectric point: the pH where the net charge of a molecule/peptide crosses zero (bisection)."""
    import numpy as np

    specs = []
    for g in groups:
        p = float(cite.num(g.get("pka")))
        if "protonated" in g:
            z_prot = float(g["protonated"])
            z_dep = float(g["deprotonated"])
        elif g.get("type") == "acid":
            z_prot, z_dep = 0.0, -1.0
        elif g.get("type") == "base":
            z_prot, z_dep = 1.0, 0.0
        else:
            raise ValueError("Each group needs 'type' in {acid, base} or explicit protonated/deprotonated charges.")
        specs.append((p, z_prot, z_dep))

    def net(pH):
        return sum(((1.0 - th) * zp + th * zd)
                   for (p, zp, zd) in specs
                   for th in [1.0 / (1.0 + 10.0 ** (p - pH))])

    lo, hi = 0.0, 14.0
    flo, fhi = net(lo), net(hi)
    if flo < 0 or fhi > 0:
        raise ValueError("Net charge does not cross zero in pH 0-14; check the pKa list.")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = net(mid)
        if abs(fm) < 1e-9 or (hi - lo) < 1e-10:
            break
        if (flo < 0) == (fm < 0):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    pi = 0.5 * (lo + hi)
    return {
        "pI": cite.value(pi, "", "derived", method="pH where net charge crosses zero (bisection over the pKa ladder)"),
        "net_charge_at_pi": net(pi),
        "notes": ["pI is computed from the supplied pKa values; for amino acids use the pKa values "
                  "that bracket the zwitterion (alpha-COOH and alpha-NH3+)."],
    }


def dG0prime_table():
    """The small standard transformed dG' table (pH 7, 25 C); unverified bibliographic values."""
    out = []
    for e in _DG0PRIME:
        out.append({
            "id": e["id"], "name": e["name"],
            "dg0_kjmol": cite.value(e["dg0"], "kJ/mol", "unverified",
                                    source=e["source"],
                                    note="standard transformed dG' at pH 7, 25 C; source not fetched during build"),
        })
    return out


def dG0prime(name):
    """Look up one standard transformed dG' by id (raises KeyError if unknown)."""
    for e in _DG0PRIME:
        if e["id"] == name:
            return cite.value(e["dg0"], "kJ/mol", "unverified",
                              source=e["source"],
                              note="standard transformed dG' at pH 7, 25 C; source not fetched during build")
    raise KeyError("Unknown dG0prime id %r; choose from %s" % (name, [e["id"] for e in _DG0PRIME]))


def biochemical_dG(dG0prime_kJ, ratio_products_reactants, T_K=298.15):
    """dG' = dG'0 + RT ln Q, with Q = [products]/[reactants]."""
    if isinstance(dG0prime_kJ, dict) and cite.is_record(dG0prime_kJ):
        dg0_rec = dG0prime_kJ
        dG0 = float(dg0_rec["value"])
    else:
        dG0 = float(dG0prime_kJ)
        dg0_rec = cite.value(dG0, "kJ/mol", "user", note="standard transformed dG' (pH 7)")
    Q = float(ratio_products_reactants)
    T = float(T_K)
    if Q <= 0 or T <= 0:
        raise ValueError("ratio (Q) and T_K must be positive.")
    dg = dG0 + R_KJ * T * math.log(Q)
    return {
        "dg0prime_kjmol": dg0_rec,
        "q": cite.value(Q, "", "user"),
        "t_k": cite.value(T, "K", "user"),
        "dg_kjmol": cite.value(dg, "kJ/mol", "derived",
                               method="dG' = dG'0 + R*T*ln(Q), R = %.6f kJ/(mol K)" % R_KJ),
        "notes": ["The biochemical standard state is pH 7, ~0.1 M ionic strength, 25 C (and 1 mM "
                  "Mg2+ for ATP); Q uses concentrations (activities of water and H+ are fixed at 1 "
                  "in the transformed convention)."],
    }

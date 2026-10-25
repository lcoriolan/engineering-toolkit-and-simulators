# DESCRIPTION: The "My Lab" reaction bench: drop atoms/molecules, press React, get the
#              equilibrium product mix, the direct reaction, energy diagram, bonds,
#              Lewis picture and hazards — all from cited NIST WebBook data.
"""My Lab reaction bench.

``react(items, ...)`` parses the user's atoms/molecules (via Task A's
``chem.parse_items``), maps each to a species in the cited thermochemistry database,
enumerates every database species whose elements are available, runs the
``thermo.equilibrium`` Gibbs minimisation, and reports the products together with the
direct reaction to the dominant product set.

Thermodynamics only: the bench reports what is favoured at equilibrium, never whether
or how fast a reaction happens (kinetics is not predicted). Gas-phase recombination
of free atoms additionally carries a third-body note (H + H + M -> H2 + M).
"""

from __future__ import annotations

import re

from . import cite, thermo

_KINETICS_NOTE = (
    "Thermodynamics only: shows what is favored at equilibrium, not whether or how "
    "fast it happens (kinetics not predicted)."
)
_THIRD_BODY_NOTE = (
    "In the gas phase two atoms cannot stay bonded without a third body "
    "(H + H + M → H₂ + M) to carry away the bond energy."
)
_METHOD = "Gibbs minimisation (%s)" % thermo.METHOD
_BOND_SYMBOL = {1: "-", 1.5: ":", 2: "=", 3: "#"}


def _mol_elements(mol: dict) -> dict:
    """Element composition {El: n} of a chem.molecule dict (from its atoms)."""
    els = {}
    for a in mol.get("atoms", []):
        els[a["el"]] = els.get(a["el"], 0) + 1
    return els


def _match_species(elements: dict, phase: str):
    """Map a molecule's element composition to a thermo species key (or None).

    Prefers the requested phase, then the gas phase, then any match. Elements are
    compared as dicts so Hill-order differences in formula strings never matter.
    """
    matches = []
    for entry in thermo.species():
        sp = thermo.species(entry["key"])
        if sp["elements"] == elements:
            matches.append(entry["key"])
    if not matches:
        return None
    for pref in (phase, "gas"):
        for k in matches:
            if thermo.species(k)["phase"] == pref:
                return k
    return matches[0]


def _explicit_smiles(smiles: str) -> str:
    """RDKit SMILES with explicit H, so bare-element SMILES ('O' = water) are not
    re-read by ``chem.molecule`` as a single atom. Local and deterministic."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(Chem.AddHs(mol), allHsExplicit=True)


def _molecule_view(spec: dict) -> dict:
    """chem.molecule dict for the 3D/Lewis view of a species (best effort)."""
    try:
        from engineering_sim.science import chem

        return chem.molecule(_explicit_smiles(spec["smiles"]))
    except Exception as exc:  # chem may be missing or a SMILES may not embed
        return {"smiles": spec["smiles"], "formula": spec["formula"], "error": str(exc)}


def _bond_types(mol: dict) -> dict:
    """Count bonds of each type ('H-H', 'O=O', ...) in a chem.molecule dict."""
    els = {a["i"]: a["el"] for a in mol.get("atoms", [])}
    out = {}
    for b in mol.get("bonds", []):
        e1, e2 = els.get(b["a"]), els.get(b["b"])
        if not e1 or not e2:
            continue
        sym = _BOND_SYMBOL.get(b.get("order"), "-")
        key = sym.join(sorted((e1, e2)))  # e.g. 'H-H', 'O=O', 'H-O'
        out[key] = out.get(key, 0) + 1
    return out


def _diatomic_bond_enthalpy(e1: str, e2: str, T: float):
    """Mean bond enthalpy (kJ/mol) for an element pair, from the cited database.

    Looks for a gas-phase diatomic species with exactly that element composition
    and reports its atomization enthalpy X-Y -> X + Y (derived from the cited
    formation enthalpies). Returns a value record or None when no such species exists.
    """
    want = {e1: 2} if e1 == e2 else {e1: 1, e2: 1}
    for entry in thermo.species():
        sp = thermo.species(entry["key"])
        if sp["elements"] == want and sp["phase"] == "gas":
            atom_counts = {}
            for el in (e1, e2):
                ak = _match_species({el: 1}, "gas")
                if ak is None:
                    return None
                atom_counts[ak] = atom_counts.get(ak, 0) + 1
            try:
                r = thermo.reaction({entry["key"]: 1}, atom_counts, T)
                return cite.value(cite.num(r["dH"]), "kJ/mol", "derived",
                                  method="Shomate/NIST WebBook atomization")
            except Exception:
                return None
    return None


def _direct_reaction(feed_counts: dict, product_amounts: dict, T: float):
    """Balanced direct reaction feed -> dominant products, as a thermo.reaction result.

    ``product_amounts`` are equilibrium moles of the newly formed species. For a
    single product the integer coefficient is solved from the element balance
    (2 H -> H2); otherwise near-integer amounts are rounded.
    """
    if not product_amounts:
        return None
    # Element totals of the feed (conserved).
    feed_elements = {}
    for k, n in feed_counts.items():
        for el, c in thermo.species(k)["elements"].items():
            feed_elements[el] = feed_elements.get(el, 0.0) + n * c
    products = {}
    if len(product_amounts) == 1:
        key, _amt = next(iter(product_amounts.items()))
        sp_els = thermo.species(key)["elements"]
        coeff = None
        ok = True
        for el, b in feed_elements.items():
            a = sp_els.get(el, 0.0)
            if a == 0.0:
                if abs(b) > 1e-9:
                    ok = False
            else:
                c = b / a
                if coeff is None:
                    coeff = c
                elif abs(c - coeff) > 1e-6:
                    ok = False
        if ok and coeff is not None and abs(coeff - round(coeff)) < 1e-6:
            products[key] = int(round(coeff))
    if not products:
        # Multiple products / non-integer: round near-integer equilibrium amounts.
        for k, amt in product_amounts.items():
            r = round(amt)
            products[k] = r if abs(amt - r) < 0.05 else amt
    try:
        return thermo.reaction(feed_counts, products, T)
    except ValueError:
        # Fall back to the raw equilibrium amounts (they balance by construction).
        try:
            return thermo.reaction(feed_counts, {k: float(v) for k, v in product_amounts.items()}, T)
        except ValueError:
            return None


def _enumerate_xtb(elements: dict, known_elements: set):
    """Small valence-saturated molecules from the available atoms (<= 6 heavy atoms),
    estimated with GFN2-xTB and separated from the cited equilibrium.

    A modest deterministic enumeration: all diatomics of the present elements plus a
    curated list of common small molecules whose elements are available. Molecules
    already in the cited database (``known_elements``, by element composition) are
    skipped.
    """
    from engineering_sim.science import xtb

    present = {el for el, n in elements.items() if n > 0}
    curated = [
        ("H2O", "O"), ("CO2", "O=C=O"), ("NH3", "N"), ("CH4", "C"),
        ("H2O2", "OO"), ("HCN", "C#N"), ("HCHO", "C=O"), ("C2H2", "C#C"),
        ("C2H4", "C=C"), ("C2H6", "CC"), ("CH3OH", "CO"), ("CO", "[C-]#[O+]"),
        ("NO2", "[N](=O)[O]"), ("N2O", "[O-][N+]#N"), ("SO2", "O=S=O"),
        ("H2S", "S"), ("HCl", "Cl"), ("HBr", "Br"), ("HF", "F"),
    ]

    def _formula_elements(formula):
        els = {}
        for m in re.finditer(r"([A-Z][a-z]?)(\d*)", formula):
            els[m.group(1)] = els.get(m.group(1), 0) + (int(m.group(2)) if m.group(2) else 1)
        return els

    smis = []
    # diatomics: A-B (and A2) for every pair of present elements
    import itertools

    for e1, e2 in itertools.combinations_with_replacement(sorted(present), 2):
        smis.append("[%s][%s]" % (e1, e2))
    # curated small molecules whose elements are all available in the feed
    for formula, smi in curated:
        if set(_formula_elements(formula)).issubset(present):
            smis.append(smi)
    out = []
    seen = set()
    for smi in smis:
        try:
            est = xtb.estimate(smi)
        except Exception:
            continue
        f = est.get("formula")
        if not f:
            continue
        els = tuple(sorted(_formula_elements(f).items()))
        if els in known_elements or els in seen:
            continue
        seen.add(els)
        est["note"] = "Enumerated candidate (not in the cited database); GFN2-xTB estimate only."
        out.append(est)
    return sorted(out, key=lambda e: e.get("formula", ""))


def react(items, T: float = 298.15, P_bar: float = 1.0, phase: str = "gas", use_xtb: bool = False) -> dict:
    """Run the My Lab bench: parse items, equilibrate, report products and reaction.

    Parameters
    ----------
    items : list[str] | str
        Atom/molecule list, e.g. ``["2 H2", "O2"]`` (count prefixes and comma
        separation accepted; see ``chem.parse_items``).
    T : float
        Temperature in K.
    P_bar : float
        Total pressure in bar (gas phase).
    phase : str
        ``'gas'`` restricts the candidate pool to gas species; anything else also
        allows condensed species (pure phases, activity 1).
    use_xtb : bool
        If True, additionally enumerate small valence-saturated molecules and
        estimate them with GFN2-xTB (returned separately, never mixed into the
        cited equilibrium).
    """
    from engineering_sim.science import chem

    T = float(T)
    P_bar = float(P_bar)
    parsed = chem.parse_items(items)

    # --- 1. map each parsed item to a thermo species and total the feed ----------
    feed_counts = {}
    feed_view = []
    feed_has_radicals = False
    unresolved = []
    for item in parsed:
        mol = item["molecule"]
        if mol.get("error") or not mol.get("smiles"):
            unresolved.append(item.get("query") or mol.get("query"))
            continue
        if mol.get("radical_electrons", 0) > 0:
            feed_has_radicals = True
        elements = _mol_elements(mol)
        key = _match_species(elements, phase)
        if key is None:
            unresolved.append(mol.get("query") or mol.get("formula"))
            continue
        feed_counts[key] = feed_counts.get(key, 0.0) + float(item["count"])
        feed_view.append({
            "query": mol.get("query"),
            "count": item["count"],
            "formula": mol.get("formula"),
            "smiles": mol.get("smiles"),
            "name": mol.get("name"),
            "species": key,
        })

    # --- 2. element inventory + candidate pool -----------------------------------
    elements = {}
    for k, n in feed_counts.items():
        for el, c in thermo.species(k)["elements"].items():
            elements[el] = elements.get(el, 0.0) + n * c
    candidates = []
    out_of_range = []  # species excluded because their data do not cover T (listed in notes)
    for entry in thermo.species():
        sp = thermo.species(entry["key"])
        if not set(sp["elements"]).issubset(set(elements)):
            continue
        if phase == "gas" and sp["phase"] != "gas":
            continue
        # Review fix (2026-09-14): a candidate whose cited Shomate data does not cover T is skipped and
        # reported, instead of aborting the whole bench (e.g. H2O2 data end at 1500 K, so H2 + O2 at 2000 K
        # previously raised). Feed species must still cover T; thermo raises for those with its message.
        try:
            thermo.props(entry["key"], T)
        except ValueError:
            out_of_range.append(entry["key"])
            continue
        candidates.append(entry["key"])

    # --- 3. equilibrium ----------------------------------------------------------
    if not feed_counts:
        eq = {"amounts": {}, "gas_mole_fractions": {}, "total_g": cite.value(0.0, "kJ", "derived", method=_METHOD),
              "n_gas_total": 0.0, "converged": True, "message": "empty feed"}
    else:
        eq = thermo.equilibrium(feed_counts, candidates, T, P_bar)

    # --- 4. products (sorted by amount) + direct reaction ------------------------
    products = []
    present = {k: amt for k, amt in eq["amounts"].items() if amt > 1e-9}
    for k, amt in sorted(present.items(), key=lambda kv: -kv[1]):
        sp = thermo.species(k)
        frac = eq["gas_mole_fractions"].get(k)
        products.append({
            "key": k,
            "name": sp["name"],
            "formula": sp["formula"],
            "smiles": sp["smiles"],
            "phase": sp["phase"],
            "amount": cite.value(amt, "mol", "derived", method=_METHOD),
            "mole_fraction": cite.value(frac, "dimensionless", "derived", method=_METHOD) if frac is not None else None,
            "molecule": _molecule_view(sp),
        })
    new_products = {k: amt for k, amt in present.items() if k not in feed_counts}
    reaction = _direct_reaction(feed_counts, new_products, T)

    if not new_products:
        verdict = "no reaction (already stable)"
    elif reaction is not None:
        verdict = "favored" if reaction["favored"] else "not favored"
    else:
        verdict = "not favored"

    # --- 5. energy diagram (H and G, reactants at 0) -----------------------------
    dH = cite.num(reaction["dH"]) if reaction else 0.0
    dG = cite.num(reaction["dG"]) if reaction else 0.0
    energy_diagram = [
        {"label": "Reactants H", "level_kJmol": 0.0, "kind": "reactants"},
        {"label": "Reactants G", "level_kJmol": 0.0, "kind": "reactants"},
        {"label": "Products H", "level_kJmol": dH, "kind": "products"},
        {"label": "Products G", "level_kJmol": dG, "kind": "products"},
    ]

    # --- 6. bonds formed/broken ---------------------------------------------------
    react_bonds, prod_bonds = {}, {}
    for item in parsed:
        mol = item["molecule"]
        if mol.get("error") or not mol.get("atoms"):
            continue
        bt = _bond_types(mol)
        for t, c in bt.items():
            react_bonds[t] = react_bonds.get(t, 0) + item["count"] * c
    for p in products:
        mol = p["molecule"]
        if mol.get("error") or not mol.get("atoms"):
            continue
        bt = _bond_types(mol)
        for t, c in bt.items():
            prod_bonds[t] = prod_bonds.get(t, 0) + cite.num(p["amount"]) * c
    bonds = {"formed": {}, "broken": {}, "bond_enthalpies": {}}
    for t in sorted(set(react_bonds) | set(prod_bonds)):
        r = react_bonds.get(t, 0)
        p = prod_bonds.get(t, 0)
        if p > r:
            bonds["formed"][t] = round(p - r, 3)
        elif r > p:
            bonds["broken"][t] = round(r - p, 3)
        # mean bond enthalpy when the element pair has a diatomic in the database
        m = re.match(r"^([A-Z][a-z]?)([-=:#])([A-Z][a-z]?)$", t)
        if m:
            be = _diatomic_bond_enthalpy(m.group(1), m.group(3), T)
            if be is not None:
                bonds["bond_enthalpies"][t] = be

    # --- 7. electron picture + notes + hazards ------------------------------------
    lewis_bits = []
    for item in parsed:
        mol = item["molecule"]
        if mol.get("lewis") and mol["lewis"].get("text"):
            lewis_bits.append("%s: %s" % (mol.get("formula") or mol.get("query"), mol["lewis"]["text"]))
    for p in products:
        mol = p["molecule"]
        if mol.get("lewis") and mol["lewis"].get("text"):
            lewis_bits.append("%s: %s" % (p["formula"], mol["lewis"]["text"]))
    electron_picture = " ".join(lewis_bits)

    notes = [_KINETICS_NOTE]
    if feed_has_radicals and any(thermo.species(k)["phase"] == "gas" for k in new_products):
        notes.append(_THIRD_BODY_NOTE)
    if out_of_range:
        notes.append("Excluded at %.1f K because their cited heat-capacity data do not cover this temperature: %s."
                     % (T, ", ".join(sorted(out_of_range))))

    hazards = []
    try:
        from engineering_sim.science import records

        for p in products:
            rec = records.lookup(p["name"], offline=True)
            if rec.get("found") and rec.get("hazards"):
                hazards.append({"key": p["key"], "name": p["name"], "hazards": rec["hazards"]})
    except Exception:
        hazards = []

    result = {
        "feed": feed_view,
        "conditions": {"T": T, "P_bar": P_bar, "phase": phase},
        "products": products,
        "reaction": reaction,
        "verdict": verdict,
        "energy_diagram": energy_diagram,
        "bonds": bonds,
        "electron_picture": electron_picture,
        "hazards": hazards,
        "notes": notes,
        "unresolved": unresolved,
        "converged": eq["converged"],
        "total_g": eq["total_g"],
    }
    if use_xtb:
        known = {tuple(sorted(thermo.species(entry["key"])["elements"].items()))
                 for entry in thermo.species()}
        result["estimated_candidates"] = _enumerate_xtb(elements, known)
    return result

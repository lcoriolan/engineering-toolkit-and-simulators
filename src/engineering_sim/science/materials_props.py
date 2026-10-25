# DESCRIPTION: Materials Science area (Task H): periodic-table element data and the
#              closed-form property calculators (radius ratio, rule of mixtures, thermal
#              mismatch stress, Hall-Petch, Brus quantum-dot band gap).
"""Materials property calculators for science toolkit.

Element data come from pymatgen's bundled ``periodic_table.json.gz`` and Shannon
ionic-radii table (``pymatgen/core/ionic_radii.json``); every number is wrapped in a
``cite`` value record. The bundled files are package data, so they are cited to the
package file and its upstream reference with ``verified=False`` (they were not
re-fetched from the original publisher during this build). Fundamental constants are
CODATA 2018 values read from the NIST all-constants ASCII table fetched during this
build, so those are ``verified=True``.

Units follow the conventions printed on the page: temperatures in K (melting points),
masses in amu, radii in Å, moduli in GPa, stress in MPa, thermal expansion in 1e-6/K
(ppm/K), energies in eV.
"""
from __future__ import annotations

import json
import os

from pymatgen.core import Element, periodic_table

from engineering_sim.science import cite

# ---- Fundamental constants (CODATA 2018, read from the NIST ASCII table) ----
# Fetched during this build from the NIST all-constants table (physics.nist.gov).
_CODATA_URL = "https://physics.nist.gov/cuu/Constants/Table/allascii.txt"
_RETRIEVED = "2026-09-15"
N_A = 6.02214076e23          # mol^-1  (exact)
M_E = 9.1093837139e-31       # kg      electron mass
E_CHG = 1.602176634e-19      # C       elementary charge (exact)
HBAR = 1.054571817e-34       # J s     reduced Planck constant (exact)
EPS0 = 8.8541878188e-12      # F/m     vacuum electric permittivity

# pymatgen-bundled data provenance (not re-fetched upstream during this build).
_PMG_SRC = "pymatgen 2026.5.4 periodic_table.json.gz (bundled package data)"
_IR_SRC = "pymatgen 2026.5.4 core/ionic_radii.json (bundled package data; upstream R. D. Shannon, Acta Cryst. A32 (1976) 751)"


def _codata(value, unit, name):
    """CODATA 2018 constant read from the NIST table during this build."""
    return cite.value(value, unit, "cited", source=_CODATA_URL, retrieved=_RETRIEVED,
                      verified=True, note="CODATA 2018: " + name)


def _pmg(value, unit, note):
    """A value read from pymatgen's bundled periodic-table data (not re-fetched)."""
    return cite.value(value, unit, "cited", source=_PMG_SRC, verified=False, note=note)


def element(symbol: str):
    """Periodic-table card for one element from pymatgen ``Element``.

    Returns the named properties as cite records, plus a ``notes`` list. ``density``
    and ``crystal_structure`` are not present in pymatgen's element table and are
    reported as ``null`` with an explanatory note (flagged missing, not invented).
    """
    sym = (symbol or "").strip().capitalize()
    try:
        el = Element(sym)
    except Exception as exc:  # noqa: BLE001 - unknown symbol -> clear message
        return {"error": "Unknown element symbol: %s (%s)" % (symbol, type(exc).__name__)}

    out = {
        "symbol": el.symbol,
        "name": el.long_name or el.name,
        "atomic_number": el.Z,
        "category": ("metal" if el.is_metal else "metalloid" if el.is_metalloid else "nonmetal"),
    }
    if el.atomic_mass is not None:
        out["atomic_mass"] = _pmg(float(el.atomic_mass), "amu",
                                  "standard atomic weight (upstream CIAAW/IUPAC)")
    if el.X is not None:
        out["electronegativity"] = _pmg(float(el.X), "Pauling",
                                         "Pauling electronegativity")
    if getattr(el, "melting_point", None) is not None:
        out["melting_point"] = _pmg(float(el.melting_point), "K",
                                    "normal melting point (upstream compiled element data)")
    for key, label in (("atomic_radius", "atomic radius"),
                       ("atomic_radius_calculated", "calculated atomic radius"),
                       ("metallic_radius", "metallic radius")):
        v = getattr(el, key, None)
        if v is not None:
            out[key] = _pmg(float(v), "Å", label)
    # Shannon ionic radii: per oxidation state (and per coordination number) from the
    # bundled ionic_radii.json. Summary lists the common oxidation state(s) and the
    # CN-6 radius where available.
    ionic = getattr(el, "ionic_radii", None) or {}
    if ionic:
        out["ionic_radii"] = {str(ox): float(r) for ox, r in sorted(ionic.items(), key=lambda kv: kv[0])}
        out["common_oxidation_states"] = [int(o) for o in el.common_oxidation_states]
    notes = [
        "density and crystal_structure are not provided by pymatgen's element table; use the crystal tab for a measured density.",
        "Radii and melting points are compiled/upstream values bundled with pymatgen (not re-fetched from the publisher); hover badges show the source note.",
    ]
    out["notes"] = notes
    return out


# ---- Shannon ionic radii (pymatgen bundled ionic_radii.json) -----------------
_IONIC_RADII = None


def _load_ionic_radii() -> dict:
    """Load pymatgen's Shannon ionic-radii table once (element -> oxidation -> CN -> Å).

    The table stores Shannon *crystal* radii (r(O2-, CN 6) = 1.26 Å convention).
    """
    global _IONIC_RADII
    if _IONIC_RADII is None:
        pkg = os.path.dirname(os.path.abspath(periodic_table.__file__))
        with open(os.path.join(pkg, "ionic_radii.json"), "r", encoding="utf-8") as f:
            _IONIC_RADII = json.load(f)
    return _IONIC_RADII


def _shannon_radius(symbol: str, oxidation: int, cn: int):
    """(radius_Å, actual_CN) for element at oxidation state, nearest available CN."""
    table = _load_ionic_radii()
    entry = table.get(symbol.capitalize(), {})
    cns = entry.get(str(oxidation), {})
    if not cns:
        return None, None
    keys = sorted(int(k) for k in cns)
    if cn in keys:
        return float(cns[str(cn)]), cn
    nearest = min(keys, key=lambda k: abs(k - cn))
    return float(cns[str(nearest)]), nearest


def ionic_radius_ratio(cation: str, anion: str, CN: int = 6):
    """Cation/anion Shannon radius ratio and the Pauling coordination-number prediction.

    Radii are Shannon crystal radii from pymatgen's ``ionic_radii.json`` at the requested
    coordination number (nearest tabulated CN when the exact one is absent). The cation
    uses its highest common oxidation state, the anion its lowest, which is the usual
    ionic-ratio convention and is noted in the result.
    """
    cat, an = cation.strip().capitalize(), anion.strip().capitalize()
    try:
        cat_el, an_el = Element(cat), Element(an)
    except Exception as exc:  # noqa: BLE001
        return {"error": "Unknown element: %s (%s)" % (str(exc), type(exc).__name__)}
    cn = int(CN)
    if cn < 2 or cn > 12:
        return {"error": "CN must be an integer in 2..12 (got %d)." % cn}
    cat_ox = max(cat_el.common_oxidation_states or (1,))
    an_ox = min(an_el.common_oxidation_states or (-1,))
    r_cat, cn_cat = _shannon_radius(cat, cat_ox, cn)
    r_an, cn_an = _shannon_radius(an, an_ox, cn)
    if r_cat is None or r_an is None:
        return {"error": "No Shannon radius available for %s(%+d) or %s(%+d)."
                         % (cat, cat_ox, an, an_ox)}
    ratio = r_cat / r_an
    predicted, label = _pauling_cn(ratio)
    matched = predicted == cn
    return {
        "cation": cat, "cation_oxidation": cat_ox,
        "anion": an, "anion_oxidation": an_ox,
        "requested_cn": cn,
        "r_cation": cite.value(r_cat, "Å", "cited", source=_IR_SRC, verified=False,
                               note="Shannon crystal radius, CN %d (requested %d)" % (cn_cat, cn)),
        "r_anion": cite.value(r_an, "Å", "cited", source=_IR_SRC, verified=False,
                              note="Shannon crystal radius, CN %d (requested %d)" % (cn_an, cn)),
        "ratio": cite.value(ratio, "", "derived", method="r(cation)/r(anion)"),
        "predicted_cn": cite.value(predicted, "", "derived",
                                   method="Pauling radius-ratio rule (geometric thresholds)"),
        "predicted_label": label,
        "matches_requested_cn": bool(matched),
        "notes": [
            "Radii are Shannon crystal radii (r(O2-, VI)=1.26 Å convention), not effective ionic radii; the ratio is close to the effective-ionic ratio.",
            "Cation uses its highest common oxidation state, anion its lowest.",
        ],
    }


def _pauling_cn(ratio: float):
    """Pauling coordination prediction from r(cation)/r(anion).

    Thresholds are exact from cation/anion contact geometry:
    CN 2 < 0.155, CN 3 0.155-0.225, CN 4 0.225-0.414, CN 6 0.414-0.732,
    CN 8 0.732-1.0, CN 12 > 1.0.
    """
    if ratio < 0.155:
        return 2, "linear"
    if ratio < 0.225:
        return 3, "trigonal planar"
    if ratio < 0.414:
        return 4, "tetrahedral"
    if ratio < 0.732:
        return 6, "octahedral"
    if ratio < 1.0:
        return 8, "cubic (body-centred)"
    return 12, "cuboctahedral / close-packed"


# ---- Rule of mixtures for a two-phase composite ------------------------------
def rule_of_mixtures(props, fractions, bound="voigt"):
    """Voigt / Reuss / Hashin-Shtrikman bounds on the effective elastic moduli.

    ``props`` maps two phase names to ``{'K': bulk GPa, 'G': shear GPa, 'E': Young GPa}``
    (K and G required; E optional). ``fractions`` are the two volume fractions (sum 1).
    ``bound`` selects the headline bound: 'voigt', 'reuss', or 'hashin_shtrikman'.

    Returns every bound as cite records so the caller (and the ordering test) can
    compare them. Hashin-Shtrikman assumes isotropic phases ordered K2>=K1, G2>=G1
    (phase 2 stiffer); if the caller's phases are not ordered the phases are swapped
    internally and noted.
    """
    names = list(props.keys())
    if len(names) != 2:
        return {"error": "rule_of_mixtures needs exactly two phases (got %d)." % len(names)}
    if len(fractions) != 2 or abs(sum(fractions) - 1.0) > 1e-6:
        return {"error": "fractions must be two numbers summing to 1 (got %s)." % fractions}
    for n in names:
        p = props[n] or {}
        for k in ("K", "G"):
            if not isinstance(p.get(k), (int, float)):
                return {"error": "phase %r is missing a numeric '%s' (GPa)." % (n, k)}
    f = [float(fractions[0]), float(fractions[1])]
    P = [props[names[0]], props[names[1]]]
    # Order phases so that phase 2 is the stiffer one (required by the HS formula).
    if P[0]["K"] > P[1]["K"] or (P[0]["K"] == P[1]["K"] and P[0]["G"] > P[1]["G"]):
        f = [f[1], f[0]]
        P = [P[1], P[0]]
        names = [names[1], names[0]]
        swapped = True
    else:
        swapped = False
    K1, K2 = P[0]["K"], P[1]["K"]
    G1, G2 = P[0]["G"], P[1]["G"]

    def _rec(v, unit, method):
        return cite.value(v, unit, "derived", method=method)

    def _voigt(x1, x2):
        return f[0] * x1 + f[1] * x2

    def _reuss(x1, x2):
        return 1.0 / (f[0] / x1 + f[1] / x2)

    # Hashin-Shtrikman bounds (isotropic, K2>=K1, G2>=G1).
    def _hs_k_upper():
        return K2 + f[0] / (1.0 / (K1 - K2) + 3.0 * f[1] / (3.0 * K2 + 4.0 * G2))

    def _hs_k_lower():
        return K1 + f[1] / (1.0 / (K2 - K1) + 3.0 * f[0] / (3.0 * K1 + 4.0 * G1))

    def _hs_g_upper():
        return G2 + f[0] / (1.0 / (G1 - G2)
                            + 6.0 * f[1] * (K2 + 2.0 * G2) / (5.0 * G2 * (3.0 * K2 + 4.0 * G2)))

    def _hs_g_lower():
        return G1 + f[1] / (1.0 / (G2 - G1)
                            + 6.0 * f[0] * (K1 + 2.0 * G1) / (5.0 * G1 * (3.0 * K1 + 4.0 * G1)))

    bounds = {
        "voigt": {"K": _rec(_voigt(K1, K2), "GPa", "Voigt rule of mixtures (isostrain upper bound)"),
                  "G": _rec(_voigt(G1, G2), "GPa", "Voigt rule of mixtures (isostrain upper bound)")},
        "reuss": {"K": _rec(_reuss(K1, K2), "GPa", "Reuss rule of mixtures (isostress lower bound)"),
                  "G": _rec(_reuss(G1, G2), "GPa", "Reuss rule of mixtures (isostress lower bound)")},
        "hs_upper": {"K": _rec(_hs_k_upper(), "GPa", "Hashin-Shtrikman upper bound (isotropic)"),
                     "G": _rec(_hs_g_upper(), "GPa", "Hashin-Shtrikman upper bound (isotropic)")},
        "hs_lower": {"K": _rec(_hs_k_lower(), "GPa", "Hashin-Shtrikman lower bound (isotropic)"),
                     "G": _rec(_hs_g_lower(), "GPa", "Hashin-Shtrikman lower bound (isotropic)")},
    }
    for b in ("voigt", "reuss"):
        if isinstance(P[0].get("E"), (int, float)) and isinstance(P[1].get("E"), (int, float)):
            E = _voigt(P[0]["E"], P[1]["E"]) if b == "voigt" else _reuss(P[0]["E"], P[1]["E"])
            bounds[b]["E"] = _rec(E, "GPa", "%s rule of mixtures" % ("Voigt" if b == "voigt" else "Reuss"))

    if bound not in ("voigt", "reuss", "hashin_shtrikman"):
        bound = "voigt"
    headline = bounds["voigt"] if bound == "voigt" else bounds["reuss"] if bound == "reuss" else bounds["hs_upper"]
    return {
        "bound": bound,
        "headline": headline,
        "bounds": bounds,
        "phases": names,
        "fractions": [f[0], f[1]],
        "notes": [
            "Voigt/Reuss are isostrain/isostress bounds on the effective modulus of an ideal two-phase composite.",
            "Hashin-Shtrikman assumes isotropic, well-ordered phases (phase 2 stiffer); it is tighter than Voigt/Reuss but still a bound, not the exact modulus.",
        ] + (["Phases were reordered so the stiffer phase is phase 2 (HS convention)."] if swapped else []),
    }


# ---- Thermal expansion mismatch stress --------------------------------------
def thermal_expansion_mismatch(alpha1, alpha2, dT, E, nu):
    """Biaxial residual stress from a thermal expansion mismatch.

    Film-on-thick-substrate model: sigma = E * (alpha1 - alpha2) * dT / (1 - nu).
    ``alpha`` in 1e-6/K (ppm/K), ``dT`` in K (cooling = negative), ``E`` in GPa,
    ``nu`` dimensionless; result in MPa (E[GPa] * dalpha[1e-6/K] * dT[K] / (1-nu)
    gives kPa, so the MPa result carries a 1e-3 factor).
    """
    for name, val in (("alpha1", alpha1), ("alpha2", alpha2), ("dT", dT), ("E", E), ("nu", nu)):
        if not isinstance(val, (int, float)):
            return {"error": "%s must be numeric (got %r)." % (name, val)}
    if not (-1.0 < nu < 0.5):
        return {"error": "Poisson ratio nu must be in (-1, 0.5) (got %r)." % nu}
    da = float(alpha1) - float(alpha2)
    sigma = float(E) * da * float(dT) / (1.0 - float(nu)) * 1e-3  # MPa
    return {
        "stress": cite.value(sigma, "MPa", "estimated",
                             method="sigma = E*(alpha1-alpha2)*dT/(1-nu) (film on thick substrate)"),
        "delta_alpha": cite.value(da, "1e-6/K", "derived", method="alpha1 - alpha2"),
        "notes": [
            "Assumes an ideal elastic film bonded to a thick, unyielding substrate; no plastic relaxation, no creep, no substrate constraint factor.",
            "A negative dT (cooling) with a positive delta_alpha gives a negative (compressive) stress.",
        ],
    }


# ---- Hall-Petch --------------------------------------------------------------
def hall_petch(sigma0, k, d):
    """Yield strength from the Hall-Petch relation: sigma_y = sigma0 + k / sqrt(d).

    ``sigma0`` (MPa, friction stress), ``k`` (MPa·sqrt(m), locking parameter),
    ``d`` (m, grain size). Result in MPa. Empirical; validity ends at very small
    grain sizes (inverse Hall-Petch) which is noted but not modelled.
    """
    for name, val in (("sigma0", sigma0), ("k", k), ("d", d)):
        if not isinstance(val, (int, float)):
            return {"error": "%s must be numeric (got %r)." % (name, val)}
    if float(d) <= 0:
        return {"error": "grain size d must be positive (got %r)." % d}
    sigma = float(sigma0) + float(k) / (float(d) ** 0.5)
    return {
        "yield_strength": cite.value(sigma, "MPa", "estimated",
                                     method="Hall-Petch: sigma_y = sigma0 + k/sqrt(d)"),
        "notes": [
            "Empirical grain-boundary strengthening relation; assumes d in metres and k in MPa·sqrt(m).",
            "Breaks down (inverse Hall-Petch softening) below roughly 10-20 nm, which is not modelled.",
        ],
    }


# ---- Brus quantum-dot band gap ----------------------------------------------
def band_gap_brus(bulk_gap_eV, radius_nm, me, mh, eps_r):
    """Quantum-dot optical gap from the Brus effective-mass equation.

    E(R) = E_g + (hbar^2 pi^2 / (2 R^2 m0)) (1/me + 1/mh) - (1.786 e / (4 pi eps0 eps_r R)),
    with effective masses in units of the electron mass, R the dot radius, eps_r the
    semiconductor relative permittivity. As R -> infinity the corrections vanish and
    E -> bulk gap (tested). Effective-mass approximation: valid for radii larger than
    a few lattice constants; the confined exciton Coulomb term uses the classical
    1.786 prefactor.
    """
    for name, val in (("bulk_gap_eV", bulk_gap_eV), ("radius_nm", radius_nm),
                      ("me", me), ("mh", mh), ("eps_r", eps_r)):
        if not isinstance(val, (int, float)):
            return {"error": "%s must be numeric (got %r)." % (name, val)}
    R = float(radius_nm) * 1e-9  # m
    if R <= 0:
        return {"error": "radius_nm must be positive (got %r)." % radius_nm}
    if float(me) <= 0 or float(mh) <= 0 or float(eps_r) <= 0:
        return {"error": "me, mh and eps_r must all be positive."}
    # Confinement energy coefficient (J*m^2 -> eV via /e): hbar^2 pi^2 / (2 m0 e).
    A = (HBAR ** 2) * (3.141592653589793 ** 2) / (2.0 * M_E * E_CHG)  # V*m^2
    confinement = A * (1.0 / float(me) + 1.0 / float(mh)) / (R * R)
    # Coulomb attraction coefficient (V*m): 1.786 e / (4 pi eps0 eps_r).
    coulomb = 1.786 * E_CHG / (4.0 * 3.141592653589793 * EPS0 * float(eps_r) * R)
    egap = float(bulk_gap_eV) + confinement - coulomb
    return {
        "band_gap": cite.value(egap, "eV", "estimated",
                               method="Brus effective-mass equation (quantum dot)"),
        "confinement_term": cite.value(confinement, "eV", "derived",
                                       method="(hbar^2 pi^2/(2 m0 R^2)) (1/me + 1/mh)"),
        "coulomb_term": cite.value(coulomb, "eV", "derived",
                                   method="1.786 e/(4 pi eps0 eps_r R)"),
        "notes": [
            "Effective-mass approximation; ignores surface states, shape anisotropy, and dielectric confinement.",
            "Constants (hbar, m0, e, eps0) are CODATA 2018 (NIST); the 1.786 exciton prefactor is the classical Brus value.",
        ],
    }

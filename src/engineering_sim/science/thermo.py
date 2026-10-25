# DESCRIPTION: Cited NIST WebBook thermochemistry: Shomate heat capacity, reaction
#              thermodynamics, and ideal-gas + pure-condensed-phase Gibbs equilibrium.
"""Thermochemistry engine for science toolkit.

Data: ``engineering_sim/science/data/thermo_species.json`` (fetched from the NIST WebBook;
see the ``_meta`` block and per-value ``source`` / ``retrieved`` / ``verified``).

Conventions
-----------
* Standard state: 1 bar, ideal gas; pure condensed phases have activity 1.
* ``props`` returns the standard-state molar properties at temperature ``T``
  (kJ/mol for h and g; J/mol*K for cp and s) as cite value records.
* Shomate equation (NIST WebBook): t = T/1000,
    Cp = A + B t + C t^2 + D t^3 + E/t^2                        (J/mol*K)
    H - H298 = A t + B t^2/2 + C t^3/3 + D t^4/4 - E/t + F - H  (kJ/mol)
    S = A ln t + B t + C t^2/2 + D t^3/3 - E/(2 t^2) + G        (J/mol*K)
  The "H" coefficient equals dHf(298.15 K), so the Shomate fit is anchored so that
  H - H298 == 0 at 298.15 K. Species without Shomate data use a flagged constant
  Cp with H - H298 = Cp*(T-298.15)/1000 and S = S298 + Cp*ln(T/298.15).
* ``reaction`` is the direct stoichiometric reaction between two species sets and
  reports equilibrium-relevant quantities (thermodynamics, not kinetics).
* ``equilibrium`` minimises the total Gibbs energy over a fixed candidate pool with
  element conservation. It is a solver: amounts/fractions are plain floats, not
  value records (the page-facing ``bench.react`` wraps the reported numbers).

Only standard-library + numpy/scipy are imported; RDKit/tblite live in ``xtb.py``.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

from . import cite

# Gas constant in the two unit systems used here. R (J/mol*K) appears in entropy
# and Cp terms; R_kJ (kJ/mol*K) appears in Gibbs-energy and equilibrium constants.
R = 8.314462618  # J/(mol K), CODATA 2018
R_KJ = R / 1000.0  # kJ/(mol K)
T_REF = 298.15  # K — Shomate anchor temperature
P_STD = 1.0  # bar — standard pressure
METHOD = "Shomate/NIST WebBook"

_DATA = None
_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "thermo_species.json")


def _load() -> dict:
    """Load the species database once and cache it (module-level)."""
    global _DATA
    if _DATA is None:
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            _DATA = json.load(f)
    return _DATA


def species(key: str | None = None):
    """Look up one species by key (``'H2O(g)'``) or list all species.

    With no argument, returns a sorted list of ``{key, name, formula, phase}``
    dictionaries (used by the ``species`` worker op). With a key, returns the full
    species record (or ``None`` if unknown).
    """
    data = _load()
    if key is None:
        return [
            {"key": k, "name": v["name"], "formula": v["formula"], "phase": v["phase"]}
            for k, v in sorted(data.items())
            if k != "_meta"
        ]
    return data.get(key)


def _plain(rec):
    """Number from a value record (or a bare float)."""
    return cite.num(rec)


def _shomate_range(spec: dict, T: float) -> dict:
    """Pick the Shomate coefficient range for temperature T.

    Raises ValueError outside [298.15 K, max T]. Between 298.15 K and the first
    range's lower bound the first range is used (the fit is anchored so that
    H - H298 == 0 and S == S298 at 298.15 K).
    """
    ranges = spec["shomate"]
    tmax = max(r["Tmax"] for r in ranges)
    if T < T_REF - 1e-9:
        raise ValueError(
            "%s: T=%.1f K is below 298.15 K (Shomate anchor); data range starts at %.1f K"
            % (spec["formula"], T, ranges[0]["Tmin"])
        )
    if T > tmax:
        raise ValueError(
            "%s: T=%.1f K exceeds the Shomate data range (max %.1f K)"
            % (spec["formula"], T, tmax)
        )
    for r in ranges:
        if r["Tmin"] <= T < r["Tmax"]:
            return r
    if T == tmax:
        return ranges[-1]
    # 298.15 <= T < first range lower bound -> anchor region, use the first range.
    return ranges[0]


def props(key: str, T: float):
    """Standard-state molar properties of ``key`` at ``T`` as value records.

    Returns ``{cp, h, s, g}`` (all ``kind='derived'``, method "Shomate/NIST WebBook"):
    cp in J/mol*K, h = dHf298 + (H-H298) in kJ/mol, s in J/mol*K,
    g = h - T*s/1000 in kJ/mol. Raises KeyError for an unknown species and
    ValueError outside the valid temperature range.
    """
    spec = species(key)
    if spec is None:
        raise KeyError("unknown thermo species: %r" % (key,))
    dhf = _plain(spec["dHf298_kJmol"])  # kJ/mol
    s298 = _plain(spec["S298_JmolK"])  # J/mol*K
    T = float(T)
    if spec["cp_model"] == "shomate":
        r = _shomate_range(spec, T)
        t = T / 1000.0
        A, B, C, D, E, F, G, H = (float(r[k]) for k in "ABCDEFGH")
        cp = A + B * t + C * t * t + D * t ** 3 + E / (t * t)
        dh = A * t + B * t * t / 2 + C * t ** 3 / 3 + D * t ** 4 / 4 - E / t + F - H
        s = A * math.log(t) + B * t + C * t * t / 2 + D * t ** 3 / 3 - E / (2 * t * t) + G
    else:
        # Constant-Cp approximation (species without Shomate data; flagged in the DB).
        cp = _plain(spec["cp298_JmolK"])
        dh = cp / 1000.0 * (T - T_REF)
        s = s298 + cp * math.log(T / T_REF)
    h = dhf + dh
    g = h - T * s / 1000.0
    return {
        "cp": cite.value(cp, "J/mol*K", "derived", method=METHOD),
        "h": cite.value(h, "kJ/mol", "derived", method=METHOD),
        "s": cite.value(s, "J/mol*K", "derived", method=METHOD),
        "g": cite.value(g, "kJ/mol", "derived", method=METHOD),
    }


def _elements(spec: dict) -> dict:
    """Element composition {El: n} of a species record."""
    return spec["elements"]


def _balance(reactants: dict, products: dict):
    """Return ``(balanced, imbalance)`` for a stoichiometric reaction.

    ``imbalance`` maps element -> (reactant_atoms - product_atoms). Unknown species
    keys raise KeyError (also from the side dicts here).
    """
    imbalance = {}
    for key, coeff in reactants.items():
        if species(key) is None:
            raise KeyError("unknown thermo species: %r" % (key,))
        for el, n in species(key)["elements"].items():
            imbalance[el] = imbalance.get(el, 0.0) + coeff * n
    for key, coeff in products.items():
        if species(key) is None:
            raise KeyError("unknown thermo species: %r" % (key,))
        for el, n in species(key)["elements"].items():
            imbalance[el] = imbalance.get(el, 0.0) - coeff * n
    imbalance = {el: d for el, d in imbalance.items() if abs(d) > 1e-9}
    return (not imbalance), imbalance


def reaction(reactants: dict, products: dict, T: float = 298.15):
    """Direct reaction ``reactants -> products`` at temperature T.

    ``reactants`` / ``products`` are ``{species_key: stoichiometric_coefficient}``
    (e.g. ``{"H2": 2, "O2": 1} -> {"H2O(g)": 2}``). Element balance is checked and a
    ValueError is raised (stating the imbalance) if it fails.

    Returns value records: dH (kJ/mol), dS (J/mol*K), dG (kJ/mol),
    K = exp(-dG/RT) (dimensionless), ``favored`` (bool, dG < 0), and per-species
    ``contributions``. All derived by the Shomate/NIST WebBook method.
    """
    T = float(T)
    balanced, imbalance = _balance(reactants, products)
    if not balanced:
        detail = ", ".join("%s: %+.4g" % (el, d) for el, d in sorted(imbalance.items()))
        raise ValueError(
            "reaction is not element-balanced (%s); reactants and products have different atom counts"
            % detail
        )
    dh = ds = dg = 0.0
    contributions = {}
    for label, side in (("reactants", reactants), ("products", products)):
        for key, coeff in side.items():
            p = props(key, T)
            h = _plain(p["h"])
            s = _plain(p["s"])
            g = _plain(p["g"])
            contributions[key] = {
                "coefficient": coeff,
                "h": p["h"],
                "s": p["s"],
                "g": p["g"],
            }
            sign = -1.0 if label == "reactants" else 1.0
            dh += sign * coeff * h
            ds += sign * coeff * s
            dg += sign * coeff * g
    K = math.exp(-dg * 1000.0 / (R * T))
    return {
        "T": T,
        "balanced": True,
        "reactants": reactants,
        "products": products,
        "dH": cite.value(dh, "kJ/mol", "derived", method=METHOD),
        "dS": cite.value(ds, "J/mol*K", "derived", method=METHOD),
        "dG": cite.value(dg, "kJ/mol", "derived", method=METHOD),
        "K": cite.value(K, "dimensionless", "derived", method=METHOD),
        "favored": bool(dg < 0.0),
        "contributions": contributions,
    }


def equilibrium(feed: dict, candidates: list, T: float, P_bar: float):
    """Gibbs-energy minimisation over ``candidates`` with element conservation.

    Parameters
    ----------
    feed : dict
        ``{species_key: moles}`` of the initial material (e.g. ``{"H2": 1.0}``).
    candidates : list[str]
        Species keys allowed to form (a subset of the database). Feed keys are
        added automatically if missing, so the initial state is always reachable.
    T : float
        Temperature in K (298.15 <= T <= 6000).
    P_bar : float
        Total pressure in bar (used only for the gas phase; condensed phases are
        pure with activity 1).

    Returns
    -------
    dict with ``amounts`` (mol per species, trace < 1e-12 clipped to 0),
    ``gas_mole_fractions`` (gas species with positive amount), ``total_g``
    (kJ for the whole system), ``n_gas_total`` (mol), ``converged`` (bool),
    ``message``, ``T`` and ``P_bar``. Condensed phases are allowed to vanish.
    """
    T = float(T)
    P_bar = float(P_bar)
    data = _load()
    cand = list(candidates)
    for k in feed:
        if k not in cand:
            cand.append(k)
    # Partition into gas and condensed candidates and resolve species records.
    gas_keys, cond_keys = [], []
    for k in cand:
        spec = data.get(k)
        if spec is None:
            raise KeyError("unknown thermo species: %r" % (k,))
        (gas_keys if spec["phase"] == "gas" else cond_keys).append(k)
    if not gas_keys:
        raise ValueError("equilibrium needs at least one gas-phase candidate")
    # Element inventory of the feed (conserved quantities).
    elements = []
    b = {}
    for k, moles in feed.items():
        for el, n in data[k]["elements"].items():
            if el not in b:
                elements.append(el)
            b[el] = b.get(el, 0.0) + moles * n
    elements = sorted(elements)
    b_vec = np.array([b.get(el, 0.0) for el in elements], dtype=float)
    # Stoichiometric matrix rows for every candidate over the sorted element list.
    def row(key):
        spec = data[key]
        return np.array([spec["elements"].get(el, 0.0) for el in elements], dtype=float)

    K = len(gas_keys)
    M = len(cond_keys)
    A_gas = np.zeros((K, len(elements)))
    for i, k in enumerate(gas_keys):
        A_gas[i] = row(k)
    A_cond = np.zeros((M, len(elements)))
    for j, k in enumerate(cond_keys):
        A_cond[j] = row(k)
    g_gas = np.array([_plain(props(k, T)["g"]) for k in gas_keys])   # kJ/mol
    g_cond = np.array([_plain(props(k, T)["g"]) for k in cond_keys])  # kJ/mol
    RT = R_KJ * T  # kJ/mol

    # Variables v = [ln(n_gas_1), ..., ln(n_gas_K), n_cond_1, ..., n_cond_M].
    # Gas species are optimised in log-moles so amounts stay positive and the
    # ln(n_i / n_g) ideal-gas mixing term is well behaved down to ~1e-13 mol.
    def unpack(v):
        x = v[:K]
        n_gas = np.exp(x)
        n_cond = v[K:]
        return n_gas, n_cond

    def energy(v):
        n_gas, n_cond = unpack(v)
        n_tot = n_gas.sum()
        # G = sum_i n_i [g_i + RT ln(n_i/n_tot * P)] + sum_j n_j g_j  (kJ)
        mix = RT * (np.log(np.maximum(n_gas, 1e-300)) - math.log(n_tot) + math.log(P_bar))
        return float(np.dot(n_gas, g_gas + mix) + np.dot(n_cond, g_cond))

    def jac(v):
        n_gas, n_cond = unpack(v)
        n_tot = n_gas.sum()
        mu = g_gas + RT * (np.log(np.maximum(n_gas, 1e-300)) - math.log(n_tot) + math.log(P_bar))
        return np.concatenate([n_gas * mu, g_cond])

    def cons(v):
        n_gas, n_cond = unpack(v)
        return A_gas.T @ n_gas + A_cond.T @ n_cond - b_vec

    def cons_jac(v):
        n_gas, n_cond = unpack(v)
        # d cons_e / d x_i = a_ei * n_i ; d cons_e / d n_j = a_ej
        J = np.zeros((len(elements), K + M))
        J[:, :K] = A_gas.T * n_gas[None, :]
        if M:
            J[:, K:] = A_cond.T
        return J

    # Initial guess: put the feed back into its own species (gas), tiny amount of
    # every other gas, condensed absent. Satisfies element balance to ~1e-6 mol.
    x0 = np.full(K, math.log(1e-6))
    for i, k in enumerate(gas_keys):
        if k in feed and feed[k] > 0:
            x0[i] = math.log(float(feed[k]))
    n0 = np.zeros(M)
    v0 = np.concatenate([x0, n0])
    # Physical upper bound: no species can hold more moles than the total number of
    # atoms fed (each species has >= 1 atom). Capping the log-moles here keeps the
    # ideal-gas objective bounded (it is linear under a uniform scaling of all
    # amounts) so the solver cannot run away to spurious large amounts at high T.
    atom_total = float(sum(b.values()))
    hi = math.log(max(atom_total, 1.0))
    bounds = [(math.log(1e-13), hi)] * K + [(0.0, atom_total)] * M

    # SLSQP can stall when the pure-reactant start leaves some product species at a
    # trace amount (their constraint rows then look negligible). Run from a few
    # deterministic starts and keep the lowest-energy converged solution.
    n_species_total = K + M
    share = max(atom_total / max(n_species_total, 1), 1e-6)
    starts = [
        v0,                                                        # feed in its own species
        np.concatenate([np.full(K, math.log(share)), np.full(M, 0.0)]),  # equal split of atoms
        np.concatenate([np.full(K, math.log(1e-3)), np.zeros(M)]),  # small uniform gas
    ]
    best = None
    best_g = float("inf")
    messages = []
    for start in starts:
        r = minimize(
            energy,
            start,
            method="SLSQP",
            jac=jac,
            bounds=bounds,
            constraints=NonlinearConstraint(cons, 0.0, 0.0, jac=cons_jac),
            options={"maxiter": 5000, "ftol": 1e-14},
        )
        g_val = float(energy(r.x))
        messages.append(str(r.message))
        if g_val < best_g:
            best_g = g_val
            best = r
    res = best
    v_opt = res.x
    n_gas, n_cond = unpack(v_opt)
    amounts = {}
    for i, k in enumerate(gas_keys):
        amounts[k] = 0.0 if n_gas[i] < 1e-12 else float(n_gas[i])
    for j, k in enumerate(cond_keys):
        amounts[k] = 0.0 if n_cond[j] < 1e-12 else float(n_cond[j])
    n_tot = float(n_gas[n_gas >= 1e-12].sum())
    gas_fractions = {}
    for i, k in enumerate(gas_keys):
        if n_gas[i] >= 1e-12 and n_tot > 0:
            gas_fractions[k] = float(n_gas[i] / n_tot)
    total_g = float(energy(v_opt))
    return {
        "T": T,
        "P_bar": P_bar,
        "amounts": amounts,
        "gas_mole_fractions": gas_fractions,
        "total_g": cite.value(total_g, "kJ", "derived", method="Gibbs minimisation (%s)" % METHOD),
        "n_gas_total": n_tot,
        "converged": bool(res.success),
        "message": str(res.message),
    }

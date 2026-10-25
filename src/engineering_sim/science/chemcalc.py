# DESCRIPTION: Task J: Chemical Calculator engine — deterministic breadth-first edit search,
#              recipe identity, PubChem snapshot, portable export and verify.
"""Chemical Calculator: nearest valid compounds to a start molecule.

Determinism is the product. The same recipe always produces a byte-identical
``result.json``: no randomness, no dict/set iteration order leaking into output
(everything is sorted explicitly or driven by the recipe's own list order), no
timestamps inside hashed content, and fixed RDKit options. Network state may
change whether a run reaches PubChem, but an offline replay over a saved
snapshot reproduces the original bytes exactly.

Nearest combines three signals:
  1. structural edit distance (primary; breadth-first over graph edits),
  2. RDKit atom-pair *count* Tanimoto similarity against the start (an
     element-count Tanimoto fallback covers single-atom molecules whose
     atom-pair fingerprint is empty),
  3. a one-reaction link using science toolkit's gas-phase NIST WebBook thermochemistry.

Validity is RDKit sanitisation plus the recipe constraints; it is NOT a claim
of stability or synthesizability, and "hypothetical" is not a claim of existence.

The engine is import-robust so a portable copy can vendor ``cite``/``paths``/
``thermo`` as a flat ``science_bundle`` package (see ``export``).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator

try:  # normal checkout: package imports
    from engineering_sim.science import cite, paths, thermo
except ImportError:  # pragma: no cover - portable vendored copy (flat `science_bundle` package)
    from science_bundle import cite, paths, thermo

HARNESS = 'science_bundle_chemcalc'
HARNESS_VERSION = '1.0.0'


MAIN_GROUP = [
    'H', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
    'Cl', 'K', 'Ca', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'In', 'Sn', 'Sb',
    'Te', 'I', 'Cs', 'Ba', 'Tl', 'Pb', 'Bi', 'He', 'Ne', 'Ar', 'Kr', 'Xe',
]

DEFAULT_ELEMENTS = list(MAIN_GROUP)
DEFAULT_MOVES = ['add_atom', 'remove_atom', 'bond_order_up', 'bond_order_down',
                 'add_bond', 'remove_bond', 'swap_element', 'charge_up', 'charge_down']
DEFAULT_REACTION_PARTNERS = ['H2', 'O2', 'H2O', 'CO2', 'N2', 'NH3', 'HCl', 'CH4']

SIZE_GUARD = 2_000_000  # refuse a distance level that would exceed this many candidates

PUB_BASE = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
UA = 'EngineeringSim-chemcalc/1.0'
_PUB_TIMEOUT = 20  # seconds
_PUB_CHUNK = 100  # batch lookup chunk size
_PUB_SLEEP = 0.25  # per-key fallback rate limit (~4 requests/s, under the 5/s cap)

_PT = Chem.GetPeriodicTable()
_ATOMIC = {el: _PT.GetAtomicNumber(el) for el in MAIN_GROUP}
_ORDER_NAME = {1: 'single', 2: 'double', 3: 'triple'}
# Standard covalent bond types the harness keeps. RDKit can reinterpret a bond as
# dative/ionic when a metal atom is swapped in; those are valence-model artifacts
# without a meaningful InChI, so they are rejected below.
_COVALENT = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
             Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)

# Deterministic offline common-name table (names the contract/tests rely on).
# General names still fall back to engineering_sim.science.chem.molecule (PubChem lookup),
# but these stay deterministic and network-free.
_NAME_SMILES = {
    'water': 'O', 'hydrogen': '[H][H]', 'ammonia': 'N', 'methane': 'C',
    'methanol': 'CO', 'oxygen': 'O=O', 'nitrogen': 'N#N', 'carbon dioxide': 'O=C=O',
    'carbon monoxide': '[C-]#[O+]', 'hydrogen peroxide': 'OO', 'hydroxyl': '[OH]',
}

# Hill-formula -> canonical SMILES for the small, unambiguous formulas the harness
# resolves offline. Arbitrary molecular formulas are genuinely ambiguous (e.g.
# C2H6O) and are not resolved; that is an honest limit, not an oversight.
_FORMULA_SMILES = {
    'H2O': 'O', 'H2O2': 'OO', 'CO': 'CO', 'CO2': 'O=C=O', 'CH4': 'C', 'H3N': 'N',
    'HCl': 'Cl', 'H2S': 'S', 'CHN': 'C#N', 'CH2O': 'C=O', 'CH4O': 'CO',
    'HF': 'F', 'HBr': 'Br', 'HI': 'I', 'C2H6': 'CC', 'C2H4': 'C=C', 'C2H2': 'C#C',
    'C2H6O': 'CCO',
}

# Single-element diatomic canonical SMILES (count == 2). Hydrogen is handled
# separately ([H][H]); other homonuclear diatomics use the common bond order.
_DIATOMIC = {'O': 'O=O', 'N': 'N#N', 'F': 'FF', 'Cl': 'ClCl', 'Br': 'BrBr', 'I': 'II'}

_LIMITS = [
    'Valence-valid by RDKit sanitisation is not a claim of stability or synthesizability.',
    'Hypothetical entries were not found in PubChem and are not claims of existence.',
    'Nearest combines structural edit distance (primary), atom-pair count similarity, and a one-reaction link.',
    'Reaction links use gas-phase species from science toolkit NIST WebBook thermochemistry only; a missing link means no cited thermochemistry, not that no reaction is possible.',
]

_FP_GENERATOR = None  # one atom-pair count generator, fixed options


try:
    from engineering_sim.execution import operation as _simulation_operation, source_files as _simulation_sources
except ModuleNotFoundError as exc:
    if exc.name not in ('engineering_sim', 'engineering_sim.execution'):
        raise
    # Portable chemical harnesses carry the same dispatcher beside their launcher.
    from simulation_bridge import operation as _simulation_operation, source_files as _simulation_sources

def _fp_generator():
    """Lazily build the atom-pair *count* fingerprint generator (fixed options)."""
    global _FP_GENERATOR
    if _FP_GENERATOR is None:
        _FP_GENERATOR = rdFingerprintGenerator.GetAtomPairGenerator()
    return _FP_GENERATOR


# ---------------------------------------------------------------------------
# Canonical JSON and hashing (the byte identity of the harness)
# ---------------------------------------------------------------------------

def canonical_json(obj) -> str:
    """Stable canonical JSON: sorted keys, compact separators, unicode preserved."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def serialize_result(result) -> str:
    """The exact byte form written to ``result.json`` (hashed content)."""
    return json.dumps(result, sort_keys=True, indent=1, ensure_ascii=False)


def serialize_snapshot(snapshot) -> str:
    """The exact byte form written to ``pubchem_snapshot.json``."""
    return json.dumps(snapshot, sort_keys=True, indent=1, ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _read_json(path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _write_json(path, obj):
    """Write deterministic (sorted, indented, no trailing newline) JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False), encoding='utf-8')


def _stamp() -> str:
    """Display timestamps in UTC, Mountain and Eastern time (not hashed)."""
    now = datetime.now(timezone.utc)
    return ' | '.join(now.astimezone(ZoneInfo(z)).strftime('%Y-%m-%d %H:%M %Z')
                      for z in ('America/Denver', 'America/New_York', 'UTC')).replace('UTC', 'Zulu')


# ---------------------------------------------------------------------------
# Start resolution and recipe identity
# ---------------------------------------------------------------------------

def _parse_formula(text) -> dict:
    """Parse ``H2O``/``NaCl`` into {element: count}, or None when not a formula."""
    t = text.strip()
    if not t or not re.fullmatch(r'(?:[A-Z][a-z]?\d*)+', t):
        return None
    comp = {}
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', t):
        comp[el] = comp.get(el, 0) + (int(n) if n else 1)
    return comp


def _hill_formula(comp) -> str:
    """Hill system formula: C first, H second, remaining elements alphabetically."""
    c = comp.get('C', 0)
    h = comp.get('H', 0)
    others = sorted((el, n) for el, n in comp.items() if el not in ('C', 'H'))
    out = ''
    if c:
        out += 'C' + (str(c) if c > 1 else '')
    if h:
        out += 'H' + (str(h) if h > 1 else '')
    for el, n in others:
        out += el + (str(n) if n > 1 else '')
    return out


def _single_atom_smiles(el) -> str:
    """Canonical SMILES of a single neutral heavy atom (implicit hydrogens)."""
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom(el))
    with rdBase.BlockLogs():
        Chem.SanitizeMol(rw)
        return Chem.MolToSmiles(Chem.RemoveHs(rw), isomericSmiles=True)


def _formula_to_smiles(comp) -> str:
    """Resolve a parsed formula to canonical SMILES, or None when not covered."""
    items = sorted(comp.items())
    if len(items) == 1:
        el, n = items[0]
        if el == 'H':
            if n == 1:
                return '[H]'
            if n == 2:
                return '[H][H]'
            return None
        if n == 1:
            return _single_atom_smiles(el)
        if n == 2 and el in _DIATOMIC:
            return _DIATOMIC[el]
        return None
    return _FORMULA_SMILES.get(_hill_formula(comp))


def _canonical_smiles(mol) -> str:
    """Canonical isomeric SMILES of a sanitized molecule (explicit H folded)."""
    with rdBase.BlockLogs():
        m = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(m, isomericSmiles=True)


def resolve_start(text, offline=False):
    """Resolve a start written in any notation to canonical SMILES.

    Delegates to ``engineering_sim.science.chem_input.resolve`` (2026-09-14 review), which accepts SMILES, a
    formula (``H2O``, ``H₂O``, ``H_2O``, LaTeX-decorated), a name, InChI, InChIKey or a CAS number, and
    can be forced with a prefix (``smiles:CO`` vs ``formula:CO``). Every notation for one molecule gives
    the same canonical SMILES, so the recipe id is identical whichever way the user types the start.
    Multi-fragment starts (salts such as ``NaCl``) are rejected: the edit search walks one connected graph.
    The portable exported harness vendors chem_input, so this import also works outside the checkout.
    """
    from engineering_sim.science.chem_input import resolve as _resolve
    detail = _resolve(text, offline=offline)
    mol = Chem.MolFromSmiles(detail['smiles'])
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('chemcalc start must be a single connected molecule: %r resolved to %s'
                         % (text, detail['smiles']))
    return _canonical_smiles(mol)


def resolve_start_detail(text, offline=False):
    """Same resolution, but returns the full record (smiles, via, PubChem cid/name) for the page."""
    from engineering_sim.science.chem_input import resolve as _resolve
    detail = _resolve(text, offline=offline)
    detail['smiles'] = resolve_start(text, offline=offline)
    return detail


def normalise_recipe(fields):
    """Fill defaults and validate; return the recipe with its computed ``recipe_id``.

    ``start`` is stored as the canonical SMILES of the resolved start, so ``H2O``,
    ``water`` and ``O`` produce the same recipe. ``rdkit_version`` is always taken
    from the running RDKit, never from user input.
    """
    fields = dict(fields or {})

    start = resolve_start(fields.get('start', 'O'))
    k = int(fields.get('k', 1000))
    if not 1 <= k <= 5000:
        raise ValueError('k must be 1..5000')
    # mode 'k'        : stop once k valid neighbours exist (finishing the shell) and return the nearest k.
    # mode 'complete' : return EVERY valid neighbour of every fully expanded shell, nothing truncated
    #                   (owner, 2026-09-15: "it should list all neighbors"). k then only sets the floor
    #                   that decides how many shells are expanded; max_distance caps it instead when given.
    mode = fields.get('mode', 'k')
    if mode not in ('k', 'complete'):
        raise ValueError("mode must be 'k' or 'complete'")
    max_distance = fields.get('max_distance')
    if max_distance is not None:
        max_distance = int(max_distance)
        if not 1 <= max_distance <= 6:
            raise ValueError('max_distance must be 1..6')

    elements = list(fields.get('elements', DEFAULT_ELEMENTS))
    if not elements:
        raise ValueError('elements must not be empty')
    if len(set(elements)) != len(elements):
        raise ValueError('elements must not repeat')
    for el in elements:
        if el not in MAIN_GROUP:
            raise ValueError('element %r is not a main-group element' % el)

    max_charge = int(fields.get('max_charge', 1))
    if not 0 <= max_charge <= 2:
        raise ValueError('max_charge must be 0..2')

    allow_radicals = bool(fields.get('allow_radicals', True))

    max_heavy_atoms = int(fields.get('max_heavy_atoms', 12))
    if not 1 <= max_heavy_atoms <= 30:
        raise ValueError('max_heavy_atoms must be 1..30')

    moves = list(fields.get('moves', DEFAULT_MOVES))
    if not moves or len(set(moves)) != len(moves):
        raise ValueError('moves must be a non-empty list without repeats')
    for m in moves:
        if m not in DEFAULT_MOVES:
            raise ValueError('unknown move %r' % m)

    known_check = fields.get('known_check', 'pubchem')
    if known_check not in ('pubchem', None):
        raise ValueError("known_check must be 'pubchem' or None")

    reaction_partners = list(fields.get('reaction_partners', DEFAULT_REACTION_PARTNERS))
    if not all(isinstance(p, str) and p.strip() for p in reaction_partners):
        raise ValueError('reaction_partners must be non-empty strings')

    recipe = {
        'harness': HARNESS,
        'harness_version': HARNESS_VERSION,
        'start': start,
        'k': k,
        'mode': mode,
        'max_distance': max_distance,
        'elements': elements,
        'max_charge': max_charge,
        'allow_radicals': allow_radicals,
        'max_heavy_atoms': max_heavy_atoms,
        'moves': moves,
        'known_check': known_check,
        'reaction_partners': reaction_partners,
        'rdkit_version': Chem.rdBase.rdkitVersion,
    }
    recipe['recipe_id'] = _recipe_id(recipe)
    return recipe


def _recipe_id(recipe_without_id) -> str:
    """SHA-256 of the canonical JSON of the recipe without its own ``recipe_id``."""
    body = {k: v for k, v in recipe_without_id.items() if k != 'recipe_id'}
    return _sha256(canonical_json(body))


def default_recipe():
    """The default recipe (canonical contract), normalised and with its ``recipe_id``."""
    return normalise_recipe({})


# ---------------------------------------------------------------------------
# Search: deterministic breadth-first edit search
# ---------------------------------------------------------------------------

def _sanitize(mol):
    """Return a freshly sanitized copy, or None when RDKit rejects the valence."""
    try:
        with rdBase.BlockLogs():
            m = Chem.Mol(mol)
            Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def _apply_moves(mol, recipe):
    """Yield ``(edited_rwmol, move_name, detail)`` for every move in a fixed order.

    Ordering is the determinism contract: atom index order, then element order as
    listed in the recipe, then bond order 1 -> 2 -> 3. Radicals only when
    ``allow_radicals`` is true; charge moves respect ``max_charge``.
    """
    moves = recipe['moves']
    elements = recipe['elements']
    max_charge = recipe['max_charge']
    allow_radicals = recipe['allow_radicals']
    n = mol.GetNumAtoms()

    for i in range(n):
        atom = mol.GetAtomWithIdx(i)
        sym = atom.GetSymbol()

        if 'swap_element' in moves:
            for X in elements:
                if X == sym or X == 'H':
                    continue  # identity, or swap-to-H (ill-defined in the implicit-H model)
                rw = Chem.RWMol(mol)
                a = rw.GetAtomWithIdx(i)
                a.SetAtomicNum(_ATOMIC[X])
                a.SetNumRadicalElectrons(0)
                a.SetNoImplicit(False)
                yield rw, 'swap_element', '%s -> %s' % (sym, X)

        if 'add_atom' in moves:
            for X in elements:
                if X == 'H':
                    # Adding H to a heavy atom folds into implicit H (a no-op); it
                    # only matters between hydrogen atoms (H2 formation from [H]).
                    if sym == 'H':
                        rw = Chem.RWMol(mol)
                        ni = rw.AddAtom(Chem.Atom('H'))
                        rw.AddBond(i, ni, Chem.BondType.SINGLE)
                        yield rw, 'add_atom', 'H-H single'
                    continue
                for o in (1, 2, 3):
                    rw = Chem.RWMol(mol)
                    ni = rw.AddAtom(Chem.Atom(X))
                    rw.AddBond(i, ni, Chem.BondType.values[o])
                    yield rw, 'add_atom', '%s-%s %s' % (X, sym, _ORDER_NAME[o])

        q = atom.GetFormalCharge()
        if 'charge_up' in moves and q < max_charge:
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(i).SetFormalCharge(q + 1)
            yield rw, 'charge_up', '%s charge %d->%d' % (sym, q, q + 1)
        if 'charge_down' in moves and q > -max_charge:
            rw = Chem.RWMol(mol)
            rw.GetAtomWithIdx(i).SetFormalCharge(q - 1)
            yield rw, 'charge_down', '%s charge %d->%d' % (sym, q, q - 1)

        if 'remove_atom' in moves:
            rw = Chem.RWMol(mol)
            rw.RemoveAtom(i)
            yield rw, 'remove_atom', 'remove %s' % sym
            # Removing one implicit hydrogen leaves a radical ("remove a terminal
            # atom" where the terminal atom is an implicit H). This is the only
            # radical source in the move set, and only when allowed.
            if allow_radicals and sym != 'H' and atom.GetTotalNumHs() >= 1:
                rw = Chem.RWMol(mol)
                a = rw.GetAtomWithIdx(i)
                a.SetNumRadicalElectrons(a.GetNumRadicalElectrons() + 1)
                yield rw, 'remove_atom', 'remove H from %s' % sym

    if 'add_bond' in moves:
        for i in range(n):
            for j in range(i + 1, n):
                if mol.GetBondBetweenAtoms(i, j) is not None:
                    continue
                rw = Chem.RWMol(mol)
                rw.AddBond(i, j, Chem.BondType.SINGLE)
                yield rw, 'add_bond', 'add bond %d-%d' % (i, j)

    for bond in mol.GetBonds():
        bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        o = int(bond.GetBondTypeAsDouble())
        if 'bond_order_up' in moves and o < 3:
            rw = Chem.RWMol(mol)
            rw.GetBondBetweenAtoms(bi, bj).SetBondType(Chem.BondType.values[o + 1])
            yield rw, 'bond_order_up', 'bond %d-%d order %d->%d' % (bi, bj, o, o + 1)
        if 'bond_order_down' in moves and o > 1:
            rw = Chem.RWMol(mol)
            rw.GetBondBetweenAtoms(bi, bj).SetBondType(Chem.BondType.values[o - 1])
            yield rw, 'bond_order_down', 'bond %d-%d order %d->%d' % (bi, bj, o, o - 1)
        if 'remove_bond' in moves:
            rw = Chem.RWMol(mol)
            rw.RemoveBond(bi, bj)
            yield rw, 'remove_bond', 'remove bond %d-%d' % (bi, bj)


def _valid(mol, recipe) -> bool:
    """Single connected fragment within the recipe's element/charge/radical limits."""
    if mol.GetNumAtoms() == 0:
        return False
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    allowed = set(recipe['elements'])
    heavy = 0
    total_charge = 0
    radical = 0
    for b in mol.GetBonds():
        if b.GetBondType() not in _COVALENT:
            return False
    for a in mol.GetAtoms():
        if a.GetSymbol() not in allowed:
            return False
        if a.GetAtomicNum() > 1:
            heavy += 1
        q = a.GetFormalCharge()
        if abs(q) > recipe['max_charge']:
            return False
        total_charge += q
        radical += a.GetNumRadicalElectrons()
    if heavy > recipe['max_heavy_atoms']:
        return False
    if abs(total_charge) > recipe['max_charge']:
        return False
    if not recipe['allow_radicals'] and radical > 0:
        return False
    return True


def _search(recipe, start_mol, start_smiles):
    """Breadth-first edit search; returns (found, generated, valence_valid).

    ``found`` is a list of ``{smiles, distance, path, mol}`` in deterministic
    first-discovery order. ``generated`` counts every candidate produced by a
    move; ``valence_valid`` counts those that RDKit sanitizes — it is a raw
    candidate count, taken before canonical-SMILES deduplication and before the
    recipe checks in ``_valid`` (2026-09-15: clarified after a complete distance-2
    run on H2O reported 16,234 sanitizing candidates for 7,143 distinct compounds;
    the same structure is reached by many different edit paths). The number of
    distinct compounds is ``counts['returned']`` / ``len(neighbours)``.
    """
    found = []
    seen = {start_smiles}
    frontier = [(start_smiles, start_mol, [])]
    distance = 0
    generated = 0
    valence_valid = 0

    while frontier:
        distance += 1
        level_candidates = 0
        nxt = []
        for (_, pmol, ppath) in frontier:
            for (rw, move_name, detail) in _apply_moves(pmol, recipe):
                level_candidates += 1
                generated += 1
                if level_candidates > SIZE_GUARD:
                    raise ValueError(
                        'chemcalc distance level %d would exceed %d candidates; '
                        'narrow the elements/moves/max_heavy_atoms or lower k'
                        % (distance, SIZE_GUARD))
                m = _sanitize(rw)
                if m is None:
                    continue
                valence_valid += 1
                with rdBase.BlockLogs():
                    m = Chem.RemoveHs(m)
                    cs = Chem.MolToSmiles(m, isomericSmiles=True)
                if cs == start_smiles or cs in seen:
                    continue
                if not _valid(m, recipe):
                    continue
                seen.add(cs)
                step = {'move': move_name, 'detail': detail, 'smiles': cs}
                path = ppath + [step]
                found.append({'smiles': cs, 'distance': distance, 'path': path, 'mol': m})
                nxt.append((cs, m, path))
        frontier = nxt
        # A shell is always finished before stopping, so 'complete' returns whole shells and the
        # result never depends on the order neighbours happened to be generated in.
        cap = recipe.get('max_distance')
        if cap is not None and distance >= cap:
            break
        if cap is None and len(found) >= recipe['k']:
            break
        if not frontier:
            break
    return found, generated, valence_valid


# ---------------------------------------------------------------------------
# Similarity (atom-pair count Tanimoto, element-count fallback)
# ---------------------------------------------------------------------------

def _composition(mol) -> dict:
    """Element counts including implicit hydrogens, {element: count}."""
    comp = {}
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        comp[sym] = comp.get(sym, 0) + 1
        comp['H'] = comp.get('H', 0) + a.GetTotalNumHs()
    return comp


def _count_tanimoto(fp_a, fp_b):
    """Tanimoto on two count-vector numpy arrays (sum min / sum max).

    Returns None when both fingerprints are empty (denominator zero), which is
    the signal for the element-count fallback.
    """
    num = float(np.minimum(fp_a, fp_b).sum())
    den = float(np.maximum(fp_a, fp_b).sum())
    return (num / den) if den > 0 else None


def _element_count_tanimoto(comp_a, comp_b):
    """Sum(min)/sum(max) over element counts (hydrogens included)."""
    els = set(comp_a) | set(comp_b)
    num = sum(min(comp_a.get(e, 0), comp_b.get(e, 0)) for e in els)
    den = sum(max(comp_a.get(e, 0), comp_b.get(e, 0)) for e in els)
    return (num / den) if den > 0 else 0.0


def _similarity_raw(mol_a, mol_b):
    """Return (similarity, method) with the atom-pair count Tanimoto and fallback.

    An atom-pair fingerprint is empty for a molecule with fewer than two heavy
    atoms, so when either side is empty we fall back to the documented
    element-count Tanimoto (hydrogens included).
    """
    gen = _fp_generator()
    fp_a = gen.GetCountFingerprintAsNumPy(mol_a)
    fp_b = gen.GetCountFingerprintAsNumPy(mol_b)
    if fp_a.sum() > 0 and fp_b.sum() > 0:
        sim = _count_tanimoto(fp_a, fp_b)
        return round(sim, 6), 'RDKit atom-pair count Tanimoto'
    comp_a = _composition(mol_a)
    comp_b = _composition(mol_b)
    return round(_element_count_tanimoto(comp_a, comp_b), 6), \
        'element-count Tanimoto (single-atom fallback, hydrogens included)'


def similarity(smiles_a, smiles_b):
    """Public similarity between two SMILES, returned as a ``derived`` value record."""
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    value, method = _similarity_raw(mol_a, mol_b)
    return cite.value(value, 'dimensionless', 'derived', method=method)


# ---------------------------------------------------------------------------
# Reaction link (science toolkit NIST WebBook, gas phase)
# ---------------------------------------------------------------------------

_THERMO_GAS_INDEX = None


def _thermo_gas_index():
    """{composition_tuple: sorted gas-phase species keys} built once from thermo data."""
    global _THERMO_GAS_INDEX
    if _THERMO_GAS_INDEX is None:
        data_path = os.path.join(os.path.dirname(thermo.__file__), 'data', 'thermo_species.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        idx = {}
        for key, rec in data.items():
            if key == '_meta' or rec.get('phase') != 'gas':
                continue
            comp = rec.get('elements') or {}
            ck = tuple(sorted(comp.items()))
            idx.setdefault(ck, []).append(key)
        for ck in idx:
            idx[ck].sort()
        _THERMO_GAS_INDEX = idx
    return _THERMO_GAS_INDEX


def _thermo_gas_key(comp):
    """First gas-phase species key for a composition, or None."""
    keys = _thermo_gas_index().get(tuple(sorted(comp.items())))
    return keys[0] if keys else None


def _balance(start, neigh, partner_comps):
    """Deterministic small-coefficient balance of ``a*S + b*L -> c*N + d*R``.

    Search order (documented): no partner, then one left partner, then one right
    partner, then both; within a case partners follow the recipe list and
    coefficients ascend. Returns ``(a, b, L_index, c, d, R_index)`` or None.
    """
    def ok(a, b, li, c, d, ri):
        bal = {}
        for el, x in start.items():
            bal[el] = bal.get(el, 0) + a * x
        if li is not None:
            for el, x in partner_comps[li].items():
                bal[el] = bal.get(el, 0) + b * x
        for el, x in neigh.items():
            bal[el] = bal.get(el, 0) - c * x
        if ri is not None:
            for el, x in partner_comps[ri].items():
                bal[el] = bal.get(el, 0) - d * x
        return all(v == 0 for v in bal.values())

    for a in range(1, 5):
        for c in range(1, 5):
            if ok(a, 0, None, c, 0, None):
                return (a, 0, None, c, 0, None)
    for li in range(len(partner_comps)):
        for a in range(1, 5):
            for b in range(1, 5):
                for c in range(1, 5):
                    if ok(a, b, li, c, 0, None):
                        return (a, b, li, c, 0, None)
    for ri in range(len(partner_comps)):
        for a in range(1, 5):
            for c in range(1, 5):
                for d in range(1, 5):
                    if ok(a, 0, None, c, d, ri):
                        return (a, 0, None, c, d, ri)
    for li in range(len(partner_comps)):
        for ri in range(len(partner_comps)):
            for a in range(1, 5):
                for b in range(1, 5):
                    for c in range(1, 5):
                        for d in range(1, 5):
                            if ok(a, b, li, c, d, ri):
                                return (a, b, li, c, d, ri)
    return None


def _reaction_for(recipe, start_comp, neigh_comp, start_formula, neigh_formula):
    """Return ``(reaction_dict, favored_bool_or_None)`` for start -> neighbour."""
    start_key = _thermo_gas_key(start_comp)
    neigh_key = _thermo_gas_key(neigh_comp)
    if start_key is None or neigh_key is None:
        return {'linked': False, 'basis': 'no cited thermochemistry for this species'}, None

    partners = []
    for pstr in recipe['reaction_partners']:
        comp = _parse_formula(pstr)
        if not comp:
            continue
        pkey = _thermo_gas_key(comp)
        if pkey is None:
            continue
        partners.append({'key': pkey, 'comp': comp, 'formula': _hill_formula(comp)})

    sol = _balance(start_comp, neigh_comp, [p['comp'] for p in partners])
    if sol is None:
        return {'linked': False, 'basis': 'no balanced one-reaction link in science toolkit thermochemistry'}, None

    def fmt(coeff, formula):
        return formula if coeff == 1 else '%d %s' % (coeff, formula)

    a, b, li, c, d, ri = sol
    reactants = {start_key: a}
    products = {neigh_key: c}
    terms_l = [fmt(a, start_formula)]
    terms_r = [fmt(c, neigh_formula)]
    if li is not None:
        reactants[partners[li]['key']] = b
        terms_l.append(fmt(b, partners[li]['formula']))
    if ri is not None:
        products[partners[ri]['key']] = d
        terms_r.append(fmt(d, partners[ri]['formula']))

    r = thermo.reaction(reactants, products, T=298.15)
    dg = round(cite.num(r['dG']), 6)
    favored = bool(r['favored'])
    rec = {
        'linked': True,
        'equation': ' + '.join(terms_l) + ' -> ' + ' + '.join(terms_r),
        'dG298_kJmol': cite.value(dg, 'kJ/mol', 'derived', method='science toolkit thermo (NIST)'),
        'favored': favored,
        'basis': 'science toolkit thermo (NIST)',
    }
    return rec, favored


# ---------------------------------------------------------------------------
# Known check (PubChem PUG REST, batched, snapshot-backed)
# ---------------------------------------------------------------------------

def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _http(url, data=None):
    headers = {'User-Agent': UA, 'Accept': 'application/json'}
    body = None
    if data is not None:
        body = data.encode('utf-8')
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=_PUB_TIMEOUT) as r:
        return json.loads(r.read().decode('utf-8'))


def _pubchem_batch(inchikeys):
    """Batch resolve InChIKeys via the property/InChIKey endpoint (min CID per key).

    The plain ``/inchikey/cids`` batch endpoint returns a flat, unmappable CID list
    (isotopologues are interleaved with canonical entries), so the harness batches
    through the ``property/InChIKey`` endpoint instead and takes the minimum CID per
    InChIKey — the canonical, non-isotopic PubChem entry. A 404 on a chunk means the
    whole chunk was not found (hypothetical), not a rejection.
    """
    out = {}
    for chunk in _chunks(inchikeys, _PUB_CHUNK):
        body = 'inchikey=' + urllib.parse.quote(','.join(chunk), safe=',')
        try:
            resp = _http(PUB_BASE + '/compound/inchikey/property/InChIKey/JSON', data=body)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                for ik in chunk:
                    out[ik] = {'cid': None}
                continue
            raise
        by_key = {}
        for row in (resp.get('PropertyTable', {}).get('Properties') or []):
            ik = row.get('InChIKey')
            cid = row.get('CID')
            if ik and cid is not None:
                by_key.setdefault(ik, []).append(cid)
        for ik in chunk:
            cids = by_key.get(ik)
            out[ik] = {'cid': min(cids) if cids else None}
    return out


def _pubchem_get(inchikey):
    """Per-key GET fallback; returns {'cid': int|None} (minimum CID, 404 = not found)."""
    try:
        resp = _http(PUB_BASE + '/compound/inchikey/' + urllib.parse.quote(inchikey) + '/cids/JSON')
        cids = (resp.get('IdentifierList', {}).get('CID') or [])
        return {'cid': min(cids) if cids else None}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {'cid': None}
        raise


def _known_check(inchikeys, snapshot, allow_network):
    """Resolve known/hypothetical for a list of InChIKeys.

    Returns ``(status, snapshot_out, note)`` where ``status[k]`` is one of
    ``'known'`` (with a CID), ``'hypothetical'`` (PubChem confirmed not found) or
    ``'unknown'`` (network failure / offline without a snapshot). The snapshot
    records each PubChem decision as ``{inchikey: {'cid': int|None}}``; a missing
    key means the decision could not be made.
    """
    snapshot_out = dict(snapshot) if snapshot else {}
    missing = [k for k in inchikeys if k not in snapshot_out]
    note = None
    if missing and allow_network:
        fetched = None
        try:
            fetched = _pubchem_batch(missing)
        except urllib.error.HTTPError:
            fetched = {}
            for k in missing:
                try:
                    fetched[k] = _pubchem_get(k)
                    time.sleep(_PUB_SLEEP)
                except urllib.error.HTTPError as e:
                    if e.code == 404:
                        fetched[k] = {'cid': None}
                    else:
                        fetched[k] = None
                except Exception:
                    fetched[k] = None
        except Exception:
            fetched = None
        if fetched is None:
            note = 'PubChem unreachable; known values are null'
        else:
            for k in missing:
                v = fetched.get(k)
                if v is not None:
                    snapshot_out[k] = v

    status = {}
    for k in inchikeys:
        v = snapshot_out.get(k)
        if v is None:
            status[k] = {'known': None, 'cid': None}
        elif v.get('cid') is None:
            status[k] = {'known': False, 'cid': None}
        else:
            status[k] = {'known': True, 'cid': v['cid']}
    return status, snapshot_out, note


# ---------------------------------------------------------------------------
# Core deterministic computation
# ---------------------------------------------------------------------------

def _rank_key(entry, use_known):
    """Deterministic ranking key: distance asc, similarity desc, favored desc,
    known desc, canonical smiles asc. (Unavailable booleans sort as False.)"""
    sim = entry['similarity']
    favored = 1 if entry.get('favored') else 0
    if use_known:
        known = 1 if entry.get('known') else 0
        return (entry['distance'], -sim, -favored, -known, entry['smiles'])
    return (entry['distance'], -sim, -favored, entry['smiles'])


def _cid_record(cid):
    """A cited PubChem CID value record (no timestamp, so it stays hash-stable)."""
    return cite.value(cid, 'dimensionless', 'cited',
                      source='https://pubchem.ncbi.nlm.nih.gov/compound/%d' % cid,
                      verified=True)


@_simulation_operation('science_bundle.chemical_search.compute', kind='study')
def compute(recipe, snapshot=None, allow_network=True):
    """Deterministic core: search, rank, annotate. Returns (result, snapshot, notes).

    ``recipe`` must already be normalised (see ``normalise_recipe``): ``start`` is
    canonical SMILES and every field is present. ``snapshot`` is a previously
    saved ``pubchem_snapshot.json`` dict; when given (or when ``allow_network`` is
    False) no network calls are made.
    """
    start_smiles = recipe['start']
    start_mol = Chem.MolFromSmiles(start_smiles)
    start_comp = _composition(start_mol)
    start_formula = _hill_formula(start_comp)
    start_inchikey = Chem.MolToInchiKey(start_mol)

    found, generated, valence_valid = _search(recipe, start_mol, start_smiles)

    # Annotate every found species (similarity + reaction) before ranking.
    for entry in found:
        mol = entry['mol']
        entry['formula'] = _hill_formula(_composition(mol))
        entry['inchikey'] = Chem.MolToInchiKey(mol)
        entry['charge'] = Chem.GetFormalCharge(mol)
        entry['radical_electrons'] = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
        entry['heavy_atoms'] = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        entry['similarity'], entry['similarity_method'] = _similarity_raw(start_mol, mol)
        reaction, favored = _reaction_for(recipe, start_comp, _composition(mol),
                                          start_formula, entry['formula'])
        entry['reaction'] = reaction
        entry['favored'] = favored

    # Rank without known status. In 'k' mode the nearest k are kept; in 'complete' mode every valid
    # neighbour of the expanded shells is kept, so the list is the full neighbourhood, not a top-k.
    found.sort(key=lambda e: _rank_key(e, use_known=False))
    selected = found if recipe.get('mode') == 'complete' else found[:recipe['k']]

    # Known check for the returned k plus the start.
    inchikeys = sorted(set([start_inchikey] + [e['inchikey'] for e in selected]))
    do_network = allow_network and recipe.get('known_check') == 'pubchem'
    status, snapshot_out, note = _known_check(inchikeys, snapshot, do_network)

    # Final ranking with known as the last tiebreak, then assign ranks.
    for e in selected:
        st = status.get(e['inchikey'], {'known': None, 'cid': None})
        e['known'] = st['known']
        e['cid'] = st['cid']
    selected.sort(key=lambda e: _rank_key(e, use_known=True))

    start_status = status.get(start_inchikey, {'known': None, 'cid': None})

    neighbours = []
    by_distance = {}
    for rank, e in enumerate(selected, start=1):
        by_distance[e['distance']] = by_distance.get(e['distance'], 0) + 1
        cid = e['cid']
        neighbours.append({
            'rank': rank,
            'smiles': e['smiles'],
            'formula': e['formula'],
            'inchikey': e['inchikey'],
            'distance': e['distance'],
            'path': e['path'],
            'similarity': cite.value(e['similarity'], 'dimensionless', 'derived',
                                     method=e['similarity_method']),
            'known': e['known'],
            'cid': _cid_record(cid) if cid else None,
            'charge': e['charge'],
            'radical_electrons': e['radical_electrons'],
            'heavy_atoms': e['heavy_atoms'],
            'reaction': e['reaction'],
        })

    known_count = sum(1 for e in selected if e['known'] is True)
    hypothetical_count = sum(1 for e in selected if e['known'] is False)

    limits = list(_LIMITS)
    if note:
        limits.append(note)

    result = {
        'recipe_id': recipe['recipe_id'],
        'start': {
            'smiles': start_smiles,
            'formula': start_formula,
            'inchikey': start_inchikey,
            'cid': _cid_record(start_status['cid']) if start_status['cid'] else None,
        },
        'counts': {
            'generated': generated,
            'valence_valid': valence_valid,
            'returned': len(selected),
            'known': known_count,
            'hypothetical': hypothetical_count,
            'by_distance': by_distance,
        },
        'neighbours': neighbours,
        'limits': limits,
    }
    notes = ([note] if note else [])
    return result, snapshot_out, notes


# ---------------------------------------------------------------------------
# Run folder, verify, export, listing
# ---------------------------------------------------------------------------

def _runs_dir():
    return paths.data_dir() / 'chemcalc' / 'runs'


def _run_dir(recipe_id):
    return _runs_dir() / recipe_id


def run(fields=None, replay=False):
    """Run a recipe end-to-end: normalise, compute, write the run folder.

    ``replay=True`` reuses a saved snapshot (no network). Returns
    ``{recipe, run, result}``.
    """
    recipe = normalise_recipe(fields or {})
    recipe_id = recipe['recipe_id']
    snapshot = None
    if replay:
        snap_path = _run_dir(recipe_id) / 'pubchem_snapshot.json'
        if snap_path.exists():
            snapshot = _read_json(snap_path)

    t0 = time.perf_counter()
    result, snapshot_out, notes = compute(recipe, snapshot=snapshot,
                                          allow_network=(not replay))
    wall = round(time.perf_counter() - t0, 4)

    folder = _run_dir(recipe_id)
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / 'recipe.json', recipe)
    result_bytes = serialize_result(result)
    (folder / 'result.json').write_text(result_bytes, encoding='utf-8')
    _write_json(folder / 'pubchem_snapshot.json', snapshot_out)

    result_sha = _sha256(result_bytes)
    snapshot_sha = _sha256(serialize_snapshot(snapshot_out))
    engine_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    run_meta = {
        'recipe_id': recipe_id,
        'created': _stamp(),
        'created_iso': datetime.now(timezone.utc).isoformat(),
        'wall_time_s': wall,
        'counts': result['counts'],
        'result_sha256': result_sha,
        'snapshot_sha256': snapshot_sha,
        'engine_sha256': engine_sha,
        'notes': notes,
    }
    _write_json(folder / 'run.json', run_meta)
    return {'recipe': recipe, 'run': run_meta, 'result': result}


def verify(recipe_id):
    """Recompute offline from the stored recipe + snapshot and compare the hash."""
    folder = _run_dir(recipe_id)
    recipe = _read_json(folder / 'recipe.json')
    run_meta = _read_json(folder / 'run.json')
    snapshot = _read_json(folder / 'pubchem_snapshot.json')
    result, _, _ = compute(recipe, snapshot=snapshot, allow_network=False)
    recomputed_sha = _sha256(serialize_result(result))
    expected_sha = run_meta.get('result_sha256')
    return {
        'ok': recomputed_sha == expected_sha,
        'expected_sha256': expected_sha,
        'recomputed_sha256': recomputed_sha,
        'snapshot_used': True,
    }


def get(recipe_id):
    """Read a stored run's recipe/run/result from its run folder."""
    folder = _run_dir(recipe_id)
    return {
        'recipe': _read_json(folder / 'recipe.json'),
        'run': _read_json(folder / 'run.json'),
        'result': _read_json(folder / 'result.json'),
    }


def list_runs():
    """Summaries of every stored run, newest first."""
    runs_dir = _runs_dir()
    if not runs_dir.exists():
        return []
    out = []
    for folder in runs_dir.iterdir():
        run_path = folder / 'run.json'
        recipe_path = folder / 'recipe.json'
        if not (run_path.is_file() and recipe_path.is_file()):
            continue
        run_meta = _read_json(run_path)
        recipe = _read_json(recipe_path)
        counts = run_meta.get('counts', {})
        out.append({
            'recipe_id': recipe.get('recipe_id', folder.name),
            'start': recipe.get('start'),
            'k': recipe.get('k'),
            'created': run_meta.get('created'),
            'created_iso': run_meta.get('created_iso'),
            'result_sha256': run_meta.get('result_sha256'),
            'returned': counts.get('returned'),
            'known': counts.get('known'),
        })
    out.sort(key=lambda r: r.get('created_iso') or '', reverse=True)
    return out


def _export_readme(recipe_id, recipe, result, created):
    return ('<!-- PROJECT: Engineering Toolkit and Simulators — science Chemical Calculator harness\n'
            'CREATED: %s\n'
            'DESCRIPTION: Portable, verifiable copy of a science toolkit chemical-space run. -->\n\n'
            '# Chemical Calculator — portable harness\n\n'
            'Recipe `%s` (start `%s`, k = %s) with its result and PubChem snapshot.\n\n'
            'Requires Python 3.13 with rdkit/numpy/scipy (see `requirements.txt`).\n\n'
            '```sh\n'
            'python run.py verify            # recompute offline and compare the result hash\n'
            'python run.py run               # recompute offline from the saved snapshot\n'
            'python run.py run --online      # recompute with a fresh PubChem known-check\n'
            '```\n\n'
            '`verify` recomputes the result from `recipe.json` + `pubchem_snapshot.json` with\n'
            'no network and compares the SHA-256 of `result.json`. Validity is RDKit\n'
            'sanitisation only — not stability or synthesizability, and hypothetical entries\n'
            'are not claims of existence.\n'
            % (created, recipe_id, recipe.get('start'), recipe.get('k')))


def _export_run_py():
    return '''#!/usr/bin/env python3
# DESCRIPTION: Portable launcher: `verify` (offline, hash-compare) or `run` (recompute).
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('ENGINEERING_SIM_DATA', str(ROOT / 'data'))
sys.path.insert(0, str(ROOT))

import chemcalc  # noqa: E402


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'verify'
    online = '--online' in sys.argv
    recipe = json.loads((ROOT / 'recipe.json').read_text())
    if cmd == 'verify':
        snapshot = json.loads((ROOT / 'pubchem_snapshot.json').read_text())
        result, _, _ = chemcalc.compute(recipe, snapshot=snapshot, allow_network=False)
        recomputed_sha = hashlib.sha256(chemcalc.serialize_result(result).encode()).hexdigest()
        expected_sha = hashlib.sha256((ROOT / 'result.json').read_text().encode()).hexdigest()
        if recomputed_sha == expected_sha:
            print('OK: result hash matches ' + recomputed_sha)
            return 0
        print('MISMATCH: expected %s, recomputed %s' % (expected_sha, recomputed_sha))
        return 1
    if cmd == 'run':
        snap_path = ROOT / 'pubchem_snapshot.json'
        snapshot = json.loads(snap_path.read_text()) if snap_path.exists() else None
        result, snapshot_out, notes = chemcalc.compute(recipe, snapshot=snapshot, allow_network=online)
        (ROOT / 'result.json').write_text(chemcalc.serialize_result(result), encoding='utf-8')
        (ROOT / 'pubchem_snapshot.json').write_text(chemcalc.serialize_snapshot(snapshot_out), encoding='utf-8')
        for n in notes:
            print('note: ' + n)
        print('result sha256: ' + hashlib.sha256(chemcalc.serialize_result(result).encode()).hexdigest())
        return 0
    print('usage: run.py verify | run [--online]')
    return 2


if __name__ == '__main__':
    sys.exit(main())
'''


def export(recipe_id):
    """Write a self-contained zip (engine + vendored deps + artifacts) and verify it."""
    folder = _run_dir(recipe_id)
    recipe = _read_json(folder / 'recipe.json')
    result = _read_json(folder / 'result.json')
    snapshot = _read_json(folder / 'pubchem_snapshot.json')
    run_meta = _read_json(folder / 'run.json')
    # The README carries the run's own creation time (stable), so re-exporting a
    # stored run produces a byte-identical zip.
    created = run_meta.get('created') or _stamp()

    engine_src = Path(__file__).read_text(encoding='utf-8')
    science_bundle_dir = Path(__file__).resolve().parent
    cite_src = (science_bundle_dir / 'cite.py').read_text(encoding='utf-8')
    paths_src = (science_bundle_dir / 'paths.py').read_text(encoding='utf-8')
    thermo_src = (science_bundle_dir / 'thermo.py').read_text(encoding='utf-8')
    thermo_data = (science_bundle_dir / 'data' / 'thermo_species.json').read_text(encoding='utf-8')

    try:
        from importlib.metadata import version as _pkg_version
        requirements = '\n'.join(['rdkit==%s' % _pkg_version('rdkit'),
                                  'numpy==%s' % _pkg_version('numpy'),
                                  'scipy==%s' % _pkg_version('scipy')]) + '\n'
    except Exception:  # pragma: no cover
        requirements = 'rdkit>=2026.03.6\nnumpy\nscipy\n'

    zip_path = folder / ('chemcalc_%s_harness.zip' % recipe_id[:12])
    files = {
        'chemcalc.py': engine_src,
        'simulation_bridge.py': _simulation_sources()[0].read_text(encoding='utf-8'),
        'unified_sim_engine.py': _simulation_sources()[1].read_text(encoding='utf-8'),
        'science_bundle/__init__.py': ('# PROJECT: Engineering Toolkit and Simulators — science\n'
                               '# DESCRIPTION: Vendored science toolkit modules for the portable Chemical Calculator harness.\n'),
        'science_bundle/cite.py': cite_src,
        'science_bundle/paths.py': paths_src,
        'science_bundle/thermo.py': thermo_src,
        'science_bundle/data/thermo_species.json': thermo_data,
        'recipe.json': serialize_result(recipe),
        'result.json': serialize_result(result),
        'pubchem_snapshot.json': serialize_snapshot(snapshot),
        'requirements.txt': requirements,
        'run.py': _export_run_py(),
        'README.md': _export_readme(recipe_id, recipe, result, created),
    }
    import zipfile
    # Fixed entry timestamps make the archive byte-identical across exports of the
    # same stored run (the result file itself is the hashed content; the archive
    # should not silently vary).
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, content)

    # Verify the zip by extracting it and running `run.py verify` with the venv.
    import tempfile
    import subprocess
    import sys
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(td)
        proc = subprocess.run([sys.executable, '-B', str(Path(td) / 'run.py'), 'verify'],
                              capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise ValueError('exported harness failed verification: ' + (proc.stdout or proc.stderr or '').strip())

    size = zip_path.stat().st_size
    return {
        'zip_path': str(zip_path),
        'bytes': size,
        'sha256': hashlib.sha256(zip_path.read_bytes()).hexdigest(),
    }

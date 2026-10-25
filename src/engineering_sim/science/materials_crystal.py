# DESCRIPTION: Materials Science area (Task H): COD crystal search/fetch, pymatgen
#              analysis (space group, cell, density, coordination) and XRD, plus ASE
#              idealised prototype builders for the unit-cell viewer.
"""Crystal-structure engine for science toolkit (Crystallography Open Database + pymatgen/ASE).

Search and download use the Crystallography Open Database REST API
(https://wiki.crystallography.net/RESTful_API/), an open, key-free database; every
CIF is cached under ``paths.data_dir()/'crystals'``. Lattice values read from the CIF
are ``cited`` to the COD URL; quantities computed from the cell (volume, Z, density,
space group via SpacegroupAnalyzer, coordination via CrystalNN, the XRD pattern) are
``derived`` with the method named. Atom coordinates are returned as viewer data (plain
floats) with a provenance note, not as per-atom cite records.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
import warnings

from engineering_sim.science import cite, paths

COD_RESULT = "https://www.crystallography.net/cod/result"
COD_CIF = "https://www.crystallography.net/cod/{id}.cif"
UA = "EngineeringSim/1.0"
TIMEOUT = 30  # seconds per HTTP call
MAX_RESULTS = 100
MAX_VIEWER_ATOMS = 2000
MAX_SUPERCELL = 3
N_A = 6.02214076e23  # mol^-1 (CODATA 2018)



def _get_json(url, timeout=TIMEOUT):
    """GET a JSON endpoint with urllib and return the parsed object."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _get_text(url, timeout=TIMEOUT):
    """GET a plain-text endpoint (a CIF) and return its text."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _hill_formula(text):
    """Best-effort Hill-notation formula (space separated) for a COD formula query."""
    try:
        from pymatgen.core import Composition
        return Composition(text.strip()).hill_formula
    except Exception:  # noqa: BLE001 - fall back to the raw string for exotic inputs
        return text.strip()


# ---- Search ------------------------------------------------------------------
def search(formula=None, elements=None, text=None, max_results=25):
    """Search the COD for crystal structures.

    ``formula`` is converted to Hill notation; ``elements`` (a list of symbols) maps to
    the COD ``el1..el8`` parameters; ``text`` searches entry metadata. Returns a list of
    candidate dicts (COD id, formula, space group, cell, year, journal, DOI, URL). Cell
    numbers are ``cited`` to the exact search URL; the rest are strings.
    """
    params = {"format": "json"}
    formula = (formula or "").strip()
    if formula:
        params["formula"] = _hill_formula(formula)
    elems = [e.strip().capitalize() for e in (elements or []) if (e or "").strip()][:8]
    for i, el in enumerate(elems, start=1):
        params["el%d" % i] = el
    txt = (text or "").strip()
    if txt:
        params["text"] = txt
    if not params or (len(params) == 1 and "format" in params):
        return {"error": "Provide a formula, a list of elements, or a text query."}
    n = max(1, min(int(max_results or 25), MAX_RESULTS))
    url = COD_RESULT + "?" + urllib.parse.urlencode(params)
    try:
        rows = _get_json(url)
    except Exception as exc:  # noqa: BLE001 - network/parse errors are a soft failure
        return {"error": "%s: %s" % (type(exc).__name__, exc)}
    if not isinstance(rows, list):
        return {"error": "Unexpected COD response (not a list)."}
    retrieved = cite.today()
    out = []
    for r in rows[:n]:
        cid = str(r.get("file", ""))
        cand = {
            "id": cid,
            "formula": r.get("formula") or r.get("calcformula") or "",
            "chemname": r.get("chemname") or "",
            "mineral": r.get("mineral") or "",
            "space_group": r.get("sg") or "",
            "space_group_number": r.get("sgNumber") or "",
            "authors": r.get("authors") or "",
            "title": r.get("title") or "",
            "journal": r.get("journal") or "",
            "year": r.get("year") or "",
            "doi": r.get("doi") or "",
            "url": COD_CIF.format(id=cid),
        }
        for key, field in (("a", "a"), ("b", "b"), ("c", "c"),
                           ("alpha", "alpha"), ("beta", "beta"), ("gamma", "gamma"),
                           ("volume", "vol")):
            v = r.get(field)
            if v not in (None, ""):
                cand[key] = cite.value(float(v), "Å" if key != "volume" else "Å³", "cited",
                                      source=url, retrieved=retrieved, verified=True,
                                      note="COD search result (cell metadata)")
        out.append(cand)
    return {"count": len(out), "url": url, "retrieved": retrieved, "results": out}


# ---- Fetch + analysis --------------------------------------------------------
def _cif_path(cod_id):
    return paths.data_dir() / "crystals" / ("%s.cif" % cod_id)


def _download_cif(cod_id, refresh=False, offline=False):
    """Return (cif_text, source_url, from_cache) for a COD id.

    Caches the downloaded CIF under the runtime data directory. ``offline=True``
    returns the cached CIF or raises a clear error.
    """
    cid = str(cod_id).strip()
    if not cid or not cid.isdigit():
        raise ValueError("COD id must be numeric (got %r)." % cod_id)
    path = _cif_path(cid)
    url = COD_CIF.format(id=cid)
    if not refresh and path.is_file():
        return path.read_text(encoding="utf-8", errors="replace"), url, True
    if offline:
        raise ValueError("Offline: no cached CIF for COD id %s." % cid)
    text = _get_text(url)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError:
        pass  # cache is best-effort
    return text, url, False


def _cif_metadata(cif_text):
    """Extract bibliographic strings from the CIF via pymatgen's CifFile parser."""
    from pymatgen.io.cif import CifParser
    parser = CifParser.from_str(cif_text)
    meta = {}
    for block in parser._cif.data.values():
        d = getattr(block, "data", block)
        meta.setdefault("authors", d.get("_publ_author_name"))
        meta.setdefault("title", d.get("_publ_section_title"))
        meta.setdefault("journal", d.get("_journal_name_full"))
        meta.setdefault("year", d.get("_journal_year"))
        meta.setdefault("doi", d.get("_journal_paper_doi"))
        meta.setdefault("volume", d.get("_journal_volume"))
        meta.setdefault("firstpage", d.get("_journal_page_first"))
        meta.setdefault("lastpage", d.get("_journal_page_last"))
        meta.setdefault("density_diffrn", d.get("_exptl_crystal_density_diffrn"))
    def _clean(v):
        if isinstance(v, (list, tuple)):
            v = "; ".join(str(x) for x in v if x)
        return str(v).strip() if v else ""
    return {k: _clean(v) for k, v in meta.items()}


def _structure_from_text(cif_text):
    """Parse a CIF with pymatgen; returns (structure, partial_occupancy_flag)."""
    from pymatgen.io.cif import CifParser
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = CifParser.from_str(cif_text)
        structs = parser.parse_structures(primitive=False)
    if not structs:
        raise ValueError("CIF parsed to no structure.")
    structure = structs[0]
    partial = False
    for site in structure:
        total = sum(site.species.values())
        if total < 1.0 - 1e-6:
            partial = True
            break
    return structure, partial


def _quiet(fn):
    """Run a function with library warnings suppressed (spglib/CrystalNN chatter)."""
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return fn(*args, **kwargs)
    return wrapper


@_quiet
def _analyze(structure, source_url, retrieved, from_cache):
    """Assemble the structure card from a parsed pymatgen Structure."""
    from pymatgen.analysis.local_env import CrystalNN
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    sa = SpacegroupAnalyzer(structure, symprec=0.01)
    try:
        sg_symbol = sa.get_space_group_symbol()
        sg_number = sa.get_space_group_number()
        crystal_system = sa.get_crystal_system()
    except Exception:  # noqa: BLE001 - symmetry detection can fail on disordered cells
        sg_symbol = structure.get_space_group_info(symprec=0.01)[0]
        sg_number = structure.get_space_group_info(symprec=0.01)[1]
        crystal_system = ""
    conventional = sa.get_conventional_standard_structure()
    lat = conventional.lattice
    reduced, factor = conventional.composition.get_reduced_composition_and_factor()
    formula_str = conventional.composition.reduced_formula  # e.g. "NaCl" (no subscripts)
    density = conventional.density  # g/cm^3 (mass of all sites / N_A / volume)

    atoms = []
    for site in conventional:
        atoms.append({
            "el": site.species_string,
            "frac": [round(float(x), 6) for x in site.frac_coords],
            "cart": [round(float(x), 4) for x in site.coords],
        })
    capped = len(atoms) > MAX_VIEWER_ATOMS

    # Coordination environments via CrystalNN (capped for runtime).
    coord = {}
    cnn = CrystalNN()
    for i, site in enumerate(conventional[:200]):
        try:
            cn = cnn.get_cn(conventional, i)
        except Exception:  # noqa: BLE001 - one site failing must not break the card
            continue
        coord.setdefault(site.species_string, []).append(round(float(cn), 3))
    coord_summary = {
        el: {"count": len(v), "cn": sum(v) / len(v)} for el, v in sorted(coord.items())
    }

    result = {
        "formula": cite.value(formula_str, "", "derived",
                              method="pymatgen reduced composition of the CIF cell"),
        "space_group_symbol": sg_symbol,
        "space_group_number": sg_number,
        "crystal_system": crystal_system,
        "lattice": {
            "a": cite.value(round(float(lat.a), 6), "Å", "cited", source=source_url,
                            retrieved=retrieved, verified=not from_cache,
                            note="CIF cell length a"),
            "b": cite.value(round(float(lat.b), 6), "Å", "cited", source=source_url,
                            retrieved=retrieved, verified=not from_cache,
                            note="CIF cell length b"),
            "c": cite.value(round(float(lat.c), 6), "Å", "cited", source=source_url,
                            retrieved=retrieved, verified=not from_cache,
                            note="CIF cell length c"),
            "alpha": cite.value(round(float(lat.alpha), 6), "°", "cited", source=source_url,
                                retrieved=retrieved, verified=not from_cache,
                                note="CIF cell angle alpha"),
            "beta": cite.value(round(float(lat.beta), 6), "°", "cited", source=source_url,
                               retrieved=retrieved, verified=not from_cache,
                               note="CIF cell angle beta"),
            "gamma": cite.value(round(float(lat.gamma), 6), "°", "cited", source=source_url,
                                retrieved=retrieved, verified=not from_cache,
                                note="CIF cell angle gamma"),
        },
        "volume": cite.value(round(float(lat.volume), 4), "Å³", "derived",
                             method="a·b·c·sin terms (conventional cell)"),
        "Z": cite.value(int(factor), "formula units", "derived",
                        method="conventional-cell sites / reduced formula atoms"),
        "density": cite.value(round(float(density), 4), "g/cm³", "derived",
                              method="cell mass / (N_A · cell volume)"),
        "atoms": atoms[:MAX_VIEWER_ATOMS],
        "atom_count": len(atoms),
        "coordination": coord_summary,
        "notes": [],
    }
    if capped:
        result["notes"].append(
            "Cell has %d atoms; only the first %d are sent to the viewer (apply a supercell cautiously)."
            % (len(atoms), MAX_VIEWER_ATOMS))
    return result


def fetch(cod_id, refresh=False, offline=False):
    """Download and analyse one COD structure.

    Returns the structure card (formula, space group, lattice, volume, Z, density,
    atoms for the viewer, coordination summary, reference). Lattice numbers are
    ``cited`` to the COD CIF URL with ``verified=True`` on a fresh fetch and
    ``verified=False`` when served from the local cache. Never raises.
    """
    try:
        cif_text, url, from_cache = _download_cif(cod_id, bool(refresh), bool(offline))
        retrieved = cite.today()
        structure, partial = _structure_from_text(cif_text)
    except Exception as exc:  # noqa: BLE001 - network/parse errors become a soft failure
        return {"error": "%s: %s" % (type(exc).__name__, exc)}

    result = _analyze(structure, url, retrieved, from_cache)
    meta = _cif_metadata(cif_text)
    result["reference"] = {
        "authors": meta.get("authors", ""),
        "title": meta.get("title", ""),
        "journal": meta.get("journal", ""),
        "year": meta.get("year", ""),
        "volume": meta.get("volume", ""),
        "pages": ("%s–%s" % (meta["firstpage"], meta["lastpage"]) if meta.get("firstpage") else ""),
        "doi": meta.get("doi", ""),
    }
    result["cod_id"] = str(cod_id)
    result["url"] = url
    result["retrieved"] = retrieved
    result["from_cache"] = bool(from_cache)
    if meta.get("density_diffrn"):
        try:
            result["density_reported"] = cite.value(
                float(meta["density_diffrn"]), "g/cm³", "cited", source=url,
                retrieved=retrieved, verified=not from_cache,
                note="CIF _exptl_crystal_density_diffrn")
        except (TypeError, ValueError):
            pass
    if partial:
        result["notes"].append("CIF contains partial site occupancies; the reported cell is the disordered average.")
    result["notes"].append("Space group from SpacegroupAnalyzer (symprec 0.01); density is a derived value from the cell and atomic masses.")
    return result


# ---- XRD ---------------------------------------------------------------------
def xrd_pattern(structure, wavelength="CuKa", two_theta=(10, 90)):
    """Powder XRD peaks for a pymatgen Structure (intensity >= 1 % of max).

    Returns ``{peaks, wavelength, two_theta_range}``; each peak carries 2θ, d-spacing,
    hkl and relative intensity. Used by ``xrd`` (COD path) and directly by the tests.
    """
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    lo, hi = float(two_theta[0]), float(two_theta[1])
    if not (0 < lo < hi <= 180):
        raise ValueError("two_theta must satisfy 0 < lo < hi <= 180.")
    calc = XRDCalculator(wavelength=wavelength)
    pat = calc.get_pattern(structure, two_theta_range=(lo, hi))
    lam = float(calc.wavelength)
    peaks = []
    for i in range(len(pat.x)):
        if pat.y[i] < 1.0:  # < 1 % of the strongest peak (pattern is normalised to 100)
            continue
        hkl = tuple(pat.hkls[i][0]["hkl"]) if pat.hkls[i] else ()
        peaks.append({
            "two_theta": round(float(pat.x[i]), 4),
            "intensity": round(float(pat.y[i]), 4),
            "d_spacing": round(float(pat.d_hkls[i]), 4),
            "hkl": list(hkl),
        })
    return {"peaks": peaks, "wavelength": lam, "two_theta_range": [lo, hi]}


def xrd(cod_id, wavelength="CuKa", two_theta=(10, 90)):
    """Simulated powder XRD pattern (Cu Kα by default) for a COD structure.

    Returns peaks with intensity >= 1 % of the maximum, each with 2θ, d-spacing, hkl
    and relative intensity, computed by pymatgen's XRDCalculator (kinematic model,
    no peak broadening, no absorption). ``wavelength`` is a pymatgen XRDCalculator key
    ('CuKa' = 1.5406 Å, 'MoKa', 'CoKa', 'FeKa', ...).
    """
    try:
        cif_text, url, from_cache = _download_cif(cod_id, refresh=False, offline=False)
        structure, _ = _structure_from_text(cif_text)
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, exc)}
    try:
        pat = xrd_pattern(structure, wavelength=wavelength, two_theta=two_theta)
    except Exception as exc:  # noqa: BLE001 - bad wavelength key or range
        return {"error": "%s: %s" % (type(exc).__name__, exc)}
    return {
        "cod_id": str(cod_id),
        "wavelength": cite.value(pat["wavelength"], "Å", "unverified",
                                 note="pymatgen XRDCalculator %s (standard Kα mean)" % wavelength),
        "two_theta_range": pat["two_theta_range"],
        "peak_count": len(pat["peaks"]),
        "peaks": pat["peaks"],
        "notes": [
            "Kinematic powder pattern (pymatgen XRDCalculator): no thermal/absorption corrections, no peak broadening, no preferred orientation.",
            "Intensities are relative to the strongest peak (100).",
        ],
    }


# ---- Supercell ---------------------------------------------------------------
def supercell(cod_id, n=2):
    """Repeat a COD cell n×n×n for the viewer (n capped at MAX_SUPERCELL)."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return {"error": "n must be an integer."}
    if not (1 <= n <= MAX_SUPERCELL):
        return {"error": "n must be in 1..%d (got %d)." % (MAX_SUPERCELL, n)}
    try:
        cif_text, url, from_cache = _download_cif(cod_id, refresh=False, offline=False)
        structure, _ = _structure_from_text(cif_text)
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, exc)}
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    conventional = SpacegroupAnalyzer(structure, symprec=0.01).get_conventional_standard_structure()
    cell = conventional * (n, n, n)
    lat = cell.lattice
    if len(cell) > MAX_VIEWER_ATOMS:
        return {"error": "%d×%d×%d supercell has %d atoms (cap %d)."
                         % (n, n, n, len(cell), MAX_VIEWER_ATOMS)}
    atoms = [{"el": s.species_string,
              "frac": [round(float(x), 6) for x in s.frac_coords],
              "cart": [round(float(x), 4) for x in s.coords]} for s in cell]
    return {
        "cod_id": str(cod_id),
        "n": n,
        "atom_count": len(atoms),
        "atoms": atoms,
        "lattice": {
            "a": round(float(lat.a), 6), "b": round(float(lat.b), 6), "c": round(float(lat.c), 6),
            "alpha": round(float(lat.alpha), 6), "beta": round(float(lat.beta), 6),
            "gamma": round(float(lat.gamma), 6),
        },
        "notes": ["%d×%d×%d repetition of the conventional cell." % (n, n, n)],
    }


# ---- Prototypes (idealised, ASE) ---------------------------------------------
def density_from_cell(elements, volume_ang3):
    """Density (g/cm³) of a cell from a list of element symbols and its volume (Å³).

    Uses CIAAW/IUPAC atomic masses (pymatgen ``Element.atomic_mass``) and the CODATA
    Avogadro constant: rho = sum(masses) / (N_A · V).
    """
    from pymatgen.core import Element
    if volume_ang3 <= 0:
        raise ValueError("cell volume must be positive (got %r)." % volume_ang3)
    mass_g = sum(Element(s).atomic_mass for s in elements)
    return mass_g / (N_A * volume_ang3 * 1e-24)


def from_formula_prototype(formula, prototype="fcc", a=4.0):
    """Build an idealised prototype structure with ASE.

    ``prototype`` in rocksalt | fcc | bcc | hcp | diamond | perovskite. ``a`` is the
    cubic lattice parameter in Å (hcp uses it for a, c = sqrt(8/3)·a). The result is
    clearly labelled an idealised prototype (``kind='user'`` for ``a``), not an
    experimental determination.
    """
    try:
        from pymatgen.core import Composition
        comp = Composition(str(formula).strip())
        symbols = [str(e) for e in comp.elements]
        counts = [int(comp[e]) for e in comp.elements]
    except Exception as exc:  # noqa: BLE001
        return {"error": "Could not parse formula %r (%s)." % (formula, type(exc).__name__)}

    proto = (prototype or "fcc").lower()
    try:
        a = float(a)
    except (TypeError, ValueError):
        return {"error": "a must be numeric (got %r)." % a}
    if a <= 0:
        return {"error": "a must be positive (got %r)." % a}

    try:
        from ase import Atoms
        from ase.build import bulk
        from ase.spacegroup import crystal
    except ImportError as exc:  # pragma: no cover - the venv ships ASE
        return {"error": "ASE unavailable: %s" % exc}

    atoms = None
    note = ""
    if proto in ("fcc", "bcc", "hcp", "diamond"):
        if len(symbols) != 1:
            return {"error": "prototype %r needs a single-element formula (got %r)." % (proto, formula)}
        atoms = bulk(symbols[0], proto, a=a)
        note = "Idealised %s packing of %s (ASE bulk builder)." % (proto, symbols[0])
    elif proto == "rocksalt":
        if len(symbols) != 2:
            return {"error": "rocksalt needs a two-element formula AB (got %r)." % formula}
        atoms = crystal([symbols[0], symbols[1]], [(0, 0, 0), (0.5, 0.5, 0.5)],
                        spacegroup=225, cellpar=[a, a, a, 90, 90, 90])
        note = "Idealised rock-salt (Fm-3m) %s (ASE)." % formula
    elif proto == "perovskite":
        if len(symbols) != 3:
            return {"error": "perovskite needs an ABX3 formula (got %r)." % formula}
        if counts != [1, 1, 3]:
            return {"error": "perovskite expects ABX3 stoichiometry (got %r)." % formula}
        atoms = crystal([symbols[0], symbols[1], symbols[2]],
                        [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
                        spacegroup=221, cellpar=[a, a, a, 90, 90, 90])
        note = "Idealised cubic perovskite (Pm-3m) %s (ASE)." % formula
    else:
        return {"error": "Unknown prototype %r (choose rocksalt, fcc, bcc, hcp, diamond, perovskite)."
                         % prototype}

    cell = atoms.get_cell()
    # Density from the ASE cell and pymatgen atomic masses.
    vol_ang3 = abs(atoms.get_volume())
    density = density_from_cell(atoms.get_chemical_symbols(), vol_ang3) if vol_ang3 > 0 else None

    out_atoms = [{"el": s, "frac": [round(float(x), 6) for x in f],
                  "cart": [round(float(x), 4) for x in c]}
                 for s, f, c in zip(atoms.get_chemical_symbols(),
                                    atoms.get_scaled_positions(), atoms.get_positions())]
    return {
        "formula": formula,
        "prototype": proto,
        "lattice_a": cite.value(a, "Å", "user", note="user-supplied lattice parameter"),
        "atom_count": len(atoms),
        "atoms": out_atoms,
        "lattice": {
            "a": round(float(cell[0][0]), 6), "b": round(float(cell[1][1]), 6),
            "c": round(float(cell[2][2]), 6),
            "alpha": round(float(atoms.cell.cellpar()[3]), 6),
            "beta": round(float(atoms.cell.cellpar()[4]), 6),
            "gamma": round(float(atoms.cell.cellpar()[5]), 6),
        },
        "density": cite.value(round(float(density), 4), "g/cm³", "derived",
                              method="cell mass / (N_A · cell volume)") if density is not None else None,
        "notes": [note, "Idealised prototype geometry, not an experimental determination."],
    }

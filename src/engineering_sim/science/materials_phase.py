# DESCRIPTION: Materials Science area (Task H): pycalphad binary phase diagrams and
#              single-point equilibria from open thermodynamic databases (data/tdb).
"""Phase-diagram engine for science toolkit (pycalphad over open CALPHAD databases).

Shipped databases live in ``engineering_sim/science/data/tdb/``, one ``.tdb`` plus a provenance
``.json`` (source URL, retrieval date, reference, elements, phases, license). The phase
diagram is computed on a bounded T-X grid; phase-region boundaries are returned as
polylines and two-phase tie-lines as horizontal segments, so the page draws an SVG plot
(no image is produced server-side). Everything here is ``derived`` (method names the
pycalphad call) from the database, which is itself cited in the provenance file.
"""
from __future__ import annotations

import json
import os
import warnings

from engineering_sim.science import cite, paths

TDB_DIR = os.path.join(os.path.dirname(__file__), "data", "tdb")
def _tdb_path(name):
    if os.environ.get("ENGINEERING_SIM_TDB_DIR"):
        return os.path.join(os.environ["ENGINEERING_SIM_TDB_DIR"], name + ".tdb")
    import pycalphad
    return os.path.join(os.path.dirname(pycalphad.__file__), "tests", "databases", name + ".tdb")
MAX_GRID = 100
DEFAULT_GRID = 60
NP_THRESHOLD = 1e-3  # phase amount below which a phase is treated as absent


def _tdb_files():
    """List (name, tdb_path, meta) for every shipped database, sorted by name."""
    out = []
    for jf in sorted(os.listdir(TDB_DIR)):
        if not jf.endswith(".json"):
            continue
        name = jf[:-5]
        tdb = _tdb_path(name)
        if not os.path.isfile(tdb):
            continue
        with open(os.path.join(TDB_DIR, jf), "r", encoding="utf-8") as f:
            meta = json.load(f)
        out.append((name, tdb, meta))
    return out


def systems():
    """List the shipped thermodynamic databases (no computation)."""
    return [{"name": n, "elements": m.get("elements", []), "phases": m.get("phases", []),
             "reference": m.get("reference", ""), "source_url": m.get("source_url", ""),
             "retrieved": m.get("retrieved", ""), "license": m.get("license", ""),
             "notes": m.get("notes", "")}
            for n, _, m in _tdb_files()]


def _load_database(name):
    """Return (pycalphad Database, meta) for a shipped database name."""
    for n, tdb, meta in _tdb_files():
        if n == name:
            from pycalphad import Database
            return Database(tdb), meta
    raise ValueError("Unknown database %r. Available: %s"
                     % (name, ", ".join(n for n, _, _ in _tdb_files())))


def _stable_phases(eq, second=None, np_threshold=NP_THRESHOLD):
    """List of (phase_name, amount, x_second) for phases above the threshold.

    ``second`` names the component whose mole fraction is reported (defaults to the
    last non-reference component, i.e. the second of a binary pair).
    """
    import numpy as np
    comps = list(eq.X.component.values)
    if second is None:
        second = comps[-1] if comps else None
    vertices = np.asarray(eq.Phase.values).ravel()
    amounts = np.asarray(eq.NP.values).ravel()
    xs = np.asarray(eq.X.sel(component=second).values).ravel() if second in comps \
        else np.zeros_like(amounts)
    out = []
    for idx, name in enumerate(vertices):
        if name and float(amounts[idx]) > np_threshold:
            out.append((str(name), float(amounts[idx]), float(xs[idx])))
    return out


def equilibrium_point(tdb, comps, conditions=None):
    """Stable phases, amounts and compositions at one (T, P, x) point.

    ``conditions`` = ``{'T': K, 'P': Pa, 'x': mole fraction of comps[1]}``. Returns the
    phase list plus the overall conditions. The caller must provide exactly two elements.
    """
    conditions = conditions or {}
    try:
        dbf, meta = _load_database(tdb)
    except ValueError as exc:
        return {"error": str(exc)}
    comps = [str(c).upper() for c in (comps or [])]
    if len(comps) != 2:
        return {"error": "equilibrium_point needs exactly two elements (got %r)." % comps}
    try:
        T = float(conditions.get("T", 298.15))
        P = float(conditions.get("P", 101325.0))
        x = float(conditions.get("x", 0.5))
    except (TypeError, ValueError):
        return {"error": "T, P and x must be numeric."}
    if not (0.0 <= x <= 1.0):
        return {"error": "x must be in [0, 1] (got %r)." % x}
    import pycalphad.variables as v
    from pycalphad import equilibrium
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # pycalphad on numpy 2.5
        try:
            eq = equilibrium(dbf, comps + ["VA"], list(dbf.phases.keys()),
                             {v.X(comps[1]): x, v.T: T, v.P: P})
        except Exception as exc:  # noqa: BLE001 - solver failure becomes a message
            return {"error": "equilibrium failed: %s: %s" % (type(exc).__name__, exc)}
    phases = [{"phase": p, "fraction": a, "x_%s" % comps[1]: xp}
              for p, a, xp in _stable_phases(eq)]
    return {
        "tdb": tdb,
        "conditions": {"T": T, "P": P, "x": x, "component": comps[1]},
        "phases": phases,
        "notes": ["Phases below %.0e amount are omitted." % NP_THRESHOLD],
    }


def _grid(dbf, comps, phases, Ts, Xs, P):
    """Compute the stable phase-set at every (T, X) grid point plus tie-lines.

    Returns (rows, tie_lines) where rows[i][j] is a frozenset of stable phase names and
    tie_lines collects one tie-line per two-phase region per temperature row.
    """
    import numpy as np
    import pycalphad.variables as v
    from pycalphad import equilibrium

    second = comps[1]
    rows = []
    tie_lines = []
    for T in Ts:
        row = []
        seen_pairs = set()
        for X in Xs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                eq = equilibrium(dbf, comps + ["VA"], phases,
                                 {v.X(second): float(X), v.T: float(T), v.P: float(P)})
            stable = _stable_phases(eq, second=second)
            row.append(frozenset(p for p, _, _ in stable))
            if len(stable) == 2:
                pair = frozenset((stable[0][0], stable[1][0]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    x_a = stable[0][2] if stable[0][0] < stable[1][0] else stable[1][2]
                    x_b = stable[1][2] if stable[0][0] < stable[1][0] else stable[0][2]
                    pa = min(stable[0][0], stable[1][0])
                    pb = max(stable[0][0], stable[1][0])
                    tie_lines.append({"T": round(float(T), 3), "x_a": round(x_a, 6),
                                      "x_b": round(x_b, 6), "phase_a": pa, "phase_b": pb})
        rows.append(row)
    return rows, tie_lines


def _boundaries(rows, Ts, Xs):
    """Extract phase-region boundary polylines from a categorical phase-set grid.

    Boundary points are placed at the midpoint of every grid edge whose two endpoints
    differ, then chained greedily by proximity (in normalized X-T space) into polylines.
    """
    ids = {}
    id_rows = []
    for row in rows:
        id_row = []
        for s in row:
            key = tuple(sorted(s))
            if key not in ids:
                ids[key] = len(ids)
            id_row.append(ids[key])
        id_rows.append(id_row)
    # name -> sorted tuple of phases
    rev = {v: k for k, v in ids.items()}
    nT, nx = len(rows), len(Xs)
    x_range = (Xs[-1] - Xs[0]) or 1.0
    t_range = (Ts[-1] - Ts[0]) or 1.0
    scale_x = 1.0 / x_range
    scale_t = 1.0 / t_range
    tol = 2.6 * max(1.0 / max(nx - 1, 1), 1.0 / max(nT - 1, 1))

    groups = {}
    # horizontal edges
    for i in range(nT):
        for j in range(nx - 1):
            if id_rows[i][j] != id_rows[i][j + 1]:
                key = tuple(sorted((id_rows[i][j], id_rows[i][j + 1])))
                groups.setdefault(key, []).append(((Xs[j] + Xs[j + 1]) / 2.0, Ts[i]))
    # vertical edges
    for i in range(nT - 1):
        for j in range(nx):
            if id_rows[i][j] != id_rows[i + 1][j]:
                key = tuple(sorted((id_rows[i][j], id_rows[i + 1][j])))
                groups.setdefault(key, []).append((Xs[j], (Ts[i] + Ts[i + 1]) / 2.0))

    boundaries = []
    for key in sorted(groups):
        pts = groups[key]
        polylines = _chain(pts, tol, scale_x, scale_t)
        set_a, set_b = rev[key[0]], rev[key[1]]
        for line in polylines:
            if len(line) >= 2:
                boundaries.append({
                    "phases_a": list(set_a), "phases_b": list(set_b),
                    "points": [[round(float(x), 6), round(float(t), 3)] for x, t in line],
                })
    return boundaries


def _chain(points, tol, scale_x, scale_t):
    """Greedily chain nearby points into polylines in normalized (X, T) space."""
    pts = [(x * scale_x, t * scale_t, x, t) for x, t in points]
    remaining = pts[:]
    lines = []

    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    while remaining:
        line = [remaining.pop(0)]
        grew = True
        while grew:
            grew = False
            for k in range(len(remaining) - 1, -1, -1):
                if dist(line[-1], remaining[k]) <= tol:
                    line.append(remaining.pop(k))
                    grew = True
                    break
        grew = True
        while grew:
            grew = False
            for k in range(len(remaining) - 1, -1, -1):
                if dist(line[0], remaining[k]) <= tol:
                    line.insert(0, remaining.pop(k))
                    grew = True
                    break
        lines.append([(p[2], p[3]) for p in line])
    return lines


def binary_phase_diagram(tdb, comps, T_range=None, x_range=None, P=101325.0,
                         nT=None, nx=None):
    """Binary T-X phase diagram as SVG-ready data (boundaries + tie-lines + labels).

    ``comps`` are the two elements (the x-axis is the mole fraction of ``comps[1]``).
    Grid is bounded at %d × %d points; defaults to %d × %d. Returns polylines for the
    phase-region boundaries, horizontal tie-lines, and a label anchor per phase region.
    """ % (MAX_GRID, MAX_GRID, DEFAULT_GRID, DEFAULT_GRID)
    try:
        dbf, meta = _load_database(tdb)
    except ValueError as exc:
        return {"error": str(exc)}
    comps = [str(c).upper() for c in (comps or [])]
    if len(comps) != 2:
        return {"error": "binary_phase_diagram needs exactly two elements (got %r)." % comps}

    def _two(v, default):
        v = v or default
        try:
            lo, hi = float(v[0]), float(v[1])
        except (TypeError, ValueError, IndexError):
            return None
        if lo >= hi:
            return None
        return lo, hi

    T_range = _two(T_range, (300.0, 1200.0))
    x_range = _two(x_range, (1e-6, 1.0 - 1e-6))
    if T_range is None or x_range is None:
        return {"error": "T_range and x_range must each be a [lo, hi] pair with lo < hi."}
    nT = min(max(int(nT or DEFAULT_GRID), 10), MAX_GRID)
    nx = min(max(int(nx or DEFAULT_GRID), 10), MAX_GRID)

    import numpy as np
    Ts = np.linspace(T_range[0], T_range[1], nT)
    Xs = np.linspace(x_range[0], x_range[1], nx)
    phases = list(dbf.phases.keys())

    try:
        rows, tie_lines = _grid(dbf, comps, phases, Ts, Xs, float(P))
    except Exception as exc:  # noqa: BLE001 - solver failure becomes a message
        return {"error": "phase diagram computation failed: %s: %s" % (type(exc).__name__, exc)}

    boundaries = _boundaries(rows, Ts, Xs)

    # Region label anchors: the average (X, T) of all grid points with each phase set.
    anchors = {}
    counts = {}
    for i in range(nT):
        for j in range(nx):
            key = tuple(sorted(rows[i][j]))
            anchors.setdefault(key, [0.0, 0.0])
            counts[key] = counts.get(key, 0) + 1
            anchors[key][0] += Xs[j]
            anchors[key][1] += Ts[i]
    regions = [{"phases": list(key), "x": round(float(anchors[key][0]) / counts[key], 6),
                "T": round(float(anchors[key][1]) / counts[key], 3)}
               for key in sorted(anchors)]

    return {
        "tdb": tdb,
        "reference": meta.get("reference", ""),
        "comps": comps,
        "P": float(P),
        "T_range": [T_range[0], T_range[1]],
        "x_range": [x_range[0], x_range[1]],
        "grid": {"nT": nT, "nx": nx},
        "boundaries": boundaries,
        "tie_lines": tie_lines,
        "regions": regions,
        "notes": [
            "Computed with pycalphad %s on a %d×%d T-X grid; boundaries are interpolated between grid points."
            % (_pycalphad_version(), nT, nx),
            "Tie-lines connect the two phase compositions at each sampled temperature.",
            "Only the phases declared by the database are considered.",
        ],
    }


def _pycalphad_version():
    try:
        from importlib.metadata import version
        return version("pycalphad")
    except Exception:  # noqa: BLE001
        return "?"

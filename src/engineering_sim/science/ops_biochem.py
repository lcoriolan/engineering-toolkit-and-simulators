# DESCRIPTION: Biochemistry area extension registry: op timeouts and handlers (standard-library
#              imports only at module level; heavy Biopython/primer3/numpy/scipy imports live in

"""science toolkit Biochemistry area (Task I): DNA/RNA tools, primers, enzyme kinetics, buffers.

Standard library only at module level so engineering_sim/science/api.py can import this under the studio's
stdlib interpreter to read TIMEOUTS. Every handler takes ``args: dict`` and returns a
JSON-serialisable dict; the heavy science modules (biochem_nucleic/biochem_enzyme/biochem_buffers)
are imported inside each handler.
"""
from __future__ import annotations

import json
import os

from . import cite

# Seconds. Primers/alignment/fitting do real optimisation and get longer budgets; everything is
# bounded by input caps inside the science modules rather than by the timeout.
TIMEOUTS = {
    "biochem_nucleic": 120,
    "biochem_primers": 300,
    "biochem_restriction": 120,
    "biochem_align": 300,
    "biochem_mutate": 120,
    "biochem_fit": 300,
    "biochem_simulate": 120,
    "biochem_efficiency": 60,
    "biochem_eyring": 60,
    "biochem_buffers": 60,
    "biochem_hh": 60,
    "biochem_prepare_buffer": 60,
    "biochem_titration": 60,
    "biochem_ionic_strength": 60,
    "biochem_pi": 60,
    "biochem_dg": 60,
}

_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "biochem_pka.json")


def _biochem_nucleic(args):
    from engineering_sim.science import biochem_nucleic
    return biochem_nucleic.analyze(args.get("seq", ""), kind=args.get("kind", "auto"))


def _biochem_primers(args):
    from engineering_sim.science import biochem_nucleic
    return biochem_nucleic.primers(
        args.get("template", ""),
        target_region=args.get("target_region"),
        size=tuple(args.get("size", (18, 25))),
        tm=tuple(args.get("tm", (57, 63))))


def _biochem_restriction(args):
    from engineering_sim.science import biochem_nucleic
    return biochem_nucleic.restriction(args.get("seq", ""), enzymes=args.get("enzymes", "common"))


def _biochem_align(args):
    from engineering_sim.science import biochem_nucleic
    return biochem_nucleic.align(args.get("seq_a", ""), args.get("seq_b", ""),
                                 mode=args.get("mode", "global"))


def _biochem_mutate(args):
    from engineering_sim.science import biochem_nucleic
    return biochem_nucleic.mutate(args.get("seq", ""), args.get("spec", []))


def _as_number_list(x):
    """Accept a JSON list or a whitespace/comma/; separated string; return a list of floats."""
    if isinstance(x, str):
        import re
        return [float(v) for v in re.split(r"[,\s;]+", x.strip()) if v]
    return [float(v) for v in list(x)]


def _fit_inputs(args):
    """Resolve the S/v/I arrays for the kinetics fit from either arrays or a pasted CSV."""
    from engineering_sim.science import biochem_enzyme
    if args.get("csv"):
        import csv as _csv
        import io
        rows = list(_csv.reader(io.StringIO(args["csv"].strip())))
        rows = [r for r in rows if r and any(c.strip() for c in r)]
        header = rows[0]
        body = rows[1:]
        sidx = next(i for i, h in enumerate(header) if h.strip().lower() in ("s", "[s]", "substrate"))
        vidx = next(i for i, h in enumerate(header) if h.strip().lower() in ("v", "v0", "rate"))
        iidx = next((i for i, h in enumerate(header) if h.strip().lower() in ("i", "[i]", "inhibitor")), None)
        S = [float(r[sidx]) for r in body]
        v = [float(r[vidx]) for r in body]
        I = [float(r[iidx]) for r in body] if iidx is not None else None
    else:
        S = _as_number_list(args.get("S", []))
        v = _as_number_list(args.get("v", []))
        I = _as_number_list(args.get("I")) if args.get("I") not in (None, "", []) else None
    return S, v, I


def _biochem_fit(args):
    from engineering_sim.science import biochem_enzyme
    S, v, I = _fit_inputs(args)
    model = args.get("model", "mm")
    if model in ("compare", "auto", "all"):
        return {"models": biochem_enzyme.compare_models(S, v, I=I)}
    return biochem_enzyme.fit_michaelis_menten(S, v, model=model, I=I)


def _biochem_simulate(args):
    from engineering_sim.science import biochem_enzyme
    return biochem_enzyme.simulate(args.get("model", "mm"), args.get("params", {}),
                                   args.get("S0"), args.get("E0"), args.get("t_end"))


def _biochem_efficiency(args):
    from engineering_sim.science import biochem_enzyme
    return biochem_enzyme.kcat_efficiency(args.get("kcat"), args.get("Km"))


def _biochem_eyring(args):
    from engineering_sim.science import biochem_enzyme
    return biochem_enzyme.arrhenius_eyring(_as_number_list(args.get("k_values", [])),
                                           _as_number_list(args.get("T_values", [])))


def _biochem_buffers(args):
    from engineering_sim.science import biochem_buffers
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    buffers = []
    for bid in sorted(data["buffers"]):
        entry = data["buffers"][bid]
        buffers.append({
            "id": bid, "name": entry["name"], "formula": entry.get("formula", ""),
            "charge_start": entry.get("charge_start", 0),
            "pkas": [{"label": p["label"], "pka": p["pka"],
                      "dT": p.get("dT"), "acid_form": p.get("acid_form"), "base_form": p.get("base_form")}
                     for p in entry["pkas"]],
            "weigh_note": entry.get("weigh_note"),
            "form_names": entry.get("form_names"),
        })
    groups = [{"id": gid, "name": g["name"], "type": g["type"], "pka": g["pka"]}
              for gid, g in sorted(data["groups"].items())]
    return {"buffers": buffers, "groups": groups,
            "dg0prime": biochem_buffers.dG0prime_table()}


def _biochem_hh(args):
    from engineering_sim.science import biochem_buffers
    return biochem_buffers.henderson_hasselbalch(
        args.get("buffer"), pH=args.get("pH"), ratio=args.get("ratio"),
        T_C=args.get("T_C", 25.0), I=args.get("I"))


def _biochem_prepare_buffer(args):
    from engineering_sim.science import biochem_buffers
    return biochem_buffers.prepare_buffer(
        args.get("buffer"), args.get("pH"), args.get("conc_M"), args.get("volume_L"),
        form=args.get("form", "acid+conjugate"), T_C=args.get("T_C", 25.0),
        I=args.get("I"), titrant_M=args.get("titrant_M", 1.0))


def _biochem_titration(args):
    from engineering_sim.science import biochem_buffers
    return biochem_buffers.titration(
        args.get("buffer"), conc_M=args.get("conc_M", 0.1), I=args.get("I"),
        T_C=args.get("T_C", 25.0), ph_range=tuple(args.get("ph_range", (1.0, 13.0))),
        points=int(args.get("points", 121)))


def _biochem_ionic_strength(args):
    from engineering_sim.science import biochem_buffers
    return biochem_buffers.ionic_strength(args.get("solutes", []))


def _biochem_pi(args):
    from engineering_sim.science import biochem_buffers
    return biochem_buffers.pI(args.get("groups", []))


def _biochem_dg(args):
    from engineering_sim.science import biochem_buffers
    dg0 = args.get("dG0prime_kJ")
    if dg0 is None and args.get("dG0prime"):
        dg0 = biochem_buffers.dG0prime(args["dG0prime"])
    if dg0 is None:
        raise ValueError("Provide dG0prime_kJ (number) or dG0prime (table id).")
    return biochem_buffers.biochemical_dG(dg0, args.get("ratio", 1.0), T_K=args.get("T_K", 298.15))


OPS = {
    "biochem_nucleic": _biochem_nucleic,
    "biochem_primers": _biochem_primers,
    "biochem_restriction": _biochem_restriction,
    "biochem_align": _biochem_align,
    "biochem_mutate": _biochem_mutate,
    "biochem_fit": _biochem_fit,
    "biochem_simulate": _biochem_simulate,
    "biochem_efficiency": _biochem_efficiency,
    "biochem_eyring": _biochem_eyring,
    "biochem_buffers": _biochem_buffers,
    "biochem_hh": _biochem_hh,
    "biochem_prepare_buffer": _biochem_prepare_buffer,
    "biochem_titration": _biochem_titration,
    "biochem_ionic_strength": _biochem_ionic_strength,
    "biochem_pi": _biochem_pi,
    "biochem_dg": _biochem_dg,
}

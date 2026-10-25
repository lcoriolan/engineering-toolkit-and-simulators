# DESCRIPTION: Materials Science area (Task H): op handlers and timeouts for the
#              materials extension. Standard-library imports only at module level.
"""science toolkit ``materials`` area: op handlers and timeouts.

This module is imported by the stdlib server (``api.extension_timeouts``) to read
timeouts, so it must not import pymatgen/pycalphad/ASE at module level. Each handler
imports its science module lazily, inside the worker subprocess.
"""

# Op timeouts in seconds (max 900). Phase diagrams run a bounded T-X grid of pycalphad
# equilibria, so they get the longest budget; COD network fetches get a modest budget.
TIMEOUTS = {
    "materials_search": 60,
    "materials_fetch": 120,
    "materials_xrd": 300,
    "materials_supercell": 120,
    "materials_prototype": 120,
    "materials_systems": 60,
    "materials_phase_diagram": 900,
    "materials_equilibrium": 300,
    "materials_element": 60,
    "materials_radius_ratio": 60,
    "materials_mixtures": 60,
    "materials_mismatch": 60,
    "materials_hall_petch": 60,
    "materials_brus": 60,
}


def _materials_search(args):
    from engineering_sim.science import materials_crystal
    return materials_crystal.search(formula=args.get("formula"), elements=args.get("elements"),
                                    text=args.get("text"), max_results=args.get("max_results", 25))


def _materials_fetch(args):
    from engineering_sim.science import materials_crystal
    return materials_crystal.fetch(args.get("cod_id"), refresh=bool(args.get("refresh")),
                                   offline=bool(args.get("offline")))


def _materials_xrd(args):
    from engineering_sim.science import materials_crystal
    return materials_crystal.xrd(args.get("cod_id"), wavelength=args.get("wavelength", "CuKa"),
                                 two_theta=args.get("two_theta", (10, 90)))


def _materials_supercell(args):
    from engineering_sim.science import materials_crystal
    return materials_crystal.supercell(args.get("cod_id"), n=args.get("n", 2))


def _materials_prototype(args):
    from engineering_sim.science import materials_crystal
    return materials_crystal.from_formula_prototype(args.get("formula"),
                                                    prototype=args.get("prototype", "fcc"),
                                                    a=args.get("a", 4.0))


def _materials_systems(args):
    from engineering_sim.science import materials_phase
    return materials_phase.systems()


def _materials_phase_diagram(args):
    from engineering_sim.science import materials_phase
    return materials_phase.binary_phase_diagram(
        args.get("tdb"), args.get("comps"), T_range=args.get("T_range"),
        x_range=args.get("x_range"), P=args.get("P", 101325.0),
        nT=args.get("nT"), nx=args.get("nx"))


def _materials_equilibrium(args):
    from engineering_sim.science import materials_phase
    return materials_phase.equilibrium_point(args.get("tdb"), args.get("comps"),
                                             conditions=args.get("conditions"))


def _materials_element(args):
    from engineering_sim.science import materials_props
    return materials_props.element(args.get("symbol"))


def _materials_radius_ratio(args):
    from engineering_sim.science import materials_props
    return materials_props.ionic_radius_ratio(args.get("cation"), args.get("anion"),
                                              CN=args.get("CN", 6))


def _materials_mixtures(args):
    from engineering_sim.science import materials_props
    return materials_props.rule_of_mixtures(args.get("props"), args.get("fractions"),
                                            bound=args.get("bound", "voigt"))


def _materials_mismatch(args):
    from engineering_sim.science import materials_props
    return materials_props.thermal_expansion_mismatch(
        args.get("alpha1"), args.get("alpha2"), args.get("dT"), args.get("E"), args.get("nu"))


def _materials_hall_petch(args):
    from engineering_sim.science import materials_props
    return materials_props.hall_petch(args.get("sigma0"), args.get("k"), args.get("d"))


def _materials_brus(args):
    from engineering_sim.science import materials_props
    return materials_props.band_gap_brus(args.get("bulk_gap_eV"), args.get("radius_nm"),
                                         args.get("me"), args.get("mh"), args.get("eps_r"))


OPS = {
    "materials_search": _materials_search,
    "materials_fetch": _materials_fetch,
    "materials_xrd": _materials_xrd,
    "materials_supercell": _materials_supercell,
    "materials_prototype": _materials_prototype,
    "materials_systems": _materials_systems,
    "materials_phase_diagram": _materials_phase_diagram,
    "materials_equilibrium": _materials_equilibrium,
    "materials_element": _materials_element,
    "materials_radius_ratio": _materials_radius_ratio,
    "materials_mixtures": _materials_mixtures,
    "materials_mismatch": _materials_mismatch,
    "materials_hall_petch": _materials_hall_petch,
    "materials_brus": _materials_brus,
}

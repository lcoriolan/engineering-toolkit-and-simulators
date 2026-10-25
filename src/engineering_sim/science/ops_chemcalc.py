# DESCRIPTION: Task J: Chemical Calculator area op registry — nearest-valid-compound runs,
#              listing, retrieval, offline verify, portable export and the default recipe.
"""Chemical Calculator extension area.

Imported by ``engineering_sim/science/api.py`` under the standard-library studio interpreter to
read timeouts, and by ``engineering_sim/science/worker.py`` under configured science environment to run handlers.
Standard library only at module level; the heavy ``chemcalc`` module is imported
lazily inside each handler. Every op name starts with ``chemcalc_``.
"""

TIMEOUTS = {
    'chemcalc_run': 900,
    'chemcalc_runs': 60,
    'chemcalc_get': 60,
    'chemcalc_verify': 900,
    'chemcalc_export': 120,
    'chemcalc_default_recipe': 60,
}


def _run(args):
    from engineering_sim.science import chemcalc
    return chemcalc.run(args, replay=bool(args.get('replay')))


def _runs(args):
    from engineering_sim.science import chemcalc
    return chemcalc.list_runs()


def _get(args):
    from engineering_sim.science import chemcalc
    return chemcalc.get(args.get('recipe_id'))


def _verify(args):
    from engineering_sim.science import chemcalc
    return chemcalc.verify(args.get('recipe_id'))


def _export(args):
    from engineering_sim.science import chemcalc
    return chemcalc.export(args.get('recipe_id'))


def _default_recipe(args):
    from engineering_sim.science import chemcalc
    return chemcalc.default_recipe()


OPS = {
    'chemcalc_run': _run,
    'chemcalc_runs': _runs,
    'chemcalc_get': _get,
    'chemcalc_verify': _verify,
    'chemcalc_export': _export,
    'chemcalc_default_recipe': _default_recipe,
}

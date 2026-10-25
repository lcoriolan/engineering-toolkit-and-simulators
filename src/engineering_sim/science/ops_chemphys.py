# DESCRIPTION: Task G: Chemical Physics area op registry — quantum chemistry, orbitals,
#              spectra (IR / UV-vis / rotational) and statistical thermodynamics.
"""Chemical Physics extension area.

Imported by ``engineering_sim/science/api.py`` under the standard-library studio interpreter to
read timeouts, and by ``engineering_sim/science/worker.py`` under configured science environment to run handlers.
Standard library only at module level; heavy imports (PySCF/tblite/RDKit) live inside
each handler. Every op name starts with ``chemphys_``.
"""

TIMEOUTS = {
    'chemphys_scf': 900,
    'chemphys_orbital': 900,
    'chemphys_ir': 900,
    'chemphys_uvvis': 900,
    'chemphys_rotational': 300,
    'chemphys_statmech': 600,
    'chemphys_compare_nist': 600,
    'chemphys_boltzmann': 60,
    'chemphys_models': 60,
    'chemphys_constants': 60,
}


def _scf(args):
    from engineering_sim.science import chemphys_qm
    return chemphys_qm.scf(args.get('query', ''), method=args.get('method', 'RHF'),
                           basis=args.get('basis', 'sto-3g'), optimize=bool(args.get('optimize')),
                           preopt=args.get('preopt', 'none'))


def _orbital(args):
    from engineering_sim.science import chemphys_qm
    return chemphys_qm.orbital_cube(args.get('query', ''), orbital=args.get('orbital', 'HOMO'),
                                    method=args.get('method', 'RHF'), basis=args.get('basis', 'sto-3g'),
                                    grid=int(args.get('grid', 40) or 40))


def _ir(args):
    from engineering_sim.science import chemphys_spectra
    return chemphys_spectra.ir(args.get('query', ''), method=args.get('method', 'B3LYP'),
                               basis=args.get('basis', '6-31g*'), intensities=bool(args.get('intensities', True)),
                               fwhm=float(args.get('fwhm', 20.0)))


def _uvvis(args):
    from engineering_sim.science import chemphys_spectra
    return chemphys_spectra.uv_vis(args.get('query', ''), functional=args.get('functional', 'B3LYP'),
                                   basis=args.get('basis', '6-31g*'), nstates=int(args.get('nstates', 10)))


def _rotational(args):
    from engineering_sim.science import chemphys_spectra
    return chemphys_spectra.rotational(args.get('query', ''), preopt=args.get('preopt', 'none'))


def _statmech(args):
    from engineering_sim.science import chemphys_statmech
    return chemphys_statmech.partition_functions(args.get('query', ''), T_K=float(args.get('T_K', 298.15)),
                                                 P_bar=float(args.get('P_bar', 1.0)))


def _compare_nist(args):
    from engineering_sim.science import chemphys_statmech
    return chemphys_statmech.compare_nist(args.get('species_key', 'N2'), T_K=float(args.get('T_K', 298.15)))


def _boltzmann(args):
    from engineering_sim.science import chemphys_statmech
    levels = args.get('levels_eV', [])
    if not isinstance(levels, list) or not levels:
        raise ValueError('levels_eV: provide a non-empty list of {energy_eV, degeneracy} (or bare floats).')
    return chemphys_statmech.boltzmann(levels, T_K=float(args.get('T_K', 298.15)))


def _models(args):
    from engineering_sim.science import chemphys_statmech
    kind = args.get('model', 'particle_in_box')
    if kind == 'particle_in_box':
        return chemphys_statmech.particle_in_box(n_max=int(args.get('n_max', 5)),
                                                 L_nm=float(args.get('L_nm', 1.0)))
    if kind == 'harmonic_oscillator':
        return chemphys_statmech.harmonic_oscillator(n_max=int(args.get('n_max', 5)),
                                                     nu_cm1=float(args.get('nu_cm1', 1000.0)))
    if kind == 'hydrogen_atom':
        return chemphys_statmech.hydrogen_atom(n_max=int(args.get('n_max', 6)))
    raise ValueError('unknown model %r (particle_in_box | harmonic_oscillator | hydrogen_atom)' % kind)


def _constants(args):
    from engineering_sim.science import chemphys_statmech
    return chemphys_statmech.constants()


OPS = {
    'chemphys_scf': _scf,
    'chemphys_orbital': _orbital,
    'chemphys_ir': _ir,
    'chemphys_uvvis': _uvvis,
    'chemphys_rotational': _rotational,
    'chemphys_statmech': _statmech,
    'chemphys_compare_nist': _compare_nist,
    'chemphys_boltzmann': _boltzmann,
    'chemphys_models': _models,
    'chemphys_constants': _constants,
}

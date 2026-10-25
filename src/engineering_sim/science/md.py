# DESCRIPTION: Task E: OpenMM energy minimization and short MD for a fetched protein structure.
"""OpenMM minimization / short MD for a structure fetched by protein.fetch.

Input is a ``structure_ref`` cache key (never a user file path); the cached
structure is loaded, reduced to the selected polymer chains (waters and ligands
removed), fixed up with PDBFixer (missing heavy atoms + hydrogens at pH 7.0; no
missing loops are modelled — gaps are reported), and run under amber14 + GBn2
implicit solvent (or vacuum).

Energies are force-field estimates, not free energies; implicit solvent; no pKa
or loop modelling. Deterministic: MD uses seed 0xf00d.
"""
import math
import tempfile
from pathlib import Path

from engineering_sim.science import cite
from engineering_sim.science import protein


from engineering_sim.execution import operation as _simulation_operation

def _write_filtered_pdb(st, selected_chains, path):
    """Write a PDB holding only the selected polymer chains (first model).

    Returns (removed_waters, removed_ligands).
    """
    model = st[0]
    removed_waters = 0
    removed_ligands = []
    for chain in model:
        for res in chain:
            if res.is_water():
                removed_waters += 1
            elif res.het_flag == 'H':
                removed_ligands.append(res.name)

    out = gemmi_structure_clone_with_chains(st, selected_chains)
    out.write_pdb(str(path))
    return removed_waters, sorted(set(removed_ligands))


def gemmi_structure_clone_with_chains(st, selected_chains):
    """Clone the first model, keeping only polymer residues of the named chains."""
    import gemmi
    out = gemmi.Structure()
    out.name = st.name
    out.cell = st.cell
    model = st[0]
    om = gemmi.Model('1')
    for chain in model:
        if chain.name not in selected_chains:
            continue
        poly = gemmi.Chain(chain.name)
        for res in chain:
            if not res.is_water() and res.het_flag != 'H':
                poly.add_residue(res)
        if len(poly) > 0:
            om.add_chain(poly)
    out.add_model(om)
    return out


def _selected_chain_names(st, chain):
    """Chain names to keep: the requested one, or every chain with polymer residues."""
    model = st[0]
    names = [c.name for c in model if any(not r.is_water() and r.het_flag != 'H' for r in c)]
    if not names:
        return []
    if chain is not None:
        return [chain] if chain in names else [names[0]]
    return names


def _chain_gaps(st, selected_chains):
    """Report residue-number jumps (missing residues) we will NOT model."""
    gaps = []
    model = st[0]
    for chain in model:
        if chain.name not in selected_chains:
            continue
        prev = None
        for res in chain:
            if res.is_water() or res.het_flag == 'H':
                continue
            num = res.seqid.num
            if prev is not None and num > prev + 1:
                gaps.append({'chain': chain.name, 'from': prev, 'to': num})
            prev = num
    return gaps


def _input_ca_positions(st, selected_chains):
    """C-alpha positions from the input structure, keyed by (chain, resi)."""
    ref = {}
    model = st[0]
    for chain in model:
        if chain.name not in selected_chains:
            continue
        for res in chain:
            if res.is_water() or res.het_flag == 'H':
                continue
            for atom in res:
                if atom.name == 'CA':
                    ref[(chain.name, res.seqid.num)] = (atom.pos.x, atom.pos.y, atom.pos.z)
    return ref


def _simulation_ca_positions(sim, fixer):
    """C-alpha positions from the simulation, keyed by (chain, resi)."""
    positions = sim.context.getState(getPositions=True).getPositions()
    got = {}
    for atom, pos in zip(fixer.topology.atoms(), positions):
        if atom.name != 'CA':
            continue
        resi = int(''.join(ch for ch in atom.residue.id if ch.isdigit()) or 0)
        x = pos.x * 10.0  # OpenMM nm -> Å
        y = pos.y * 10.0
        z = pos.z * 10.0
        got[(atom.residue.chain.id, resi)] = (x, y, z)
    return got


def _ca_rmsd(ref, got):
    """RMSD over the C-alpha positions common to both dicts (Å)."""
    common = [k for k in ref if k in got]
    if not common:
        return None, 0
    total = 0.0
    for k in common:
        a, b = ref[k], got[k]
        total += (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    return math.sqrt(total / len(common)), len(common)


def _ca_trace(sim, fixer):
    """C-alpha trace (Å) from the current simulation state, for the viewer."""
    positions = sim.context.getState(getPositions=True).getPositions()
    trace = []
    for atom, pos in zip(fixer.topology.atoms(), positions):
        if atom.name != 'CA':
            continue
        resi = int(''.join(ch for ch in atom.residue.id if ch.isdigit()) or 0)
        trace.append({'chain': atom.residue.chain.id, 'resi': resi,
                      'resn': atom.residue.name,
                      'x': round(pos.x * 10.0, 3), 'y': round(pos.y * 10.0, 3),
                      'z': round(pos.z * 10.0, 3)})
    return trace


@_simulation_operation('science_bundle.molecular_dynamics', kind='transient')
def minimize(structure_ref, chain=None, steps=500, md_ps=0.0, temperature_K=300,
             implicit_solvent=True, max_atoms=15000):
    """Minimize (and optionally run short MD on) a fetched structure.

    Returns energies (kJ/mol), C-alpha RMSD, platform and wall time; the minimized
    C-alpha trace updates the viewer. Refuses atom counts above ``max_atoms`` or
    ``md_ps`` above 100 ps with a clear error message.
    """
    from openmm import app, unit
    from openmm.app import ForceField, Simulation
    from openmm import LangevinMiddleIntegrator, VerletIntegrator, Platform

    if md_ps > 100:
        raise ValueError('md_ps must be at most 100 ps.')
    st, source = protein.resolve_structure(structure_ref)
    if st is None:
        raise ValueError('structure not in cache: %s (run a protein fetch first)' % structure_ref)

    selected = _selected_chain_names(st, chain)
    if not selected:
        raise ValueError('the structure has no polymer chains to minimize')

    # Build a PDB with only the selected polymer chains; count what was removed.
    tmp = tempfile.TemporaryDirectory()
    pdb_path = Path(tmp.name) / 'input.pdb'
    removed_waters, removed_ligands = _write_filtered_pdb(st, selected, pdb_path)
    gaps = _chain_gaps(st, selected)

    from pdbfixer import PDBFixer
    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()   # detected for gap reporting only; never added
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    n_atoms = fixer.topology.getNumAtoms()
    if n_atoms > max_atoms:
        tmp.cleanup()
        raise ValueError('%d atoms exceed the max_atoms limit (%d).' % (n_atoms, max_atoms))

    ff_name = 'amber14-all + GBn2' if implicit_solvent else 'amber14-all (vacuum)'
    forcefield = ForceField('amber14-all.xml', *(('implicit/gbn2.xml',) if implicit_solvent else ()))
    system = forcefield.createSystem(fixer.topology, nonbondedMethod=app.NoCutoff,
                                     constraints=app.HBonds, removeCMMotion=True)

    # Platform order OpenCL -> CPU -> Reference; report which one actually ran.
    platform = None
    for name in ('OpenCL', 'CPU', 'Reference'):
        try:
            platform = Platform.getPlatformByName(name)
            break
        except Exception:  # noqa: BLE001 - a platform may be missing; try the next
            continue

    integrator = (LangevinMiddleIntegrator(temperature_K * unit.kelvin,
                                           1.0 / unit.picosecond, 0.002 * unit.picoseconds)
                  if md_ps > 0 else VerletIntegrator(0.001 * unit.picoseconds))
    if md_ps > 0:
        integrator.setRandomNumberSeed(0xf00d)

    simulation = Simulation(fixer.topology, system, integrator, platform)
    simulation.context.setPositions(fixer.positions)

    energy_before = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    import time
    wall = time.time()
    simulation.minimizeEnergy(maxIterations=int(steps))
    energy_after = simulation.context.getState(getEnergy=True).getPotentialEnergy()

    ref_ca = _input_ca_positions(st, selected)
    got_ca = _simulation_ca_positions(simulation, fixer)
    rmsd, n_matched = _ca_rmsd(ref_ca, got_ca)

    method = 'OpenMM 8.6.1 %s' % ff_name
    result = {
        'ok': True,
        'ref': structure_ref,
        'chains': selected,
        'removed': {'waters': removed_waters, 'ligands': removed_ligands},
        'gaps': gaps,
        'atoms': n_atoms,
        'platform': platform.getName() if platform else 'Reference',
        'forcefield': ff_name,
        'energy_before_kJmol': cite.value(round(float(energy_before.value_in_unit(unit.kilojoule_per_mole)), 2),
                                          'kJ/mol', 'estimated', method=method),
        'energy_after_kJmol': cite.value(round(float(energy_after.value_in_unit(unit.kilojoule_per_mole)), 2),
                                         'kJ/mol', 'estimated', method=method),
        'rmsd_Ca_A': cite.value(round(rmsd, 3), 'Å', 'derived',
                                method='Cα RMSD, minimized vs input (%d atoms)' % n_matched),
        'md': None,
        'trace': _ca_trace(simulation, fixer),
        'notes': [
            'Force-field energies are not free energies; implicit solvent; no pKa or loop modelling.',
            'Missing residues were detected but not modelled; gaps are reported above.',
            'Minimized with %d steps under %s.' % (int(steps), ff_name),
        ],
    }

    if md_ps > 0:
        md_steps = int(round(md_ps * 1000 / 2))  # 2 fs per step
        sample_every = int(1000 / 2)             # one energy sample per ps
        energies = []
        for i in range(md_steps):
            simulation.step(1)
            if (i + 1) % sample_every == 0:
                e = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                energies.append(cite.value(
                    round(float(e.value_in_unit(unit.kilojoule_per_mole)), 2),
                    'kJ/mol', 'estimated', method=method))
        got_ca = _simulation_ca_positions(simulation, fixer)
        final_rmsd, n_matched = _ca_rmsd(ref_ca, got_ca)
        result['md'] = {
            'ps': md_ps,
            'steps': md_steps,
            'temperature_K': temperature_K,
            'energies_kJmol': energies,
            'final_rmsd_Ca_A': cite.value(round(final_rmsd, 3), 'Å', 'derived',
                                          method='Cα RMSD, MD final vs input (%d atoms)' % n_matched),
            'wall_time_s': round(time.time() - wall, 2),
        }
        result['trace'] = _ca_trace(simulation, fixer)

    tmp.cleanup()
    return result

# DESCRIPTION: Task E: AutoDock Vina 1.2.7 docking of a ligand into a fetched protein structure.
"""AutoDock Vina 1.2.7 docking for a structure fetched by protein.fetch.

Receptor: cached structure, selected polymer chains, waters/ligands removed,
hydrogens added (PDBFixer pH 7.0), then converted to PDBQT. Meeko 0.8.0 ships no
receptor-preparation pipeline, so a minimal documented PDBQT writer is used here:
AutoDock4 atom types from an element/valence heuristic and Gasteiger charges via
RDKit (state this clearly — it is a simplification of the ADFR preparation).

Ligand: a SMILES string, a PubChem name (resolved to SMILES via records.lookup),
or a hetero-group code present in the structure (e.g. MK1 in 1HSG). Hetero-code
ligands are built from the crystal coordinates, so the docked pose's Vina RMSD
lower-bound is a redocking RMSD against the crystal ligand.

Docking scores rank poses; they are not measured binding free energies.
Deterministic: single-threaded Vina (cpu=1) with a fixed seed.
"""
import tempfile
from pathlib import Path

from engineering_sim.science import cite
from engineering_sim.science import protein


# --- receptor preparation ---------------------------------------------------
def _ad4_type(atom):
    """AutoDock4 atom type for one RDKit atom (documented heuristic)."""
    sym = atom.GetSymbol()
    hs = atom.GetTotalNumHs()
    if sym == 'H':
        return 'H'
    if sym == 'C':
        return 'C'
    if sym == 'O':
        return 'OA'
    if sym == 'N':
        return 'N' if hs > 0 else 'NA'
    if sym == 'S':
        return 'S' if hs > 0 else 'SA'
    # Metals and halogens keep their element symbol (Vina accepts them).
    return sym


def _build_receptor_rdkit(fixer):
    """Build an RDKit molecule (all single bonds) from a PDBFixer topology.

    Returns the sanitized molecule with Gasteiger charges computed. Over-valent
    nitrogens (protonated N-termini) are given a +1 formal charge so RDKit can
    sanitize; this does not change Gasteiger's electronegativity-based charges.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    atoms = list(fixer.topology.atoms())
    mol = Chem.RWMol()
    key = {}
    for i, a in enumerate(atoms):
        key[id(a)] = i
        ra = Chem.Atom(a.element.atomic_number)
        ra.SetNoImplicit(True)
        mol.AddAtom(ra)
    for a, b in fixer.topology.bonds():
        mol.AddBond(key[id(a)], key[id(b)], Chem.BondType.SINGLE)
    m = mol.GetMol()
    m.UpdatePropertyCache(strict=False)
    for atom in m.GetAtoms():
        if atom.GetAtomicNum() == 7:
            degree = sum(1 for _ in atom.GetBonds()) + atom.GetNumImplicitHs()
            if degree > 3:
                atom.SetFormalCharge(degree - 3)
                atom.SetNoImplicit(True)
    Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL)
    AllChem.ComputeGasteigerCharges(m)
    return m


def _write_receptor_pdbqt(fixer, path):
    """Write the receptor PDBQT (minimal writer: AD4 types + Gasteiger charges).

    Returns (n_atoms, n_hydrogens). Line format matches Meeko's PDBQT writer so
    the Vina parser reads it exactly as it reads Meeko output.
    """
    from openmm import unit

    mol = _build_receptor_rdkit(fixer)
    atoms = list(fixer.topology.atoms())
    lines = []
    n_h = 0
    for i, a in enumerate(atoms):
        ra = mol.GetAtomWithIdx(i)
        sym = a.element.symbol
        charge = float(ra.GetDoubleProp('_GasteigerCharge'))
        atype = _ad4_type(ra)
        if sym == 'H':
            n_h += 1
            # Polar hydrogens (bonded to N/O/S) are HD; carbon hydrogens stay H.
            nbrs = [x for x in ra.GetNeighbors()]
            if nbrs and nbrs[0].GetSymbol() in ('N', 'O', 'S'):
                atype = 'HD'
        p = fixer.positions[i].value_in_unit(unit.angstrom)
        resi = int(''.join(ch for ch in a.residue.id if ch.isdigit()) or 0)
        lines.append('{:6s}{:5d} {:<4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}  {:+8.3f} {:<2s}'.format(
            'ATOM', i + 1, a.name, ' ', a.residue.name, a.residue.chain.id, resi, '',
            p.x, p.y, p.z, 1.0, 0.0, charge, atype))
    path.write_text('\n'.join(lines) + '\n')
    return len(lines), n_h


def _prepare_receptor(st, selected_chains, tmp):
    """Prepare the receptor PDBQT file; returns (pdbqt_path, n_atoms, n_hydrogens)."""
    from pdbfixer import PDBFixer

    # Filtered PDB (polymer residues of the selected chains only).
    pdb_path = Path(tmp) / 'receptor.pdb'
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
    out.write_pdb(str(pdb_path))

    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)

    rec_path = Path(tmp) / 'receptor.pdbqt'
    n_atoms, n_h = _write_receptor_pdbqt(fixer, rec_path)
    return str(rec_path), n_atoms, n_h


# --- ligand preparation -----------------------------------------------------
def _hetero_codes(st):
    """Set of non-water hetero-group codes present in the structure."""
    codes = set()
    model = st[0]
    for chain in model:
        for res in chain:
            if not res.is_water() and res.het_flag == 'H':
                codes.add(res.name)
    return codes


def _ligand_from_hetero(st, code, ref):
    """Prepare a ligand from a hetero group in the structure (crystal coordinates).

    Returns (pdbqt_string, is_crystal=True). Raises ValueError if the code is absent.
    """
    from rdkit import Chem
    from rdkit.Chem import rdGeometry

    # Crystal heavy atoms.
    atoms = []
    for chain in st[0]:
        for res in chain:
            if res.name == code:
                for atom in res:
                    if atom.element.name != 'H':
                        atoms.append((atom.name, atom.element.name,
                                      atom.pos.x, atom.pos.y, atom.pos.z))
    if not atoms:
        raise ValueError('hetero group %s not found in the structure' % code)

    # Bonds from the CCD definition carried inside the cached mmCIF.
    bonds = []
    path = protein.structure_file_path(ref)
    if path is not None and path.suffix == '.cif':
        import gemmi
        block = gemmi.cif.read(str(path))[0]
        for a1, a2, order, arom in protein._chemcomp_bonds(block, code):
            bonds.append((a1, a2, order, arom))

    name2idx = {n: i for i, (n, _el, _x, _y, _z) in enumerate(atoms)}
    mol = Chem.RWMol()
    for (_n, el, _x, _y, _z) in atoms:
        ra = Chem.Atom(el)
        ra.SetNoImplicit(True)
        mol.AddAtom(ra)
    for a1, a2, order, arom in bonds:
        if a1 in name2idx and a2 in name2idx:
            b = Chem.BondType.AROMATIC if arom == 'Y' else (
                Chem.BondType.DOUBLE if order == 'doub' else (
                    Chem.BondType.TRIPLE if order == 'trip' else Chem.BondType.SINGLE))
            mol.AddBond(name2idx[a1], name2idx[a2], b)
    mol = mol.GetMol()
    conf = Chem.Conformer(len(atoms))
    for i, (_n, _el, x, y, z) in enumerate(atoms):
        conf.SetAtomPosition(i, rdGeometry.Point3D(x, y, z))
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)
    return _meeko_pdbqt(mol, is_crystal=True)


def _meeko_pdbqt(mol, is_crystal):
    """Add hydrogens and run Meeko preparation; return (pdbqt_string, is_crystal)."""
    from rdkit import Chem
    mol = Chem.AddHs(mol)
    from meeko import MoleculePreparation
    prep = MoleculePreparation()
    setups = prep.prepare(mol)
    from meeko.writer import PDBQTWriterLegacy
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise ValueError('Meeko ligand preparation failed: %s' % (err or 'unknown error'))
    return pdbqt, is_crystal


def _ligand_from_smiles(smiles):
    """Prepare a ligand from SMILES (RDKit 3D embed -> Meeko)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('invalid SMILES: %s' % smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    AllChem.UFFOptimizeMolecule(mol)
    return _meeko_pdbqt(mol, is_crystal=False)


def _resolve_ligand(ligand, st, ref):
    """Interpret the ligand argument: hetero code -> crystal, else SMILES, else PubChem name."""
    codes = _hetero_codes(st)
    upper = (ligand or '').strip().upper()
    if upper in codes:
        return _ligand_from_hetero(st, upper, ref)
    from rdkit import Chem
    if Chem.MolFromSmiles(ligand or '') is not None:
        return _ligand_from_smiles(ligand)
    # Fall back to a PubChem name lookup for a SMILES.
    from engineering_sim.science.records import lookup
    rec = lookup(ligand)
    if not rec.get('found') or not rec.get('smiles'):
        raise ValueError('could not resolve ligand "%s" as a hetero code, SMILES, or PubChem name'
                         % ligand)
    return _ligand_from_smiles(rec['smiles'])


# --- box handling -----------------------------------------------------------
def _ligand_box(st, code):
    """Box from a hetero group: centroid center, extent + 8 Å (min 16 Å per side)."""
    pts = []
    for chain in st[0]:
        for res in chain:
            if res.name == code:
                for atom in res:
                    if atom.element.name != 'H':
                        pts.append([atom.pos.x, atom.pos.y, atom.pos.z])
    if not pts:
        raise ValueError('hetero group %s not found for box_from_ligand' % code)
    import numpy as np
    arr = np.array(pts, dtype=float)
    center = arr.mean(axis=0)
    size = arr.max(axis=0) - arr.min(axis=0) + 8.0
    size = np.clip(size, 16.0, None)
    if (size > 40.0).any():
        raise ValueError('box exceeds 40 Å per side (%.1f, %.1f, %.1f)' % tuple(size))
    return [round(float(v), 2) for v in center], [round(float(v), 2) for v in size]


# AutoDock atom types (PDBQT columns 78-79) -> element. The atom-name field (columns 13-16) is not an
# element: names like "HN1" or "CA" would be misread, so elements come from the type column.
_AD_ELEMENT = {'A': 'C', 'C': 'C', 'N': 'N', 'NA': 'N', 'NS': 'N', 'OA': 'O', 'OS': 'O', 'O': 'O', 'SA': 'S', 'S': 'S',
               'H': 'H', 'HD': 'H', 'HS': 'H', 'P': 'P', 'F': 'F', 'Cl': 'Cl', 'CL': 'Cl', 'Br': 'Br', 'BR': 'Br', 'I': 'I',
               'Mg': 'Mg', 'MG': 'Mg', 'Zn': 'Zn', 'ZN': 'Zn', 'Fe': 'Fe', 'FE': 'Fe', 'Ca': 'Ca', 'CA': 'Ca', 'Mn': 'Mn', 'MN': 'Mn'}


def _crystal_heavy_atoms(st, code):
    """Heavy atoms [(element, x, y, z)] of the first instance of a hetero group (reference pose for redocking)."""
    for chain in st[0]:
        for res in chain:
            if res.name == code:
                return [(a.element.name.capitalize(), a.pos.x, a.pos.y, a.pos.z) for a in res if a.element.name != 'H']
    return []


def _rmsd_to_reference(pose_atoms, ref_atoms):
    """Heavy-atom RMSD (Å) of a docked pose to the crystal ligand, in the receptor frame (no superposition).

    Symmetry handling: within each element the pose-to-reference atom correspondence is the minimum-cost
    assignment on squared distance (scipy linear_sum_assignment), so equivalent atoms (e.g. a flipped phenyl)
    are not penalised. That makes the value a lower bound on the graph-isomorphism RMSD; it is None when the
    heavy-atom element counts differ (different protonation/tautomer handling or a partial crystal ligand).
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if not pose_atoms or not ref_atoms:
        return None
    by_el = {}
    for el, *xyz in ref_atoms:
        by_el.setdefault(el, [[], []])[1].append(xyz)
    for a in pose_atoms:
        if a['el'] not in by_el:
            return None
        by_el[a['el']][0].append([a['x'], a['y'], a['z']])
    total, n = 0.0, 0
    for pose_xyz, ref_xyz in by_el.values():
        if len(pose_xyz) != len(ref_xyz):
            return None
        P, R = np.array(pose_xyz), np.array(ref_xyz)
        cost = ((P[:, None, :] - R[None, :, :]) ** 2).sum(axis=2)
        rows, cols = linear_sum_assignment(cost)
        total += cost[rows, cols].sum(); n += len(rows)
    return float(np.sqrt(total / n))


def _parse_poses(poses_str, heavy_only=True):
    """Parse Vina's poses() PDBQT into per-pose {affinity, rmsd_lb, rmsd_ub, atoms}.

    rmsd_lb/rmsd_ub are Vina's RMSDs relative to the best pose of this run (pose 1 is always 0), not to a crystal.
    """
    poses = []
    current = None
    for line in poses_str.splitlines():
        if line.startswith('MODEL'):
            current = {'affinity': None, 'rmsd_lb': None, 'rmsd_ub': None, 'atoms': []}
            poses.append(current)
        elif line.startswith(('ATOM', 'HETATM')) and current is not None:
            ad_type = line[77:79].strip()
            element = _AD_ELEMENT.get(ad_type, ad_type.capitalize())
            if heavy_only and element == 'H':
                continue
            current['atoms'].append({
                'el': element,
                'x': round(float(line[30:38]), 3),
                'y': round(float(line[38:46]), 3),
                'z': round(float(line[46:54]), 3),
            })
        elif 'VINA RESULT' in line and current is not None:
            parts = line.split()
            try:
                current['affinity'] = float(parts[3])
                current['rmsd_lb'] = float(parts[4])
                current['rmsd_ub'] = float(parts[5])
            except (IndexError, ValueError):
                pass
    return [p for p in poses if p.get('affinity') is not None]


# --- public API -------------------------------------------------------------
def dock(structure_ref, ligand, center=None, size=(20, 20, 20), chain=None,
         exhaustiveness=8, n_poses=5, seed=0xf00d, box_from_ligand=None):
    """Dock a ligand into a fetched structure with AutoDock Vina.

    Returns poses (affinity kcal/mol, RMSD lower/upper bound, heavy-atom coords for
    the viewer), the box used, timing, and the notes on interpretation.
    """
    st, source = protein.resolve_structure(structure_ref)
    if st is None:
        raise ValueError('structure not in cache: %s (run a protein fetch first)' % structure_ref)

    # Selected chains (default: every polymer chain).
    model = st[0]
    all_chains = [c.name for c in model if any(not r.is_water() and r.het_flag != 'H' for r in c)]
    selected = [chain] if chain is not None else all_chains
    if chain is not None and chain not in all_chains:
        raise ValueError('chain %s not present in the structure' % chain)
    if not selected:
        raise ValueError('the structure has no polymer chains')

    tmp = tempfile.TemporaryDirectory()
    try:
        rec_path, n_atoms, n_h = _prepare_receptor(st, selected, tmp.name)

        ligand_pdbqt, is_crystal = _resolve_ligand(ligand, st, structure_ref)

        # Box: explicit center+size, or derived from a hetero ligand's extent.
        if box_from_ligand:
            box_center, box_size = _ligand_box(st, box_from_ligand.upper())
        else:
            box_center = [float(v) for v in center] if center else None
            box_size = [float(v) for v in size]
        if box_center is None:
            raise ValueError('provide a center (x, y, z) or a box_from_ligand code')
        if len(box_size) != 3 or any(v <= 0 for v in box_size):
            raise ValueError('size must be three positive numbers')
        if any(v > 40.0 for v in box_size):
            raise ValueError('box exceeds 40 Å per side')

        from vina import Vina
        v = Vina(sf_name='vina', cpu=1, seed=int(seed), verbosity=0)
        v.set_receptor(rec_path)
        v.set_ligand_from_string(ligand_pdbqt)
        import time
        wall = time.time()
        v.compute_vina_maps(center=box_center, box_size=box_size)
        v.dock(exhaustiveness=int(exhaustiveness), n_poses=int(n_poses))
        wall_s = round(time.time() - wall, 2)
        poses = _parse_poses(v.poses(n_poses=int(n_poses), energy_range=10.0))

        method = 'AutoDock Vina 1.2.7 scoring function'
        # Redocking reference: the native hetero group's crystal heavy atoms, when the ligand came from it.
        crystal_ref = _crystal_heavy_atoms(st, (ligand or '').strip().upper()) if is_crystal else None
        result = {
            'ok': True,
            'ref': structure_ref,
            'chains': selected,
            'receptor_atoms': n_atoms,
            'receptor_hydrogens': n_h,
            'ligand': ligand,
            'ligand_from_crystal': is_crystal,
            'box': {'center': box_center, 'size': box_size},
            'poses': [{
                'affinity': cite.value(round(p['affinity'], 3), 'kcal/mol', 'estimated', method=method),
                'rmsd_lb': cite.value(round(p['rmsd_lb'], 3), 'Å', 'derived',
                                      method='Vina RMSD lower bound relative to the best pose of this run'),
                'rmsd_ub': cite.value(round(p['rmsd_ub'], 3), 'Å', 'derived',
                                      method='Vina RMSD upper bound relative to the best pose of this run'),
                'rmsd_to_crystal': (None if crystal_ref is None or (r := _rmsd_to_reference(p['atoms'], crystal_ref)) is None
                                    else cite.value(round(r, 3), 'Å', 'derived',
                                                    method='Heavy-atom RMSD to the crystal ligand in the receptor frame, element-wise optimal assignment (lower bound on symmetry-corrected RMSD)')),
                'atoms': p['atoms'],
            } for p in poses],
            'wall_time_s': wall_s,
            'notes': [
                'Docking scores rank poses; they are not measured binding free energies.',
                'Receptor prepared with a minimal PDBQT writer: AutoDock4 atom types + RDKit '
                'Gasteiger charges (Meeko 0.8.0 has no receptor-preparation pipeline).',
                'Deterministic single-threaded Vina (cpu=1, seed=0x%x).' % int(seed),
            ],
        }
        if is_crystal:
            result['notes'].append(
                'Redocking of the native hetero group: rmsd_to_crystal compares each pose with the crystal '
                'ligand (success is commonly taken as ≤ 2 Å). rmsd_lb/rmsd_ub are relative to the best pose of this run.')
        return result
    finally:
        tmp.cleanup()

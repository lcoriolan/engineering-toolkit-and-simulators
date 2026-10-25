# DESCRIPTION: Task G: PySCF 2.14.0 quantum chemistry for the Chemical Physics area —
#              SCF energies/orbitals/charges/dipole, geometry optimisation, molecular-orbital
#              isosurfaces (own marching-tetrahedra), and bond-dissociation energies.
"""PySCF quantum chemistry for the Chemical Physics area.

Input geometries come from ``engineering_sim.science.chem.molecule`` (RDKit ETKDG + MMFF)
or, when ``preopt='xtb'``, from a GFN2-xTB relaxation via ``engineering_sim.science.xtb``.
Every reported number is a ``cite`` value record: energies are ``estimated`` with
method ``"PySCF 2.14.0 <method>/<basis>"``; unit conversions are ``derived``.

Units and conversions (CODATA 2022 recommended values, fetched from
physics.nist.gov/cuu/Constants/ during this build — see chemphys_statmech.constants):
  1 hartree (E_h)   = 2625.499639… kJ/mol = 27.211386… eV
  1 bohr (a0)       = 0.52917721… Angstrom
  1 e·bohr          = 2.54174647… Debye

Size guard: molecules with more than 30 heavy atoms or more than 400 basis
functions are refused before the SCF is started (with a clear message), so the
op bounds its own runtime instead of timing out.
"""
from __future__ import annotations

import time

import numpy as np

from . import cite

# CODATA 2022 recommended values (exact-defined where noted), fetched from
# https://physics.nist.gov/cuu/Constants/ during this build.
BOHR = 0.529177210544           # a0, Angstrom
EV_PER_HARTREE = 27.211386245981167
KJMOL_PER_HARTREE = 2625.499639479163
DEBYE_PER_EA0 = 2.541746473      # Debye per e·bohr (dipole moment unit conversion)

# User-facing basis names -> PySCF basis keys.
_BASIS_MAP = {'sto-3g': 'sto-3g', '6-31g*': '6-31g*', 'def2-svp': 'def2svp',
              'cc-pvdz': 'ccpvdz'}

# PySCF exchange-correlation names for the DFT methods.
_XC = {'B3LYP': 'b3lyp', 'PBE0': 'pbe0', 'wB97X-D': 'wb97x'}

_MAX_HEAVY = 30     # heavy atoms
_MAX_NBASIS = 400   # contracted basis functions

_ELEMENT_MASS = None  # lazy atomic-mass table (amu) shared with the spectra module


def atomic_masses():
    """Standard atomic masses (amu) for the elements science toolkit can parse.

    Most-abundant-isotope masses (rounded), from the standard tables of atomic
    weights (CIAAW 2021). Not fetched during this build, so downstream rotational
    constants cite them as ``unverified`` approximations rather than ``cited``.
    """
    global _ELEMENT_MASS
    if _ELEMENT_MASS is None:
        _ELEMENT_MASS = {
            'H': 1.007825, 'He': 4.002603, 'Li': 7.016004, 'Be': 9.012182,
            'B': 11.009305, 'C': 12.000000, 'N': 14.003074, 'O': 15.994915,
            'F': 18.998403, 'Ne': 19.992440, 'Na': 22.989770, 'Mg': 23.985042,
            'Al': 26.981538, 'Si': 27.976927, 'P': 30.973762, 'S': 31.972071,
            'Cl': 34.968853, 'Ar': 39.962383, 'K': 38.963707, 'Ca': 39.962591,
            'Fe': 55.934936, 'Cu': 62.929598, 'Zn': 63.929142, 'Br': 78.918338,
            'I': 126.904468,
        }
    return _ELEMENT_MASS


def _heavy_count(symbols):
    """Number of atoms with Z > 1."""
    return sum(1 for s in symbols if s != 'H')


def _estimate_n_basis(symbols, basis):
    """Quick upper estimate of the contracted basis size, before building the Mole.

    Rough per-atom counts for the four supported bases (enough to refuse oversize
    inputs early; the exact ``mol.nao`` is checked after the Mole is built).
    """
    per = {'sto-3g': {'H': 1, 'C': 5, 'N': 5, 'O': 5, 'F': 5},  # 1s..2p
           '6-31g*': {'H': 2, 'C': 15, 'N': 15, 'O': 15, 'F': 15},
           'def2-svp': {'H': 5, 'C': 14, 'N': 14, 'O': 14, 'F': 14},
           'cc-pvdz': {'H': 5, 'C': 14, 'N': 14, 'O': 14, 'F': 14}}[basis]
    return sum(per.get(s, 14 if basis != 'sto-3g' else 5) for s in symbols)


def resolve_geometry(query, preopt='none'):
    """Resolve a query into atoms/charge/spin for a PySCF calculation.

    ``preopt='xtb'`` relaxes the geometry with GFN2-xTB (tblite) first, which is
    more robust for radicals and unusual valences; otherwise the RDKit ETKDG+MMFF
    geometry from ``chem.molecule`` is used unchanged. Returns a dict with
    ``atoms`` ([{el,x,y,z}], Angstrom), ``charge``, ``spin_mult`` (2S+1),
    ``smiles``, ``formula``, and a ``notes`` list.
    """
    from engineering_sim.science import chem

    mol = chem.molecule(query)
    if mol.get('error') or not mol.get('smiles'):
        raise ValueError('could not resolve query %r: %s' % (query, mol.get('error') or 'no molecule'))
    if preopt == 'xtb':
        from engineering_sim.science import xtb
        est = xtb.estimate(mol['smiles'])
        atoms = [{'el': _sym(int(z)), 'x': c[0], 'y': c[1], 'z': c[2]}
                 for z, c in zip(_numbers_of(mol['smiles']), est['coordinates_angstrom'])]
        return {'atoms': atoms, 'charge': est['charge'], 'spin_mult': est['multiplicity'],
                'smiles': mol['smiles'], 'formula': mol['formula'],
                'geometry': 'GFN2-xTB (tblite 0.7.0) optimised',
                'notes': [est.get('note', '')] if est.get('note') else []}
    atoms = [{'el': a['el'], 'x': a['x'], 'y': a['y'], 'z': a['z']} for a in mol['atoms']]
    return {'atoms': atoms, 'charge': mol['charge'], 'spin_mult': mol['spin_multiplicity'],
            'smiles': mol['smiles'], 'formula': mol['formula'],
            'geometry': 'RDKit ETKDG + MMFF',
            'notes': list(mol.get('notes') or [])}


def _numbers_of(smiles):
    """Atomic numbers of a SMILES (with hydrogens added), matching xtb.estimate's order."""
    from rdkit import Chem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return [a.GetAtomicNum() for a in m.GetAtoms()]


def _sym(z):
    """Element symbol from atomic number."""
    from rdkit import Chem
    return Chem.GetPeriodicTable().GetElementSymbol(int(z))


def atoms_to_atom_string(atoms):
    """PySCF atom string ('El x y z; ...') from [{el,x,y,z}] (Angstrom)."""
    return '; '.join('%s %.8f %.8f %.8f' % (a['el'], float(a['x']), float(a['y']), float(a['z']))
                     for a in atoms)


def _make_mf(mol, method, spin):
    """Build the mean-field object for a method/spin; RHF auto-upgrades to UHF for open shells."""
    from pyscf import dft, scf

    if method in ('RHF', 'UHF'):
        want = method
        if spin > 0 and method == 'RHF':
            want = 'UHF'  # RHF cannot describe an open shell
        return scf.UHF(mol) if want == 'UHF' else scf.RHF(mol)
    xc = _XC[method]
    if spin == 0:
        return dft.RKS(mol, xc=xc)
    return dft.UKS(mol, xc=xc)


def _opt_method_label(method, spin):
    """The method actually used (after RHF->UHF upgrade for open shells)."""
    if method in ('RHF', 'UHF'):
        return 'UHF' if (spin > 0 and method == 'RHF') else method
    return method


def _optimize_geometry(mol, method, spin, max_steps):
    """Optimise the geometry with scipy BFGS on PySCF analytic gradients.

    pyscf.geomopt requires the optional `geometric`/`berny` packages, which are not
    installed in this venv; a deterministic scipy BFGS over Cartesian coordinates
    with analytic gradients is used instead (reported honestly in the result).
    """
    from scipy.optimize import minimize

    coords0 = np.asarray([list(row) for row in mol.atom_coords()], dtype=float)

    def energy_and_grad(x):
        coords = np.asarray(x).reshape(-1, 3)
        m = mol.copy()
        m.set_geom_(coords, unit='Bohr')
        mf = _make_mf(m, method, spin)
        e = mf.kernel()
        grad = mf.nuc_grad_method().kernel()
        return float(e), np.asarray(grad, dtype=float).ravel()

    res = minimize(energy_and_grad, coords0.ravel(), method='BFGS', jac=True,
                   options={'maxiter': max_steps, 'gtol': 3e-4})
    coords = np.asarray(res.x).reshape(-1, 3)
    mol.set_geom_(coords, unit='Bohr')
    mf = _make_mf(mol, method, spin)
    mf.kernel()
    return mol, mf, 'scipy-bfgs (pyscf geomopt unavailable: no geometric/berny)', bool(res.success), int(res.nit)


def _orbitals(mf):
    """Normalised list of (energy_eV, occupation) for every MO, energy-ordered.

    UHF/UKS return alpha/beta arrays; they are concatenated into one list. Sorted
    by energy so the HOMO/LUMO assignment and the page diagram are deterministic.
    """
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    if isinstance(mo_energy, (list, tuple)) and len(mo_energy) == 2 and not np.isscalar(mo_energy[0]):
        energies = np.concatenate([np.asarray(mo_energy[0]), np.asarray(mo_energy[1])])
        occs = np.concatenate([np.asarray(mo_occ[0]), np.asarray(mo_occ[1])])
    else:
        energies = np.asarray(mo_energy)
        occs = np.asarray(mo_occ)
    pairs = sorted(zip(energies, occs), key=lambda t: float(t[0]))
    return [(float(e) * EV_PER_HARTREE, float(o)) for e, o in pairs]


def _homo_lumo(orbitals):
    """(homo, lumo) indices into the sorted orbital list, or (None, None)."""
    occ_idx = [i for i, (e, o) in enumerate(orbitals) if o > 0.5]
    vir_idx = [i for i, (e, o) in enumerate(orbitals) if o < 0.5]
    if not occ_idx or not vir_idx:
        return None, None
    return occ_idx[-1], vir_idx[0]


def _lowdin_charges(mol, dm):
    """Lowdin charges: q_A = Z_A - sum_i (S^{1/2} D S^{1/2})_{ii} over A's AOs.

    ``aoslice_by_atom()`` returns [p0, p1, c0, c1] per atom where ``c0:c1`` are the
    contracted-AO indices of that atom (``p0:p1`` are shell indices, not AOs).
    """
    S = mol.intor_symmetric('int1e_ovlp')
    w, v = np.linalg.eigh(S)
    S_half = (v * np.sqrt(w)) @ v.T
    PSP = S_half @ dm @ S_half
    ao = mol.aoslice_by_atom()
    charges = []
    for ia, row in enumerate(ao):
        c0, c1 = int(row[2]), int(row[3])
        charges.append(mol.atom_charge(ia) - float(np.trace(PSP[c0:c1, c0:c1])))
    return np.asarray(charges)


def _density(mf):
    """Total (alpha+beta) density matrix."""
    dm = mf.make_rdm1()
    if isinstance(dm, (list, tuple)):
        return dm[0] + dm[1]
    return dm


def _charges(mf, mol):
    """(mulliken, lowdin) per-atom charge arrays; either may be None if unavailable."""
    mulliken = None
    lowdin = None
    dm = _density(mf)
    try:
        pop = mf.mulliken_pop()
        # RHF/RKS -> (pop, charges); UHF/UKS -> ((pop_a, chg_a), (pop_b, chg_b)).
        if isinstance(pop[0], (list, tuple)) and not np.isscalar(pop[0]):
            mulliken = np.asarray(pop[0][1]) + np.asarray(pop[1][1])
        else:
            mulliken = np.asarray(pop[1])
    except Exception:
        mulliken = None
    try:
        lowdin = _lowdin_charges(mol, dm)
    except Exception:
        lowdin = None
    return mulliken, lowdin


def _dipole_debye(mf, mol):
    """|dipole| in Debye (None if it cannot be computed)."""
    try:
        dip = mf.dip_moment(unit='Debye', verbose=0)
        return float(np.linalg.norm(np.asarray(dip)))
    except Exception:
        return None


def run_scf(atoms, charge, spin_mult, method='RHF', basis='sto-3g', optimize=False, max_steps=50):
    """Run one SCF calculation on an explicit atom list.

    Parameters
    ----------
    atoms : list of {el, x, y, z} (Angstrom)
    charge : int
    spin_mult : int (2S+1)
    method : 'RHF' | 'UHF' | 'B3LYP' | 'PBE0' | 'wB97X-D'
    basis : 'sto-3g' | '6-31g*' | 'def2-svp' | 'cc-pvdz'
    optimize : run a geometry optimisation first (scipy BFGS, analytic gradients)

    Returns a dict of cite records; raises ValueError for oversize inputs.
    """
    from pyscf import gto

    symbols = [a['el'] for a in atoms]
    if _heavy_count(symbols) > _MAX_HEAVY:
        raise ValueError('%d heavy atoms exceeds the %d heavy-atom limit for quantum chemistry'
                         % (_heavy_count(symbols), _MAX_HEAVY))
    spin = int(spin_mult) - 1
    if spin < 0:
        raise ValueError('spin multiplicity must be >= 1')
    basis_key = _BASIS_MAP.get(basis)
    if basis_key is None:
        raise ValueError('unknown basis %r (use sto-3g, 6-31g*, def2-svp, cc-pvdz)' % basis)
    if _estimate_n_basis(symbols, basis) > _MAX_NBASIS:
        raise ValueError('~%d basis functions would exceed the %d basis-function limit'
                         % (_estimate_n_basis(symbols, basis), _MAX_NBASIS))

    atom_str = atoms_to_atom_string(atoms)
    mol = gto.M(atom=atom_str, basis=basis_key, charge=int(charge), spin=spin, verbose=0)
    if mol.nao > _MAX_NBASIS:
        raise ValueError('%d basis functions exceeds the %d basis-function limit'
                         % (mol.nao, _MAX_NBASIS))

    optimizer = None
    optimized = False
    if optimize and len(atoms) > 1:
        mol, mf, optimizer, converged_geom, steps = _optimize_geometry(mol, method, spin, max_steps)
        optimized = True
    else:
        mf = _make_mf(mol, method, spin)

    label = _opt_method_label(method, spin)
    t0 = time.time()
    mf.kernel()
    wall = time.time() - t0
    converged = bool(mf.converged)

    orbitals = _orbitals(mf)
    homo, lumo = _homo_lumo(orbitals)
    mulliken, lowdin = _charges(mf, mol)
    dipole = _dipole_debye(mf, mol)

    method_str = 'PySCF 2.14.0 %s/%s' % (label, basis)
    notes = []
    if method == 'wB97X-D':
        notes.append('wB97X-D: the -D dispersion correction was not applied '
                     '(pyscf-dispersion is not installed); this is a wB97X energy.')
    if spin > 0 and method == 'RHF':
        notes.append('RHF requested for an open shell; UHF was used instead.')
    if not converged:
        notes.append('SCF did not converge within the cycle limit; treat energies as unreliable.')
    if optimized and not converged_geom:
        notes.append('geometry optimisation stopped at the step limit.')

    def est(v, unit, nd):
        return cite.value(round(v, nd), unit, 'estimated', method=method_str)

    out = {
        'formula': _formula(symbols),
        'charge': int(charge),
        'spin_multiplicity': int(spin_mult),
        'spin': spin,
        'method': label,
        'basis': basis,
        'converged': converged,
        'n_basis': int(mol.nao),
        'wall_time_s': round(wall, 3),
        'energy_hartree': est(float(mf.e_tot), 'hartree', 8),
        'energy_kjmol': est(float(mf.e_tot) * KJMOL_PER_HARTREE, 'kJ/mol', 3),
        'orbitals': [{'index': i, 'energy_eV': est(e, 'eV', 4), 'occupation': o}
                     for i, (e, o) in enumerate(orbitals)],
        'atoms': atoms,
    }
    if optimized:
        out['optimized'] = True
        out['optimizer'] = optimizer
    if homo is not None:
        out['homo'] = {'index': homo, 'energy_eV': est(orbitals[homo][0], 'eV', 4)}
        out['lumo'] = {'index': lumo, 'energy_eV': est(orbitals[lumo][0], 'eV', 4)}
        out['gap_eV'] = est(orbitals[lumo][0] - orbitals[homo][0], 'eV', 4)
    charges_out = []
    for ia, a in enumerate(atoms):
        row = {'el': a['el'], 'index': ia}
        if mulliken is not None:
            row['mulliken'] = cite.value(round(float(mulliken[ia]), 4), 'e', 'estimated',
                                         method='PySCF Mulliken population %s/%s' % (label, basis))
        if lowdin is not None:
            row['lowdin'] = cite.value(round(float(lowdin[ia]), 4), 'e', 'estimated',
                                       method='PySCF Lowdin population %s/%s' % (label, basis))
        charges_out.append(row)
    out['charges'] = charges_out
    if dipole is not None:
        out['dipole'] = est(dipole, 'debye', 3)
    out['notes'] = notes
    return out


def _formula(symbols):
    """Hill formula from an atom symbol list (H/C first, then alphabetical)."""
    counts = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    order = sorted(counts, key=lambda s: (s != 'C', s != 'H', s))
    return ''.join('%s%d' % (s, counts[s]) if counts[s] > 1 else s for s in order)


def scf(query, method='RHF', basis='sto-3g', optimize=False, preopt='none'):
    """Query-facing SCF wrapper: resolve geometry, then :func:`run_scf`."""
    geo = resolve_geometry(query, preopt=preopt)
    result = run_scf(geo['atoms'], geo['charge'], geo['spin_mult'], method=method,
                     basis=basis, optimize=optimize)
    result['query'] = query
    result['smiles'] = geo['smiles']
    result['geometry_source'] = geo['geometry']
    result['notes'] = list(geo['notes']) + result.get('notes', [])
    return result


# ---- Molecular-orbital isosurface -------------------------------------------

def _march(grid, iso, origin, spacing):
    """Marching tetrahedra over a cubic scalar grid; returns (vertices, triangles).

    Each cube is split into 6 tetrahedra around its 0-7 body diagonal (the standard
    crack-free decomposition). Edge crossings are cached by the pair of grid-point
    linear indices so adjacent tetrahedra share vertices (watertight mesh).
    """
    n = grid.shape[0]
    flat = grid.reshape(-1)
    verts = []
    tris = []
    cache = {}

    def idx(ix, iy, iz):
        return (iz * n + iy) * n + ix

    def vertex(a, b):
        key = (a, b) if a < b else (b, a)
        if key in cache:
            return cache[key]
        fa, fb = flat[a], flat[b]
        t = (iso - fa) / (fb - fa)
        za, rem = divmod(a, n * n)
        ya, xa = divmod(rem, n)
        zb, rem = divmod(b, n * n)
        yb, xb = divmod(rem, n)
        p = np.array([xa + t * (xb - xa), ya + t * (yb - ya), za + t * (zb - za)], dtype=float)
        cache[key] = len(verts)
        verts.append(origin + p * spacing)
        return cache[key]

    corners = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
               (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    # 6 tetrahedra around the main diagonal 0-7.
    tets = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    for iz in range(n - 1):
        for iy in range(n - 1):
            for ix in range(n - 1):
                ids = [idx(ix + cx, iy + cy, iz + cz) for cx, cy, cz in corners]
                for tet in tets:
                    p = [ids[j] for j in tet]
                    val = np.array([flat[j] for j in p])
                    inside = val < iso
                    ni = int(inside.sum())
                    if ni == 0 or ni == 4:
                        continue
                    ev = {}
                    for (a, b) in edges:
                        if inside[a] != inside[b]:
                            ev[(a, b)] = vertex(p[a], p[b])
                    if ni == 1 or ni == 3:
                        s = next(j for j in range(4) if (inside[j] if ni == 1 else not inside[j]))
                        tri = [ev[(min(s, o), max(s, o))] for o in range(4) if o != s]
                        tris.append(tri)
                    else:  # ni == 2: quad split into two triangles
                        ia = [j for j in range(4) if inside[j]]
                        io = [j for j in range(4) if not inside[j]]
                        e0 = ev[(min(ia[0], io[0]), max(ia[0], io[0]))]
                        e1 = ev[(min(io[0], ia[1]), max(io[0], ia[1]))]
                        e2 = ev[(min(ia[1], io[1]), max(ia[1], io[1]))]
                        e3 = ev[(min(io[1], ia[0]), max(io[1], ia[0]))]
                        tris.append([e0, e1, e2])
                        tris.append([e0, e2, e3])
    return verts, tris


def _cap_triangles(verts, tris, cap):
    """Deterministic uniform stride of triangles down to ``cap`` (keeps the surface sparse)."""
    if len(tris) <= cap:
        return verts, tris, False
    step = -(-len(tris) // cap)  # ceil
    return verts, tris[::step], True


def orbital_cube(query, orbital='HOMO', method='RHF', basis='sto-3g', grid=40, isovalue=0.05):
    """Evaluate one MO on a grid and return ± isosurface meshes for the page viewer.

    The AO values are evaluated with ``pyscf.dft.numint.eval_ao`` and the surface is
    extracted with the module's own marching-tetrahedra (skimage is not installed).
    Triangles are capped at 60 000 across both lobes. Coordinates are Angstrom to
    overlay directly on the ball-and-stick molecule.
    """
    from pyscf import gto
    from pyscf.dft import numint

    geo = resolve_geometry(query, preopt='none')
    atoms = geo['atoms']
    result = run_scf(atoms, geo['charge'], geo['spin_mult'], method=method, basis=basis)
    if not result.get('converged'):
        raise ValueError('SCF did not converge; no orbital surface available.')
    if 'homo' not in result:
        raise ValueError('no virtual/occupied orbitals to plot for this molecule.')

    basis_key = _BASIS_MAP[basis]
    symbols = [a['el'] for a in atoms]
    atom_str = atoms_to_atom_string(atoms)
    mol = gto.M(atom=atom_str, basis=basis_key, charge=int(geo['charge']),
                spin=int(geo['spin_mult']) - 1, verbose=0)
    mf = _make_mf(mol, method, int(geo['spin_mult']) - 1)
    mf.kernel()

    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    if isinstance(mo_coeff, (list, tuple)) and len(mo_coeff) == 2 and not np.isscalar(mo_coeff[0]):
        mo_coeff = mo_coeff[0]  # plot the alpha orbital set for open shells
        mo_energy = mo_energy[0]
    orbitals = _orbitals(mf)
    homo_idx, lumo_idx = _homo_lumo(orbitals)
    occ_indices = [i for i, (e, o) in enumerate(orbitals) if o > 0.5]
    vir_indices = [i for i, (e, o) in enumerate(orbitals) if o < 0.5]
    if isinstance(orbital, int):
        mo_index = orbital
    elif orbital == 'LUMO' and vir_indices:
        mo_index = vir_indices[0]
    elif orbital == 'HOMO' and occ_indices:
        mo_index = occ_indices[-1]
    elif isinstance(orbital, str) and orbital.isdigit():
        mo_index = int(orbital)
    else:
        raise ValueError('orbital %r not available' % (orbital,))
    if mo_index < 0 or mo_index >= len(mo_energy):
        raise ValueError('orbital index %d out of range' % mo_index)

    # Grid box: atoms + 3.5 bohr margin, N points per axis.
    coords_bohr = np.asarray([[a['x'], a['y'], a['z']] for a in atoms], dtype=float) / BOHR
    margin = 3.5
    lo = coords_bohr.min(axis=0) - margin
    hi = coords_bohr.max(axis=0) + margin
    n = max(16, min(int(grid), 48))
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    zs = np.linspace(lo[2], hi[2], n)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    ao = numint.eval_ao(mol, points, deriv=0)
    mo_vals = ao @ np.asarray(mo_coeff)[:, mo_index]
    grid_vals = mo_vals.reshape(n, n, n)

    spacing = np.array([xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]]) * BOHR  # Angstrom
    origin = np.array([lo[0], lo[1], lo[2]]) * BOHR
    cap = 60000
    surfaces = []
    notes = []
    total_tris = 0
    for sign in ('+', '-'):
        iso = isovalue if sign == '+' else -isovalue
        verts, tris = _march(grid_vals, iso, origin, spacing)
        verts, tris, decimated = _cap_triangles(verts, tris, cap - total_tris)
        total_tris += len(tris)
        surfaces.append({'sign': sign, 'isovalue': round(iso, 4),
                         'vertices': [[round(float(v[0]), 4), round(float(v[1]), 4), round(float(v[2]), 4)]
                                      for v in verts],
                         'triangles': [list(map(int, t)) for t in tris]})
        if decimated:
            notes.append('%s lobe decimated to %d triangles (60 000 cap).' % (sign, len(tris)))
    notes.append('Orbital %d (%s, %.3f eV) at isovalue ±%.2f on a %d³ grid.'
                 % (mo_index, orbital, orbitals[mo_index][0], isovalue, n))

    return {
        'query': query, 'smiles': geo['smiles'], 'formula': result['formula'],
        'method': result['method'], 'basis': basis, 'converged': result['converged'],
        'orbital': orbital, 'orbital_index': mo_index,
        'orbital_energy_eV': cite.value(round(orbitals[mo_index][0], 4), 'eV', 'estimated',
                                        method='PySCF 2.14.0 %s/%s' % (result['method'], basis)),
        'occupation': float(orbitals[mo_index][1]),
        'atoms': atoms,
        'surfaces': surfaces,
        'n_triangles': total_tris,
        'notes': notes,
    }


# ---- Bond dissociation energy ------------------------------------------------

def bond_dissociation(query_AB, fragments, method='RHF', basis='sto-3g', with_zpe=False):
    """Homolytic bond-dissociation energy: E(A) + E(B) - E(AB), optional ZPE correction.

    ``fragments`` is a list of two fragment queries. Geometries are the RDKit
    ETKDG+MMFF structures; when ``with_zpe`` is requested the zero-point energy of
    each species is computed from B3LYP/6-31G* harmonic frequencies (the same method
    as :mod:`chemphys_spectra`). D0 = De - ΔZPE. All energies are estimates.
    """
    from engineering_sim.science import chemphys_spectra

    ab = resolve_geometry(query_AB, preopt='none')
    e_ab = run_scf(ab['atoms'], ab['charge'], ab['spin_mult'], method=method, basis=basis)
    frag_data = [resolve_geometry(f, preopt='none') for f in fragments]
    e_frags = [run_scf(f['atoms'], f['charge'], f['spin_mult'], method=method, basis=basis)
               for f in frag_data]

    de_kjmol = (float(e_frags[0]['energy_kjmol']['value']) + float(e_frags[1]['energy_kjmol']['value'])
                - float(e_ab['energy_kjmol']['value']))
    method_str = 'PySCF 2.14.0 %s/%s' % (method, basis)
    out = {
        'query': query_AB, 'fragments': list(fragments),
        'method': method, 'basis': basis,
        'converged': bool(e_ab['converged']) and all(f['converged'] for f in e_frags),
        'e_ab': e_ab['energy_kjmol'],
        'e_fragments': [f['energy_kjmol'] for f in e_frags],
        'bde_kjmol': cite.value(round(de_kjmol, 3), 'kJ/mol', 'estimated', method=method_str,
                                note='E(A)+E(B)-E(AB), electronic (De); RDKit geometries'),
        'notes': [],
    }
    if with_zpe:
        zpe_ab = chemphys_spectra.zpe(query_AB)
        zpe_frags = [chemphys_spectra.zpe(f) for f in fragments]
        zpe_total = zpe_ab['value'] - (zpe_frags[0]['value'] + zpe_frags[1]['value'])
        d0 = de_kjmol - zpe_total
        out['zpe_ab_kjmol'] = zpe_ab
        out['zpe_fragments_kjmol'] = zpe_frags
        out['zpe_correction_kjmol'] = cite.value(round(zpe_total, 3), 'kJ/mol', 'estimated',
                                                 method='B3LYP/6-31g* scaled ZPE')
        out['bde0_kjmol'] = cite.value(round(d0, 3), 'kJ/mol', 'estimated', method=method_str,
                                       note='D0 = De - ΔZPE (B3LYP/6-31g* scaled zero-point)')
        out['notes'].append('ZPE correction from B3LYP/6-31g* harmonic frequencies '
                            '(scale factor 0.9613); D0 = De - ΔZPE.')
    return out

# DESCRIPTION: GFN2-xTB (tblite 0.7.0) geometry relaxation and energy estimates for
#              species without cited thermochemical data.
"""GFN2-xTB estimates via tblite.

``estimate(smiles)`` parses a SMILES with RDKit (ETKDG embed, fixed seed 0xf00d),
relaxes the geometry with a BFGS optimisation on tblite GFN2-xTB energies/gradients
(<= 200 steps), and returns the electronic energy, coordinates, HOMO-LUMO gap,
dipole, spin state and a (caveated) atomization energy.

All reported numbers are ``kind='estimated'`` with method "GFN2-xTB (tblite 0.7.0)"
and the standing caveat that xTB energies are NOT thermochemically accurate — this
module is only for species that have no cited data in ``thermo_species.json``.
"""

from __future__ import annotations

import numpy as np

from . import cite

METHOD = "GFN2-xTB (tblite 0.7.0)"
CAVEAT = "Not thermochemically accurate; for species without cited data only."
BOHR = 0.529177210903  # Bohr radius in Angstrom (CODATA 2018)
EV_PER_HARTREE = 27.211386245988
DEBYE_PER_E_BOHR = 2.541746473
KJMOL_PER_HARTREE = 2625.499639

# Ground-state number of unpaired electrons for the isolated atom (used only for the
# atomization-energy estimate). Main-group elements follow the spin of the valence
# p-shell; a few common transition metals are included.
_ATOM_UHF = {
    1: 1, 2: 0, 3: 1, 4: 0, 5: 1, 6: 2, 7: 3, 8: 2, 9: 1, 10: 0,
    11: 1, 12: 0, 13: 1, 14: 2, 15: 3, 16: 2, 17: 1, 18: 0, 19: 1, 20: 0,
    26: 4, 35: 1, 53: 1,
}


def _make_calculator(numbers, positions_bohr, charge, uhf):
    """Build a silent GFN2-xTB calculator for the given structure."""
    from tblite.interface import Calculator

    calc = Calculator("GFN2-xTB", numbers, np.asarray(positions_bohr, dtype=float),
                      charge=charge, uhf=uhf)
    calc.set("verbosity", 0)
    return calc


def _initial_geometry(mol):
    """Deterministic 3D starting geometry for an RDKit molecule (Angstrom).

    Uses ETKDG embedding with a fixed seed; falls back to atoms laid out along z
    (a robust start for single atoms, diatomics and radical fragments that ETKDG
    cannot embed).
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    n = mol.GetNumAtoms()
    coords = None
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xF00D
        if AllChem.EmbedMolecule(mol, params) == 0:
            # A quick force-field polish is fine for closed-shell organics; radicals
            # and isolated atoms are simply left at the embedded geometry.
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
            conf = mol.GetConformer()
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n)], dtype=float)
    except Exception:
        coords = None
    if coords is None:
        coords = np.zeros((n, 3))
        for i in range(n):
            coords[i, 2] = i * 1.5  # 1.5 A spacing along z; BFGS will relax it
    return coords


def _homo_lumo_gap(result):
    """HOMO-LUMO gap in eV, or None when the result has no virtual orbitals."""
    try:
        occ = np.asarray(result.get("orbital-occupations"), dtype=float)
        ener = np.asarray(result.get("orbital-energies"), dtype=float)
    except Exception:
        return None
    occ_highest = np.where(occ > 0.5)[0]
    occ_lowest = np.where(occ < 0.5)[0]
    if len(occ_highest) == 0 or len(occ_lowest) == 0:
        return None
    homo = float(ener[occ_highest].max())
    lumo = float(ener[occ_lowest].min())
    return (lumo - homo) * EV_PER_HARTREE


def _atom_energy(atomic_number, uhf):
    """GFN2-xTB electronic energy of an isolated atom (Hartree)."""
    calc = _make_calculator([atomic_number], [[0.0, 0.0, 0.0]], 0.0, uhf)
    return float(calc.singlepoint().get("energy"))


def estimate(smiles: str) -> dict:
    """Run a GFN2-xTB geometry optimisation and return estimated properties.

    Parameters
    ----------
    smiles : str
        SMILES of the species (radicals kept explicit; ``[H][H]`` is H2).

    Returns
    -------
    dict with canonical SMILES, formula, charge, multiplicity, unpaired electron
    count (``uhf``), electronic energy (Hartree), optimised coordinates (Angstrom),
    HOMO-LUMO gap (eV), dipole (Debye), atomization energy (kJ/mol), and
    convergence info. All numbers are ``kind='estimated'`` GFN2-xTB records.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("cannot parse SMILES: %r" % (smiles,))
    charge = Chem.GetFormalCharge(mol)
    uhf = int(sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()))
    mol = Chem.AddHs(mol)
    n_atoms = mol.GetNumAtoms()
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    canonical = Chem.MolToSmiles(Chem.RemoveHs(mol))
    formula = rdMolDescriptors.CalcMolFormula(mol)

    coords_ang = _initial_geometry(mol)
    positions_bohr = coords_ang / BOHR

    converged = False
    steps = 0
    if n_atoms > 1:
        from scipy.optimize import minimize

        def objective(x):
            pos = np.asarray(x).reshape(-1, 3)
            calc = _make_calculator(numbers, pos, charge, uhf)
            res = calc.singlepoint()
            e = float(res.get("energy"))
            g = np.asarray(res.get("gradient"), dtype=float).ravel()
            return e, g

        result = minimize(
            objective,
            positions_bohr.ravel(),
            method="BFGS",
            jac=True,
            options={"maxiter": 200, "gtol": 1e-5},
        )
        positions_bohr = np.asarray(result.x).reshape(-1, 3)
        converged = bool(result.success)
        steps = int(result.nit)

    calc = _make_calculator(numbers, positions_bohr, charge, uhf)
    res = calc.singlepoint()
    energy_hartree = float(res.get("energy"))
    coords_final_ang = positions_bohr * BOHR
    gap_eV = _homo_lumo_gap(res)
    dipole_debye = None
    try:
        dipole_debye = float(np.linalg.norm(np.asarray(res.get("dipole")))) * DEBYE_PER_E_BOHR
    except Exception:
        dipole_debye = None

    # Atomization energy: E(molecule) - sum of isolated-atom GFN2 energies, computed
    # for each unique element at its ground-state spin. Only an estimate (xTB atom
    # and molecule energies carry a systematic error), hence the caveat.
    atomization_kjmol = None
    try:
        atom_sum = 0.0
        for z in sorted(set(numbers)):
            atom_sum += numbers.count(z) * _atom_energy(z, _ATOM_UHF.get(z, 0))
        atomization_kjmol = (atom_sum - energy_hartree) * KJMOL_PER_HARTREE
    except Exception:
        atomization_kjmol = None

    out = {
        "smiles": canonical,
        "formula": formula,
        "n_atoms": n_atoms,
        "charge": charge,
        "uhf": uhf,
        "multiplicity": uhf + 1,
        "electronic_energy": cite.value(energy_hartree, "hartree", "estimated",
                                         method=METHOD, note=CAVEAT),
        "coordinates_angstrom": [list(map(float, row)) for row in coords_final_ang],
        "converged": converged,
        "steps": steps,
        "note": CAVEAT,
    }
    if gap_eV is not None:
        out["homo_lumo_gap"] = cite.value(gap_eV, "eV", "estimated", method=METHOD, note=CAVEAT)
    if dipole_debye is not None:
        out["dipole"] = cite.value(dipole_debye, "debye", "estimated", method=METHOD, note=CAVEAT)
    if atomization_kjmol is not None:
        out["atomization_energy"] = cite.value(
            atomization_kjmol, "kJ/mol", "estimated", method=METHOD,
            note="Atomization energy from isolated-atom GFN2 energies; %s" % CAVEAT)
    return out

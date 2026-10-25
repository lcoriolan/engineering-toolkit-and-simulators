# DESCRIPTION: Task A: RDKit molecule parser (atoms/bonds/Lewis/functional groups/Joback estimates).
"""Parse a molecule query into a full structural + thermodynamic picture.

Detection order (see module contract):
  1. a bare element symbol (H, O, Na, Cl, ...) -> a single neutral atom/radical;
  2. a SMILES string (CCO, [H][H], O=O);
  3. otherwise a chemical name -> records.lookup() for a SMILES.

Deterministic: RDKit ETKDG uses randomSeed=0xf00d and every list/set is sorted
so the same query always yields byte-identical output.
"""
import re

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Geometry import Point3D

from engineering_sim.science import cite

_PT = Chem.GetPeriodicTable()

# English names for bare-atom queries (name is only attached when known).
_ELEMENT_NAMES = {
    'H': 'Hydrogen', 'He': 'Helium', 'Li': 'Lithium', 'Be': 'Beryllium', 'B': 'Boron',
    'C': 'Carbon', 'N': 'Nitrogen', 'O': 'Oxygen', 'F': 'Fluorine', 'Ne': 'Neon',
    'Na': 'Sodium', 'Mg': 'Magnesium', 'Al': 'Aluminium', 'Si': 'Silicon', 'P': 'Phosphorus',
    'S': 'Sulfur', 'Cl': 'Chlorine', 'Ar': 'Argon', 'K': 'Potassium', 'Ca': 'Calcium',
    'Fe': 'Iron', 'Cu': 'Copper', 'Zn': 'Zinc', 'Br': 'Bromine', 'I': 'Iodine',
}

# Functional groups detected by SMARTS, in report order (alphabetical in output).
_GROUPS = (
    ('carboxylic_acid', '[CX3](=O)[OX2H1]'),
    ('hydroxyl', '[OX2H1][#6]'),
    ('primary_amine', '[NX3H2][#6]'),
    ('secondary_amine', '[NX3H1]([#6])[#6]'),
    ('epoxide', '[#6]1[#8][#6]1'),
    ('anhydride', '[CX3](=O)[OX2][CX3](=O)'),
    ('isocyanate', '[NX2]=[CX2]=[OX1]'),
    ('oxazoline', '[#8]1[#6][#6][#7]=[#6]1'),
    ('carbodiimide', '[NX2]=[CX2]=[NX2]'),
    ('ester', '[#6][CX3](=O)[OX2][#6]'),
    ('amide', '[NX3][CX3](=O)'),
    ('aldehyde', '[CX3H1](=O)[#6]'),
    ('ketone', '[#6][CX3](=O)[#6]'),
    # Exclude oxygens that are directly bonded to a carbonyl carbon so esters/amides
    # do not also light up as "ether"; C-O-C where neither C is carbonyl.
    ('ether', '[OD2]([#6;!$(C=O)])[#6;!$(C=O)]'),
    ('alkene', '[CX3]=[CX3]'),
    ('alkyne', '[CX2]#[CX2]'),
    ('aromatic_ring', 'a'),
    ('halide', '[F,Cl,Br,I][#6]'),
    ('nitrile', '[#6]#[#7]'),
    ('thiol', '[SX2H1][#6]'),
    ('silanol', '[Si][OX2H1]'),
)

# Joback-Reid group contributions. Each tuple: (dTb [K], dTm [K], dTc, dPc [bar],
# dVc [cm^3/mol], dHform [kJ/mol], dGform [kJ/mol], dHvap [kJ/mol]).
# Source: Joback, K. G.; Reid, R. C. "Estimation of Pure-Component Properties
# from Group-Contributions", Chem. Eng. Commun. 1987, 57, 233-243, as tabulated
# in Poling, Prausnitz & O'Connell, "The Properties of Gases and Liquids",
# 5th ed., McGraw-Hill, 2001, Table C-1.
_JOBACK = {
    # carbon, nonring
    '-CH3':    (23.58, -5.10, 0.0141, -0.0012, 65, -76.45, -43.96, 2.373),
    '-CH2-':   (22.88, 11.27, 0.0189,  0.0000, 56, -20.64,   8.42, 2.226),
    '>CH-':    (21.74, 12.64, 0.0164,  0.0020, 41,  29.89,  58.36, 1.691),
    '>C<':     (18.25, 46.43, 0.0067,  0.0043, 27,  82.23, 116.02, 0.636),
    '=CH2':    (18.18, -4.32, 0.0113, -0.0028, 56,  -9.63,   3.77, 1.724),
    '=CH-':    (24.96,  8.73, 0.0129, -0.0006, 46,  37.97,  48.53, 2.205),
    '=C<':     (24.14, 11.14, 0.0117,  0.0011, 38,  83.99,  92.36, 2.138),
    '=C=':     (26.15, 17.78, 0.0026,  0.0028, 36, 142.14, 136.70, 2.661),
    '#CH':     ( 9.20, -11.18, 0.0027, -0.0008, 46,  79.30,  77.71, 1.155),
    '#C-':     (27.38, 64.32, 0.0020,  0.0016, 36, 115.51, 109.82, 3.302),
    # carbon, ring
    '-CH2-(r)': (27.15,  7.75, 0.0100,  0.0025, 48, -26.80,  -3.68, 2.398),
    '>CH-(r)':  (21.78, 19.88, 0.0122,  0.0004, 38,   8.67,  40.99, 1.942),
    '>C<(r)':   (21.32, 60.15, 0.0042,  0.0061, 27,  79.72,  87.88, 0.644),
    '=CH-(r)':  (26.73,  8.13, 0.0082,  0.0011, 41,   2.09,  11.30, 2.544),
    '=C<(r)':   (31.01, 37.02, 0.0143,  0.0008, 32,  46.43,  54.05, 3.059),
    # halogens
    '-F':      (-0.03, -15.78, 0.0111, -0.0057, 27, -251.92, -247.19, -0.670),
    '-Cl':     (38.13, 13.55, 0.0105, -0.0049, 58, -71.55, -64.31, 4.532),
    '-Br':     (66.86, 43.43, 0.0133,  0.0057, 71, -29.48, -38.06, 6.582),
    '-I':      (93.84, 41.69, 0.0068, -0.0034, 97,  21.06,   5.74, 9.520),
    # oxygen
    '-OH':     (92.88, 44.45, 0.0741,  0.0112, 28, -208.04, -189.20, 16.826),
    '-OH(ph)': (76.34, 82.83, 0.0240,  0.0184, -25, -221.65, -197.37, 12.499),
    '-O-':     (22.42, 22.23, 0.0168,  0.0015, 18, -132.22, -105.00, 2.410),
    '-O-(r)':  (31.22, 23.05, 0.0098,  0.0048, 13, -138.16, -98.22, 4.682),
    '>C=O':    (76.75, 61.20, 0.0380,  0.0031, 62, -133.22, -120.50, 8.972),
    '>C=O(r)': (94.97, 75.97, 0.0284,  0.0028, 55, -164.50, -126.27, 6.645),
    'O=CH-':   (72.24, 36.90, 0.0379,  0.0030, 82, -162.03, -143.48, 9.093),
    '-COOH':   (169.09, 155.50, 0.0791,  0.0077, 89, -426.72, -387.87, 19.537),
    '-COO-':   (81.10, 53.60, 0.0481,  0.0005, 82, -337.92, -301.95, 9.633),
    '=O(other)': (-10.50, 2.08, 0.0143,  0.0101, 36, 247.61, 250.83, 5.909),
    # nitrogen
    '-NH2':    (73.23, 66.89, 0.0243,  0.0109, 38, -22.02,  14.07, 10.788),
    '>NH':     (50.17, 52.66, 0.0295,  0.0077, 35,  53.47,  89.39, 6.436),
    '>NH(r)':  (52.82, 101.51, 0.0130,  0.0114, 30,  31.65,  75.61, 5.099),
    '>N-':     (11.74, 48.84, 0.0169,  0.0074,  9, 123.34, 163.16, 1.896),
    '-CN':     (125.66, 59.89, 0.0497, -0.0101, 91,  88.43,  89.22, 12.851),
    '-NO2':    (152.54, 127.24, 0.0437,  0.0064, 91, -66.57, -16.83, 9.441),
    # sulfur
    '-SH':     (63.56, 20.09, 0.0031,  0.0084, 63, -17.33, -22.99, 6.884),
    '-S-':     (68.78, 34.40, 0.0119,  0.0049, 54,  41.87,  33.12, 5.984),
    '-S-(r)':  (52.10, 79.93, 0.0019,  0.0051, 38,  39.10,  27.76, 6.717),
}

# Joback-Reid equation constants (same source as _JOBACK).
_JB_TB, _JB_TM = 198.2, 122.5          # K
_JB_TC_A, _JB_TC_B = 0.584, 0.965       # dimensionless
_JB_PC_A, _JB_PC_B = 0.113, 0.0032      # bar
_JB_VC, _JB_HF, _JB_GF, _JB_HV = 17.5, 68.29, 53.88, 15.30  # cm^3/mol, kJ/mol

_JB_REFERENCE = ('Joback K. G., Reid R. C., "Estimation of Pure-Component Properties '
                 'from Group-Contributions", Chem. Eng. Commun. 1987, 57, 233-243.')


class _Uncovered(Exception):
    """A molecule (or atom) is outside the groups the Joback table covers."""


def _is_element_symbol(q):
    """True when `q` is a bare element symbol (H, O, Na, Cl, ...)."""
    if not re.fullmatch(r'[A-Z][a-z]?', q):
        return False
    try:
        _PT.GetAtomicNumber(q)
        return True
    except Exception:
        return False


def _bonds(atom, mol):
    """List of (neighbor atom, bond order) for one atom."""
    out = []
    for nb in atom.GetNeighbors():
        order = mol.GetBondBetweenAtoms(atom.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble()
        out.append((nb, order))
    return out


def _classify(atom, mol):
    """Map one heavy atom to a Joback group key; returns (key, extra_consumed_idx).

    Raises _Uncovered when the atom (or a multi-atom group it leads) is outside
    the Joback table. `extra_consumed_idx` are neighbor atoms swallowed by a
    multi-atom group (carbonyl oxygens, the N of a nitrile, ...) so they are not
    double-counted. Callers iterate atoms in ascending atomic number so the
    claiming atom always precedes the atoms it claims.
    """
    z = atom.GetAtomicNum()
    n_h = atom.GetTotalNumHs()
    ring = atom.IsInRing()
    arom = atom.GetIsAromatic()
    bs = _bonds(atom, mol)

    if z == 6:  # carbon
        tri_n = [nb for nb, o in bs if o == 3 and nb.GetAtomicNum() == 7]
        if tri_n:
            return '-CN', {tri_n[0].GetIdx()}  # nitrile carbon claims its N
        d_ox = [nb for nb, o in bs if o == 2 and nb.GetAtomicNum() == 8]
        if d_ox:
            n_double = sum(1 for _, o in bs if o == 2)
            if n_double > 1:
                raise _Uncovered('carbonyl carbon with an extra double bond not covered')
            others = [(nb, o) for nb, o in bs if not (o == 2 and nb.GetAtomicNum() == 8)]
            o_nb = d_ox[0]
            if n_h == 1 and len(others) == 1 and others[0][1] == 1 and others[0][0].GetAtomicNum() == 6:
                return 'O=CH-', {o_nb.GetIdx()}  # aldehyde
            if n_h == 0 and len(others) == 2 and all(o == 1 for _, o in others):
                els = sorted(nb.GetAtomicNum() for nb, _ in others)
                if els == [6, 6]:
                    return ('>C=O(r)' if ring else '>C=O'), {o_nb.GetIdx()}  # ketone
                if els == [6, 8]:
                    o2 = [nb for nb, _ in others if nb.GetAtomicNum() == 8][0]
                    other = [nb for nb in o2.GetNeighbors() if nb.GetIdx() != atom.GetIdx()]
                    if other and any(nb.GetAtomicNum() == 8 and
                                     mol.GetBondBetweenAtoms(other[0].GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() == 2
                                     for nb in other[0].GetNeighbors()):
                        raise _Uncovered('anhydride not covered by Joback groups')
                    if o2.GetTotalNumHs() >= 1:
                        return '-COOH', {o_nb.GetIdx(), o2.GetIdx()}  # carboxylic acid
                    return '-COO-', {o_nb.GetIdx(), o2.GetIdx()}      # ester
                if 7 in els:
                    raise _Uncovered('amide not covered by Joback groups')
            raise _Uncovered('carbonyl carbon not covered by Joback groups')
        if arom:
            return ('=CH-(r)' if n_h >= 1 else '=C<(r)'), set()  # aromatic ring carbon
        n_triple = sum(1 for _, o in bs if o == 3)
        n_double = sum(1 for _, o in bs if o == 2)
        n_single = sum(1 for _, o in bs if o == 1)
        if n_triple == 1:
            if n_h == 1:
                return '#CH', set()
            if n_h == 0:
                return '#C-', set()
            raise _Uncovered('triple-bonded carbon not covered by Joback groups')
        if n_double == 2 and n_h == 0:
            return '=C=', set()  # allene central carbon
        if n_double == 1:
            if n_h == 2:
                return '=CH2', set()
            if n_h == 1:
                return ('=CH-(r)' if ring else '=CH-'), set()
            if n_h == 0:
                return ('=C<(r)' if ring else '=C<'), set()
            raise _Uncovered('double-bonded carbon not covered by Joback groups')
        if n_h == 3 and n_single == 1:
            return '-CH3', set()
        if n_h == 2 and n_single == 2:
            return ('-CH2-(r)' if ring else '-CH2-'), set()
        if n_h == 1 and n_single == 3:
            return ('>CH-(r)' if ring else '>CH-'), set()
        if n_h == 0 and n_single == 4:
            return ('>C<(r)' if ring else '>C<'), set()
        raise _Uncovered('carbon atom not covered by Joback groups')

    if z == 7:  # nitrogen
        tri_c = [nb for nb, o in bs if o == 3 and nb.GetAtomicNum() == 6]
        if tri_c:
            return '-CN', {tri_c[0].GetIdx()}  # nitrile nitrogen claims its C
        d_ox = [nb for nb, o in bs if o == 2 and nb.GetAtomicNum() == 8]
        if len(d_ox) == 2 and n_h == 0:
            return '-NO2', {d_ox[0].GetIdx(), d_ox[1].GetIdx()}  # nitro
        if arom:
            raise _Uncovered('aromatic (pyridine-type) nitrogen not covered by Joback groups')
        n_single = sum(1 for _, o in bs if o == 1)
        n_double = sum(1 for _, o in bs if o == 2)
        if n_double == 1:
            raise _Uncovered('imine nitrogen (-N= / =NH) not covered by Joback groups')
        if n_h == 2 and n_single == 1:
            return '-NH2', set()
        if n_h == 1 and n_single == 2:
            return ('>NH(r)' if ring else '>NH'), set()
        if n_h == 0 and n_single == 3:
            return '>N-', set()
        raise _Uncovered('nitrogen atom not covered by Joback groups')

    if z == 8:  # oxygen
        if arom:
            return '-O-(r)', set()  # furan-type aromatic oxygen (approximation)
        n_single = sum(1 for _, o in bs if o == 1)
        n_double = sum(1 for _, o in bs if o == 2)
        if n_double == 1:
            raise _Uncovered('=O not part of a covered carbonyl/nitro group')
        if n_h == 1 and n_single == 1:
            nb = [nb for nb, o in bs if o == 1][0]
            if nb.GetAtomicNum() == 6 and nb.GetIsAromatic():
                return '-OH(ph)', set()  # phenol
            return '-OH', set()          # alcohol
        if n_h == 0 and n_single == 2:
            if any(nb.GetAtomicNum() == 8 for nb, o in bs if o == 1):
                raise _Uncovered('peroxide (-O-O-) not covered by Joback groups')
            return ('-O-(r)' if ring else '-O-'), set()
        raise _Uncovered('oxygen atom not covered by Joback groups')

    if z in (9, 17, 35, 53):  # halogens
        if n_h == 0 and len(bs) == 1 and bs[0][1] == 1:
            return {9: '-F', 17: '-Cl', 35: '-Br', 53: '-I'}[z], set()
        raise _Uncovered('halogen atom not covered by Joback groups')

    if z == 16:  # sulfur
        if arom:
            return '-S-(r)', set()  # thiophene-type aromatic sulfur (approximation)
        n_single = sum(1 for _, o in bs if o == 1)
        if n_h == 1 and n_single == 1:
            return '-SH', set()
        if n_h == 0 and n_single == 2:
            if any(nb.GetAtomicNum() == 16 for nb, o in bs if o == 1):
                raise _Uncovered('disulfide (-S-S-) not covered by Joback groups')
            return ('-S-(r)' if ring else '-S-'), set()
        raise _Uncovered('sulfur atom not covered by Joback groups')

    raise _Uncovered('atom %d (%s) not covered by Joback groups' % (atom.GetIdx(), atom.GetSymbol()))


def _joback(mol):
    """Joback-Reid group-contribution estimates, or {available: False, reason: ...}.

    `mol` is the sanitized molecule (implicit hydrogens). Iterating heavy atoms
    in ascending atomic number guarantees the claiming atom of a multi-atom
    group (carbonyl C, nitro N, nitrile C) is processed before the atoms it
    consumes, so carbonyl/nitro oxygens are never double-counted.
    """
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        return {'available': False, 'reason': 'no heavy atoms (Joback needs an organic skeleton)'}
    counts = {}
    consumed = set()
    try:
        for atom in sorted(heavy, key=lambda a: (a.GetAtomicNum(), a.GetIdx())):
            if atom.GetIdx() in consumed:
                continue
            key, extra = _classify(atom, mol)
            counts[key] = counts.get(key, 0) + 1
            consumed.update(extra)
    except _Uncovered as e:
        return {'available': False, 'reason': str(e)}

    sums = [0.0] * 8
    for key, cnt in counts.items():
        row = _JOBACK[key]
        for j in range(8):
            sums[j] += cnt * row[j]
    na = sum(1 + a.GetTotalNumHs() for a in mol.GetAtoms())  # atoms including H

    tb = _JB_TB + sums[0]
    tm = _JB_TM + sums[1]
    tc = tb / (_JB_TC_A + _JB_TC_B * sums[2] - sums[2] ** 2)
    pc = (_JB_PC_A + _JB_PC_B * na - sums[3]) ** -2
    vc = _JB_VC + sums[4]
    hf = _JB_HF + sums[5]
    gf = _JB_GF + sums[6]
    hv = _JB_HV + sums[7]

    def est(v, unit, nd):
        return cite.value(round(v, nd), unit, 'estimated', method='Joback-Reid 1987')

    return {
        'available': True,
        'Tb_K': est(tb, 'K', 2),
        'Tm_K': est(tm, 'K', 2),
        'Tc_K': est(tc, 'K', 2),
        'Pc_bar': est(pc, 'bar', 3),
        'Vc_cm3mol': est(vc, 'cm^3/mol', 1),
        'Hf_gas_kJmol': est(hf, 'kJ/mol', 2),
        'Gf_gas_kJmol': est(gf, 'kJ/mol', 2),
        'Hvap_Tb_kJmol': est(hv, 'kJ/mol', 2),
        'reference': _JB_REFERENCE,
    }


def _lewis(mol_h):
    """Lewis electron bookkeeping from a Kekulized, explicit-H molecule.

    Per atom: formal charge q = V - lone - unpaired - bond_order_sum, so
    lone_electrons = V - unpaired - bond_order_sum - q (V = outer-shell
    electrons). Aromatic bonds are Kekulized first so heteroaromatic atoms get
    integer bond orders (furan O -> 2 single bonds -> 2 lone pairs).
    """
    mol_k = Chem.Mol(mol_h)
    try:
        Chem.Kekulize(mol_k, clearAromaticFlags=False)
    except Exception:
        mol_k = mol_h  # un-Kekulizable aromatic system: fall back to 1.5 orders

    per_atom = []
    total_valence = 0
    total_lone_pairs = 0
    total_unpaired = 0
    for atom in mol_k.GetAtoms():
        i = atom.GetIdx()
        el = atom.GetSymbol()
        v = _PT.GetNOuterElecs(atom.GetAtomicNum())
        q = atom.GetFormalCharge()
        unpaired = atom.GetNumRadicalElectrons()
        bond_sum = 0
        for nb in atom.GetNeighbors():
            bond_sum += mol_k.GetBondBetweenAtoms(i, nb.GetIdx()).GetBondTypeAsDouble()
        lone_e = v - unpaired - bond_sum - q
        lone_pairs = int(round(lone_e / 2.0)) if lone_e >= 0 else 0
        # Octet (or duet for H/He) counting: shared pairs count for both atoms.
        shell = 2 * bond_sum + lone_e + unpaired
        satisfied = (shell >= 2) if atom.GetAtomicNum() in (1, 2) else (shell >= 8)
        total_valence += v
        total_lone_pairs += lone_pairs
        total_unpaired += unpaired
        per_atom.append({
            'i': i, 'el': el, 'valence_electrons': v,
            'bonds': int(round(bond_sum)), 'lone_pairs': lone_pairs,
            'unpaired': unpaired, 'octet_or_duet_satisfied': bool(satisfied),
        })

    bonding_pairs = 0
    for bond in mol_k.GetBonds():
        bonding_pairs += int(round(bond.GetBondTypeAsDouble()))
    return per_atom, total_valence, bonding_pairs, total_lone_pairs, total_unpaired


def _lewis_text(valence, bonding_pairs, lone_pairs, unpaired, formula):
    """One plain-English sentence explaining the electron picture."""
    if formula == 'H2':
        return 'H· + ·H → H:H — two 1s electrons pair into one σ bond'
    parts = ['%d valence electrons: %d bonding pair(s), %d lone pair(s)' % (valence, bonding_pairs, lone_pairs)]
    if unpaired:
        parts.append('%d unpaired electron(s) (a radical)' % unpaired)
    return '; '.join(parts) + '.'


def _embed_3d(mol_h):
    """Embed 3D coordinates: ETKDG (randomSeed=0xf00d) then MMFF (UFF fallback).

    Single atoms sit at the origin; diatomics are aligned along +z at the
    force-field bond length. Returns the optimization method name (or None).
    """
    n = mol_h.GetNumAtoms()
    if n == 1:
        conf = Chem.Conformer(1)
        conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
        mol_h.AddConformer(conf)
        return None
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 0xf00d
    if AllChem.EmbedMolecule(mol_h, ps) != 0:
        # Distance geometry failed (rare, e.g. strained cages): zeroed coords
        # keep the record well-formed rather than raising.
        conf = Chem.Conformer(n)
        for i in range(n):
            conf.SetAtomPosition(i, Point3D(0.0, 0.0, 0.0))
        mol_h.AddConformer(conf)
        return 'embed_failed'
    if AllChem.MMFFHasAllMoleculeParams(mol_h):
        AllChem.MMFFOptimizeMolecule(mol_h)
        method = 'MMFF'
    else:
        AllChem.UFFOptimizeMolecule(mol_h)
        method = 'UFF'
    if n == 2:
        conf = mol_h.GetConformer()
        p0, p1 = conf.GetAtomPosition(0), conf.GetAtomPosition(1)
        d = p0.Distance(p1)
        conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
        conf.SetAtomPosition(1, Point3D(0.0, 0.0, d))
    return method


def _from_mol(mol, query, name=None, note=None):
    """Assemble the full molecule dict from a parsed, sanitized RDKit Mol."""
    smiles = Chem.MolToSmiles(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = round(Descriptors.MolWt(mol), 3)
    charge = Chem.GetFormalCharge(mol)
    radical_electrons = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    spin_multiplicity = radical_electrons + 1

    mol_h = Chem.AddHs(mol)
    embed_method = _embed_3d(mol_h)
    conf = mol_h.GetConformer()

    per_atom, lew_valence, lew_bonding, lew_lone, lew_unpaired = _lewis(mol_h)
    lone_by_idx = {p['i']: p['lone_pairs'] for p in per_atom}
    unpaired_by_idx = {p['i']: p['unpaired'] for p in per_atom}

    atoms = []
    for atom in mol_h.GetAtoms():
        i = atom.GetIdx()
        pos = conf.GetAtomPosition(i)
        atoms.append({
            'i': i, 'el': atom.GetSymbol(),
            'x': round(pos.x, 4), 'y': round(pos.y, 4), 'z': round(pos.z, 4),
            'formal_charge': atom.GetFormalCharge(),
            'lone_pairs': lone_by_idx.get(i, 0),
            'unpaired': unpaired_by_idx.get(i, 0),
        })

    bonds = []
    for bond in mol_h.GetBonds():
        bonds.append({'a': bond.GetBeginAtomIdx(), 'b': bond.GetEndAtomIdx(),
                      'order': bond.GetBondTypeAsDouble()})

    notes = [n for n in (note,) if n]
    if formula == 'O2' and radical_electrons == 0:
        notes.append(
            'O=O is drawn here as a singlet (double bond, S=0); the ground state of O2 is the '
            'triplet diradical ³Σ⁻g with two unpaired electrons (S=1, spin multiplicity 3). '
            'Thermodynamic and spectroscopic data for O2 use the triplet state.')
    if embed_method == 'embed_failed':
        notes.append('3D embedding failed; coordinates fall back to zeroed positions.')

    return {
        'query': query,
        'smiles': smiles,
        'name': name,
        'formula': formula,
        'mw': cite.value(mw, 'g/mol', 'derived', method='RDKit MolWt (average atomic masses)'),
        'charge': charge,
        'radical_electrons': radical_electrons,
        'spin_multiplicity': spin_multiplicity,
        'atoms': atoms,
        'bonds': bonds,
        'groups': _detect_groups(mol),
        'lewis': {
            'valence_electrons_total': lew_valence,
            'bonding_pairs': lew_bonding,
            'lone_pairs_total': lew_lone,
            'per_atom': per_atom,
            'text': _lewis_text(lew_valence, lew_bonding, lew_lone, lew_unpaired, formula),
        },
        'joback': _joback(mol),
        'notes': notes,
    }


def _detect_groups(mol):
    """Sorted list of functional-group names matched by SMARTS on the heavy-atom mol."""
    found = []
    for name, smarts in _GROUPS:
        pat = Chem.MolFromSmarts(smarts)
        if pat is not None and mol.HasSubstructMatch(pat):
            found.append(name)
    return sorted(found)


def _from_smiles(smiles, query, name=None, note=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'query': query, 'smiles': None, 'name': name, 'error': 'unparseable SMILES: ' + smiles,
                'notes': ['unparseable SMILES: ' + smiles]}
    return _from_mol(mol, query, name=name, note=note)


def molecule(query):
    """Resolve a query (element symbol / SMILES / name) into a molecule dict.

    See module docstring for the detection order and the returned structure.
    Numeric properties are cite.value records; structural integers are plain.
    """
    q = (query or '').strip()
    if not q:
        return {'query': '', 'smiles': None, 'name': None, 'error': 'empty query',
                'notes': ['empty query']}
    # 1. bare element symbol -> single neutral atom/radical ([H], [O], [Na], ...)
    if _is_element_symbol(q):
        sym = q.capitalize()
        note = 'Single atom interpreted as the neutral radical [%s] (RDKit valence model).' % sym
        return _from_smiles('[' + sym + ']', q, name=_ELEMENT_NAMES.get(sym), note=note)
    # 2. SMILES
    mol = Chem.MolFromSmiles(q)
    if mol is not None:
        return _from_mol(mol, q)
    # 3. name -> records.lookup for a SMILES (cache first, then network)
    from engineering_sim.science import records
    rec = records.lookup(q)
    if rec.get('found') and rec.get('smiles'):
        return _from_smiles(rec['smiles'], q, name=rec.get('name'), note='Resolved from PubChem name lookup.')
    msg = rec.get('error') or ('offline' if rec.get('offline') else 'name not found in PubChem')
    return {'query': q, 'smiles': None, 'name': q, 'error': msg,
            'notes': ['could not resolve "%s" as an element, SMILES, or PubChem name: %s' % (q, msg)]}


def parse_items(items):
    """Split a reactant list into counted molecules.

    Each item may carry a count: "2 H", "2H" (leading number) or "O2 x3"
    (trailing "xN"). A single comma-separated string is accepted too. Returns
    [{"count": n, "molecule": molecule(q)}].
    """
    if isinstance(items, str):
        items = [s for s in re.split(r'[,\n]', items) if s.strip()]
    out = []
    for item in items:
        s = item.strip()
        if not s:
            continue
        count = 1
        m = re.fullmatch(r'(.+?)\s*[xX]\s*(\d+)\s*$', s)
        if m:
            s = m.group(1).strip()
            count = int(m.group(2))
        else:
            m = re.fullmatch(r'(\d+)\s+(.+)$', s)
            if m:
                count, s = int(m.group(1)), m.group(2).strip()
            else:
                m = re.fullmatch(r'(\d+)(\D.*)$', s)
                if m:
                    count, s = int(m.group(1)), m.group(2).strip()
        out.append({'count': count, 'molecule': molecule(s)})
    return out

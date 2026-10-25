# DESCRIPTION: Stability ladder — five separate verdicts for one molecule:
#              1 valence-valid (RDKit), 2 true energy minimum (GFN2-xTB -> B3LYP-D3(BJ)/def2-SVP opt + harmonic
#              frequencies), 3 thermodynamic stability (formation energy, energy above the cited-reference hull,
#              weakest bond energies), 4 kinetic stability (HOMO-LUMO gap, homolysis and concerted xTB-NEB
#              barriers, Eyring half-life at 298 K), 5 synthesizability (SA score, AiZynthFinder route, stock check).

"""Stability ladder engine.

Every number is labelled COMPUTED (method, basis, program version), REFERENCE (cited record from
data/thermo_species.json or a literature constant) or ESTIMATE (screening model with a stated error).
The job directory holds request.json (inputs), state.json (progress + rung results) and log.txt;
its name is the recipe hash of the inputs plus every method version, so a rerun with the same recipe
returns the stored result unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

# ---- Constants (CODATA 2018 via NIST) ----
HARTREE_KJ = 2625.499639479
HARTREE_EV = 27.211386245988
EV_KJ = 96.48533212
BOHR_A = 0.529177210903
KB = 1.380649e-23
H_PLANCK = 6.62607015e-34
R_KJ = 8.314462618e-3
T0 = 298.15
CM1_PER_AU = 219474.6313632  # cm-1 per Hartree

DFT_METHOD = 'B3LYP-D3(BJ)/def2-SVP'
DFT_XC, DFT_DISP, DFT_BASIS = 'b3lyp', 'd3bj', 'def2-svp'
XTB_METHOD = 'GFN2-xTB'
IMAG_CM1 = 50.0          # |nu| >= 50 cm-1 imaginary => not a minimum
SOFT_CM1 = 20.0          # 20-50i reported as numerical-noise soft modes
DFT_HEAVY_MAX = 20       # DFT optimisation + frequencies
DFT_BDE_HEAVY_MAX = 12   # DFT bond energies (UKS fragments)
NEB_HEAVY_MAX = 15       # xTB NEB concerted channel
QRRHO_W0 = 100.0         # Grimme quasi-RRHO cutoff (cm-1)
CACHE_VERSION = 2        # bump when the species evaluation or thermochemistry changes (invalidates _cache)

# Verdict thresholds (ASSUMED screening bins; each verdict shows the number it was judged on)
HULL_PASS, HULL_WARN = 25.0, 100.0          # kJ/mol per heavy atom above the hull
HALF_PASS_S, HALF_WARN_S = 3.156e7, 3600.0  # 1 year, 1 hour
GAP_WARN_EV = 0.5
SA_PASS, SA_WARN = 4.5, 6.5

# Atom-hydride reference scheme for formation enthalpies (error cancellation vs plain atomization).
HYDRIDE = {'C': ('C', 'CH4', 4), 'N': ('N', 'NH3', 3), 'O': ('O', 'H2O(g)', 2), 'F': ('F', 'HF', 1),
           'Cl': ('Cl', 'HCl', 1), 'Br': ('Br', 'HBr', 1), 'S': ('S', 'H2S', 2)}
HYDRIDE_SMILES = {'CH4': 'C', 'H2O(g)': 'O', 'NH3': 'N', 'HF': 'F', 'HCl': 'Cl', 'HBr': 'Br', 'H2S': 'S', 'H2': '[H][H]'}
# Element standard states (S298 from the cited species records; Br2(l) entropy NIST WebBook, CODATA)
ELEMENT_STATE = {'H': ('H2', 2), 'C': ('C(gr)', 1), 'N': ('N2', 2), 'O': ('O2', 2), 'F': ('F2', 2),
                 'Cl': ('Cl2', 2), 'S': ('S(s)', 1)}
BR2_LIQUID_S = {'kind': 'cited', 'value': 152.21, 'unit': 'J/mol*K',
                'source': 'https://webbook.nist.gov/cgi/cbook.cgi?ID=C7726956&Mask=2', 'note': 'Br2(l), CODATA 1984'}


def versions():
    import importlib.metadata as md
    out = {'python': sys.version.split()[0]}
    for mod in ('rdkit', 'pyscf', 'tblite', 'ase', 'geometric', 'numpy', 'scipy', 'pyscf-dispersion'):
        try:
            out[mod] = md.version(mod)
        except Exception:
            try:
                out[mod] = __import__(mod).__version__
            except Exception:
                out[mod] = None
    return out


def rec(value, unit, kind, method=None, source=None, note=None, uncertainty=None):
    r = {'value': value, 'unit': unit, 'kind': kind}
    for k, v in (('method', method), ('source', source), ('note', note), ('uncertainty', uncertainty)):
        if v is not None:
            r[k] = v
    return r


def computed(value, unit, method, note=None):
    return rec(value, unit, 'computed', method=method, note=note)


def estimate(value, unit, method, note=None, uncertainty=None):
    return rec(value, unit, 'estimated', method=method, note=note, uncertainty=uncertainty)


# =====================================================================================================
# Structure handling
# =====================================================================================================
SYMBOLS = None


def _sym(z):
    from rdkit import Chem
    return Chem.GetPeriodicTable().GetElementSymbol(int(z))


def parse_xyz(text):
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if lines and lines[0].strip().isdigit():
        lines = lines[2:] if len(lines) > 1 and not _is_atom_line(lines[1]) else lines[1:]
    atoms = []
    for ln in lines:
        p = ln.split()
        if len(p) < 4:
            raise ValueError('XYZ line needs "El x y z": %r' % ln)
        atoms.append((p[0].capitalize() if len(p[0]) <= 2 else p[0], float(p[1]), float(p[2]), float(p[3])))
    if not atoms:
        raise ValueError('XYZ geometry is empty.')
    return atoms


def _is_atom_line(ln):
    p = ln.split()
    try:
        return len(p) >= 4 and all(float(x) or True for x in p[1:4])
    except ValueError:
        return False


def rung1(smiles):
    """Valence-valid via RDKit sanitisation (the existing science toolkit check)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return {'verdict': 'fail', 'summary': 'RDKit cannot parse the SMILES.', 'evidence': {}}, None
    problems = Chem.DetectChemistryProblems(mol)
    if problems:
        return {'verdict': 'fail', 'summary': 'Valence/aromaticity problems: ' +
                '; '.join(p.Message() for p in problems), 'evidence': {'problems': [p.Message() for p in problems]}}, None
    Chem.SanitizeMol(mol)
    nrad = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    charge = Chem.GetFormalCharge(mol)
    molh = Chem.AddHs(mol)
    ev = {'canonical_smiles': Chem.MolToSmiles(mol), 'formula': rdMolDescriptors.CalcMolFormula(mol),
          'charge': charge, 'multiplicity': nrad + 1, 'heavy_atoms': mol.GetNumHeavyAtoms(),
          'atoms': molh.GetNumAtoms(), 'mw': round(Descriptors.MolWt(mol), 3),
          'inchikey': Chem.MolToInchiKey(mol) or None}
    return {'verdict': 'pass', 'summary': 'RDKit sanitisation passed (valence, aromaticity, charges).',
            'label': 'COMPUTED (RDKit %s)' % __import__('rdkit').__version__, 'evidence': ev}, molh


def embed(molh, nconf=10):
    """ETKDGv3 conformers (seed 0xF00D) + MMFF; returns list of coordinate arrays (Angstrom)."""
    from rdkit.Chem import AllChem
    p = AllChem.ETKDGv3()
    p.randomSeed = 0xF00D
    nrot = AllChem.CalcNumRotatableBonds(molh) if hasattr(AllChem, 'CalcNumRotatableBonds') else 3
    n = 1 if molh.GetNumAtoms() <= 4 else min(nconf, 3 + 3 * nrot)
    ids = list(AllChem.EmbedMultipleConfs(molh, numConfs=n, params=p))
    if not ids:
        p.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(molh, numConfs=n, params=p))
    if not ids:
        raise ValueError('RDKit could not embed a 3-D structure.')
    try:
        AllChem.MMFFOptimizeMoleculeConfs(molh, maxIters=500)
    except Exception:
        pass
    return [np.array(molh.GetConformer(i).GetPositions()) for i in ids]


# =====================================================================================================
# GFN2-xTB (tblite) level
# =====================================================================================================
def _xtb_calc(numbers, pos_A, charge, mult):
    from tblite.interface import Calculator
    c = Calculator('GFN2-xTB', np.asarray(numbers), np.asarray(pos_A, float) / BOHR_A,
                   charge=float(charge), uhf=int(mult) - 1)
    c.set('verbosity', 0)
    return c


def xtb_point(numbers, pos_A, charge, mult):
    r = _xtb_calc(numbers, pos_A, charge, mult).singlepoint()
    e = float(r.get('energy'))
    g = np.asarray(r.get('gradient'), float)  # Hartree/Bohr
    occ = np.asarray(r.get('orbital-occupations'), float)
    en = np.asarray(r.get('orbital-energies'), float)
    gap = None
    if (occ > 0.5).any() and (occ < 0.5).any():
        gap = float((en[occ < 0.5].min() - en[occ > 0.5].max()) * HARTREE_EV)
    return e, g, gap


def _ase_atoms(numbers, pos_A, charge, mult):
    from ase import Atoms
    from tblite.ase import TBLite
    a = Atoms(numbers=list(numbers), positions=np.asarray(pos_A, float))
    a.calc = TBLite(method='GFN2-xTB', charge=float(charge), multiplicity=int(mult), verbosity=0)
    return a


def xtb_opt(numbers, pos_A, charge, mult, fmax=0.005, steps=600):
    from ase.optimize import BFGS
    a = _ase_atoms(numbers, pos_A, charge, mult)
    if len(a) > 1:
        opt = BFGS(a, logfile=None)
        opt.run(fmax=fmax, steps=steps)
    e, g, gap = xtb_point(numbers, a.get_positions(), charge, mult)
    return a.get_positions(), e, gap


def xtb_hessian(numbers, pos_A, charge, mult, step_A=0.005):
    """Central finite-difference Hessian from analytic xTB gradients, Hartree/Bohr^2, shape (N,N,3,3)."""
    n = len(numbers)
    x0 = np.asarray(pos_A, float)
    H = np.zeros((n, 3, n, 3))
    hb = step_A / BOHR_A
    for i in range(n):
        for k in range(3):
            xp = x0.copy(); xp[i, k] += step_A
            xm = x0.copy(); xm[i, k] -= step_A
            gp = xtb_point(numbers, xp, charge, mult)[1]
            gm = xtb_point(numbers, xm, charge, mult)[1]
            H[i, k] = (gp - gm) / (2 * hb)
    H = 0.5 * (H + H.transpose(2, 3, 0, 1))
    return H.transpose(0, 2, 1, 3)


# =====================================================================================================
# DFT level (PySCF)
# =====================================================================================================
def _pyscf_mol(numbers, pos_A, charge, mult, basis=DFT_BASIS):
    from pyscf import gto
    return gto.M(atom=[(_sym(z), tuple(p)) for z, p in zip(numbers, pos_A)], basis=basis, charge=int(charge),
                 spin=int(mult) - 1, unit='Angstrom', verbose=0)


def _dft_mf(mol):
    from pyscf import dft
    mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
    mf.xc = DFT_XC
    mf.disp = DFT_DISP
    mf.conv_tol = 1e-10
    mf.grids.level = 4
    mf.max_cycle = 200
    return mf


def dft_opt(numbers, pos_A, charge, mult, log):
    from pyscf.geomopt.geometric_solver import optimize
    mol = _pyscf_mol(numbers, pos_A, charge, mult)
    if mol.natm == 1:
        mf = _dft_mf(mol); mf.kernel()
        return pos_A, mf
    mf = _dft_mf(mol)
    mol_eq = optimize(mf, maxsteps=150, convergence_set='GAU_TIGHT')
    mf = _dft_mf(mol_eq)
    mf.kernel()
    if not mf.converged:
        log('DFT SCF did not converge at the optimised geometry')
    return mol_eq.atom_coords(unit='Angstrom'), mf


def dft_hessian(mf):
    h = mf.Hessian().kernel()  # includes the D3(BJ) term through pyscf-dispersion when supported
    return h


def dft_gap(mf):
    if mf.mol.spin == 0:
        e, o = mf.mo_energy, mf.mo_occ
    else:
        e, o = np.concatenate(mf.mo_energy), np.concatenate(mf.mo_occ)
    if (o > 0.5).any() and (o < 0.5).any():
        return float((e[o < 0.5].min() - e[o > 0.5].max()) * HARTREE_EV)
    return None


# =====================================================================================================
# Harmonic analysis + quasi-RRHO thermochemistry (ideal gas, 1 bar)
# =====================================================================================================
def _sigma(numbers, pos_A):
    """Rotational symmetry number from the detected point group (PySCF symmetry detection, tolerance 0.05 Bohr)."""
    from pyscf.symm import geom
    old = geom.TOLERANCE
    geom.TOLERANCE = 0.05   # Bohr; optimised geometries carry ~1e-4 Bohr residual asymmetry
    try:
        g = geom.detect_symm([(_sym(z), tuple(np.asarray(p) / BOHR_A)) for z, p in zip(numbers, pos_A)])[0]
    except Exception:
        return 1, 'C1'
    finally:
        geom.TOLERANCE = old
    g = str(g)
    if g in ('Dooh',):
        return 2, g
    if g in ('Coov', 'C1', 'Ci', 'Cs', 'SO3'):
        return 1, g
    if g in ('T', 'Td', 'Th'):
        return 12, g
    if g in ('O', 'Oh'):
        return 24, g
    if g in ('I', 'Ih'):
        return 60, g
    import re
    m = re.match(r'^([CDS])(\d+)', g)
    if m:
        n = int(m.group(2))
        return {'C': n, 'D': 2 * n, 'S': max(1, n // 2)}[m.group(1)], g
    return 1, g


def harmonic(numbers, pos_A, hess):
    """Frequencies (cm-1, imaginary negative) with translations/rotations projected out (PySCF)."""
    from pyscf.hessian import thermo
    mol = _pyscf_mol(numbers, pos_A, 0, 1 + (sum(numbers) % 2), basis='sto-3g')
    info = thermo.harmonic_analysis(mol, hess, exclude_trans=True, exclude_rot=True, imaginary_freq=False)
    f = np.asarray(info['freq_wavenumber'])
    modes = np.asarray(info.get('norm_mode')) if info.get('norm_mode') is not None else None
    return f, modes


def rrho(numbers, pos_A, freqs_cm1, e_elec_h, mult, T=T0, P=1.0e5):
    """Ideal-gas thermochemistry (kJ/mol, J/mol/K). Grimme quasi-RRHO entropy for modes below 100 cm-1
    (Grimme, Chem. Eur. J. 2012, 18, 9955). Imaginary modes are excluded."""
    from rdkit import Chem
    pt = Chem.GetPeriodicTable()
    m_amu = np.array([pt.GetMostCommonIsotopeMass(int(z)) for z in numbers])
    M = m_amu.sum() * 1.66053906660e-27
    kT = KB * T
    q_tr = (2 * math.pi * M * kT / H_PLANCK ** 2) ** 1.5 * kT / P
    S_tr = R_KJ * 1000 * (math.log(q_tr) + 2.5)
    x = np.asarray(pos_A) * 1e-10
    com = (m_amu[:, None] * x).sum(0) / m_amu.sum()
    x = x - com
    mkg = m_amu * 1.66053906660e-27
    I = np.zeros((3, 3))
    for mi, r in zip(mkg, x):
        I += mi * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    Iv = np.sort(np.linalg.eigvalsh(I))
    sigma, pg = _sigma(numbers, pos_A)
    n = len(numbers)
    linear = n == 2 or (n > 2 and Iv[0] < 1e-3 * Iv[2])
    if n == 1:
        S_rot, H_rot = 0.0, 0.0
    elif linear:
        q_rot = 8 * math.pi ** 2 * Iv[2] * kT / (sigma * H_PLANCK ** 2)
        S_rot, H_rot = R_KJ * 1000 * (math.log(q_rot) + 1), R_KJ * T
    else:
        q_rot = (math.pi ** 0.5 / sigma) * np.prod([(8 * math.pi ** 2 * Ii * kT / H_PLANCK ** 2) ** 0.5 for Ii in Iv])
        S_rot, H_rot = R_KJ * 1000 * (math.log(q_rot) + 1.5), 1.5 * R_KJ * T
    fr = [f for f in freqs_cm1 if f > 0]
    zpe = 0.5 * sum(fr) / CM1_PER_AU * HARTREE_KJ
    H_vib, S_vib = 0.0, 0.0
    Bav = 1e-44  # Grimme's average molecular moment of inertia, kg m^2
    for f in fr:
        theta = f * 100 * 2.99792458e8 * H_PLANCK / KB
        u = theta / T
        e_th = R_KJ * theta * (1 / math.expm1(u))
        s_h = R_KJ * 1000 * (u / math.expm1(u) - math.log(1 - math.exp(-u)))
        # free-rotor entropy with effective moment mu' = mu*Bav/(mu+Bav), mu = h/(8 pi^2 nu)
        nu = f * 100 * 2.99792458e8
        mu = H_PLANCK / (8 * math.pi ** 2 * nu)
        mu_eff = mu * Bav / (mu + Bav)
        s_r = R_KJ * 1000 * (0.5 + math.log((8 * math.pi ** 3 * mu_eff * kT / H_PLANCK ** 2) ** 0.5))
        w = 1 / (1 + (QRRHO_W0 / f) ** 4)
        H_vib += e_th
        S_vib += w * s_h + (1 - w) * s_r
    S_el = R_KJ * 1000 * math.log(mult)
    E = e_elec_h * HARTREE_KJ
    H = E + zpe + H_vib + H_rot + 1.5 * R_KJ * T + R_KJ * T
    S = S_tr + S_rot + S_vib + S_el
    return {'E': E, 'ZPE': zpe, 'H': H, 'S': S, 'G': H - T * S / 1000, 'sigma': sigma, 'point_group': pg,
            'linear': bool(linear)}


# =====================================================================================================
# Species evaluation (shared by the target, hydrides and fragments)
# =====================================================================================================
class Ctx:
    def __init__(self, job_dir, cache_dir, log, progress):
        self.job_dir, self.cache_dir, self.log, self.progress = Path(job_dir), Path(cache_dir), log, progress


def evaluate(ctx, key, numbers, pos_A, charge, mult, level, want_hess=True):
    """Optimise + frequencies at 'xtb' or 'dft'. Cached by (level, key) for reference species."""
    cache = ctx.cache_dir / ('v%s_%s_%s.json' % (CACHE_VERSION, level, hashlib.sha256(key.encode()).hexdigest()[:16]))
    if key.startswith('ref:') and cache.is_file():
        return json.loads(cache.read_text())
    t0 = time.time()
    pos, e, gap = xtb_opt(numbers, pos_A, charge, mult)
    out = {'key': key, 'level': level, 'numbers': [int(z) for z in numbers], 'charge': charge, 'mult': mult}
    if level == 'xtb':
        out.update(e_h=e, gap_xtb_eV=gap, pos=pos.tolist())
        if want_hess and len(numbers) > 1:
            out['freqs'] = harmonic(numbers, pos, xtb_hessian(numbers, pos, charge, mult))[0].tolist()
        else:
            out['freqs'] = []
    else:
        out['gap_xtb_eV'] = gap
        pos2, mf = dft_opt(numbers, pos, charge, mult, ctx.log)
        out.update(e_h=float(mf.e_tot), gap_dft_eV=dft_gap(mf), pos=np.asarray(pos2).tolist(),
                   converged=bool(mf.converged))
        if want_hess and len(numbers) > 1:
            try:
                h = dft_hessian(mf)
                out['hess_note'] = 'analytic B3LYP Hessian; D3(BJ) included'
            except Exception as ex:  # dispersion Hessian unsupported: drop it for the frequencies only
                ctx.log('dispersion Hessian unavailable (%s); frequencies from B3LYP without D3' % ex)
                mf2 = _dft_mf(mf.mol); mf2.disp = None; mf2.kernel()
                h = mf2.Hessian().kernel()
                out['hess_note'] = 'analytic B3LYP Hessian without the D3(BJ) term (unsupported)'
            f, modes = harmonic(numbers, np.asarray(pos2), h)
            out['freqs'] = f.tolist()
            out['_modes'] = None if modes is None else modes.tolist()
        else:
            out['freqs'] = []
    out['thermo'] = rrho(numbers, np.asarray(out['pos']), [f for f in out['freqs'] if f > 0], out['e_h'], mult)
    out['wall_s'] = round(time.time() - t0, 1)
    if key.startswith('ref:'):
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({k: v for k, v in out.items() if k != '_modes'}))
    return out


def species_from_smiles(smiles):
    from rdkit import Chem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    nums = [a.GetAtomicNum() for a in m.GetAtoms()]
    if m.GetNumAtoms() == 1:
        pos = np.zeros((1, 3))
    elif m.GetNumAtoms() == 2:
        pos = np.array([[0, 0, 0], [0, 0, 0.9 if 1 in nums else 1.2]])
    else:
        pos = embed(m, nconf=1)[0]
    nrad = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())
    return nums, pos, Chem.GetFormalCharge(m), nrad + 1


# =====================================================================================================
# Rung 2
# =====================================================================================================
def rung2(ctx, molh, xyz_atoms, charge, mult, heavy):
    from rdkit import Chem
    level = 'dft' if heavy <= DFT_HEAVY_MAX else 'xtb'
    if xyz_atoms:
        nums = [Chem.GetPeriodicTable().GetAtomicNumber(a[0]) for a in xyz_atoms]
        starts = [np.array([a[1:] for a in xyz_atoms], float)]
        src = 'user geometry (no conformer search; optimised from the given coordinates)'
    else:
        nums = [a.GetAtomicNum() for a in molh.GetAtoms()]
        starts = embed(molh)
        src = 'RDKit ETKDGv3 seed 0xF00D, %d conformer(s), MMFF pre-relaxed' % len(starts)
    ctx.progress('rung2', 'xTB conformer screen (%d)' % len(starts))
    best = None
    for s in starts:
        try:
            p, e, gap = xtb_opt(nums, s, charge, mult)
        except Exception as ex:
            ctx.log('xTB opt failed for a conformer: %s' % ex)
            continue
        if best is None or e < best[1]:
            best = (p, e, gap)
    if best is None:
        raise RuntimeError('GFN2-xTB optimisation failed for every start geometry.')
    ctx.progress('rung2', 'optimise + frequencies at %s' % (DFT_METHOD if level == 'dft' else XTB_METHOD))
    sp = evaluate(ctx, 'target', nums, best[0], charge, mult, level)
    f = np.array(sp['freqs'])
    imag = [float(x) for x in f if x <= -IMAG_CM1]
    soft = [float(x) for x in f if -IMAG_CM1 < x <= -SOFT_CM1]
    method = DFT_METHOD if level == 'dft' else XTB_METHOD
    ev = {'level': level, 'method': method, 'start': src, 'n_imaginary': len(imag), 'imaginary_cm1': imag,
          'soft_modes_cm1': soft, 'frequencies_cm1': [round(x, 1) for x in f.tolist()],
          'energy_hartree': computed(sp['e_h'], 'Hartree', method), 'point_group': sp['thermo']['point_group'],
          'geometry_A': [[_sym(z)] + [round(c, 6) for c in p] for z, p in zip(sp['numbers'], sp['pos'])],
          'threshold': 'imaginary if nu <= -%g cm-1; -%g..-%g cm-1 = soft/numerical noise (DFT grid, residual '
                       'gradient on flat torsions); justified because spurious modes of this size are typical for '
                       'grid level 4 and GAU_TIGHT convergence' % (IMAG_CM1, SOFT_CM1, IMAG_CM1),
          'wall_s': sp['wall_s']}
    if level == 'xtb':
        ev['note'] = 'more than %d heavy atoms: xTB only (DFT skipped by the size cap)' % DFT_HEAVY_MAX
    if sp.get('hess_note'):
        ev['hessian'] = sp['hess_note']
    if len(nums) == 1:
        verdict, summ = 'pass', 'Single atom: trivially a minimum.'
    elif imag:
        verdict = 'fail'
        summ = 'Not a minimum: %d imaginary frequenc%s (%s cm-1); the optimised structure is a saddle point.' % (
            len(imag), 'y' if len(imag) == 1 else 'ies', ', '.join('%.0fi' % -x for x in imag))
        ctx.progress('rung2', 'following the imaginary mode')
        try:
            fol = follow_mode(ctx, sp, level)
            ev['mode_follow'] = fol
            summ += ' Displacing along the mode relaxes to a structure %.1f kJ/mol lower with %d imaginary mode(s).' % (
                fol['drop_kJmol'], fol['n_imaginary'])
        except Exception as ex:
            ev['mode_follow'] = {'error': str(ex)}
    else:
        verdict = 'warn' if soft else 'pass'
        summ = 'True minimum at %s: no imaginary frequencies%s.' % (
            method, '' if not soft else ' (soft modes %s cm-1 within numerical noise)' % ', '.join('%.0fi' % -x for x in soft))
    lab = 'COMPUTED (%s; tblite %s, PySCF %s, geomeTRIC %s)' % (method, versions()['tblite'], versions()['pyscf'],
                                                               versions()['geometric'])
    return {'verdict': verdict, 'summary': summ, 'label': lab, 'evidence': ev}, sp


def follow_mode(ctx, sp, level):
    modes = sp.get('_modes')
    f = np.array(sp['freqs'])
    i = int(np.argmin(f))
    if modes is None:
        raise RuntimeError('no normal modes available')
    d = np.array(modes[i]).reshape(-1, 3)
    d = d / np.abs(d).max() * 0.25
    best = None
    for sgn in (1, -1):
        p0 = np.array(sp['pos']) + sgn * d
        r = evaluate(ctx, 'follow%+d' % sgn, sp['numbers'], p0, sp['charge'], sp['mult'], level)
        if best is None or r['e_h'] < best['e_h']:
            best = r
    fb = np.array(best['freqs'])
    return {'drop_kJmol': round((sp['e_h'] - best['e_h']) * HARTREE_KJ, 2),
            'n_imaginary': int((fb <= -IMAG_CM1).sum()), 'point_group': best['thermo']['point_group'],
            'frequencies_cm1': [round(x, 1) for x in fb.tolist()]}


# =====================================================================================================
# Rung 3
# =====================================================================================================
def _thermo_db():
    from engineering_sim.science import thermo
    return thermo._load()


def _ref(key):
    d = _thermo_db()[key]
    return d


def ref_dfG(key):
    """Standard Gibbs energy of formation (kJ/mol) from cited dHf298 and S298 records."""
    if key == 'Br2(l)':
        return 0.0, None
    d = _ref(key)
    dH = d['dHf298_kJmol']['value']
    S = d['S298_JmolK']['value']
    Sel = 0.0
    for el, n in d['elements'].items():
        Sel += n * _element_S(el)
    return dH - T0 * (S - Sel) / 1000.0, d['dHf298_kJmol']


def _element_S(el):
    if el == 'Br':
        return BR2_LIQUID_S['value'] / 2
    key, n = ELEMENT_STATE[el]
    return _ref(key)['S298_JmolK']['value'] / n


def _counts(numbers):
    out = {}
    for z in numbers:
        s = _sym(z)
        out[s] = out.get(s, 0) + 1
    return out


def rung3(ctx, sp, smiles, molh, heavy):
    level = sp['level']
    method = DFT_METHOD if level == 'dft' else XTB_METHOD
    cnt = _counts(sp['numbers'])
    ev = {'level': level, 'method': method, 'T_K': T0, 'P_bar': 1.0,
          'model': 'ideal gas, harmonic + Grimme quasi-RRHO (100 cm-1)'}
    unsupported = [e for e in cnt if e != 'H' and e not in HYDRIDE]
    if unsupported or sp['charge'] != 0 or sp['mult'] != 1:
        why = ('elements %s have no cited hydride reference' % ', '.join(unsupported)) if unsupported else \
            'charged or open-shell species are outside the closed-shell neutral reference scheme'
        return {'verdict': 'unknown', 'summary': 'Thermodynamic rung not available: ' + why + '.',
                'evidence': dict(ev, reason=why)}, None
    # --- formation enthalpy/Gibbs via the atom-hydride scheme ---
    ctx.progress('rung3', 'reference hydrides at %s' % method)
    hyd = {}
    for key in ['H2'] + [HYDRIDE[e][1] for e in cnt if e != 'H']:
        nums, pos, q, mlt = species_from_smiles(HYDRIDE_SMILES[key])
        hyd[key] = evaluate(ctx, 'ref:' + key, nums, pos, q, mlt, level)
    x = (sum(HYDRIDE[e][2] * n for e, n in cnt.items() if e != 'H') - cnt.get('H', 0)) / 2.0
    prod = {HYDRIDE[e][1]: n for e, n in cnt.items() if e != 'H'}
    dH_r = sum(n * hyd[k]['thermo']['H'] for k, n in prod.items()) - sp['thermo']['H'] - x * hyd['H2']['thermo']['H']
    dG_r = sum(n * hyd[k]['thermo']['G'] for k, n in prod.items()) - sp['thermo']['G'] - x * hyd['H2']['thermo']['G']
    dHf = sum(n * _ref(k)['dHf298_kJmol']['value'] for k, n in prod.items()) - dH_r
    dGf = sum(n * ref_dfG(k)[0] for k, n in prod.items()) - dG_r
    rxn = '%s + %g H2 -> %s' % (sp_formula(cnt), x, ' + '.join('%d %s' % (n, k) for k, n in prod.items()))
    ev['formation_scheme'] = {'reaction': rxn, 'dH_rxn_kJmol': computed(round(dH_r, 2), 'kJ/mol', method),
                              'note': 'dHf(M) = sum dHf_ref(hydrides) - dH_rxn(computed); hydride dHf are cited '
                                      'NIST WebBook/CODATA records (data/thermo_species.json)'}
    ev['dHf298'] = estimate(round(dHf, 1), 'kJ/mol', method + ' + cited hydride references',
                            uncertainty=('~10-30 kJ/mol' if level == 'dft' else '~30-80 kJ/mol'))
    ev['dGf298'] = estimate(round(dGf, 1), 'kJ/mol', method + ' + cited hydride references')
    ref_key = _ref_key_for(smiles)
    if ref_key:
        r = _ref(ref_key)['dHf298_kJmol']
        ev['reference_dHf298'] = dict(r, species=ref_key)
        ev['method_error_dHf_kJmol'] = round(dHf - r['value'], 1)
        if prod == {ref_key: 1} and abs(x) < 1e-9:
            ev['method_error_note'] = 'trivial: the target is itself the reference hydride'
    # --- energy above the cited-reference hull (molecular + elemental standard states) ---
    hull = hull_lp(cnt, dGf)
    per = hull['above_kJmol'] / max(1, heavy)
    ev['hull'] = {'above_hull_kJmol': estimate(round(hull['above_kJmol'], 1), 'kJ/mol', 'LP over cited species'),
                  'above_hull_per_heavy_atom_kJmol': round(per, 1), 'decomposes_to': hull['products'],
                  'candidates': hull['n_candidates'],
                  'note': 'minimum-Gibbs mixture of the cited NIST/CODATA species and elemental standard states with '
                          'the same composition (the molecular analogue of energy above hull); gas phase, 298 K, 1 bar'}
    ev['solid_hull'] = 'not available: pymatgen energy-above-hull needs a key-free phase-diagram dataset; none is used ' \
                       '(Materials Project requires an API key)'
    # --- weakest bonds ---
    ctx.progress('rung3', 'bond dissociation energies')
    try:
        ev['bonds'] = bond_energies(ctx, sp, molh, heavy)
    except Exception as ex:
        ev['bonds'] = {'error': str(ex)}
    if per <= HULL_PASS:
        v = 'pass'
        s = 'On or near the hull: %.1f kJ/mol per heavy atom above the most stable mix of reference species.' % per
    elif per <= HULL_WARN:
        v = 'warn'
        s = 'Metastable: %.1f kJ/mol per heavy atom above %s; survives only if kinetically protected.' % (
            per, _fmt_products(hull['products']))
    else:
        v = 'fail'
        s = 'High-energy: %.1f kJ/mol per heavy atom above %s (strongly favoured decomposition).' % (
            per, _fmt_products(hull['products']))
    s += ' dHf298 ~ %.0f kJ/mol.' % dHf
    return {'verdict': v, 'summary': s, 'label': 'ESTIMATE (%s + cited references)' % method, 'evidence': ev,
            'thresholds': 'pass <= %g, warn <= %g kJ/mol per heavy atom above hull (ASSUMED screening bins)' % (
                HULL_PASS, HULL_WARN)}, {'dHf': dHf, 'dGf': dGf, 'hull': hull, 'hyd': hyd}


def sp_formula(cnt):
    order = ['C', 'H'] + sorted(e for e in cnt if e not in ('C', 'H'))
    return ''.join('%s%s' % (e, cnt[e] if cnt[e] > 1 else '') for e in order if e in cnt)


def _fmt_products(p):
    return ' + '.join('%g %s' % (round(v['n'], 3), k) for k, v in p.items()) or 'its elements'


def _ref_key_for(smiles):
    can = _canon(smiles)
    for k, d in _thermo_db().items():
        if k == '_meta' or not isinstance(d, dict) or d.get('phase') != 'gas' or not d.get('smiles'):
            continue
        try:
            if _canon(d['smiles']) == can:
                return k
        except Exception:
            continue
    return None


def hull_lp(cnt, dGf_target):
    from scipy.optimize import linprog
    db = _thermo_db()
    cands = []
    for k, d in db.items():
        if k == '_meta' or not isinstance(d, dict) or 'elements' not in d:
            continue
        if not set(d['elements']) <= set(cnt):
            continue
        if d.get('phase') == 'liquid':
            continue
        if 'dHf298_kJmol' not in d or 'S298_JmolK' not in d:
            continue
        try:
            g = ref_dfG(k)[0]
        except Exception:
            continue
        cands.append((k, d['elements'], g))
    if 'Br' in cnt:
        cands.append(('Br2(l)', {'Br': 2}, 0.0))
    cands.append(('TARGET', cnt, dGf_target))
    els = sorted(cnt)
    A = np.array([[c[1].get(e, 0) for c in cands] for e in els], float)
    b = np.array([cnt[e] for e in els], float)
    cost = np.array([c[2] for c in cands])
    res = linprog(cost, A_eq=A, b_eq=b, bounds=[(0, None)] * len(cands), method='highs')
    if not res.success:
        return {'above_kJmol': 0.0, 'products': {}, 'n_candidates': len(cands) - 1}
    prods = {cands[i][0]: {'n': float(res.x[i]), 'dGf_kJmol': round(cands[i][2], 2)}
             for i in range(len(cands)) if res.x[i] > 1e-9}
    return {'above_kJmol': max(0.0, dGf_target - float(res.fun)), 'products': prods,
            'n_candidates': len(cands) - 1, 'g_min': float(res.fun)}


def bond_energies(ctx, sp, molh, heavy):
    """Homolytic bond enthalpies (298 K) for each symmetry-unique acyclic bond; xTB for all, DFT for the weakest."""
    from rdkit import Chem
    if molh is None:
        return {'note': 'no connectivity (geometry-only input)'}
    ranks = list(Chem.CanonicalRankAtoms(molh, breakTies=False))
    seen, cand = set(), []
    for b in molh.GetBonds():
        if b.IsInRing():
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        key = tuple(sorted((ranks[i], ranks[j]))) + (str(b.GetBondType()),)
        if key in seen:
            continue
        seen.add(key)
        cand.append(b.GetIdx())
    rings_only = not cand
    if rings_only:
        cand = [b.GetIdx() for b in molh.GetBonds()][:4]
    xtb_level = []
    pos = np.array(sp['pos'])
    for bi in cand[:12]:
        b = molh.GetBondWithIdx(bi)
        frags = _split(molh, bi, pos)
        if frags is None:
            continue
        r = _bde(ctx, sp, frags, 'xtb', b)
        if r:
            xtb_level.append(r)
    xtb_level.sort(key=lambda r: r['dH298_kJmol'])
    out = {'xtb': xtb_level, 'ring_bonds_only': rings_only,
           'note': 'homolysis enthalpies; ring bonds are not broken (ring opening is not a dissociation)'}
    if sp['level'] == 'dft' and heavy <= DFT_BDE_HEAVY_MAX and xtb_level:
        dl = []
        for r in xtb_level[:2]:
            b = molh.GetBondWithIdx(r['bond_index'])
            frags = _split(molh, r['bond_index'], pos)
            d = _bde(ctx, sp, frags, 'dft', b)
            if d:
                dl.append(d)
        out['dft'] = dl
    best = (out.get('dft') or xtb_level or [None])[0]
    out['weakest'] = best
    return out


def _split(molh, bi, pos):
    from rdkit import Chem
    b = molh.GetBondWithIdx(bi)
    if b.GetBondType() != Chem.BondType.SINGLE:
        return None
    em = Chem.RWMol(molh)
    i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
    em.RemoveBond(i, j)
    em.GetAtomWithIdx(i).SetNumRadicalElectrons(em.GetAtomWithIdx(i).GetNumRadicalElectrons() + 1)
    em.GetAtomWithIdx(j).SetNumRadicalElectrons(em.GetAtomWithIdx(j).GetNumRadicalElectrons() + 1)
    m = em.GetMol()
    frags = Chem.GetMolFrags(m, asMols=False, sanitizeFrags=False)
    if len(frags) != 2:
        return None
    out = []
    for idx in frags:
        nums = [m.GetAtomWithIdx(k).GetAtomicNum() for k in idx]
        nrad = sum(m.GetAtomWithIdx(k).GetNumRadicalElectrons() for k in idx)
        sm = Chem.MolFragmentToSmiles(m, idx, allHsExplicit=False)
        out.append({'idx': list(idx), 'numbers': nums, 'pos': pos[list(idx)], 'mult': nrad + 1, 'smiles': sm})
    return out


def _bde(ctx, sp, frags, level, bond):
    try:
        H = 0.0
        names = []
        for f in frags:
            r = evaluate(ctx, 'ref:frag:%s:%s' % (level, f['smiles']), f['numbers'], f['pos'], 0, f['mult'], level)
            H += r['thermo']['H']
            names.append(f['smiles'])
        if level == 'xtb':
            parent_H = sp['thermo']['H'] if sp['level'] == 'xtb' else _xtb_parent(ctx, sp)['thermo']['H']
        else:
            parent_H = sp['thermo']['H']
        a = bond.GetBeginAtom().GetSymbol() + '-' + bond.GetEndAtom().GetSymbol()
        out = {'bond': a, 'bond_index': bond.GetIdx(), 'fragments': names, 'level': level,
               'dH298_kJmol': round(H - parent_H, 1)}
        ref = _bde_reference(names, sp)
        if ref:
            out['reference_kJmol'] = ref
            out['error_kJmol'] = round(out['dH298_kJmol'] - ref['value'], 1)
        return out
    except Exception as ex:
        ctx.log('BDE failed (%s, %s): %s' % (level, bond.GetIdx(), ex))
        return None


_XTB_PARENT = {}


def _xtb_parent(ctx, sp):
    if 'p' not in _XTB_PARENT:
        _XTB_PARENT['p'] = evaluate(ctx, 'target-xtb', sp['numbers'], sp['pos'], sp['charge'], sp['mult'], 'xtb')
    return _XTB_PARENT['p']


def _bde_reference(frag_smiles, sp):
    """Cited reaction enthalpy when parent and both radicals are in the reference set (e.g. H2O -> H + OH)."""
    from rdkit import Chem
    keys = []
    for s in frag_smiles:
        can = _canon(s)
        k = None
        for kk, d in _thermo_db().items():
            if kk == '_meta' or not isinstance(d, dict) or not d.get('smiles') or d.get('phase') != 'gas':
                continue
            if _canon(d['smiles']) == can:
                k = kk
                break
        if k is None:
            return None
        keys.append(k)
    return _bde_ref_parent(keys, sp)


def _canon(s):
    """Canonical SMILES with hydrogens implicit, so '[H][O]' (a fragment) and '[OH]' (a record) compare equal."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(s)
    if m is None:
        m = Chem.MolFromSmiles(s, sanitize=False)
        return Chem.MolToSmiles(m) if m is not None else s
    return Chem.MolToSmiles(Chem.RemoveHs(m))


def _bde_ref_parent(keys, sp):
    cnt = _counts(sp['numbers'])
    for kk, d in _thermo_db().items():
        if kk == '_meta' or not isinstance(d, dict) or d.get('phase') != 'gas':
            continue
        if d.get('elements') == cnt and 'dHf298_kJmol' in d:
            val = sum(_ref(k)['dHf298_kJmol']['value'] for k in keys) - d['dHf298_kJmol']['value']
            return {'value': round(val, 1), 'unit': 'kJ/mol', 'kind': 'cited',
                    'note': 'from cited dHf298 of %s and %s' % (kk, ' + '.join(keys)),
                    'source': d['dHf298_kJmol'].get('source')}
    return None


# =====================================================================================================
# Rung 4
# =====================================================================================================
def eyring_half_life(dG_kJ, T=T0):
    k = KB * T / H_PLANCK * math.exp(-dG_kJ / (R_KJ * T))
    return math.log(2) / k


def _fmt_time(s):
    for unit, n in (('years', 3.156e7), ('days', 86400), ('hours', 3600), ('minutes', 60), ('s', 1)):
        if s >= n:
            return '%.3g %s' % (s / n, unit)
    return '%.2g s' % s


def rung4(ctx, sp, r3, molh, heavy):
    ev = {}
    gaps = {}
    if sp.get('gap_xtb_eV') is not None:
        gaps['xtb'] = computed(round(sp['gap_xtb_eV'], 3), 'eV', XTB_METHOD)
    if sp.get('gap_dft_eV') is not None:
        gaps['dft'] = computed(round(sp['gap_dft_eV'], 3), 'eV', DFT_METHOD)
    ev['homo_lumo_gap'] = gaps
    channels = []
    # (a) homolysis: barrier ~ bond enthalpy (radical recombination has ~no barrier)
    w = None
    if r3 and isinstance(r3.get('bonds'), dict):
        w = r3['bonds'].get('weakest')
    if w:
        channels.append({'channel': 'homolysis %s -> %s' % (w['bond'], ' + '.join(w['fragments'])),
                         'dG_barrier_kJmol': w['dH298_kJmol'], 'level': w['level'],
                         'model': 'barrier taken as the 298 K bond enthalpy (no reverse barrier; loose-TS entropy '
                                  'ignored, so the true rate can be up to ~1e3 faster)'})
    # (b) concerted fragmentation to the hull products (xTB NEB) when atoms can be partitioned without H transfer
    hull = (r3 or {}).get('hull_products')
    if hull and heavy <= NEB_HEAVY_MAX and molh is not None:
        ctx.progress('rung4', 'xTB NEB toward %s' % ', '.join(hull))
        try:
            ch = concerted_neb(ctx, sp, molh, hull)
            if ch:
                channels.append(ch)
        except Exception as ex:
            ev['neb_error'] = str(ex)
            ctx.log('NEB failed: %s' % traceback.format_exc())
    elif hull and heavy > NEB_HEAVY_MAX:
        ev['neb_note'] = 'concerted channel skipped: more than %d heavy atoms' % NEB_HEAVY_MAX
    for c in channels:
        c['half_life_298K_s'] = eyring_half_life(c['dG_barrier_kJmol']) if c['dG_barrier_kJmol'] > 0 else 0.0
        c['half_life_298K'] = _fmt_time(c['half_life_298K_s'])
    ev['channels'] = channels
    ev['eyring'] = 'k = (kB T / h) exp(-dG_barrier / RT), t1/2 = ln 2 / k, T = 298.15 K, transmission coefficient 1'
    if not channels:
        return {'verdict': 'unknown', 'summary': 'No decomposition channel could be evaluated.', 'evidence': ev,
                'label': 'ESTIMATE'}
    low = min(channels, key=lambda c: c['dG_barrier_kJmol'])
    ev['lowest'] = low['channel']
    t = low['half_life_298K_s']
    g = (gaps.get('dft') or gaps.get('xtb') or {}).get('value')
    if t >= HALF_PASS_S:
        v = 'pass'
    elif t >= HALF_WARN_S:
        v = 'warn'
    else:
        v = 'fail'
    if g is not None and g < GAP_WARN_EV and v == 'pass':
        v = 'warn'
    s = 'Lowest barrier %.0f kJ/mol (%s): half-life ~%s at 298 K.' % (low['dG_barrier_kJmol'], low['channel'],
                                                                     low['half_life_298K'])
    if g is not None:
        s += ' HOMO-LUMO gap %.2f eV.' % g
    return {'verdict': v, 'summary': s, 'evidence': ev,
            'label': 'ESTIMATE (screening: xTB barriers carry ~+/-30 kJ/mol, i.e. several orders of magnitude in t1/2)',
            'thresholds': 'pass t1/2 >= 1 year, warn >= 1 hour, else fail; gap < %g eV downgrades to warn' % GAP_WARN_EV}


def _skeleton(m):
    """Copy with every bond single and no aromatic flags, so substructure matching ignores bond order
    (an N-N single bond in the cage maps onto the N#N of a product)."""
    from rdkit import Chem
    rw = Chem.RWMol(m)
    for a in rw.GetAtoms():
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def concerted_neb(ctx, sp, molh, hull_products):
    """Partition the target's atoms into the hull's molecular products by disjoint heavy-atom substructure matches
    (no hydrogen transfer), build the separated product complex and run a climbing-image NEB on GFN2-xTB."""
    from rdkit import Chem
    from ase.mep import NEB
    from ase.optimize import FIRE
    prods = []
    for k, v in hull_products.items():
        if k in ('TARGET', 'C(gr)', 'S(s)', 'Br2(l)'):
            return None
        n = v['n']
        if abs(n - round(n)) > 1e-6:
            return None
        prods += [k] * int(round(n))
    used = set()
    groups = []
    target_skel = _skeleton(molh)
    for k in prods:
        smi = _ref(k).get('smiles')
        pm = Chem.MolFromSmiles(smi)
        q = _skeleton(Chem.RemoveHs(pm))
        ok = None
        for match in target_skel.GetSubstructMatches(q, uniquify=False, maxMatches=2000):
            if not used.intersection(match):
                ok = match
                break
        if ok is None:
            return None
        hs = [nb.GetIdx() for i in ok for nb in molh.GetAtomWithIdx(i).GetNeighbors() if nb.GetAtomicNum() == 1]
        if len(hs) != sum(a.GetTotalNumHs() for a in pm.GetAtoms()):
            return None
        grp = list(ok) + hs
        used.update(grp)
        groups.append((k, grp))
    if len(used) != molh.GetNumAtoms():
        return None
    nums = sp['numbers']
    R = np.array(sp['pos'])
    P = R.copy()
    cen = R.mean(0)
    for k, grp in groups:
        sub = [nums[i] for i in grp]
        p, e, _ = xtb_opt(sub, R[grp], 0, 1)
        c = R[grp].mean(0)
        dirv = c - cen
        dirv = dirv / (np.linalg.norm(dirv) or 1.0)
        P[grp] = p - p.mean(0) + c + 1.5 * dirv
    # relax the product complex lightly so the endpoint is a stationary-ish point on the same surface
    Pp, eP, _ = xtb_opt(nums, P, 0, 1, fmax=0.05, steps=150)
    nimg = 11
    images = [_ase_atoms(nums, R, 0, 1)]
    for _ in range(nimg - 2):
        images.append(_ase_atoms(nums, R, 0, 1))
    images.append(_ase_atoms(nums, Pp, 0, 1))
    neb = NEB(images, climb=True, method='improvedtangent', allow_shared_calculator=False)
    neb.interpolate('idpp')
    conv = bool(FIRE(neb, logfile=None).run(fmax=0.05, steps=1500))
    E = [im.get_potential_energy() for im in images]
    i = int(np.argmax(E))
    out = {'channel': 'concerted -> ' + ' + '.join(prods), 'level': 'xtb', 'neb_converged': conv,
           'xtb_barrier_kJmol': round((E[i] - E[0]) * EV_KJ, 1),
           'xtb_reaction_energy_kJmol': round((E[-1] - E[0]) * EV_KJ, 1),
           'neb_profile_kJmol': [round((e - E[0]) * EV_KJ, 1) for e in E], 'ts_image': i}
    model = 'GFN2-xTB climbing-image NEB (improved tangent, %d images, IDPP start, fmax 0.05 eV/A)' % nimg
    # Re-score the band with DFT single points when the target was treated at DFT (DFT//xTB barrier).
    if sp['level'] == 'dft':
        ctx.progress('rung4', 'DFT single points on the NEB band')
        Ed = []
        for im in images:
            mf = _dft_mf(_pyscf_mol(nums, im.get_positions(), 0, 1))
            Ed.append(float(mf.kernel()))
        j = int(np.argmax(Ed))
        out.update(dft_profile_kJmol=[round((e - Ed[0]) * HARTREE_KJ, 1) for e in Ed],
                   dft_barrier_kJmol=round((Ed[j] - Ed[0]) * HARTREE_KJ, 1),
                   dft_reaction_energy_kJmol=round((Ed[-1] - Ed[0]) * HARTREE_KJ, 1), dft_ts_image=j)
        out['dG_barrier_kJmol'] = out['dft_barrier_kJmol']
        out['level'] = 'dft//xtb'
        model += '; barrier from %s single points on the xTB band (path not re-optimised at DFT; B3LYP/double-zeta ' \
                 'tends to underestimate barriers; validation: Td N4 -> 2 N2 gives 174 vs 255 kJ/mol ab initio, ' \
                 'Lee & Rice 1991, NASA NTRS 19910039643)' % DFT_METHOD
        if abs(out['dft_reaction_energy_kJmol'] - out['xtb_reaction_energy_kJmol']) > 100:
            out['warning'] = ('xTB and DFT disagree on the reaction energy by %.0f kJ/mol: the xTB path may not '
                              'resemble the DFT one' % abs(out['dft_reaction_energy_kJmol'] - out['xtb_reaction_energy_kJmol']))
    else:
        out['dG_barrier_kJmol'] = out['xtb_barrier_kJmol']
    if not conv:
        out['warning'] = (out.get('warning', '') + '; ' if out.get('warning') else '') + \
            'NEB not converged in 1500 steps: barrier is approximate'
    out['model'] = model + '; electronic barrier (ZPE/entropy ignored)'
    return out


# =====================================================================================================
# Rung 5
# =====================================================================================================
def rung5(ctx, smiles, run_retro=True):
    from rdkit import Chem
    ev = {}
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except Exception:
        import importlib.util
        from rdkit import RDConfig
        spec = importlib.util.spec_from_file_location('sascorer', os.path.join(RDConfig.RDContribDir, 'SA_Score',
                                                                                  'sascorer.py'))
        sascorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sascorer)
    m = Chem.MolFromSmiles(smiles)
    sa = float(sascorer.calculateScore(m))
    ev['sa_score'] = computed(round(sa, 2), '1 (easy) .. 10 (hard)', 'RDKit Contrib SA_Score (Ertl & Schuffenhauer, '
                                                                  'J. Cheminform. 2009, 1, 8)')
    retro = None
    if run_retro:
        ctx.progress('rung5', 'AiZynthFinder retrosynthesis')
        retro = run_aizynth(ctx, smiles)
        ev['retrosynthesis'] = retro
    ev['askcos'] = 'not used: ASKCOS runs as a multi-service Docker deployment (MongoDB, Redis, RabbitMQ, model ' \
                   'servers); this optional backend is not bundled. Configure a supported retrosynthesis backend separately.'
    ctx.progress('rung5', 'PubChem vendor check')
    ev['pubchem_vendors'] = pubchem_vendors(ctx, smiles)
    buy = bool(retro and retro.get('target_in_stock'))
    solved = bool(retro and retro.get('solved'))
    pv0 = ev['pubchem_vendors'] if isinstance(ev['pubchem_vendors'], dict) else {}
    if buy:
        v, s = 'pass', 'Buyable: the molecule itself is in the ZINC stock.'
    elif (pv0.get('vendors') or 0) > 0 and not solved:
        v, s = 'warn', 'Possibly buyable: PubChem lists chemical-vendor sources (secondary evidence only; not found in ' \
                       'the ZINC stock or no route search was run). SA score %.2f.' % sa
    elif solved and sa <= SA_WARN:
        v, s = 'pass', 'Route found: %d step(s) from in-stock ZINC chemicals; SA score %.2f.' % (retro['steps'], sa)
    elif sa <= SA_PASS:
        v, s = 'warn', 'No route from stock found, but the SA score %.2f says it should be easy to make.' % sa
    elif sa <= SA_WARN:
        v, s = 'warn', 'No route found; SA score %.2f (moderate).' % sa
    else:
        v, s = 'fail', 'No route found and SA score %.2f (hard).' % sa
    if retro and retro.get('error'):
        s += ' (Retrosynthesis unavailable: %s.)' % retro['error']
    pv = ev['pubchem_vendors']
    if isinstance(pv, dict) and pv.get('vendors') is not None:
        s += ' PubChem lists %d chemical-vendor source(s).' % pv['vendors']
    return {'verdict': v, 'summary': s, 'evidence': ev,
            'label': 'COMPUTED (SA score) + ESTIMATE (retrosynthesis policy, USPTO-trained)',
            'thresholds': 'SA <= %g easy, <= %g moderate (ASSUMED bins, Ertl 2009 distribution)' % (SA_PASS, SA_WARN)}


from . import paths as _runtime_paths
AZ_DIR = Path(os.environ.get('ENGINEERING_SIM_RETRO_DATA') or (_runtime_paths.data_dir() / 'retrosynthesis'))
AZ_PY = Path(os.environ.get('ENGINEERING_SIM_RETRO_PYTHON') or sys.executable)


def run_aizynth(ctx, smiles):
    import subprocess
    if not AZ_PY.exists() or not (AZ_DIR / 'config.yml').is_file():
        return {'error': 'AiZynthFinder data or interpreter is unavailable; configure ENGINEERING_SIM_RETRO_DATA and ENGINEERING_SIM_RETRO_PYTHON'}
    out = ctx.job_dir / 'retro.json'
    script = Path(__file__).with_name('stability_retro.py')
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONHOME', 'PYTHONPATH')}
    try:
        p = subprocess.run([str(AZ_PY), '-B', str(script), smiles, str(AZ_DIR / 'config.yml'), str(out)],
                           capture_output=True, text=True, timeout=900, env=env, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {'error': 'retrosynthesis timed out after 900 s'}
    if not out.is_file():
        return {'error': 'retrosynthesis failed: ' + (p.stderr or p.stdout)[-400:]}
    return json.loads(out.read_text())


def pubchem_vendors(ctx, smiles):
    import urllib.parse
    import urllib.request
    cache = ctx.cache_dir / ('pubchem_vendor_%s.json' % hashlib.sha256(smiles.encode()).hexdigest()[:16])
    if cache.is_file():
        return json.loads(cache.read_text())
    try:
        u = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/cids/JSON?smiles=' + urllib.parse.quote(smiles)
        cid = json.loads(urllib.request.urlopen(u, timeout=20).read())['IdentifierList']['CID'][0]
        if not cid:
            out = {'cid': None, 'vendors': 0, 'note': 'not in PubChem'}
        else:
            u2 = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/categories/compound/%d/JSON' % cid
            j = json.loads(urllib.request.urlopen(u2, timeout=30).read())
            n = 0
            for cat in j.get('SourceCategories', {}).get('Categories', []):
                if cat.get('Category') == 'Chemical Vendors':
                    n = len(cat.get('Sources', []))
            out = {'cid': cid, 'vendors': n, 'source': u2, 'retrieved': time.strftime('%Y-%m-%d'),
                   'kind': 'cited', 'note': 'secondary flag: vendor listing, not a stock or price check'}
    except Exception as ex:
        return {'error': 'PubChem lookup failed: %s' % ex}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


# =====================================================================================================
# Job runner
# =====================================================================================================
def recipe(req):
    v = versions()
    body = {'input': {k: req.get(k) for k in ('smiles', 'xyz', 'charge', 'multiplicity')},
            'options': {k: req.get(k) for k in ('retro',)},
            'methods': {'cache_version': CACHE_VERSION, 'dft': DFT_METHOD, 'xtb': XTB_METHOD, 'imag_cm1': IMAG_CM1, 'qrrho_w0': QRRHO_W0,
                        'caps': [DFT_HEAVY_MAX, DFT_BDE_HEAVY_MAX, NEB_HEAVY_MAX],
                        'bins': [HULL_PASS, HULL_WARN, HALF_PASS_S, HALF_WARN_S, GAP_WARN_EV, SA_PASS, SA_WARN]},
            'versions': v, 'engine': 'stability_ladder/1',
            'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                            for p in (Path(__file__), Path(__file__).with_name('stability_retro.py')) if p.is_file()}}
    s = json.dumps(body, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(s.encode()).hexdigest()[:20], body


def run_job(job_dir):
    job_dir = Path(job_dir)
    req = json.loads((job_dir / 'request.json').read_text())
    state_p = job_dir / 'state.json'
    state = json.loads(state_p.read_text()) if state_p.is_file() else {}
    logf = open(job_dir / 'log.txt', 'a')

    def save():
        tmp = state_p.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, default=_plain, indent=1))
        os.replace(tmp, state_p)

    def log(msg):
        logf.write(time.strftime('%H:%M:%S ') + msg + '\n')
        logf.flush()

    def progress(rung, msg):
        state['progress'] = {'rung': rung, 'message': msg, 't': time.time()}
        log('[%s] %s' % (rung, msg))
        save()

    ctx = Ctx(job_dir, job_dir.parent / '_cache', log, progress)
    state.update(state='running', started=time.time(), pid=os.getpid())
    state.setdefault('rungs', {})
    save()
    try:
        smiles = req.get('smiles')
        xyz = parse_xyz(req['xyz']) if req.get('xyz') else None
        r1, molh = rung1(smiles) if smiles else ({'verdict': 'unknown', 'summary': 'Geometry-only input: no '
                                                  'connectivity to check.', 'evidence': {}}, None)
        state['rungs']['1'] = r1
        save()
        if smiles and r1['verdict'] == 'fail':
            raise StopIteration
        heavy = r1['evidence'].get('heavy_atoms') if smiles else sum(1 for a in xyz if a[0] != 'H')
        charge = int(req.get('charge') if req.get('charge') is not None else r1['evidence'].get('charge', 0))
        mult = int(req.get('multiplicity') or r1['evidence'].get('multiplicity', 1))
        r2, sp = rung2(ctx, molh, xyz, charge, mult, heavy)
        state['rungs']['2'] = r2
        save()
        r3extra = None
        if smiles:
            r3, r3extra = rung3(ctx, sp, smiles, molh, heavy)
        else:
            r3 = {'verdict': 'unknown', 'summary': 'Geometry-only input: no reference composition scheme.',
                  'evidence': {}}
        state['rungs']['3'] = r3
        save()
        r3ctx = {'bonds': r3['evidence'].get('bonds'),
                 'hull_products': (r3extra or {}).get('hull', {}).get('products')}
        if r2['verdict'] == 'fail':
            r4 = {'verdict': 'unknown', 'summary': 'Not evaluated: the structure is not an energy minimum (rung 2).',
                  'evidence': {}}
        else:
            r4 = rung4(ctx, sp, r3ctx, molh, heavy)
        state['rungs']['4'] = r4
        save()
        if smiles:
            state['rungs']['5'] = rung5(ctx, r1['evidence']['canonical_smiles'], run_retro=req.get('retro', True))
        else:
            state['rungs']['5'] = {'verdict': 'unknown', 'summary': 'Geometry-only input.', 'evidence': {}}
        state['state'] = 'done'
    except StopIteration:
        state['state'] = 'done'
    except Exception as ex:
        state['state'] = 'error'
        state['error'] = '%s: %s' % (type(ex).__name__, ex)
        log(traceback.format_exc())
    state['finished'] = time.time()
    state['progress'] = {'rung': None, 'message': state['state']}
    save()
    logf.close()


def _plain(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (set, tuple)):
        return list(v)
    raise TypeError(type(v))


if __name__ == '__main__':
    import fcntl
    jd = Path(sys.argv[1])
    lock = open(jd.parent / '.run.lock', 'w')
    st = jd / 'state.json'
    s = json.loads(st.read_text()) if st.is_file() else {}
    s.update(state='queued', progress={'rung': None, 'message': 'waiting for the previous stability job'})
    st.write_text(json.dumps(s))
    fcntl.flock(lock, fcntl.LOCK_EX)   # one ladder job at a time (DFT is CPU-heavy)
    run_job(jd)

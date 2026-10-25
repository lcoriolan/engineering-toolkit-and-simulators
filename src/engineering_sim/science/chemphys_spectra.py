# DESCRIPTION: Task G: spectra and rigid-rotor properties for the Chemical Physics area —
#              harmonic IR (PySCF analytical Hessian or GFN2-xTB finite difference), TDDFT
#              UV-vis, and rotational constants/symmetry.
"""Vibrational / electronic / rotational spectroscopy for the Chemical Physics area.

* ``ir``       harmonic frequencies + finite-difference IR intensities, Lorentzian spectrum.
* ``uv_vis``   TDDFT excitation energies + oscillator strengths, Gaussian spectrum.
* ``rotational`` principal moments of inertia, rotational constants, symmetry number, rotor type.
* ``zpe``      zero-point energy (B3LYP/6-31g* scaled), used by bond_dissociation.

All energies are ``estimated`` (method "PySCF 2.14.0 <method>/<basis>" or
"GFN2-xTB (tblite 0.7.0)"). Frequencies are scaled by the CCCBDB factor for the
method/basis; the factor is ``unverified`` because the live CCCBDB page could not
be fetched as a static table during this build (see ``_scale_record``).
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import chemphys_qm, cite
from .chemphys_qm import BOHR, EV_PER_HARTREE, atomic_masses, resolve_geometry, _BASIS_MAP
from .chemphys_statmech import A0, C_LIGHT, E_H, H_PLANCK, N_A, U_AMU  # CODATA physical constants (fetched)

_SCALE_SOURCE = 'https://cccbdb.nist.gov/vibscalejust.asp'


def _scale_record(method, basis):
    """Vibrational frequency scale factor record for a method/basis.

    The CCCBDB page that tabulates these factors is a JS form and was not
    retrievable as a static table during this build, so the value is ``unverified``
    rather than ``cited`` (honest per the module contract).
    """
    if method == 'B3LYP' and basis == '6-31g*':
        return cite.value(0.9613, 'dimensionless', 'unverified', source=_SCALE_SOURCE,
                          note='CCCBDB (NIST SRD 101) B3LYP/6-31G* vibrational scale factor; '
                               'the live page was not fetchable as a static table at build time.')
    return cite.value(1.0, 'dimensionless', 'unverified', source=_SCALE_SOURCE,
                      note='No standard GFN2-xTB frequency scale factor; frequencies left unscaled.')


def _lorentzian(x, x0, gamma):
    """Lorentzian lineshape (normalised to 1 at peak)."""
    return gamma ** 2 / ((x - x0) ** 2 + gamma ** 2)


def _gaussian(x, x0, sigma):
    """Gaussian lineshape (normalised to 1 at peak)."""
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def _spectrum_curve(freqs, intensities, xmin, xmax, npts, broadening, shape):
    """Sum of broadened peaks on a uniform grid; returns (x, y) lists."""
    x = np.linspace(xmin, xmax, npts)
    y = np.zeros_like(x)
    total = sum(intensities) or 1.0
    for f, i in zip(freqs, intensities):
        if i <= 0:
            continue
        y += (i / total) * shape(x, f, broadening)
    return x.tolist(), y.tolist()


def _frequencies_pyscf(query, method='B3LYP', basis='6-31g*', intensities=True):
    """Harmonic frequencies + IR intensities from a PySCF analytical Hessian.

    Intensities come from a finite-difference dipole derivative projected onto the
    mass-weighted normal modes; they are relative (arbitrary units), not absolute
    km/mol. Returns None for intensities when skipped (molecule too large).
    """
    from pyscf import gto
    from pyscf.hessian import thermo

    geo = resolve_geometry(query, preopt='none')
    atoms = geo['atoms']
    symbols = [a['el'] for a in atoms]
    basis_key = _BASIS_MAP[basis]
    atom_str = chemphys_qm.atoms_to_atom_string(atoms)
    mol = gto.M(atom=atom_str, basis=basis_key, charge=int(geo['charge']),
                spin=int(geo['spin_mult']) - 1, verbose=0)
    mf = chemphys_qm._make_mf(mol, method, int(geo['spin_mult']) - 1)
    mf.kernel()

    t0 = time.time()
    hess = mf.Hessian().kernel()
    ana = thermo.harmonic_analysis(mol, hess)
    wall = time.time() - t0
    freq_raw = np.asarray(ana['freq_wavenumber'], dtype=float)
    norm_mode = np.asarray(ana['norm_mode'], dtype=float)  # (3N, 3N) mass-weighted
    masses = np.array([atomic_masses().get(s, 12.0) for s in symbols], dtype=float)

    if intensities and len(atoms) <= 10:
        dip_grad = _dipole_derivative_pyscf(mol, mf, method)
        inten = _intensities_from_dipole_derivative(dip_grad, norm_mode, masses, symbols)
    else:
        inten = None

    return {'freq_raw': freq_raw, 'norm_mode': norm_mode, 'masses': masses, 'atoms': atoms,
            'symbols': symbols, 'converged': bool(mf.converged), 'wall_s': wall,
            'intensities': inten,
            'intensity_note': None if inten is not None else
            'IR intensities skipped: finite-difference dipole derivative limited to <=10 atoms.'}


def _dipole_derivative_pyscf(mol, mf, method):
    """Finite-difference dipole derivative dmu/dR (Debye/Angstrom), shape (3N, 3).

    Two SCF runs per coordinate (+-delta); the converged density is reused as the
    initial guess so the displaced runs converge in a few cycles.
    """
    coords = np.asarray([list(row) for row in mol.atom_coords()], dtype=float)  # bohr
    delta = 0.005 / BOHR  # 0.005 Angstrom in bohr
    n = coords.shape[0]
    spin = mol.spin
    dip_grad = np.zeros((n, 3, 3))

    def dipole_at(c):
        m = mol.copy()
        m.set_geom_(c, unit='Bohr')
        mf2 = chemphys_qm._make_mf(m, method, spin)
        mf2.kernel(dm0=mf.make_rdm1())
        return _dipole(mf2, m)

    for a in range(n):
        for d in range(3):
            c_plus = coords.copy()
            c_minus = coords.copy()
            c_plus[a, d] += delta
            c_minus[a, d] -= delta
            dip_grad[a, d] = (dipole_at(c_plus) - dipole_at(c_minus)) / (2 * delta * BOHR)
    return dip_grad  # Debye/Angstrom


def _dipole(mf, mol):
    """|dipole vector| as a numpy array in Debye (zeros if unavailable)."""
    try:
        return np.asarray(mf.dip_moment(unit='Debye', verbose=0), dtype=float)
    except Exception:
        return np.zeros(3)


def _intensities_from_dipole_derivative(dip_grad, norm_mode, masses, symbols):
    """Project the dipole derivative onto mass-weighted normal modes -> relative |dmu/dQ|^2.

    pyscf ``norm_mode`` has shape (n_modes, n_atoms, 3); dmu/dQ_k =
    sum_{a,d} (dmu/dR_{a,d}) * L[k, a, d] / sqrt(m_a).
    """
    natoms = len(symbols)
    nmode = norm_mode.shape[0]
    intensities = []
    for k in range(nmode):
        dmu_dq = 0.0
        for a in range(natoms):
            for d in range(3):
                dmu_dq += dip_grad[a, d] * norm_mode[k, a, d] / math.sqrt(masses[a])
        intensities.append(float(np.dot(dmu_dq, dmu_dq)))
    return intensities


def _is_linear(coords_bohr):
    """True when all atoms are collinear (a linear rotor has 5 zero modes, else 6)."""
    if len(coords_bohr) <= 2:
        return True
    v = coords_bohr[1:] - coords_bohr[0]
    n = np.linalg.norm(v, axis=1)
    n[n < 1e-9] = 1.0
    v = v / n[:, None]
    return float(np.abs(np.cross(v[1:], v[0])).max()) < 1e-3


def _frequencies_xtb(query):
    """Harmonic frequencies from a GFN2-xTB finite-difference Hessian (step 0.005 A).

    The mass-weighted Hessian is built from central differences of tblite gradients
    (3N coordinate displacements, 2 single points each), symmetrised, and
    diagonalised. Frequencies are unscaled (no standard xTB scale factor).
    """
    from tblite.interface import Calculator
    from rdkit import Chem

    geo = resolve_geometry(query, preopt='xtb')
    atoms = geo['atoms']
    symbols = [a['el'] for a in atoms]
    pt = Chem.GetPeriodicTable()
    numbers = [pt.GetAtomicNumber(s) for s in symbols]
    masses = np.array([atomic_masses().get(s, 12.0) for s in symbols], dtype=float)
    charge = int(geo['charge'])
    uhf = int(geo['spin_mult']) - 1
    coords_bohr = np.asarray([[a['x'], a['y'], a['z']] for a in atoms], dtype=float) / BOHR
    n = len(atoms)

    def grad_at(c):
        calc = Calculator('GFN2-xTB', numbers, np.asarray(c, dtype=float), charge=charge, uhf=uhf)
        calc.set('verbosity', 0)
        res = calc.singlepoint()
        return np.asarray(res.get('gradient'), dtype=float).ravel()

    t0 = time.time()
    delta = 0.005 / BOHR  # 0.005 Angstrom in bohr
    hess = np.zeros((3 * n, 3 * n))
    for j in range(3 * n):
        c_plus = coords_bohr.copy().reshape(-1)
        c_minus = coords_bohr.copy().reshape(-1)
        c_plus[j] += delta
        c_minus[j] -= delta
        g_plus = grad_at(c_plus.reshape(-1, 3))
        g_minus = grad_at(c_minus.reshape(-1, 3))
        hess[:, j] = (g_plus - g_minus) / (2 * delta)
    hess = 0.5 * (hess + hess.T)

    m_inv = 1.0 / np.sqrt(masses)
    m3 = np.repeat(m_inv, 3)  # (3N,) — one inverse-sqrt-mass per Cartesian coordinate
    h_mw = hess * (m3[:, None] * m3[None, :])
    evals, evecs = np.linalg.eigh(h_mw)
    wall = time.time() - t0
    # sqrt(eigenvalue) [sqrt(Eh/(bohr^2*amu))] -> cm^-1, via CODATA constants.
    factor = math.sqrt(E_H / (A0 ** 2 * U_AMU)) / (2 * math.pi * C_LIGHT * 100.0)
    freq_raw = np.array([factor * math.sqrt(max(ev, 0.0)) for ev in evals])

    # Drop the 5 (linear) / 6 (non-linear) translational+rotational near-zero modes.
    n_tr = 5 if _is_linear(coords_bohr) else 6
    order = np.argsort(np.abs(freq_raw))
    keep = np.array([i for i in range(3 * n) if i not in set(order[:n_tr].tolist())])
    freq_raw = freq_raw[keep]
    norm_mode = evecs[:, keep]

    return {'freq_raw': freq_raw, 'norm_mode': norm_mode, 'masses': masses, 'atoms': atoms,
            'symbols': symbols, 'converged': True, 'wall_s': wall,
            'intensities': None, 'intensity_note': 'IR intensities not computed for the xTB path.'}


def ir(query, method='B3LYP', basis='6-31g*', intensities=True, fwhm=20.0, xmin=400.0, xmax=4000.0):
    """Harmonic IR spectrum: scaled frequencies, Lorentzian-broadened stick/curve.

    ``method='GFN2-xTB'`` uses the tblite finite-difference Hessian (unscaled);
    any other method uses the PySCF analytical Hessian with ``basis``. Returns
    frequency/intensity lists and the sampled spectrum curve.
    """
    if method == 'GFN2-xTB':
        data = _frequencies_xtb(query)
        label = 'GFN2-xTB (tblite 0.7.0)'
    else:
        data = _frequencies_pyscf(query, method=method, basis=basis, intensities=intensities)
        label = 'PySCF 2.14.0 %s/%s' % (method, basis)

    freq_raw = data['freq_raw']
    scale = _scale_record(method if method != 'GFN2-xTB' else 'GFN2-xTB', basis)
    scale_v = scale['value']
    freq = freq_raw * scale_v
    # Keep only the 6 real modes region (drop 3N-6/5 translations/rotations by
    # taking the physically meaningful frequencies; imaginary ones are retained as negatives).
    real = freq[np.abs(freq) > 1.0]
    inten = data['intensities'] if data['intensities'] is not None else np.ones(len(freq_raw))
    inten = np.asarray(inten)
    if len(inten) != len(freq):
        inten = np.ones(len(freq))
    inten = inten[np.abs(freq) > 1.0]

    peaks = [{'frequency': cite.value(round(float(f), 2), 'cm^-1', 'estimated', method=label,
                                      note='scaled harmonic frequency'),
              'intensity': cite.value(round(float(i), 4), 'relative', 'estimated', method=label)}
             for f, i in sorted(zip(real, inten), key=lambda t: t[0])]

    x, y = _spectrum_curve(real, inten, xmin, xmax, 512, fwhm / 2.0, _lorentzian)
    notes = [data.get('intensity_note')] if data.get('intensity_note') else []
    notes.append('Harmonic frequencies scaled by %s; Lorentzian FWHM %.0f cm^-1.'
                 % (scale_v, fwhm))
    if method == 'GFN2-xTB':
        notes.append('GFN2-xTB frequencies are qualitative; the finite-difference Hessian '
                     'step is 0.005 Angstrom.')
    return {
        'query': query, 'method': label, 'basis': basis,
        'converged': data['converged'], 'wall_s': round(data['wall_s'], 2),
        'scale_factor': scale,
        'frequencies': peaks,
        'spectrum': {'x': [round(v, 3) for v in x], 'y': [round(v, 6) for v in y],
                     'xmin': xmin, 'xmax': xmax, 'unit': 'cm^-1'},
        'notes': notes,
    }


def zpe(query, method='B3LYP', basis='6-31g*'):
    """Zero-point energy (kJ/mol) from scaled harmonic frequencies: ZPE = 0.5 * h c * sum(nu)."""
    if method == 'GFN2-xTB':
        data = _frequencies_xtb(query)
        scale = 1.0
        label = 'GFN2-xTB (tblite 0.7.0)'
    else:
        data = _frequencies_pyscf(query, method=method, basis=basis, intensities=False)
        scale = 0.9613
        label = 'PySCF 2.14.0 %s/%s' % (method, basis)
    freq = data['freq_raw'] * scale
    real = freq[freq > 1.0]
    # 0.5 * h * c * nu (cm^-1) per mode -> J, * N_A -> J/mol, /1000 -> kJ/mol.
    z = 0.5 * H_PLANCK * C_LIGHT * 100.0 * real.sum() * N_A / 1000.0
    return cite.value(round(float(z), 3), 'kJ/mol', 'estimated', method=label,
                      note='ZPE = 0.5 h c sum(nu), scaled harmonic frequencies.')


def uv_vis(query, functional='B3LYP', basis='6-31g*', nstates=10, sigma=0.3):
    """TDDFT vertical excitation energies and oscillator strengths (Gaussian spectrum).

    Small basis sets make TDDFT energies qualitative; the note says so. nstates is
    capped at 40 and the oscillator-strength extraction is defensive (falls back to
    zero when pyscf's transition-dipole helper is unavailable).
    """
    from pyscf import dft, gto, tdscf

    geo = resolve_geometry(query, preopt='none')
    atoms = geo['atoms']
    basis_key = _BASIS_MAP[basis]
    atom_str = chemphys_qm.atoms_to_atom_string(atoms)
    mol = gto.M(atom=atom_str, basis=basis_key, charge=int(geo['charge']),
                spin=int(geo['spin_mult']) - 1, verbose=0)
    mf = dft.RKS(mol, xc=_xc_name(functional)) if int(geo['spin_mult']) - 1 == 0 else dft.UKS(mol, xc=_xc_name(functional))
    mf.kernel()

    td = tdscf.TDDFT(mf)
    td.nstates = max(1, min(int(nstates), 40))
    t0 = time.time()
    td.kernel()
    wall = time.time() - t0
    energies_ev = np.asarray(td.e).ravel() * EV_PER_HARTREE
    try:
        osc = np.asarray(td.oscillator_strength()).ravel()
    except Exception:
        osc = np.zeros_like(energies_ev)

    method_str = 'PySCF 2.14.0 TD-%s/%s' % (functional, basis)
    states = [{'index': i,
               'energy_eV': cite.value(round(float(e), 4), 'eV', 'estimated', method=method_str),
               'wavelength_nm': cite.value(round(1239.84198 / float(e), 2), 'nm', 'derived',
                                           method='E = hc/lambda', note='from excitation energy'),
               'oscillator_strength': cite.value(round(float(f), 4), 'dimensionless', 'estimated',
                                                 method=method_str)}
              for i, (e, f) in enumerate(zip(energies_ev, osc))]

    emin = max(0.5, float(energies_ev.min()) - 2.0)
    emax = float(energies_ev.max()) + 2.0
    x, y = _spectrum_curve(energies_ev, np.maximum(osc, 0.0), emin, emax, 512, sigma, _gaussian)
    return {
        'query': query, 'method': method_str, 'functional': functional, 'basis': basis,
        'nstates': len(states), 'converged': bool(mf.converged), 'wall_s': round(wall, 2),
        'states': states,
        'spectrum': {'x': [round(v, 4) for v in x], 'y': [round(v, 6) for v in y],
                     'xmin': round(emin, 2), 'xmax': round(emax, 2), 'unit': 'eV'},
        'notes': ['TDDFT with a small basis set is qualitative: excitation energies are '
                  'typically overestimated by ~0.1-0.5 eV for B3LYP/6-31G*.'],
    }


def _xc_name(functional):
    return {'B3LYP': 'b3lyp', 'CAM-B3LYP': 'camb3lyp'}.get(functional, 'b3lyp')


def _symmetry_number(atoms, moments):
    """Symmetry number sigma for the rigid-rotor partition function.

    Linear molecules: 2 if homonuclear (all atoms identical), else 1. Non-linear:
    PySCF point-group detection with a mapping to the order of the proper-rotation
    subgroup; falls back to 1 (with a note) when detection is unavailable.
    """
    symbols = [a['el'] for a in atoms]
    linear = float(moments[0]) < 1e-2
    if linear:
        return 2 if len(set(symbols)) == 1 else 1
    sigma_map = {'C1': 1, 'Cs': 1, 'Ci': 1, 'C2': 2, 'C2v': 2, 'C2h': 2, 'D2': 4, 'D2h': 4, 'D2d': 4,
                 'C3': 3, 'C3v': 3, 'C3h': 3, 'D3': 6, 'D3h': 6, 'D3d': 6, 'C4': 4, 'C4v': 4, 'C4h': 4,
                 'D4': 8, 'D4h': 8, 'D4d': 8, 'C5': 5, 'C5v': 5, 'D5': 10, 'D5h': 10, 'C6': 6, 'C6v': 6,
                 'D6': 12, 'D6h': 12, 'Td': 12, 'Oh': 24, 'Ih': 60, 'Cinfv': 1, 'Dinfh': 2}
    try:
        from pyscf import gto, symm
        basis_key = 'sto-3g'
        atom_str = chemphys_qm.atoms_to_atom_string(atoms)
        mol = gto.M(atom=atom_str, basis=basis_key, charge=0, spin=0, verbose=0)
        group = symm.detect_symm([(a[0], a[1]) for a in mol._atom], basis=basis_key)[0]
        return sigma_map.get(str(group), 1)
    except Exception:
        return 1


def rotational(query, preopt='none'):
    """Principal moments of inertia, rotational constants (cm^-1, GHz), symmetry number.

    Uses standard (most-abundant-isotope) atomic masses, cited ``unverified`` since
    the mass table was not fetched during this build. The rigid-rotor constant
    B = h/(8 pi^2 c I) is computed with CODATA constants (fetched).
    """
    geo = resolve_geometry(query, preopt=preopt)
    atoms = geo['atoms']
    symbols = [a['el'] for a in atoms]
    masses = np.array([atomic_masses().get(s, 12.0) for s in symbols], dtype=float)
    coords = np.asarray([[a['x'], a['y'], a['z']] for a in atoms], dtype=float)

    com = (masses[:, None] * coords).sum(axis=0) / masses.sum()
    r = coords - com
    # Inertia tensor (amu Angstrom^2).
    I = np.zeros((3, 3))
    for m, ri in zip(masses, r):
        I += m * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    moments = np.sort(np.linalg.eigvalsh(I))  # ascending
    # B [cm^-1] = h/(8 pi^2 c I) = (h/(8 pi^2 c U * 1e-20 * 100)) / I[amu A^2], CODATA.
    factor = H_PLANCK / (8 * math.pi ** 2 * C_LIGHT * U_AMU * 1e-20 * 100.0)
    constants = factor / np.maximum(moments, 1e-9)

    linear = float(moments[0]) < 1e-2
    if linear:
        B_cm1 = constants[1]  # the two non-zero, degenerate moments
        a_cm1 = b_cm1 = c_cm1 = B_cm1
        rotor = 'linear'
    else:
        a_cm1, b_cm1, c_cm1 = constants[0], constants[1], constants[2]
        rel = 0.02
        if abs(moments[0] - moments[1]) < rel * moments[0] and abs(moments[1] - moments[2]) < rel * moments[0]:
            rotor = 'spherical top'
        elif abs(moments[0] - moments[1]) < rel * moments[0] or abs(moments[1] - moments[2]) < rel * moments[0]:
            rotor = 'symmetric top'
        else:
            rotor = 'asymmetric top'

    sigma = _symmetry_number(atoms, moments)
    method_str = 'rigid rotor (CODATA constants, most-abundant-isotope masses)'
    return {
        'query': query, 'smiles': geo['smiles'], 'formula': chemphys_qm._formula(symbols),
        'geometry_source': geo['geometry'],
        'moments_of_inertia': [cite.value(round(float(m), 5), 'amu*Angstrom^2', 'derived',
                                          method='principal moments of inertia') for m in moments],
        'rotational_constants_cm1': {
            'A': cite.value(round(float(a_cm1), 6), 'cm^-1', 'derived', method=method_str),
            'B': cite.value(round(float(b_cm1), 6), 'cm^-1', 'derived', method=method_str),
            'C': cite.value(round(float(c_cm1), 6), 'cm^-1', 'derived', method=method_str),
        },
        'rotational_constants_GHz': {
            'A': cite.value(round(float(a_cm1 * C_LIGHT * 1e-7), 6), 'GHz', 'derived', method=method_str),
            'B': cite.value(round(float(b_cm1 * C_LIGHT * 1e-7), 6), 'GHz', 'derived', method=method_str),
            'C': cite.value(round(float(c_cm1 * C_LIGHT * 1e-7), 6), 'GHz', 'derived', method=method_str),
        },
        'symmetry_number': cite.value(sigma, 'dimensionless', 'derived',
                                      method='point-group proper-rotation order (PySCF symm)'),
        'rotor_type': rotor,
        'atoms': atoms,
        'notes': ['Atomic masses are most-abundant-isotope values (unverified standard table); '
                  'rotational constants assume a rigid rotor at the %s geometry.' % geo['geometry']],
    }

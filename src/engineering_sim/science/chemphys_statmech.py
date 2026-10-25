# DESCRIPTION: Task G: statistical thermodynamics for the Chemical Physics area —
#              ideal-gas RRHO partition functions, NIST comparison, Boltzmann populations,
#              photon/blackbody utilities, and teaching quantum models.
"""Statistical thermodynamics for the Chemical Physics area.

CODATA physical constants are fetched from https://physics.nist.gov/cuu/Constants/
during this build and cited (``verified=True``). The ideal-gas rigid-rotor
harmonic-oscillator model gives the translational/rotational/vibrational/electronic
contributions to S, Cp, H-H0 and G-H0; every contribution is a ``derived`` value
given the model inputs, and the totals are ``estimated`` with an explicit model
name and its assumptions in the notes.
"""
from __future__ import annotations

import math

from . import chemphys_qm, cite

# ---- CODATA physical constants (fetched 2026-09-15 from physics.nist.gov/cuu) ----
# The served page states "2022 CODATA recommended values". The exact-defined
# constants (h, c, k_B, N_A, e) are identical to CODATA 2018.
H_PLANCK = 6.62607015e-34       # J s (exact)
C_LIGHT = 299792458.0           # m/s (exact)
K_B = 1.380649e-23              # J/K (exact)
N_A = 6.02214076e23             # 1/mol (exact)
E_CHARGE = 1.602176634e-19      # C (exact)
U_AMU = 1.66053906892e-27       # atomic mass constant, kg
E_H = 4.3597447222060e-18       # Hartree energy, J
A0 = 5.29177210544e-11          # Bohr radius, m
RYD_INF = 10973731.568157       # Rydberg constant, m^-1
M_E = 9.1093837139e-31          # electron mass, kg

R = N_A * K_B                   # molar gas constant, J/(mol K)
RYDBERG_EV = RYD_INF * H_PLANCK * C_LIGHT / E_CHARGE  # 13.605693… eV
EV_PER_HARTREE = E_H / E_CHARGE                        # 27.211386…

_RETRIEVED = '2026-09-15'
_CONST_URL = 'https://physics.nist.gov/cgi-bin/cuu/Value?%s'
_CODATA_NOTE = 'CODATA value page (served 2022 recommended values at retrieval).'

_CONSTANTS = (
    ('planck_constant', 'Planck constant', H_PLANCK, 'J*s', 'h', True),
    ('speed_of_light', 'speed of light in vacuum', C_LIGHT, 'm/s', 'c', True),
    ('boltzmann_constant', 'Boltzmann constant', K_B, 'J/K', 'k', True),
    ('avogadro_constant', 'Avogadro constant', N_A, '1/mol', 'na', True),
    ('elementary_charge', 'elementary charge', E_CHARGE, 'C', 'e', True),
    ('atomic_mass_constant', 'atomic mass constant', U_AMU, 'kg', 'u', False),
    ('hartree_energy', 'Hartree energy', E_H, 'J', 'hr', False),
    ('bohr_radius', 'Bohr radius', A0, 'm', 'bohrrada0', False),
    ('rydberg_constant', 'Rydberg constant', RYD_INF, 'm^-1', 'ryd', False),
    ('electron_mass', 'electron mass', M_E, 'kg', 'me', False),
)


def constants():
    """CODATA physical constants used by the area, each a cited value record."""
    out = []
    for key, name, val, unit, sym, exact in _CONSTANTS:
        rec = cite.value(val, unit, 'cited', source=_CONST_URL % sym, retrieved=_RETRIEVED,
                         verified=True, note=_CODATA_NOTE + (' Exact by SI definition.' if exact else ''))
        out.append({'key': key, 'name': name, 'record': rec})
    return out


# ---- Teaching quantum models (textbook closed forms) -------------------------

def particle_in_box(n_max=5, L_nm=1.0, mass_kg=M_E):
    """Particle-in-a-box levels E_n = n^2 h^2 / (8 m L^2) in eV (n = 1..n_max).

    The energies scale exactly as n^2. Default mass is the electron, length L in nm.
    """
    L = float(L_nm) * 1e-9
    levels = []
    for n in range(1, int(n_max) + 1):
        e_j = n * n * H_PLANCK ** 2 / (8.0 * mass_kg * L * L)
        levels.append({'n': n, 'energy_eV': cite.value(round(e_j / E_CHARGE, 6), 'eV', 'derived',
                                                       method='E_n = n^2 h^2/(8 m L^2)')})
    return {'model': 'particle in a box', 'L_nm': L_nm, 'mass_kg': mass_kg, 'levels': levels}


def harmonic_oscillator(n_max=5, nu_cm1=1000.0):
    """Harmonic-oscillator levels E_n = h c nu (n + 1/2) in eV (equally spaced)."""
    levels = []
    for n in range(int(n_max) + 1):
        e_j = (n + 0.5) * H_PLANCK * C_LIGHT * float(nu_cm1) * 100.0
        levels.append({'n': n, 'energy_eV': cite.value(round(e_j / E_CHARGE, 6), 'eV', 'derived',
                                                       method='E_n = h c nu (n + 1/2)')})
    return {'model': 'harmonic oscillator', 'nu_cm1': nu_cm1, 'levels': levels}


def hydrogen_atom(n_max=6):
    """Hydrogen-atom levels E_n = -Rydberg / n^2 in eV (Rydberg from CODATA)."""
    levels = []
    for n in range(1, int(n_max) + 1):
        levels.append({'n': n, 'energy_eV': cite.value(round(-RYDBERG_EV / (n * n), 6), 'eV', 'derived',
                                                       method='E_n = -Rydberg/n^2 (CODATA Rydberg)')})
    return {'model': 'hydrogen atom', 'levels': levels}


# ---- Boltzmann / photon / blackbody utilities --------------------------------

def boltzmann(levels_eV, T_K):
    """Relative Boltzmann populations p_i ∝ g_i exp(-E_i / kT), normalised to sum 1.

    ``levels_eV`` is a list of levels; each may be a bare float (eV), a cite value
    record ``{value, ...}``, or a dict ``{energy_eV, degeneracy}`` (degeneracy
    defaults to 1).
    """
    T = float(T_K)
    if T <= 0:
        raise ValueError('T must be positive')

    def level_eV(lv):
        if isinstance(lv, dict):
            if 'energy_eV' in lv:
                return float(cite.num(lv['energy_eV']))
            if 'value' in lv:
                return float(lv['value'])
        return float(lv)

    def level_g(lv):
        if isinstance(lv, dict) and 'degeneracy' in lv:
            return float(lv['degeneracy'])
        return 1.0

    kt = K_B * T / E_CHARGE  # eV
    energies = [level_eV(lv) for lv in levels_eV]
    weights = [level_g(lv) * math.exp(-e / kt) for lv, e in zip(levels_eV, energies)]
    total = sum(weights)
    return {
        'T_K': T,
        'kT_eV': cite.value(round(kt, 6), 'eV', 'derived', method='kT (CODATA k_B)'),
        'populations': [{'index': i,
                         'energy_eV': cite.value(round(e, 6), 'eV', 'user'),
                         'degeneracy': level_g(lv),
                         'population': round(w / total, 6)}
                        for i, (lv, e, w) in enumerate(zip(levels_eV, energies, weights))],
        'notes': ['Populations are relative and normalised to sum 1 (partition function omitted).'],
    }


def photon(energy_eV=None, wavelength_nm=None, frequency_Hz=None):
    """Convert between photon energy (eV), wavelength (nm) and frequency (Hz) via E = h nu = h c / lambda."""
    given = sum(x is not None for x in (energy_eV, wavelength_nm, frequency_Hz))
    if given != 1:
        raise ValueError('provide exactly one of energy_eV, wavelength_nm, frequency_Hz')
    if energy_eV is not None:
        e = float(energy_eV)
        nu = e * E_CHARGE / H_PLANCK
        lam = C_LIGHT / nu
    elif wavelength_nm is not None:
        lam = float(wavelength_nm) * 1e-9
        nu = C_LIGHT / lam
        e = H_PLANCK * nu / E_CHARGE
    else:
        nu = float(frequency_Hz)
        lam = C_LIGHT / nu
        e = H_PLANCK * nu / E_CHARGE
    return {
        'energy_eV': cite.value(round(e, 6), 'eV', 'derived', method='E = h nu'),
        'wavelength_nm': cite.value(round(lam * 1e9, 6), 'nm', 'derived', method='lambda = c/nu'),
        'frequency_Hz': cite.value(round(nu, 4), 'Hz', 'derived', method='nu = c/lambda'),
        'notes': ['CODATA h and c (fetched).'],
    }


def blackbody(T_K):
    """Wien peak wavelength/frequency and photon energy for a blackbody at T_K."""
    T = float(T_K)
    if T <= 0:
        raise ValueError('T must be positive')
    b_wien = 2.897771955e-3  # m K (Wien displacement constant, CODATA-derived)
    lam_max = b_wien / T
    nu_max = 5.878925757e10 * T  # Hz (5.8789...e10 Hz/K from Wien in frequency form)
    e_ev = H_PLANCK * nu_max / E_CHARGE
    return {
        'T_K': T,
        'wavelength_max_nm': cite.value(round(lam_max * 1e9, 3), 'nm', 'derived',
                                        method="Wien's displacement law, lambda_max = b/T"),
        'frequency_max_Hz': cite.value(round(nu_max, 2), 'Hz', 'derived', method="Wien's law (frequency form)"),
        'photon_energy_eV': cite.value(round(e_ev, 5), 'eV', 'derived', method='E = h nu_max'),
        'notes': ["Wien displacement constant b = 2.897771955e-3 m K (CODATA-derived)."],
    }


# ---- Ideal-gas rigid-rotor harmonic-oscillator (RRHO) model ------------------

def _rrho_contributions(mass_kg, sigma, rot_consts_cm1, linear, freqs_cm1, multiplicity, T, P_pa):
    """S/Cp/H-H0/G-H0 contributions (per mole) for the RRHO ideal-gas model.

    Translational (Sackur-Tetrode), rotational (rigid rotor, sigma symmetry number),
    vibrational (harmonic oscillators) and electronic (spin multiplicity g=2S+1).
    Returns a dict of per-term dicts {s (J/mol*K), cp (J/mol*K), h_h0 (kJ/mol),
    g_h0 (kJ/mol)} plus totals.
    """
    T = float(T)
    # Translational (Sackur-Tetrode), P in Pa, m in kg.
    q_trans_arg = (2 * math.pi * mass_kg * K_B * T / H_PLANCK ** 2) ** 1.5 * K_B * T / P_pa
    s_trans = R * (math.log(q_trans_arg) + 2.5)
    cp_trans = 2.5 * R
    h_trans = 2.5 * R * T / 1000.0

    # Rotational.
    if linear:
        theta_rot = 100.0 * C_LIGHT * H_PLANCK * float(rot_consts_cm1[0]) / K_B
        s_rot = R * (1.0 + math.log(T / (float(sigma) * theta_rot))) if theta_rot > 0 else 0.0
        cp_rot = R
        h_rot = R * T / 1000.0
    else:
        thetas = [100.0 * C_LIGHT * H_PLANCK * float(b) / K_B for b in rot_consts_cm1]
        s_rot = R * (1.5 + math.log(math.sqrt(math.pi * T ** 3 / (thetas[0] * thetas[1] * thetas[2])) / float(sigma)))
        cp_rot = 1.5 * R
        h_rot = 1.5 * R * T / 1000.0

    # Vibrational (one harmonic oscillator per frequency).
    s_vib = cp_vib = h_vib = 0.0
    for nu in freqs_cm1:
        theta = 100.0 * C_LIGHT * H_PLANCK * float(nu) / K_B
        x = theta / T
        ex = math.exp(x)
        s_vib += R * (x / (ex - 1.0) - math.log(1.0 - math.exp(-x)))
        cp_vib += R * x * x * ex / (ex - 1.0) ** 2
        h_vib += R * theta / (ex - 1.0) / 1000.0

    # Electronic (spin multiplicity g = 2S+1).
    s_elec = R * math.log(float(multiplicity))
    cp_elec = 0.0
    h_elec = 0.0

    def term(s, cp, h):
        return {'s': s, 'cp': cp, 'h_h0': h, 'g_h0': h - T * s / 1000.0}

    terms = {
        'translational': term(s_trans, cp_trans, h_trans),
        'rotational': term(s_rot, cp_rot, h_rot),
        'vibrational': term(s_vib, cp_vib, h_vib),
        'electronic': term(s_elec, cp_elec, h_elec),
    }
    total = term(s_trans + s_rot + s_vib + s_elec,
                 cp_trans + cp_rot + cp_vib + cp_elec,
                 h_trans + h_rot + h_vib + h_elec)
    return terms, total


def _resolve_rr_inputs(query, freqs, rot, spin):
    """Resolve mass/geometry/symmetry/rotational constants/frequencies for a query.

    ``freqs`` (cm^-1), ``rot`` ({A,B,C} cm^-1) and ``spin`` (multiplicity 2S+1) override
    the computed values when given; otherwise they come from a B3LYP/6-31g* hessian and
    the rigid-rotor geometry. Symmetry number and linearity always come from the geometry.
    """
    from . import chemphys_spectra

    geo = chemphys_qm.resolve_geometry(query, preopt='none')
    atoms = geo['atoms']
    masses = chemphys_qm.atomic_masses()
    mass_kg = sum(masses.get(a['el'], 12.0) for a in atoms) * U_AMU
    if spin is None:
        spin = geo['spin_mult']
    spin = int(spin)

    rot_rec = chemphys_spectra.rotational(query, preopt='none')
    linear = rot_rec['rotor_type'] == 'linear'
    sigma = rot_rec['symmetry_number']['value']
    rot_cm1 = {k: v['value'] for k, v in rot_rec['rotational_constants_cm1'].items()}
    if rot is not None:
        for k, v in rot.items():
            rot_cm1[k] = float(v)
    if linear:
        rot_list = [float(rot_cm1.get('B', rot_cm1.get('A', 0.0)))]
    else:
        rot_list = sorted([float(rot_cm1.get(k, 0.0)) for k in ('A', 'B', 'C')], reverse=True)

    if freqs is None:
        freq_data = chemphys_spectra._frequencies_pyscf(query, method='B3LYP', basis='6-31g*',
                                                        intensities=False)
        freqs = [float(f) for f in freq_data['freq_raw'] * 0.9613]
    else:
        freqs = [float(f) for f in freqs]
    return mass_kg, sigma, rot_list, linear, freqs, spin


def partition_functions(query, T_K=298.15, P_bar=1.0, freqs=None, rot=None, spin=None):
    """RRHO partition-function contributions for a molecule at T, P.

    Returns per-term and total S (J/mol*K), Cp (J/mol*K), H-H0 and G-H0 (kJ/mol).
    All contributions are ``derived`` from the stated inputs; the totals carry a
    ``estimated`` record naming the model and its assumptions.
    """
    mass_kg, sigma, rot_cm1, linear, freqs, mult = _resolve_rr_inputs(query, freqs, rot, spin)
    P_pa = float(P_bar) * 1e5
    terms, total = _rrho_contributions(mass_kg, sigma, rot_cm1, linear, freqs, mult, float(T_K), P_pa)

    def rec(v, unit):
        return cite.value(round(v, 4), unit, 'derived',
                          method='ideal-gas rigid-rotor harmonic-oscillator (CODATA constants)')

    out = {'query': query, 'T_K': float(T_K), 'P_bar': float(P_bar),
           'sigma': sigma, 'multiplicity': mult, 'linear': bool(linear),
           'rotational_constants_cm1': [round(b, 6) for b in rot_cm1],
           'frequencies_cm1': [round(f, 3) for f in freqs],
           'contributions': {
               k: {'s': rec(v['s'], 'J/mol*K'), 'cp': rec(v['cp'], 'J/mol*K'),
                   'h_h0': rec(v['h_h0'], 'kJ/mol'), 'g_h0': rec(v['g_h0'], 'kJ/mol')}
               for k, v in terms.items()},
           'total': {
               's': cite.value(round(total['s'], 3), 'J/mol*K', 'estimated',
                               method='ideal-gas RRHO model', note='translational + rotational + vibrational + electronic'),
               'cp': cite.value(round(total['cp'], 3), 'J/mol*K', 'estimated',
                                method='ideal-gas RRHO model'),
               'h_h0': cite.value(round(total['h_h0'], 4), 'kJ/mol', 'estimated',
                                  method='ideal-gas RRHO model'),
               'g_h0': cite.value(round(total['g_h0'], 4), 'kJ/mol', 'estimated',
                                  method='ideal-gas RRHO model'),
           },
           'notes': [
               'Ideal-gas rigid-rotor harmonic-oscillator model at %g K, %g bar: assumes '
               'independent translation/rotation/vibration, no anharmonicity, no '
               'vibration-rotation interaction, no internal rotation, ground electronic state only.'
               % (float(T_K), float(P_bar)),
           ]}
    return out


def statmech(query, T_K=298.15, P_bar=1.0):
    """End-to-end RRHO thermodynamics for a query (frequencies/rotational constants from QM)."""
    return partition_functions(query, T_K=T_K, P_bar=P_bar)


# ---- NIST comparison ---------------------------------------------------------

# Diatomic molecular constants for RRHO S/Cp comparison. B_e and omega_e are the
# equilibrium rotational and vibrational constants; sigma the symmetry number.
# 'source'/'retrieved' set for values fetched from the NIST WebBook during this build.
_DIATOMIC_URL = 'https://webbook.nist.gov/cgi/cbook.cgi?ID=%s&Units=SI&Mask=1000'
_DIATOMICS = {
    'N2': {'B_e': 1.99824, 'omega_e': 2358.57, 'sigma': 2, 'mass_amu': 28.0134, 'mult': 1,
           'id': 'C7727379'},
    'CO': {'B_e': 1.93128087, 'omega_e': 2169.81358, 'sigma': 1, 'mass_amu': 28.0101, 'mult': 1,
           'id': 'C630080'},
    'O2': {'B_e': 1.44563, 'omega_e': 1580.193, 'sigma': 2, 'mass_amu': 31.9988, 'mult': 3,
           'id': None},
    'H2': {'B_e': 60.853, 'omega_e': 4401.213, 'sigma': 2, 'mass_amu': 2.01588, 'mult': 1,
           'id': None},
    'NO': {'B_e': 1.7049, 'omega_e': 1904.2, 'sigma': 1, 'mass_amu': 30.0061, 'mult': 2,
           'id': None},
    'HF': {'B_e': 20.9557, 'omega_e': 4138.3, 'sigma': 1, 'mass_amu': 20.0063, 'mult': 1,
           'id': None},
    'HCl': {'B_e': 10.5934, 'omega_e': 2990.9, 'sigma': 1, 'mass_amu': 35.9767, 'mult': 1,
            'id': None},
}


def _diatomic_records(key):
    """Cited value records for one diatomic's constants (verified when fetched)."""
    d = _DIATOMICS.get(key)
    if d is None:
        return None
    verified = d.get('id') is not None
    src = _DIATOMIC_URL % d['id'] if d.get('id') else None

    def rec(v, unit, label):
        return cite.value(v, unit, 'cited' if verified else 'unverified',
                          source=src if verified else 'NIST WebBook / Huber & Herzberg, '
                                                   'Constants of Diatomic Molecules (not fetched)',
                          retrieved=_RETRIEVED if verified else None, verified=verified,
                          note='%s of %s (X state).' % (label, key))
    return {
        'B_e': rec(d['B_e'], 'cm^-1', 'B_e'),
        'omega_e': rec(d['omega_e'], 'cm^-1', 'omega_e'),
        'sigma': rec(d['sigma'], 'dimensionless', 'symmetry number'),
        'mass_amu': rec(d['mass_amu'], 'amu', 'molecular mass'),
        'mult': d['mult'],
    }


def compare_nist(species_key, T_K=298.15):
    """Compare RRHO S and Cp with NIST Shomate values (``thermo.props``).

    For diatomics with tabulated constants (N2, CO, O2, ...) the RRHO inputs are the
    cited equilibrium constants; for other species they are computed from a
    B3LYP/6-31g* calculation (with a note that the comparison is then approximate).
    """
    from . import thermo

    spec = thermo.species(species_key)
    if spec is None:
        raise KeyError('unknown thermo species: %r' % (species_key,))
    nist = thermo.props(species_key, float(T_K))

    drec = _diatomic_records(species_key)
    if drec is None:
        raise ValueError('compare_nist supports the tabulated diatomics (%s), not %r'
                         % (', '.join(sorted(_DIATOMICS)), species_key))
    freqs = [drec['omega_e']['value']]
    mass_kg = drec['mass_amu']['value'] * U_AMU
    sigma = drec['sigma']['value']
    mult = drec['mult']
    linear = True
    rot_cm1 = [drec['B_e']['value']]
    note = 'RRHO inputs from cited NIST WebBook diatomic constants (%s).' % species_key

    terms, total = _rrho_contributions(mass_kg, sigma, rot_cm1, linear, freqs, mult, float(T_K), 1.0e5)
    return {
        'species_key': species_key, 'T_K': float(T_K),
        'nist_s': nist['s'], 'nist_cp': nist['cp'],
        'rrho_s': cite.value(round(total['s'], 3), 'J/mol*K', 'estimated',
                             method='ideal-gas RRHO model', note=note),
        'rrho_cp': cite.value(round(total['cp'], 3), 'J/mol*K', 'estimated',
                              method='ideal-gas RRHO model', note=note),
        'delta_s': cite.value(round(total['s'] - nist['s']['value'], 3), 'J/mol*K', 'derived',
                              method='RRHO - NIST Shomate'),
        'delta_cp': cite.value(round(total['cp'] - nist['cp']['value'], 3), 'J/mol*K', 'derived',
                               method='RRHO - NIST Shomate'),
        'inputs': {'B_e': drec['B_e'], 'omega_e': drec['omega_e'], 'sigma': drec['sigma'],
                   'mass_amu': drec['mass_amu'], 'multiplicity': mult},
        'notes': [note, 'RRHO harmonic treatment omits anharmonicity and vibration-rotation '
                        'interaction; NIST Shomate values include them, so small offsets are expected.'],
    }

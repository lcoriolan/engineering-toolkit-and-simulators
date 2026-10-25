# DESCRIPTION: Chemical-engineering kinetics on Cantera 3.2.0: ignition, equilibrium,
#              laminar flame speed, reaction-rate screening, and the "will it happen"
#              My Lab link (bench_kinetics). venv imports only (Cantera + numpy).
"""Chemical kinetics and reacting-flow calculations (Cantera 3.2.0).

Mechanisms are limited to those bundled with Cantera (``mechanisms()``). The 0-D
reactor / flame tools only accept homogeneous ideal-gas mechanisms; surface and
real-gas (Redlich-Kwong) mechanisms are listed but refused with a clear reason.

Units at the public API: temperature K, pressure bar (converted to Pa internally),
laminar flame speed cm/s. Rates of progress are Cantera's kmol/(m3 s); rate
constants keep Cantera's standard units (they vary with reaction order).

Everything returned is a ``cite`` value record (kind ``estimated``, method
"Cantera 3.2.0, <mechanism>"); the honest limits — mechanism validity range, no
heat loss, no external ignition source, no surface chemistry, homogeneous 0-D —
are stated in every result's ``notes``.
"""
from __future__ import annotations

import numpy as np
import cantera as ct

from engineering_sim.science import cite

# ---- Mechanism catalog ---------------------------------------------------------
# Only files bundled with Cantera's data directory. The label / reference / validity
# text is transcribed from each file's own header (read during this build, 2026-09-14);
# species and reaction counts are measured from the installed file at call time.
MECHANISMS = {
    'gri30': {
        'label': 'GRI-Mech 3.0',
        'description': 'Natural gas / H2 / CO / NOx combustion (53 species, 325 reactions)',
        'reference': 'Smith et al., GRI-Mech 3.0, http://combustion.berkeley.edu/gri-mech/version30/text30.html',
        'validity': 'GRI-Mech 3.0 is optimised for natural-gas combustion, ~1000-2500 K, 10 Torr-10 atm, phi 0.1-5',
        'homogeneous_gas': True,
    },
    'h2o2': {
        'label': 'H2/O2 submechanism',
        'description': 'Hydrogen-oxygen submechanism extracted from GRI-Mech 3.0 (+N2)',
        'reference': 'GRI-Mech 3.0 (H2/O2 subset), http://combustion.berkeley.edu/gri-mech/version30/text30.html',
        'validity': 'H2/O2 subset of GRI-Mech 3.0; see GRI-Mech 3.0 validity range',
        'homogeneous_gas': True,
    },
    'gri30_highT': {
        'label': 'GRI-Mech 3.0 (high-T)',
        'description': 'GRI-Mech 3.0 with high-temperature NASA polynomials (McBride et al., NASA TM-4513)',
        'reference': 'GRI-Mech 3.0 + high-T thermo: McBride et al., NASA TM-4513, https://ntrs.nasa.gov/search.jsp?R=19940013151',
        'validity': 'GRI-Mech 3.0 kinetics with extended high-temperature thermochemistry',
        'homogeneous_gas': True,
    },
    'gri30_ion': {
        'label': 'GRI-Mech 3.0 + ions',
        'description': 'GRI-Mech 3.0 extended with ion chemistry (HCO+, H3O+, e-)',
        'reference': 'GRI-Mech 3.0 + Han, Jie et al., Combustion Theory and Modelling 19.6 (2015) 744-772, DOI 10.1080/13647830.2015.1090018',
        'validity': 'GRI-Mech 3.0 kinetics plus ion transport; see GRI-Mech 3.0 validity range',
        'homogeneous_gas': True,
    },
    'ohn': {
        'label': 'O/H/N submechanism',
        'description': 'O/H/N submechanism extracted from GRI-Mech 3.0',
        'reference': 'GRI-Mech 3.0 (O/H/N subset), http://combustion.berkeley.edu/gri-mech/version30/text30.html',
        'validity': 'O/H/N subset of GRI-Mech 3.0; no carbon chemistry',
        'homogeneous_gas': True,
    },
    'air': {
        'label': 'Air (N/O chemistry)',
        'description': 'Ideal-gas air with N/O dissociation and NOx reactions',
        'reference': 'Cantera bundled air.yaml (thermo/transport from GRI-Mech 3.0), http://combustion.berkeley.edu/gri-mech/version30/text30.html',
        'validity': 'N/O/Ar only; no fuel species. Air dissociation and NOx formation',
        'homogeneous_gas': True,
    },
    'nDodecane_Reitz': {
        'label': 'n-dodecane/PAH (real gas)',
        'description': 'Reduced n-dodecane-PAH mechanism (Redlich-Kwong real-gas EOS)',
        'reference': 'Wang, Ra, Jia, Reitz, Fuel 136 (2014) 25-36, doi:10.1016/j.fuel.2014.07.028',
        'validity': 'n-dodecane soot predictions; real-gas EOS not supported by the 0-D ideal-gas reactor tools',
        'homogeneous_gas': False,
    },
    'ptcombust': {
        'label': 'CH4/O2 on Pt (surface)',
        'description': 'CH4-O2 surface mechanism on platinum (catalytic combustion)',
        'reference': 'Deutschmann et al., 26th Symp. (Intl.) on Combustion (1996) 1747-1754, https://www.detchem.com/mechanisms',
        'validity': 'Catalytic surface mechanism; not a homogeneous gas mechanism',
        'homogeneous_gas': False,
    },
    'methane_pox_on_pt': {
        'label': 'CH4 partial oxidation on Pt (surface)',
        'description': 'Surface mechanism for partial oxidation of CH4 on Pt wire gauze',
        'reference': 'Quiceno, Perez-Ramirez, Warnatz, Deutschmann, Appl. Catal. A (2006), https://www.detchem.com/mechanisms',
        'validity': 'Catalytic surface mechanism; not a homogeneous gas mechanism',
        'homogeneous_gas': False,
    },
}

GAS_MECHANISMS = sorted(name for name, meta in MECHANISMS.items() if meta['homogeneous_gas'])

# Integration bounds shared by the reactor ops.
MAX_T_END_S = 10.0          # refuse detailed ignition windows longer than 10 s
MAX_BENCH_T_S = 1e6         # the "will it happen" check accepts long windows (e.g. 1 h)
MAX_STEPS = 2_000_000       # refuse runs that need more integrator steps than this
MAX_SAMPLES = 400           # time-series samples returned to the page
IGNITION_DTDT_K_S = 100.0   # self-heating rate (K/s) that marks a real ignition
IGNITION_DT_K = 5.0         # minimum temperature rise (K) that marks a real ignition
TOP_SPECIES = 12            # major species kept in every profile

METHOD = 'Cantera 3.2.0'


def _cite(value, unit, mech, note=None):
    """One ``estimated`` cite record naming the model and mechanism."""
    return cite.value(float(value), unit, 'estimated',
                      method='%s, %s.yaml (%s)' % (METHOD, mech, MECHANISMS[mech]['label']),
                      note=note)


def _cite_none(unit, mech, note):
    """A ``None``-valued estimated record for a quantity that did not occur."""
    return cite.value(None, unit, 'estimated',
                      method='%s, %s.yaml' % (METHOD, mech), note=note)


def mechanisms():
    """Bundled Cantera mechanisms with species/reaction counts and references.

    Counts are measured from the installed files at call time; metadata (label,
    reference, validity, homogeneous-gas capability) is transcribed from each file's
    own header. The list is sorted by mechanism name for determinism.
    """
    out = []
    for name in sorted(MECHANISMS):
        meta = MECHANISMS[name]
        rec = {key: meta[key] for key in
               ('label', 'description', 'reference', 'validity', 'homogeneous_gas')}
        rec['name'] = name
        try:
            gas = ct.Solution(name + '.yaml')
            rec['species'] = int(gas.n_species)
            rec['reactions'] = int(gas.n_reactions)
            rec['elements'] = int(gas.n_elements)
            rec['thermo_model'] = str(gas.thermo_model)
            rec['phase'] = str(gas.name)
        except Exception as exc:  # a mechanism that does not load is reported, not hidden
            rec['species'] = None
            rec['reactions'] = None
            rec['elements'] = None
            rec['thermo_model'] = None
            rec['phase'] = None
            rec['error'] = type(exc).__name__ + ': ' + str(exc)
        out.append(rec)
    return out


def _load_gas(mech):
    """Load ``mech`` as a homogeneous ideal-gas Solution, or raise with the reason.

    Guards against surface mechanisms, real-gas (Redlich-Kwong) mechanisms and
    mechanisms with no gas-phase reactions, all of which the ideal-gas 0-D reactor
    and freely-propagating flame tools cannot use.
    """
    meta = MECHANISMS.get(mech)
    if meta is None:
        raise ValueError('Unknown mechanism %r; available: %s'
                         % (mech, ', '.join(sorted(MECHANISMS))))
    if not meta['homogeneous_gas']:
        raise ValueError('Mechanism %r (%s) is not a homogeneous ideal-gas mechanism '
                         'and cannot be used by the 0-D reactor / flame tools.'
                         % (mech, meta['label']))
    gas = ct.Solution(mech + '.yaml')
    if str(gas.thermo_model) != 'ideal-gas':
        raise ValueError('Mechanism %r uses the %s thermo model, not ideal gas; '
                         'the homogeneous reactor tools need ideal gas.'
                         % (mech, gas.thermo_model))
    if gas.n_reactions == 0:
        raise ValueError('Mechanism %r has no gas-phase reactions.' % mech)
    return gas


def _mole_fractions(gas, mixture, mech):
    """Map a ``{species: amount}`` dict to mole fractions (amounts normalised to 1).

    Accepts either relative mole numbers ({"H2": 2, "O2": 1}) or already-normalised
    fractions ({"H2": 0.667, "O2": 0.333}) — both give the same result.
    """
    if not isinstance(mixture, dict) or not mixture:
        raise ValueError('mixture must be a non-empty {species: amount} dict.')
    names = set(gas.species_names)
    total = 0.0
    amounts = {}
    for key, val in mixture.items():
        if key not in names:
            raise ValueError('Species %r is not in mechanism %r (species: %s).'
                             % (key, mech, ', '.join(sorted(names))))
        v = float(val)
        if not np.isfinite(v) or v < 0.0:
            raise ValueError('Amount for species %r must be a finite non-negative number.' % key)
        amounts[key] = v
        total += v
    if total <= 0.0:
        raise ValueError('mixture amounts sum to zero.')
    return {k: v / total for k, v in amounts.items()}


def _major_indices(X, n=TOP_SPECIES):
    """Species indices with the largest peak mole fraction, most prominent first."""
    peak = np.asarray(X).max(axis=0)
    return np.argsort(-peak)[: min(n, len(peak))]


def _notes(mech, extra=()):
    """Common honesty notes: model class + mechanism validity + what is not modelled."""
    notes = [
        'Estimated with Cantera 3.2.0, %s.yaml (%s).' % (mech, MECHANISMS[mech]['label']),
        MECHANISMS[mech]['validity'] + '.',
        '0-D adiabatic and homogeneous: no heat loss, no external ignition source, '
        'no surface chemistry and no buoyancy/transport are modelled.',
    ]
    notes.extend(extra)
    return notes


def _equilibrium_composition(gas, mech, n=TOP_SPECIES):
    """Major species of an equilibrated Solution, most abundant first, as cite records."""
    X = gas.X
    order = np.argsort(-X)[: n]
    out = []
    for i in order:
        if X[i] < 1e-6:
            continue
        out.append({'species': gas.species_names[int(i)],
                    'X': _cite(float(X[i]), 'dimensionless', mech)})
    return out


# ---- ignition ------------------------------------------------------------------
def ignition(mech, mixture, T0_K, P_bar, t_end_s, reactor='const_pressure'):
    """0-D adiabatic ignition: temperature, pressure and species vs time.

    Parameters
    ----------
    mech : str
        Mechanism name from ``mechanisms()`` (homogeneous ideal-gas only).
    mixture : dict
        ``{species: amount}`` (amounts normalised to mole fractions).
    T0_K, P_bar : float
        Initial temperature (K) and pressure (bar).
    t_end_s : float
        Integration window (s). Refused above ``MAX_T_END_S`` (10 s).
    reactor : str
        ``'const_pressure'`` or ``'const_volume'``.

    Returns ignition delay from two independent criteria (time of maximum dT/dt and
    time of the OH mole-fraction peak), the final equilibrium state, major-species
    time series (<= 400 samples) and a ``happens_within`` boolean.
    """
    gas = _load_gas(mech)
    X = _mole_fractions(gas, mixture, mech)
    T0_K = float(T0_K)
    P_bar = float(P_bar)
    t_end_s = float(t_end_s)
    if not 200.0 <= T0_K <= 6000.0:
        raise ValueError('T0_K must be within [200, 6000] K, got %r.' % T0_K)
    if not 1e-3 <= P_bar <= 1e3:
        raise ValueError('P_bar must be within [1e-3, 1e3] bar, got %r.' % P_bar)
    if not 1e-6 < t_end_s <= MAX_T_END_S:
        raise ValueError('t_end_s must be within (0, %g] s, got %r.' % (MAX_T_END_S, t_end_s))
    reactor = reactor or 'const_pressure'
    if reactor not in ('const_pressure', 'const_volume'):
        raise ValueError("reactor must be 'const_pressure' or 'const_volume', got %r." % reactor)

    gas.TPX = T0_K, P_bar * 1e5, X
    if reactor == 'const_pressure':
        rct = ct.IdealGasConstPressureReactor(gas, clone=False)
    else:
        rct = ct.IdealGasReactor(gas, clone=False)
    net = ct.ReactorNet([rct])
    net.max_steps = MAX_STEPS

    states = ct.SolutionArray(gas, extra=['t'])
    states.append(rct.phase.state, t=0.0)
    last_t = 0.0
    for target in np.linspace(0.0, t_end_s, MAX_SAMPLES)[1:]:
        try:
            t = net.advance(float(target))
        except ct.CanteraError:
            break  # integrator gave up (stiffness / max_steps); keep what we have
        if t <= last_t:      # integrator stalled before the next sample time
            break
        states.append(rct.phase.state, t=t)
        last_t = t
        if t >= t_end_s - 1e-9:
            break

    t = np.asarray(states.t, dtype=float)
    T = np.asarray(states.T, dtype=float)
    P = np.asarray(states.P, dtype=float)
    Xm = np.asarray(states.X, dtype=float)

    # Ignition delay, criterion 1: time of maximum self-heating rate.
    dTdt = np.gradient(T, t)
    i_peak = int(np.argmax(dTdt))
    ignited = bool(dTdt[i_peak] > IGNITION_DTDT_K_S and (T[i_peak] - T0_K) > IGNITION_DT_K)
    delay_dTdt = float(t[i_peak]) if ignited else None

    # Ignition delay, criterion 2: time of the OH mole-fraction peak.
    delay_OH = None
    oh_peak = None
    if 'OH' in gas.species_names:
        iOH = int(gas.species_index('OH'))
        i_oh = int(np.argmax(Xm[:, iOH]))
        delay_OH = float(t[i_oh])
        oh_peak = float(Xm[i_oh, iOH])

    happens = bool(ignited and delay_dTdt is not None and delay_dTdt < t_end_s)

    # Final equilibrium state for this reactor constraint (adiabatic).
    gas_eq = ct.Solution(mech + '.yaml')
    gas_eq.TPX = T0_K, P_bar * 1e5, X
    if reactor == 'const_pressure':
        gas_eq.equilibrate('HP')
        eq_label = 'fixed H, P (adiabatic flame / constant-pressure)'
    else:
        gas_eq.equilibrate('UV')
        eq_label = 'fixed U, V (constant-volume explosion)'

    keep = _major_indices(Xm)
    species_names = [gas.species_names[i] for i in keep]

    return {
        'mechanism': mech,
        'conditions': {'T0_K': T0_K, 'P_bar': P_bar, 't_end_s': t_end_s, 'reactor': reactor},
        'happens_within': happens,
        'ignition_delay': {
            'dTdt_max': _cite(delay_dTdt, 's', mech, note='time of maximum dT/dt') if delay_dTdt is not None
                        else _cite_none('s', mech, 'no ignition detected'),
            'OH_peak': _cite(delay_OH, 's', mech, note='time of OH mole-fraction peak') if delay_OH is not None
                       else _cite_none('s', mech, 'OH not present in mechanism'),
        },
        'OH_peak_fraction': oh_peak,
        'T_final': _cite(float(T[-1]), 'K', mech),
        'P_final': _cite(float(P[-1]) / 1e5, 'bar', mech),
        'series': {
            't_s': t.tolist(),
            'T_K': T.tolist(),
            'P_bar': (P / 1e5).tolist(),
            'species': species_names,
            'X': Xm[:, keep].tolist(),
        },
        'equilibrium': {
            'constraint': eq_label,
            'T': _cite(float(gas_eq.T), 'K', mech),
            'P_bar': _cite(float(gas_eq.P) / 1e5, 'bar', mech),
            'major': _equilibrium_composition(gas_eq, mech),
        },
        'notes': _notes(mech, (
            '%s 0-D adiabatic reactor, %d time samples.' % (reactor, len(t)),
        )),
    }


# ---- equilibrium ---------------------------------------------------------------
def equilibrate(mech, mixture, T, P, constraint='TP'):
    """Equilibrium composition; ``constraint='HP'`` gives the adiabatic flame temperature.

    ``constraint`` is ``'TP'`` (fixed T, P), ``'HP'`` (fixed enthalpy, P — adiabatic
    flame temperature) or ``'UV'`` (fixed internal energy, volume — constant-volume
    explosion temperature).
    """
    gas = _load_gas(mech)
    X = _mole_fractions(gas, mixture, mech)
    T = float(T)
    P = float(P)
    if not 200.0 <= T <= 6000.0:
        raise ValueError('T must be within [200, 6000] K, got %r.' % T)
    if not 1e-3 <= P <= 1e3:
        raise ValueError('P must be within [1e-3, 1e3] bar, got %r.' % P)
    constraint = (constraint or 'TP').upper()
    if constraint not in ('TP', 'HP', 'UV'):
        raise ValueError("constraint must be 'TP', 'HP' or 'UV', got %r." % constraint)

    gas.TPX = T, P * 1e5, X
    if constraint == 'HP':
        gas.equilibrate('HP')
        label = 'fixed enthalpy and pressure (adiabatic flame temperature)'
    elif constraint == 'UV':
        gas.equilibrate('UV')
        label = 'fixed internal energy and volume (constant-volume explosion)'
    else:
        gas.equilibrate('TP')
        label = 'fixed temperature and pressure'

    return {
        'mechanism': mech,
        'constraint': constraint,
        'constraint_label': label,
        'T_in': _cite(T, 'K', mech),
        'P_in': _cite(P, 'bar', mech),
        'T': _cite(float(gas.T), 'K', mech,
                   note='adiabatic flame temperature' if constraint == 'HP' else 'equilibrium temperature'),
        'P_bar': _cite(float(gas.P) / 1e5, 'bar', mech),
        'rho': _cite(float(gas.density), 'kg/m3', mech),
        'major': _equilibrium_composition(gas, mech),
        'notes': _notes(mech, ('Equilibrium computed by constrained Gibbs minimisation (%s).' % constraint)),
    }


# ---- laminar flame speed -------------------------------------------------------
def flame_speed(mech, fuel, oxidizer, phi, T0_K, P_bar):
    """Freely-propagating 1-D premixed flame: laminar flame speed S_L (cm/s) + profiles.

    ``oxidizer`` uses Cantera's ``set_equivalence_ratio`` syntax (e.g. ``'O2:1, N2:3.76'``).
    ``phi`` is bounded to [0.4, 3.0]; the grid is refined with
    ``set_refine_criteria(ratio=3, slope=0.07, curve=0.14)``.
    """
    gas = _load_gas(mech)
    phi = float(phi)
    T0_K = float(T0_K)
    P_bar = float(P_bar)
    if not 0.4 <= phi <= 3.0:
        raise ValueError('phi must be within [0.4, 3.0], got %r.' % phi)
    if not 200.0 <= T0_K <= 3000.0:
        raise ValueError('T0_K must be within [200, 3000] K, got %r.' % T0_K)
    if not 1e-3 <= P_bar <= 1e2:
        raise ValueError('P_bar must be within [1e-3, 1e2] bar, got %r.' % P_bar)
    if fuel not in gas.species_names:
        raise ValueError('Fuel %r is not in mechanism %r (species: %s).'
                         % (fuel, mech, ', '.join(gas.species_names)))
    for comp in str(oxidizer).split(','):
        if ':' in comp:
            comp = comp.split(':', 1)[0].strip()
        if comp and comp not in gas.species_names:
            raise ValueError('Oxidizer species %r is not in mechanism %r.' % (comp, mech))

    gas.TP = T0_K, P_bar * 1e5
    try:
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
    except Exception as exc:
        raise ValueError('set_equivalence_ratio(phi=%r, fuel=%r, oxidizer=%r) failed: %s'
                         % (phi, fuel, oxidizer, exc))

    flame = ct.FreeFlame(gas, width=0.03)
    flame.set_refine_criteria(ratio=3, slope=0.07, curve=0.14)
    try:
        # One solve with auto=True: Cantera builds the grid and applies the refine
        # criteria in a single pass. (A second auto=True solve re-grids from scratch
        # and takes minutes, so it is deliberately not done.)
        flame.solve(loglevel=0, auto=True)
    except Exception as exc:
        raise ValueError('Freely-propagating flame did not converge for %r/air phi=%g '
                         'at %g K, %g bar (mechanism %s): %s'
                         % (fuel, phi, T0_K, P_bar, mech, exc))

    S_L = float(flame.velocity[0]) * 100.0  # m/s -> cm/s
    grid_mm = (np.asarray(flame.grid) * 1000.0).tolist()  # m -> mm
    Tprof = np.asarray(flame.T, dtype=float)
    Xprof = np.asarray(flame.X, dtype=float)

    # Downsample to <= MAX_SAMPLES points (flame grids can exceed it after refinement).
    n = len(grid_mm)
    if n > MAX_SAMPLES:
        idx = np.linspace(0, n - 1, MAX_SAMPLES).astype(int)
        grid_mm = [grid_mm[i] for i in idx]
        Tprof = Tprof[idx]
        Xprof = Xprof[idx]

    keep = _major_indices(Xprof)
    species_names = [gas.species_names[i] for i in keep]

    return {
        'mechanism': mech,
        'conditions': {'fuel': fuel, 'oxidizer': oxidizer, 'phi': phi, 'T0_K': T0_K, 'P_bar': P_bar},
        'S_L': _cite(S_L, 'cm/s', mech, note='unburned-gas laminar flame speed, freely-propagating 1-D'),
        'T_products': _cite(float(Tprof[-1]), 'K', mech, note='temperature at the burned end of the flame'),
        'series': {
            'x_mm': grid_mm,
            'T_K': Tprof.tolist(),
            'species': species_names,
            'X': Xprof[:, keep].tolist(),
        },
        'notes': _notes(mech, (
            '1-D freely-propagating premixed flame, mixture-averaged transport, '
            'grid refined with ratio=3, slope=0.07, curve=0.14.',
            'S_L is a model result: no radiation or buoyancy; results outside the mechanism '
            'validity range are extrapolated.',
        )),
    }


# ---- reaction-rate screen ------------------------------------------------------
def rate_screen(mixture, T_K, P_bar, mech='gri30'):
    """Top 10 reactions by |net rate of progress| at a state, with rate constants.

    In a radical-free (frozen) mixture the net rate of progress is ~0 because forward
    and reverse rates balance; the forward rate of progress and rate constants are
    reported alongside so the fastest possible channels are still visible.
    """
    gas = _load_gas(mech)
    X = _mole_fractions(gas, mixture, mech)
    gas.TPX = float(T_K), float(P_bar) * 1e5, X

    rop_net = np.asarray(gas.net_rates_of_progress, dtype=float)
    rop_fwd = np.asarray(gas.forward_rates_of_progress, dtype=float)
    kf = np.asarray(gas.forward_rate_constants, dtype=float)
    kr = np.asarray(gas.reverse_rate_constants, dtype=float)
    eqs = gas.reaction_equations()

    order = np.argsort(-np.abs(rop_net))[:10]
    rows = []
    for i in order:
        rows.append({
            'index': int(i),
            'equation': eqs[int(i)],
            'net_rop': float(rop_net[int(i)]),
            'forward_rop': float(rop_fwd[int(i)]),
            'k_fwd': float(kf[int(i)]),
            'k_rev': float(kr[int(i)]),
        })

    return {
        'mechanism': mech,
        'conditions': {'T_K': float(T_K), 'P_bar': float(P_bar), 'mixture': X},
        'reactions': rows,
        'notes': [
            'Rate of progress units: kmol/(m3 s) (Cantera convention). Rate constants '
            'keep Cantera standard units, which vary with reaction order.',
            'Sorted by |net rate of progress|. In a radical-free frozen state net rop '
            'is ~0 (forward and reverse balanced); forward rop shows the fastest '
            'possible channels.',
        ],
    }


# ---- "will it happen" My Lab link ----------------------------------------------
def bench_kinetics(items, T_K, P_bar, t_s, mech='gri30'):
    """Map My Lab items into a mechanism and ask: does the mixture react within ``t_s``?

    Accepts the My Lab item syntax (e.g. ``["2 H2", "O2"]``); every item is resolved
    by ``chem.parse_items`` and mapped onto a mechanism species. When any item cannot
    be mapped the result is ``{"covered": False, "missing": [...]}`` rather than a
    guess. Otherwise the mixture is run in a 0-D adiabatic constant-pressure reactor
    and ``happens`` is True when it reaches >= 50 % of its equilibrium conversion
    within ``t_s`` (e.g. H2/O2 at 300 K for 1 h -> no; at 1100 K -> ignites in ms).
    """
    from engineering_sim.science import chem

    gas = _load_gas(mech)
    T_K = float(T_K)
    P_bar = float(P_bar)
    t_s = float(t_s)
    if not 200.0 <= T_K <= 6000.0:
        raise ValueError('T_K must be within [200, 6000] K, got %r.' % T_K)
    if not 1e-3 <= P_bar <= 1e3:
        raise ValueError('P_bar must be within [1e-3, 1e3] bar, got %r.' % P_bar)
    if not 1e-6 < t_s <= MAX_BENCH_T_S:
        raise ValueError('t_s must be within (0, %g] s, got %r.' % (MAX_BENCH_T_S, t_s))

    # Species names can differ in case between mechanisms (gri30 'AR' vs lowercase
    # real-gas names); match case-insensitively but return the mechanism's own name.
    upper_to_species = {}
    for s in gas.species_names:
        upper_to_species.setdefault(s.upper(), s)

    counts = {}
    feed = []
    missing = []
    # chem.parse_items resolves items through RDKit, which logs SMILES parse
    # chatter for formula-like items like "H2"; silence RDKit's C-level logger
    # (process-local) so the worker stderr stays clean.
    from rdkit import RDLogger
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    for item in chem.parse_items(items):
        mol = item.get('molecule') or {}
        formula = mol.get('formula')
        query = mol.get('query') or item.get('query')
        if mol.get('error') or not formula:
            missing.append(query or mol.get('error') or '(unresolved)')
            continue
        sp = upper_to_species.get(str(formula).upper())
        if sp is None:
            missing.append(formula)
            continue
        counts[sp] = counts.get(sp, 0.0) + float(item.get('count', 1.0))
        feed.append({'query': query, 'count': item.get('count', 1),
                     'formula': formula, 'species': sp})

    base = {
        'mechanism': mech,
        'feed': feed,
        'conditions': {'T_K': T_K, 'P_bar': P_bar, 't_s': t_s},
        'notes': _notes(mech),
    }
    if missing or not counts:
        base['covered'] = False
        base['missing'] = missing
        base['happens'] = False
        base['reason'] = 'one or more items could not be mapped to species in %s' % mech
        return base

    total = sum(counts.values())
    X = {k: v / total for k, v in counts.items()}
    feed_species = sorted(counts.keys())

    # Equilibrium conversion of the feed species at this T, P (fixed T,P).
    gas_eq = ct.Solution(mech + '.yaml')
    gas_eq.TPX = T_K, P_bar * 1e5, X
    gas_eq.equilibrate('TP')
    Xf_eq = sum(float(gas_eq.X[gas_eq.species_index(s)]) for s in feed_species)
    conv_eq = max(0.0, 1.0 - Xf_eq)  # sum of feed-species mole fractions drops from 1 to Xf_eq

    # Already at equilibrium? Then "it happens" trivially (nothing left to convert).
    if conv_eq <= 1e-6:
        base['covered'] = True
        base['happens'] = True
        base['already_equilibrium'] = True
        base['equilibrium_conversion'] = _cite(0.0, '-', mech)
        base['conversion_reached'] = _cite(0.0, '-', mech)
        base['time_to_50pct'] = _cite(0.0, 's', mech)
        base['final_T'] = _cite(float(gas_eq.T), 'K', mech,
                                note='feed already at equilibrium; no further conversion')
        base['equilibrium'] = {'T': _cite(float(gas_eq.T), 'K', mech),
                               'major': _equilibrium_composition(gas_eq, mech)}
        return base

    conv_target = 0.5 * conv_eq

    # Integrate, tracking conversion and the max dT/dt ignition criterion online.
    gas_run = ct.Solution(mech + '.yaml')
    gas_run.TPX = T_K, P_bar * 1e5, X
    rct = ct.IdealGasConstPressureReactor(gas_run, clone=False)
    net = ct.ReactorNet([rct])
    net.max_steps = MAX_STEPS

    t_prev, T_prev = 0.0, T_K
    max_dTdt = 0.0
    delay_dTdt = None
    reached_at = None
    final_conv = 0.0
    t = 0.0
    while t < t_s:
        try:
            t = net.step()
        except ct.CanteraError:
            break
        T_new = float(rct.T)
        dt = t - t_prev
        if dt > 0.0:
            dTdt = (T_new - T_prev) / dt
            if dTdt > max_dTdt:
                max_dTdt = dTdt
                delay_dTdt = t
        t_prev, T_prev = t, T_new
        Xf = sum(float(rct.phase.X[rct.phase.species_index(s)]) for s in feed_species)
        final_conv = max(0.0, 1.0 - Xf)
        if final_conv >= conv_target:
            reached_at = t
            break

    ignited = bool(max_dTdt > IGNITION_DTDT_K_S and (T_prev - T_K) > IGNITION_DT_K)
    happens = reached_at is not None

    base.update({
        'covered': True,
        'happens': happens,
        'equilibrium_conversion': _cite(conv_eq, '-', mech,
                                        note='1 - (sum of feed-species mole fractions at equilibrium)'),
        'conversion_reached': _cite(final_conv, '-', mech),
        'time_to_50pct': _cite(reached_at, 's', mech,
                               note='time to reach 50 % of equilibrium conversion') if reached_at is not None
                         else _cite_none('s', mech, 'did not reach 50 % of equilibrium conversion within t_s'),
        'ignition_delay': _cite(delay_dTdt, 's', mech, note='time of maximum dT/dt') if ignited else None,
        'final_T': _cite(T_prev, 'K', mech),
        'equilibrium': {'T': _cite(float(gas_eq.T), 'K', mech),
                        'major': _equilibrium_composition(gas_eq, mech)},
    })
    return base

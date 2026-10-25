# DESCRIPTION: Combustion section: stoichiometric air, excess air, air-fuel ratios, products of combustion
#              and their dew point, heating values (formation-enthalpy and NIST-reaction routes), enthalpy
#              of reaction, equilibrium constant, and Cantera equilibrium / adiabatic flame temperature.
"""Combustion calculators.

Conventions: a fuel is a CxHyOz molecule described by its atom counts C, H, O (real numbers, so surrogate formulas such as C7.76H13.1 work) and molar mass
MW in kg/kmol. Air is modelled as 3.76 mol N2 per mol O2 (molar mass 28.97 kg/kmol). Enthalpies are in
J/mol; heating values are returned both molar (J/mol fuel) and mass (J/kg fuel). Adiabatic flame
temperature and equilibrium composition are computed by Cantera through ``engineering_sim.science.mech.common``
(the only adapter); reaction thermochemistry uses the NIST-WebBook Shomate engine in
``engineering_sim.science.thermo``. Nothing here imports CoolProp or Cantera directly.

Reference relations are the standard ones in combustion texts (Turns, *An Introduction to Combustion*;
Cengel & Boles, *Thermodynamics*; Heywood, *ICE Fundamentals*). Cited by relation name, recalled rather
than re-read, so no page or equation numbers are given.
"""
import math
import re

from engineering_sim.science import cite
from engineering_sim.science import thermo
from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import common

SECTION = section('combustion', 'Combustion', order=300,
                  summary='Stoichiometric air, excess air, air-fuel ratios, products of combustion and '
                          'dew point, heating values, enthalpy of reaction, equilibrium constant and '
                          'Cantera equilibrium / adiabatic flame temperature.')

PI = math.pi
R_UNIV = 8.314462618  # J/(mol·K), CODATA 2018
MW_AIR = 28.97        # kg/kmol, standard dry air


def _atoms(fuel):
    """Atom counts {C, H, O} parsed from a molecular formula over C/H/O only (e.g. 'C2H6O')."""
    counts = {'C': 0, 'H': 0, 'O': 0}
    for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', fuel):
        if el not in counts:
            raise MechError(f'formula element {el!r} not supported (only C, H, O)')
        counts[el] += int(n) if n else 1
    if sum(counts.values()) == 0:
        raise MechError(f'could not parse formula {fuel!r}')
    return counts


def _a_o2(C, H, O):
    """Moles of O2 needed to burn one mole of CxHyOz to CO2 and H2O."""
    return C + H / 4.0 - O / 2.0


# ---------------------------------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------------------------------
def _stoichiometric_air(values):
    """Stoichiometric oxygen and air requirements for a CxHyOz fuel (per kmol fuel).

    CxHyOz + a(O2 + 3.76 N2) -> x CO2 + (y/2) H2O + 3.76 a N2, with a = x + y/4 − z/2. Air-fuel ratio on a
    mass basis uses MW_air = 28.97 kg/kmol.
    """
    C, H, O = float(values['C']), float(values['H']), float(values['O'])
    MW_fuel = values['MW_fuel']
    a = _a_o2(C, H, O)
    if a <= 0.0:
        raise MechError('fuel is already oxidised beyond stoichiometry (a = x + y/4 − z/2 ≤ 0)')
    a_air = 4.76 * a                       # kmol air per kmol fuel (O2 + 3.76 N2)
    af_st = a_air * MW_AIR / MW_fuel       # kg air / kg fuel
    return {'a_O2': a, 'a_air': a_air, 'AF_st': af_st, 'FA_st': 1.0 / af_st}


def _products(values):
    """Products of combustion of CxHyOz with an excess-air ratio λ, plus the water dew point.

    Products per kmol fuel: x CO2, (y/2) H2O, (λ−1)a O2, 3.76 λ a N2. The water partial pressure
    P_H2O = y_H2O · P_total gives the dew point through the Water saturation curve (CoolProp/IAPWS-95).
    """
    C, H, O = float(values['C']), float(values['H']), float(values['O'])
    lam, P_total = values['lambda'], values['P_total']
    a = _a_o2(C, H, O)
    if a <= 0.0:
        raise MechError('fuel is already oxidised beyond stoichiometry')
    if lam < 1.0:
        raise MechError('lambda < 1 (fuel-rich) is not modelled: the CO/H2 split in rich products is not computed')
    n_co2, n_h2o = float(C), H / 2.0
    n_o2 = (lam - 1.0) * a
    n_n2 = 3.76 * lam * a
    n_wet = n_co2 + n_h2o + n_o2 + n_n2
    n_dry = n_wet - n_h2o
    y_h2o = n_h2o / n_wet if n_wet > 0.0 else 0.0
    p_h2o = y_h2o * P_total
    # A fuel with no hydrogen (CO, carbon) has no water in the products: skip the dew point
    # rather than calling saturation at P = 0.
    if n_h2o > 0.0:
        t_dew = common.saturation('Water', P=p_h2o)['T']
    else:
        t_dew = None
    out = {'a_O2': a, 'n_CO2': n_co2, 'n_H2O': n_h2o, 'n_O2': n_o2, 'n_N2': n_n2,
           'n_wet': n_wet, 'n_dry': n_dry, 'y_H2O': y_h2o, 'P_H2O': p_h2o, 'T_dew': t_dew}
    if n_h2o <= 0.0:
        out['_warnings'] = ['fuel has no hydrogen: no water in the products, so the dew point is undefined']
    return out


def _heating_values_formation(values):
    """Heating values from formation enthalpies (per mole of CxHyOz fuel).

    ΔH_comb = [x hf(CO2) + (y/2) hf(H2O)] − [hf(fuel) + a hf(O2)] with hf(O2) = 0. The LHV uses gaseous
    product water, the HHV liquid product water; both are −ΔH_comb. Mass-basis values divide by the molar
    mass (MW in kg/kmol, so kg/mol = MW/1000).
    """
    C, H, O = float(values['C']), float(values['H']), float(values['O'])
    hf_fuel = values['hf_fuel']
    hf_co2 = values['hf_CO2']
    hf_h2o_g = values['hf_H2O_g']
    hf_h2o_l = values['hf_H2O_l']
    MW = values['MW_fuel']
    a = _a_o2(C, H, O)
    if a <= 0.0:
        raise MechError('fuel is already oxidised beyond stoichiometry')
    h_products_g = C * hf_co2 + (H / 2.0) * hf_h2o_g
    h_products_l = C * hf_co2 + (H / 2.0) * hf_h2o_l
    lhv = -(h_products_g - hf_fuel)
    hhv = -(h_products_l - hf_fuel)
    return {'a_O2': a, 'LHV': lhv, 'HHV': hhv,
            'LHV_mass': lhv * 1000.0 / MW, 'HHV_mass': hhv * 1000.0 / MW}


_FUELS = ('CH4', 'C2H6', 'C3H8', 'C2H4', 'C2H2', 'H2', 'CO', 'CH3OH(g)', 'C2H5OH(g)')


def _heating_value_fuel(values):
    """Heating values of a named fuel via the NIST-WebBook Shomate engine (engineering_sim.science.thermo).

    The balanced combustion reaction (fuel + a O2 -> C CO2 + (H/2) H2O) is assembled from the formula and
    its ΔH read from ``thermo.reaction``; LHV uses H2O(g), HHV uses H2O(l). Values are per mole of fuel.
    """
    fuel = values['fuel']
    rec = thermo.species(fuel)
    if rec is None:
        raise MechError(f'unknown fuel species {fuel!r}')
    atoms = _atoms(rec['formula'])
    C, H, O = atoms['C'], atoms['H'], atoms['O']
    a = _a_o2(C, H, O)
    if a <= 0.0:
        raise MechError('fuel is already oxidised beyond stoichiometry')
    mw = 12.011 * C + 1.008 * H + 15.999 * O          # kg/kmol
    # dH of combustion from the Shomate enthalpies of formation at 298.15 K (Hess's law). Read from
    # thermo.props directly rather than thermo.reaction, whose equilibrium constant exp(-dG/RT)
    # overflows double precision for very exergonic fuels such as C3H8.
    def _hf(key):
        return cite.num(thermo.props(key, 298.15)['h'])  # kJ/mol
    d_h_g = C * _hf('CO2') + (H / 2.0) * _hf('H2O(g)') - _hf(fuel) - a * _hf('O2')   # kJ/mol
    d_h_l = C * _hf('CO2') + (H / 2.0) * _hf('H2O(l)') - _hf(fuel) - a * _hf('O2')   # kJ/mol
    lhv = -d_h_g * 1000.0                              # J/mol
    hhv = -d_h_l * 1000.0                              # J/mol
    return {'a_O2': a, 'MW': mw, 'LHV': lhv, 'HHV': hhv,
            'LHV_mass': lhv * 1000.0 / mw, 'HHV_mass': hhv * 1000.0 / mw}


def _reaction_thermo(values):
    """Enthalpy/entropy/Gibbs change and equilibrium constant of a reaction via thermo.reaction.

    reactants/products are {species_key: coefficient} dicts (e.g. {'H2': 1, 'O2': 0.5} -> {'H2O(g)': 1}).
    Returns dH, dG in J/mol, dS in J/(mol·K) and the equilibrium constant K = exp(−dG/RT), all per mole of
    reaction as written.
    """
    reactants, products, T = values['reactants'], values['products'], values['T']
    if not isinstance(reactants, dict) or not isinstance(products, dict):
        raise MechError('reactants and products must be {species: coefficient} objects')
    try:
        r = thermo.reaction(reactants, products, T)
        dH, dS, dG = cite.num(r['dH']) * 1000.0, cite.num(r['dS']), cite.num(r['dG']) * 1000.0
    except ValueError as exc:
        raise MechError(str(exc))
    except OverflowError:
        # thermo.reaction also evaluates K = exp(−ΔG/RT), which overflows for ordinary exergonic reactions such as
        # propane combustion. Assemble ΔH and ΔS from the same Shomate species data (element balance still checked

        try:
            ok, imbalance = thermo._balance(reactants, products)
        except KeyError as exc:
            raise MechError(str(exc))
        if not ok:
            raise MechError(f'reaction is not element-balanced: {imbalance}')
        def side(d, key):
            return sum(c * cite.num(thermo.props(k, T)[key]) for k, c in d.items())
        dH = (side(products, 'h') - side(reactants, 'h')) * 1000.0
        dS = side(products, 's') - side(reactants, 's')
        dG = dH - T * dS
    ln_k = -dG / (R_UNIV * T)
    try:
        K = math.exp(ln_k)
    except OverflowError:
        K = None
    out = {'dH': dH, 'dS': dS, 'dG': dG, 'K': K, 'ln_K': ln_k}
    if K is None:
        out['_warnings'] = ['K = exp(ln K) overflows double precision; use ln_K']
    return out


def _equilibrium_constant(values):
    """Equilibrium constant from a standard Gibbs-energy change, K = exp(−ΔG/RT).

    Returns K, ln K and log10 K. K itself is None when exp(ln K) overflows double precision, with a
    ``_warnings`` note pointing the caller at the log forms.
    """
    dG, T = values['dG'], values['T']
    ln_k = -dG / (R_UNIV * T)
    log10_k = ln_k / math.log(10.0)
    try:
        K = math.exp(ln_k)
    except OverflowError:
        K = None
    out = {'K': K, 'ln_K': ln_k, 'log10_K': log10_k}
    if K is None:
        out['_warnings'] = ['K = exp(ln K) overflows double precision; use ln_K or log10_K instead']
    return out


def _chemical_equilibrium(values):
    """Cantera equilibrium via common.equilibrium.

    constraint 'TP' returns the equilibrium composition at (T, P); 'HP' returns the adiabatic flame
    temperature and composition (reactant T and P fixed); 'UV' the constant-volume explosion temperature.
    Returns the equilibrium temperature T_out (K), density (kg/m³) and the major species mole fractions
    (most abundant first, X ≥ 1e-6).
    """
    mech, mixture = values['mech'], values['mixture']
    T, P = values['T'], values['P']
    constraint = values['constraint']
    if not isinstance(mixture, dict) or not mixture:
        raise MechError('mixture must be a non-empty {species: amount} object')
    try:
        r = common.equilibrium(mech, mixture, T, P, constraint)
    except ValueError as exc:
        raise MechError(str(exc))
    major = [{'species': d['species'], 'X': cite.num(d['X'])} for d in r.get('major', [])]
    return {'T_out': float(r['T']), 'rho': float(r['rho']), 'major': major}


CALCS = [
    # ---- air and products --------------------------------------------------------------------------
    Calc('stoichiometric_air', 'Stoichiometric air and air-fuel ratio for a CxHyOz fuel',
         vars=[V('C', '', 'carbon atoms per fuel molecule (surrogate fuels may be non-integer)'),
               V('H', '', 'hydrogen atoms per fuel molecule (surrogate fuels may be non-integer)'),
               V('O', '', 'oxygen atoms per fuel molecule (surrogate fuels may be non-integer)'),
               V('MW_fuel', 'kg/kmol', 'fuel molar mass'),
               V('a_O2', '', 'stoichiometric oxygen (kmol O2 / kmol fuel)'),
               V('a_air', '', 'stoichiometric air (kmol air / kmol fuel)'),
               V('AF_st', '', 'stoichiometric air-fuel ratio (mass)'),
               V('FA_st', '', 'stoichiometric fuel-air ratio (mass)')],
         procedure=_stoichiometric_air,
         outputs=['a_O2', 'a_air', 'AF_st', 'FA_st'],
         example={'C': 1.0, 'H': 4.0, 'O': 0.0, 'MW_fuel': 16.04},
         keywords=['stoichiometric', 'air fuel ratio', 'theoretical air', 'afr'],
         assumptions=['Air = 3.76 mol N2 per mol O2 (MW 28.97 kg/kmol). a = x + y/4 − z/2 must be positive.'],
         refs=['Stoichiometric combustion of CxHyOz; a = x + y/4 − z/2 (recalled)']),

    Calc('excess_air', 'Excess-air fraction',
         vars=[V('excess', '', 'excess-air fraction (AF_actual − AF_st)/AF_st', signed=True),
               V('AF_actual', '', 'actual air-fuel ratio'), V('AF_st', '', 'stoichiometric air-fuel ratio')],
         equations=['excess = (AF_actual - AF_st)/AF_st'], outputs=['excess'],
         example={'AF_actual': 17.64, 'AF_st': 14.7},
         keywords=['excess air', 'lean', 'percent excess air'],
         assumptions=['Excess-air fraction; multiply by 100 for percent. Negative means fuel-rich '
                      '(air deficiency).'],
         refs=['Excess air = (AF_actual − AF_st)/AF_st']),

    Calc('products_of_combustion', 'Products of combustion and dew point',
         vars=[V('C', '', 'carbon atoms per fuel molecule (surrogate fuels may be non-integer)'),
               V('H', '', 'hydrogen atoms per fuel molecule (surrogate fuels may be non-integer)'),
               V('O', '', 'oxygen atoms per fuel molecule (surrogate fuels may be non-integer)'),
               V('lambda', '', 'excess-air ratio λ = AF_actual/AF_st', lo=1.0, label='λ'),
               V('P_total', 'Pa', 'total pressure of the products'),
               V('a_O2', '', 'stoichiometric oxygen (kmol/kmol fuel)'),
               V('n_CO2', '', 'CO2 in products (kmol/kmol fuel)'),
               V('n_H2O', '', 'H2O in products (kmol/kmol fuel)'),
               V('n_O2', '', 'excess O2 in products (kmol/kmol fuel)'),
               V('n_N2', '', 'N2 in products (kmol/kmol fuel)'),
               V('n_wet', '', 'total wet products (kmol/kmol fuel)'),
               V('n_dry', '', 'total dry products (kmol/kmol fuel)'),
               V('y_H2O', '', 'water mole fraction in the products'),
               V('P_H2O', 'Pa', 'partial pressure of water vapour'),
               V('T_dew', 'K', 'dew-point temperature of the products')],
         procedure=_products,
         outputs=['a_O2', 'n_CO2', 'n_H2O', 'n_O2', 'n_N2', 'n_wet', 'n_dry', 'y_H2O', 'P_H2O', 'T_dew'],
         example={'C': 1.0, 'H': 4.0, 'O': 0.0, 'lambda': 1.2, 'P_total': 101325.0},
         keywords=['products', 'flue gas', 'dew point', 'condensation', 'exhaust'],
         assumptions=['Complete combustion (no CO/soot), dry air, λ ≥ 1 (fuel-rich mixtures are not '
                      'modelled). Dew point uses the Water saturation curve (CoolProp/IAPWS-95) at P_H2O; it '
                      'fails if P_H2O is below the water triple point, and is undefined (null) for a '
                      'hydrogen-free fuel such as CO or carbon.'],
         refs=['Product mole balance of CxHyOz combustion; dew point at P_H2O (recalled)']),

    # ---- heating values -----------------------------------------------------------------------------
    Calc('heating_values', 'Higher and lower heating values from formation enthalpies',
         vars=[V('C', '', 'carbon atoms per fuel molecule (surrogate fuels may be non-integer)'),
               V('H', '', 'hydrogen atoms per fuel molecule (surrogate fuels may be non-integer)'),
               V('O', '', 'oxygen atoms per fuel molecule (surrogate fuels may be non-integer)'),
               V('MW_fuel', 'kg/kmol', 'fuel molar mass'),
               V('hf_fuel', 'J/mol', 'enthalpy of formation of the fuel', signed=True),
               V('hf_CO2', 'J/mol', 'enthalpy of formation of CO2', signed=True),
               V('hf_H2O_g', 'J/mol', 'enthalpy of formation of gaseous H2O', signed=True),
               V('hf_H2O_l', 'J/mol', 'enthalpy of formation of liquid H2O', signed=True),
               V('a_O2', '', 'stoichiometric oxygen (kmol/kmol fuel)'),
               V('LHV', 'J/mol', 'lower heating value (gaseous product water)'),
               V('HHV', 'J/mol', 'higher heating value (liquid product water)'),
               V('LHV_mass', 'J/kg', 'lower heating value per unit mass'),
               V('HHV_mass', 'J/kg', 'higher heating value per unit mass')],
         procedure=_heating_values_formation,
         outputs=['a_O2', 'LHV', 'HHV', 'LHV_mass', 'HHV_mass'],
         example={'C': 1.0, 'H': 4.0, 'O': 0.0, 'MW_fuel': 16.04,
                  'hf_fuel': -74.87e3, 'hf_CO2': -393.52e3, 'hf_H2O_g': -241.82e3, 'hf_H2O_l': -285.83e3},
         keywords=['heating value', 'hhv', 'lhv', 'calorific value', 'enthalpy of combustion'],
         assumptions=['ΔH_comb = Σ ν hf(products) − Σ ν hf(reactants) with hf(O2) = 0. LHV uses gaseous '
                      'product water, HHV liquid (the difference is the water latent heat).'],
         refs=['Heating value from enthalpies of formation (Hess\'s law; recalled)']),

    Calc('heating_value_fuel', 'Heating values of a named fuel (NIST-WebBook reaction thermochemistry)',
         vars=[V('fuel', '', 'fuel species', kind='text', choices=list(_FUELS)),
               V('a_O2', '', 'stoichiometric oxygen (kmol/kmol fuel)'),
               V('MW', 'kg/kmol', 'fuel molar mass'),
               V('LHV', 'J/mol', 'lower heating value'), V('HHV', 'J/mol', 'higher heating value'),
               V('LHV_mass', 'J/kg', 'lower heating value per unit mass'),
               V('HHV_mass', 'J/kg', 'higher heating value per unit mass')],
         procedure=_heating_value_fuel,
         outputs=['a_O2', 'MW', 'LHV', 'HHV', 'LHV_mass', 'HHV_mass'],
         example={'fuel': 'CH4'},
         keywords=['heating value', 'hhv', 'lhv', 'nist', 'shomate', 'methane'],
         assumptions=['ΔH of the balanced combustion reaction read from the NIST-WebBook Shomate engine in '
                      'engineering_sim.science.thermo (not recalled tables). LHV uses H2O(g), HHV uses H2O(l).'],
         refs=['NIST-WebBook Shomate reaction thermochemistry via engineering_sim.science.thermo']),

    # ---- reaction thermochemistry and equilibrium ----------------------------------------------------
    Calc('enthalpy_of_reaction', 'Enthalpy, entropy, Gibbs energy and equilibrium constant of a reaction',
         vars=[V('reactants', '', 'reactants as {species: coefficient}', kind='json'),
               V('products', '', 'products as {species: coefficient}', kind='json'),
               V('T', 'K', 'reaction temperature', lo=298.15, hi=6000.0),
               V('dH', 'J/mol', 'enthalpy of reaction', signed=True),
               V('dS', 'J/mol/K', 'entropy of reaction', signed=True),
               V('dG', 'J/mol', 'Gibbs energy of reaction', signed=True),
               V('K', '', 'equilibrium constant exp(−ΔG/RT) (null when it overflows)'),
               V('ln_K', '', 'natural log of the equilibrium constant', signed=True)],
         procedure=_reaction_thermo,
         outputs=['dH', 'dS', 'dG', 'K', 'ln_K'],
         example={'reactants': {'H2': 1, 'O2': 0.5}, 'products': {'H2O(g)': 1}, 'T': 298.15},
         keywords=['enthalpy of reaction', 'hess', 'gibbs', 'equilibrium constant', 'nist'],
         assumptions=['Species and thermochemistry from the NIST-WebBook Shomate engine; element balance is '
                      'checked. K is dimensionless.'],
         refs=['Reaction thermodynamics via engineering_sim.science.thermo (NIST-WebBook Shomate)']),

    Calc('equilibrium_constant', 'Equilibrium constant from the standard Gibbs-energy change',
         vars=[V('K', '', 'equilibrium constant exp(−ΔG/RT); null when it overflows'),
               V('ln_K', '', 'natural log of the equilibrium constant, ln K = −ΔG/(RT)', signed=True),
               V('log10_K', '', 'base-10 log of the equilibrium constant', signed=True),
               V('dG', 'J/mol', 'standard Gibbs-energy change of the reaction', signed=True),
               V('T', 'K', 'temperature')],
         procedure=_equilibrium_constant, outputs=['K', 'ln_K', 'log10_K'],
         example={'dG': -10.0e3, 'T': 298.15},
         keywords=['equilibrium constant', 'van t hoff', 'gibbs'],
         assumptions=['K = exp(−ΔG°/(R T)) with R = 8.314462618 J/(mol·K). ln K = −ΔG°/(R T) is returned '
                      'alongside K so large −ΔG° (whose K overflows double precision) is still usable.'],
         refs=['K = exp(−ΔG/RT); ln K = −ΔG/(RT)']),

    Calc('chemical_equilibrium', 'Chemical equilibrium composition and adiabatic flame temperature (Cantera)',
         vars=[V('mech', '', 'kinetic mechanism', kind='text',
                 choices=['gri30', 'gri30_highT', 'h2o2', 'ohn', 'nDodecane_Reitz']),
               V('mixture', '', 'reactant mixture {species: moles or mole fractions}', kind='json'),
               V('T', 'K', 'reactant temperature (constraint TP/HP) or fixed T (TP)', lo=200.0, hi=6000.0),
               V('P', 'Pa', 'pressure', lo=100.0, hi=100e5),
               V('constraint', '', 'equilibrium constraint', kind='text', choices=['TP', 'HP', 'UV']),
               V('T_out', 'K', 'equilibrium temperature (adiabatic flame temperature for HP)'),
               V('rho', 'kg/m^3', 'equilibrium density'),
               V('major', '', 'major species [[species, mole fraction], ...], most abundant first', kind='json')],
         procedure=_chemical_equilibrium, outputs=['T_out', 'rho', 'major'],
         example={'mech': 'gri30', 'mixture': {'CH4': 1, 'O2': 2, 'N2': 7.52}, 'T': 298.15,
                  'P': 101325.0, 'constraint': 'HP'},
         keywords=['adiabatic flame temperature', 'equilibrium composition', 'cantera', 'gri30', 'dissociation'],
         assumptions=['Cantera 0-D homogeneous equilibrium via common.equilibrium. HP = adiabatic flame '
                      'temperature; TP = fixed-T composition; UV = constant-volume explosion. The mechanism '
                      'must contain every species in the mixture (h2o2 covers H2/O2 only).'],
         refs=['Cantera constrained Gibbs-energy minimisation via engineering_sim.science.mech.common']),
]

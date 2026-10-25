# DESCRIPTION: Fluid properties and statics section: density/specific weight/specific gravity, viscosity,
#              surface tension and capillary rise, bulk modulus and speed of sound, hydrostatic and
#              manometer pressure, hydrostatic force and centre of pressure, buoyancy and metacentric
#              height, and rigid-body motion of accelerating and rotating fluid containers.
"""Fluid properties and statics calculators.

Conventions: gauge pressure is relative to a stated atmospheric pressure; absolute pressure is the sum.
Depths are measured down from the free surface (h positive downward); vertical coordinate z in the
rigid-body-motion relations is positive upward. Fluid property lookups at a state point go through
``common.fluid_state`` (CoolProp) — relations between user-given properties are plain equations.

Relations are the standard ones in any fluid-mechanics text (e.g. Munson, Young & Okiishi, *Fundamentals
of Fluid Mechanics*; White, *Fluid Mechanics*). Cited by relation name, recalled rather than re-read.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import common

SECTION = section('fluid_statics', 'Fluid properties and statics', order=230,
                  summary='Density and viscosity relations, surface tension, bulk modulus and speed of '
                          'sound, hydrostatics, manometers, hydrostatic force, buoyancy and rigid-body motion.')

PI = math.pi
G0 = common.g0()  # 9.80665 m/s², used only to pre-fill the example inputs


def _fluid_state_lookup(values):
    """State of a CoolProp fluid from T and P via common.fluid_state (the shared adapter)."""
    fluid = values['fluid']
    if not isinstance(fluid, dict) or 'name' not in fluid:
        raise MechError('fluid: pass {"name": "Water"} with a CoolProp fluid name')
    st = common.fluid_state(str(fluid['name']), T=values['T'], P=values['P'])
    out = {}
    for key in ('rho', 'mu', 'k', 'cp', 'h', 's'):
        if st.get(key) is None:
            raise MechError(f'{key} is unavailable for this fluid/state')
        out[key] = float(st[key])
    return out


def _rigid_body_accel(values):
    """Uniformly accelerating fluid tank: free-surface tilt and pressure at a point.

    theta = atan(a_x/(g + a_z)) is the angle of the free-surface normal from vertical toward +x;
    p = p_ref − ρ(a_x·x + (g + a_z)·z) with z positive upward and (x, z) measured from the reference point.
    Requires g + a_z > 0 — in free fall the free surface does not exist, so that case is rejected.
    """
    g, a_x, a_z = values['g'], values['a_x'], values['a_z']
    if g + a_z <= 0:
        raise MechError('rigid-body acceleration needs g + a_z > 0 (free fall has no free surface)')
    theta = math.atan(a_x / (g + a_z))
    p = values['p_ref'] - values['rho'] * (a_x * values['x'] + (g + a_z) * values['z'])
    return {'theta': theta, 'p': p}


CALCS = [
    Calc('density_relations', 'Specific weight and specific gravity from density',
         vars=[V('gamma', 'N/m^3', 'specific weight', label='γ'), V('SG', '', 'specific gravity'),
               V('rho', 'kg/m^3', 'density', label='ρ'), V('rho_ref', 'kg/m^3', 'reference density (water at 4 °C = 1000)'),
               V('g', 'm/s^2', 'gravitational acceleration')],
         equations=['gamma = rho*g', 'SG = rho/rho_ref'], outputs=['gamma', 'SG'],
         example={'rho': 1000.0, 'rho_ref': 1000.0, 'g': G0},
         keywords=['specific weight', 'specific gravity', 'density'],
         assumptions=['Specific gravity is relative to the stated reference density (usually water, 1000 kg/m³).'],
         refs=['γ = ρg; SG = ρ/ρ_ref']),

    Calc('viscosity_relations', 'Kinematic and dynamic viscosity (ν = μ/ρ)',
         vars=[V('nu', 'm^2/s', 'kinematic viscosity', label='ν'), V('mu', 'Pa*s', 'dynamic (absolute) viscosity', label='μ'),
               V('rho', 'kg/m^3', 'density', label='ρ')],
         equations=['nu = mu/rho'], outputs=['nu'],
         example={'mu': 0.001002, 'rho': 998.2},
         keywords=['kinematic viscosity', 'dynamic viscosity', 'centistokes'],
         assumptions=['Newtonian fluid at the state point; 1 mm²/s = 1 cSt.'],
         refs=['ν = μ/ρ']),

    Calc('capillary_rise', 'Capillary rise in a tube (h = 2σ cosθ/(ρgr))',
         vars=[V('h', 'm', 'capillary rise (negative for a non-wetting liquid)', signed=True),
               V('sigma', 'N/m', 'surface tension', label='σ'), V('theta', 'rad', 'contact angle (wetting θ < 90°)', hi=PI, label='θ'),
               V('rho', 'kg/m^3', 'liquid density', label='ρ'), V('g', 'm/s^2', 'gravitational acceleration'),
               V('r', 'm', 'tube radius')],
         equations=['h = 2*sigma*cos(theta)/(rho*g*r)'], outputs=['h'],
         example={'sigma': 0.0728, 'theta': 0.0, 'rho': 1000.0, 'g': G0, 'r': 5e-4},
         keywords=['surface tension', 'capillarity', 'meniscus'],
         assumptions=['Circular tube, contact angle θ measured through the liquid; θ > 90° (non-wetting, e.g. '
                      'mercury in glass) gives a depression (h < 0). θ = 0 is perfect wetting.'],
         refs=['Capillary rise from the Young–Laplace balance']),

    Calc('bulk_modulus_sound', 'Bulk modulus, compressibility and speed of sound (c = √(E_v/ρ))',
         vars=[V('c', 'm/s', 'speed of sound (isentropic)'), V('beta', '1/Pa', 'compressibility β = 1/E_v', label='β'),
               V('E_v', 'Pa', 'bulk modulus', label='Ev'), V('rho', 'kg/m^3', 'density', label='ρ')],
         equations=['c = sqrt(E_v/rho)', 'beta = 1/E_v'], outputs=['c', 'beta'],
         example={'E_v': 2.2e9, 'rho': 1000.0},
         keywords=['compressibility', 'sound speed', 'bulk modulus'],
         assumptions=['Small pressure perturbation (acoustic/sonic approximation). β = 1/E_v is the compressibility '
                      'at the entered modulus. The speed of sound c = √(E_v/ρ) uses the isentropic (adiabatic) bulk '
                      'modulus: liquids have E_s ≈ E_T so one value serves both, while for gases E_s = γp (γ = '
                      'specific-heat ratio) and E_s must be entered for sonic work.'],
         refs=['c = √(E_v/ρ); β = 1/E_v']),

    Calc('hydrostatic_pressure', 'Hydrostatic pressure: gauge and absolute',
         vars=[V('p_gauge', 'Pa', 'gauge pressure at depth h'), V('p_abs', 'Pa', 'absolute pressure at depth h'),
               V('p_atm', 'Pa', 'atmospheric pressure at the free surface'),
               V('h', 'm', 'depth below the free surface'), V('rho', 'kg/m^3', 'fluid density', label='ρ'),
               V('g', 'm/s^2', 'gravitational acceleration')],
         equations=['p_gauge = rho*g*h', 'p_abs = p_gauge + p_atm'], outputs=['p_gauge', 'p_abs'],
         example={'p_atm': 101325.0, 'h': 10.0, 'rho': 1000.0, 'g': G0},
         keywords=['hydrostatic', 'gauge pressure', 'absolute pressure', 'depth'],
         assumptions=['Incompressible fluid at rest; h measured down from the free surface; constant density.'],
         refs=['p = p₀ + ρgh']),

    Calc('manometer', 'Manometer pressure difference (Δp = (ρ_m − ρ_f) g h)',
         vars=[V('dp', 'Pa', 'pressure difference p_high − p_low', signed=True, label='Δp'),
               V('rho_m', 'kg/m^3', 'manometer (gauge) fluid density'), V('rho_f', 'kg/m^3', 'line fluid density (0 for a gas)'),
               V('g', 'm/s^2', 'gravitational acceleration'), V('h', 'm', 'manometer column height difference')],
         equations=['dp = (rho_m - rho_f)*g*h'], outputs=['dp'],
         example={'rho_m': 13600.0, 'rho_f': 1000.0, 'g': G0, 'h': 0.1},
         keywords=['manometer', 'u-tube', 'pressure measurement', 'piezometer'],
         assumptions=['U-tube manometer with a heavier gauge fluid; the line fluid (ρ_f) fills the connecting '
                      'legs above the gauge fluid (ρ_f ≈ 0 for a gas). dp is the difference across the column.'],
         refs=['Manometer relation Δp = (ρ_m − ρ_f) g h']),

    Calc('hydrostatic_force_plane', 'Hydrostatic force and centre of pressure on an inclined plane surface',
         vars=[V('F', 'N', 'resultant hydrostatic force'), V('y_cp', 'm', 'centre of pressure, measured from the surface along the plane'),
               V('y_c', 'm', 'centroid distance from the free surface along the plane'),
               V('h_c', 'm', 'vertical depth of the centroid below the free surface'),
               V('A', 'm^2', 'surface area'), V('I_c', 'm^4', 'centroidal second moment of area about the horizontal centroidal axis'),
               V('theta', 'rad', 'angle of the plane with the horizontal', hi=PI, label='θ'),
               V('rho', 'kg/m^3', 'fluid density', label='ρ'), V('g', 'm/s^2', 'gravitational acceleration')],
         equations=['F = rho*g*h_c*A', 'y_c = h_c/sin(theta)', 'y_cp = y_c + I_c/(A*y_c)'],
         outputs=['F', 'y_cp', 'y_c'],
         example={'h_c': 2.0, 'A': 1.5, 'I_c': 0.28125, 'theta': PI / 2, 'rho': 1000.0, 'g': G0},
         keywords=['hydrostatic force', 'centre of pressure', 'gate', 'dam'],
         assumptions=['Resultant acts normal to the plane at y_cp (always below the centroid, since I_c > 0).',
                      'The centroidal second moment I_c about the axis parallel to the free surface is a user '
                      'input (the section-properties section owns the shape formulas).',
                      'θ > 0 is required: a horizontal plate (θ = 0) has uniform pressure acting at the centroid, '
                      'and y_c = h_c/sin θ is undefined there, so a horizontal plate must be handled separately.'],
         refs=['F = ρg h_c A; y_cp = y_c + I_c/(A y_c)']),

    Calc('buoyancy', 'Buoyant force (Archimedes)',
         vars=[V('F_B', 'N', 'buoyant force'), V('rho', 'kg/m^3', 'fluid density', label='ρ'),
               V('g', 'm/s^2', 'gravitational acceleration'), V('V', 'm^3', 'displaced volume')],
         equations=['F_B = rho*g*V'], outputs=['F_B'],
         example={'rho': 1000.0, 'g': G0, 'V': 0.01},
         keywords=['archimedes', 'buoyancy', 'submerged'],
         assumptions=['Fluid at rest; V is the volume of displaced fluid (submerged volume for a floating body).'],
         refs=['Archimedes principle, F_B = ρgV']),

    Calc('metacentric_height', 'Metacentric height of a floating body (GM = KB + BM − KG)',
         vars=[V('GM', 'm', 'metacentric height (stable if > 0)', signed=True),
               V('BM', 'm', 'metacentric radius I/V_sub'), V('I', 'm^4', 'second moment of the waterplane area'),
               V('V_sub', 'm^3', 'displaced (submerged) volume'),
               V('KB', 'm', 'height of the centre of buoyancy above the keel'),
               V('KG', 'm', 'height of the centre of gravity above the keel')],
         equations=['BM = I/V_sub', 'GM = KB + BM - KG'], outputs=['BM', 'GM'],
         example={'I': 60.0, 'V_sub': 40.0, 'KB': 0.5, 'KG': 1.1},
         keywords=['stability', 'metacentre', 'floating body', 'ship'],
         assumptions=['Small-angle stability of a floating body; I is the second moment of the waterplane area '
                      'about the axis of tilt (section-properties section owns the shape formula). KB and KG are '
                      'measured up from the keel, so GM = KM − KG = (KB + BM) − KG; GM > 0 is stable.'],
         refs=['BM = I/V_sub; GM = KB + BM − KG']),

    Calc('rigid_body_accel', 'Rigid-body motion: uniformly accelerating fluid tank',
         vars=[V('theta', 'rad', 'angle of the free-surface normal from vertical, toward +x', signed=True, lo=-PI / 2, hi=PI / 2, label='θ'),
               V('p', 'Pa', 'pressure at the point'), V('p_ref', 'Pa', 'pressure at the reference point'),
               V('rho', 'kg/m^3', 'fluid density', label='ρ'),
               V('a_x', 'm/s^2', 'horizontal acceleration (+x)', signed=True),
               V('a_z', 'm/s^2', 'vertical acceleration (up positive)', signed=True),
               V('g', 'm/s^2', 'gravitational acceleration'),
               V('x', 'm', 'horizontal position of the point relative to the reference', signed=True),
               V('z', 'm', 'vertical position of the point relative to the reference (up positive)', signed=True)],
         procedure=_rigid_body_accel, outputs=['p', 'theta'],
         example={'a_x': 3.0, 'a_z': 0.0, 'g': G0, 'p_ref': 101325.0, 'rho': 1000.0, 'x': 1.0, 'z': -1.0},
         keywords=['accelerating tank', 'rigid body motion', 'free surface tilt'],
         assumptions=['Fluid moves as a rigid body under constant acceleration (no relative motion). The free '
                      'surface is a plane normal to the pressure gradient; it rises toward −x when a_x > 0. '
                      'Requires g + a_z > 0; g + a_z ≤ 0 (free fall) raises an error.'],
         refs=['Rigid-body linear acceleration: ∇p = ρ(g − a)']),

    Calc('rigid_body_rotation', 'Rigid-body motion: rotating fluid container (forced vortex)',
         vars=[V('dh', 'm', 'free-surface rise at radius R above the centre (paraboloid)'),
               V('p', 'Pa', 'pressure at radius r, height z'), V('p0', 'Pa', 'pressure at the free-surface apex (r = 0, z = 0)'),
               V('rho', 'kg/m^3', 'fluid density', label='ρ'), V('omega', 'rad/s', 'angular speed', label='ω'),
               V('g', 'm/s^2', 'gravitational acceleration'), V('R', 'm', 'container radius'),
               V('r', 'm', 'radial position of the point'), V('z', 'm', 'vertical position of the point (up positive, apex z = 0)', signed=True)],
         equations=['dh = omega**2*R**2/(2*g)', 'p = p0 + rho*omega**2*r**2/2 - rho*g*z'],
         outputs=['dh', 'p'],
         example={'omega': 10.0, 'g': G0, 'R': 0.2, 'p0': 101325.0, 'rho': 1000.0, 'r': 0.2, 'z': 0.0},
         keywords=['rotating tank', 'forced vortex', 'paraboloid', 'solid body rotation'],
         assumptions=['Steady solid-body rotation about the vertical axis; the free surface is a paraboloid '
                      'z = ω²r²/(2g). ω is angular (enter rad/s, rpm or rps).'],
         refs=['Solid-body rotation: p = p₀ + ρω²r²/2 − ρgz']),

    Calc('fluid_state_lookup', 'Fluid properties at a state point (CoolProp via common.fluid_state)',
         vars=[V('fluid', '', '{"name": "Water"} — CoolProp fluid name', kind='json'),
               V('T', 'K', 'temperature'), V('P', 'Pa', 'pressure'),
               V('rho', 'kg/m^3', 'density', label='ρ'), V('mu', 'Pa*s', 'dynamic viscosity', label='μ'),
               V('k', 'W/(m*K)', 'thermal conductivity'), V('cp', 'J/(kg*K)', 'specific heat at constant pressure'),
               V('h', 'J/kg', 'specific enthalpy'), V('s', 'J/(kg*K)', 'specific entropy')],
         procedure=_fluid_state_lookup, outputs=['rho', 'mu', 'k', 'cp', 'h', 's'],
         example={'fluid': {'name': 'Water'}, 'T': 293.15, 'P': 101325.0},
         keywords=['coolprop', 'water', 'air', 'refrigerant', 'properties'],
         assumptions=['State defined by temperature and pressure through the shared common.fluid_state adapter '
                      '(CoolProp real-fluid model). Two-phase states where transport properties are undefined '
                      'raise an error; give T and P for a single-phase state.'],
         refs=['CoolProp property database via common.fluid_state']),
]

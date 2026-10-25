# DESCRIPTION: Hydraulics and pneumatics section: cylinder force and speed, pump displacement and
#              efficiencies, hydraulic motor torque/speed, accumulator sizing (polytropic gas),
#              line pressure drops, hydraulic power and heat, and pneumatic air consumption (orifice flow lives
#              in fluid_dynamics.obstruction_flowmeter).
"""Hydraulics and pneumatics calculators.

Conventions: pressures are gauge or absolute as stated in each description; flow rates are volumetric.
Pump/motor displacement is per revolution (cc/rev); shaft speed is therefore cyclic (rev/s = Hz) — enter
'25 Hz' for 1500 rpm, never '1500 rpm' directly, because the torque/flow relations use D·n with n in rev/s.
Line losses use the Darcy–Weisbach relation with the friction factor as a user input (see fluid_dynamics
for computing f).

Relations are the standard ones in fluid-power texts (e.g. Esposito, *Fluid Power with Applications*;
Merritt, *Hydraulic Control Systems*). Cited by relation name, recalled rather than re-read.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('hydraulics', 'Hydraulics and pneumatics', order=410,
                  summary='Cylinder force and speed, pump and motor sizing, accumulator volume, line pressure '
                          'drops, hydraulic power and heat, pneumatic air consumption.')

PI = math.pi


# ---- procedures ------------------------------------------------------------------------------------
def _cylinder_speed(v):
    """Extension and retraction speeds of a double-acting cylinder from the flow rate and geometry."""
    D, d, Q = v['D'], v['d'], v['Q']
    if D <= 0:
        raise MechError('D: bore diameter must be positive')
    if d < 0:
        raise MechError('d: rod diameter must be non-negative')
    if d > D:
        raise MechError(f'd = {d:g} exceeds the bore D = {D:g}: the rod cannot be larger than the bore')
    A_bore = PI * D * D / 4.0
    A_ann = PI * (D * D - d * d) / 4.0
    v_ext = Q / A_bore
    warnings = []
    if A_ann <= 1e-12 * A_bore:
        warnings.append('d ≈ D: the annulus area is ≈ 0, so the retraction speed is unbounded')
        v_ret = None
    else:
        v_ret = Q / A_ann
    return {'A_bore': A_bore, 'A_ann': A_ann, 'v_ext': v_ext, 'v_ret': v_ret, '_warnings': warnings}


def _accumulator(v):
    """Gas-charged accumulator sizing with the polytropic gas law P·V^n = const (pressures ABSOLUTE)."""
    dV, P0, P1, P2, n = v['dV'], v['P0'], v['P1'], v['P2'], v['n']
    if n <= 0:
        raise MechError('n: polytropic exponent must be positive')
    if not (P0 < P1 < P2):
        raise MechError(f'P0 < P1 < P2 is required (precharge below the minimum system pressure below the '
                        f'maximum); got P0 = {P0:g}, P1 = {P1:g}, P2 = {P2:g}')
    V0 = dV / ((P0 / P1) ** (1.0 / n) - (P0 / P2) ** (1.0 / n))
    V1 = V0 * (P0 / P1) ** (1.0 / n)
    V2 = V0 * (P0 / P2) ** (1.0 / n)
    return {'V0': V0, 'V1': V1, 'V2': V2}


CALCS = [
    # ---- cylinders ---------------------------------------------------------------------------------
    Calc('cylinder_force', 'Hydraulic cylinder extension and retraction force (F = p·A)',
         vars=[V('p', 'Pa', 'gauge pressure'), V('D', 'm', 'bore diameter'),
               V('d', 'm', 'rod diameter'),
               V('A_bore', 'm^2', 'piston (cap-end) area'), V('A_ann', 'm^2', 'annulus (rod-end) area'),
               V('F_ext', 'N', 'extension force'), V('F_ret', 'N', 'retraction force')],
         equations=['A_bore = pi*D**2/4', 'A_ann = pi*(D**2 - d**2)/4',
                    'F_ext = p*A_bore', 'F_ret = p*A_ann'],
         outputs=['A_bore', 'A_ann', 'F_ext', 'F_ret'],
         example={'p': 10e6, 'D': 0.05, 'd': 0.02},
         keywords=['piston', 'rod side', 'annulus', 'bore'],
         assumptions=['Double-acting cylinder; extension acts on the full bore, retraction on the annulus '
                      '(bore minus rod). Friction and seal drag neglected.'],
         refs=['F = p·A; annulus area = π(D² − d²)/4']),

    Calc('cylinder_speed', 'Cylinder extension and retraction speeds from flow rate (v = Q/A)',
         vars=[V('Q', 'm^3/s', 'flow rate into the cylinder'), V('D', 'm', 'bore diameter'),
               V('d', 'm', 'rod diameter'),
               V('A_bore', 'm^2', 'piston (cap-end) area'), V('A_ann', 'm^2', 'annulus (rod-end) area'),
               V('v_ext', 'm/s', 'extension speed Q/A_bore'), V('v_ret', 'm/s', 'retraction speed Q/A_ann')],
         procedure=_cylinder_speed, outputs=['v_ret', 'v_ext', 'A_bore', 'A_ann'],
         example={'Q': 20.0 / 60000.0, 'D': 0.05, 'd': 0.02},
         keywords=['flow rate', 'piston speed', 'rod speed', 'extension', 'retraction'],
         assumptions=['Incompressible flow; extension uses the full bore area, retraction the annulus '
                      '(bore minus rod). Leakage neglected. The rod must be smaller than the bore; when '
                      'd ≈ D the annulus vanishes and the retraction speed is unbounded (a warning is '
                      'raised).'],
         refs=['Q = A·v; bore area πD²/4, annulus area π(D² − d²)/4']),

    # ---- pumps and motors ---------------------------------------------------------------------------
    Calc('pump', 'Pump displacement, flow, efficiencies and drive torque',
         vars=[V('D_p', 'm^3/revolution', 'displacement per revolution (cc/rev = cm³/rev)'),
               V('n', 'Hz', 'pump shaft speed in rev/s (Hz); 1500 rpm = 25 rev/s'),
               V('dp', 'Pa', 'pressure rise across the pump', label='Δp'),
               V('Q_th', 'm^3/s', 'theoretical (geometric) flow rate'),
               V('Q_act', 'm^3/s', 'actual delivered flow rate'),
               V('eta_v', '', 'volumetric efficiency', hi=1.0, label='ηv'),
               V('eta_mech', '', 'mechanical (torque) efficiency', hi=1.0, label='ηm'),
               V('eta_o', '', 'overall efficiency', hi=1.0, label='ηo'),
               V('T', 'N*m', 'shaft drive torque'), V('P_shaft', 'W', 'shaft input power')],
         equations=['Q_th = D_p*n', 'eta_v = Q_act/Q_th', 'eta_o = eta_v*eta_mech',
                    'T = D_p*dp/(2*pi*eta_mech)', 'P_shaft = T*2*pi*n'],
         outputs=['Q_th', 'eta_v', 'eta_o', 'T', 'P_shaft'],
         example={'D_p': 25e-6, 'n': 25.0, 'dp': 10e6, 'Q_act': 0.0006, 'eta_mech': 0.9},
         keywords=['volumetric efficiency', 'mechanical efficiency', 'displacement', 'cc/rev', 'drive torque'],
         assumptions=['Displacement per revolution, so flow = D·n with n in rev/s. Volumetric efficiency '
                      'accounts for internal leakage; overall = volumetric × mechanical. The shaft torque is '
                      'the ideal D·Δp/(2π) divided by the mechanical efficiency; shaft power = T·2π·n = '
                      'Δp·Q_th/ηm.'],
         refs=['Q_th = D·n; ηv = Q_act/Q_th; ηo = ηv·ηm; T = D·Δp/(2π·ηm)']),

    Calc('hydraulic_motor', 'Hydraulic motor torque and speed',
         vars=[V('D_m', 'm^3/revolution', 'displacement per revolution (cc/rev)'),
               V('dp', 'Pa', 'pressure drop across the motor', label='Δp'),
               V('Q', 'm^3/s', 'flow rate into the motor'),
               V('eta_mech', '', 'mechanical efficiency', hi=1.0, label='ηm'),
               V('eta_v', '', 'volumetric efficiency', hi=1.0, label='ηv'),
               V('T_ideal', 'N*m', 'ideal (theoretical) torque'),
               V('n_th', 'Hz', 'theoretical speed Q/D_m in rev/s (Hz)'),
               V('n', 'Hz', 'actual shaft speed ηv·Q/D_m in rev/s (Hz)'),
               V('T_act', 'N*m', 'actual output torque')],
         equations=['T_ideal = D_m*dp/(2*pi)', 'n_th = Q/D_m', 'n = eta_v*Q/D_m', 'T_act = eta_mech*T_ideal'],
         outputs=['T_ideal', 'n_th', 'n', 'T_act'],
         example={'D_m': 25e-6, 'dp': 10e6, 'Q': 0.0006, 'eta_mech': 0.9, 'eta_v': 0.95},
         keywords=['displacement', 'torque', 'cc/rev', 'hydrostatic'],
         assumptions=['Ideal torque T = D·Δp/(2π) (the 2π converts per-revolution displacement to per-radian '
                      'work); actual torque reduced by the mechanical efficiency. Theoretical speed n_th = Q/D '
                      'is reduced by internal leakage: actual speed n = ηv·Q/D.'],
         refs=['T = D·Δp/(2π); n_th = Q/D; n = ηv·Q/D']),

    # ---- accumulator --------------------------------------------------------------------------------
    Calc('accumulator', 'Gas-charged accumulator volume (polytropic, Boyle when n = 1)',
         vars=[V('dV', 'm^3', 'usable oil volume delivered', label='ΔV'),
               V('P0', 'Pa', 'gas precharge pressure (absolute)'),
               V('P1', 'Pa', 'minimum system pressure (absolute)'),
               V('P2', 'Pa', 'maximum system pressure (absolute)'),
               V('n', '', 'polytropic exponent (1 = isothermal/Boyle, 1.4 = adiabatic air)', lo=1.0),
               V('V0', 'm^3', 'gas volume at precharge'), V('V1', 'm^3', 'gas volume at P1'),
               V('V2', 'm^3', 'gas volume at P2')],
         procedure=_accumulator, outputs=['V0', 'V1', 'V2'],
         example={'dV': 0.001, 'P0': 90e5, 'P1': 100e5, 'P2': 200e5, 'n': 1.4},
         keywords=['bladder', 'bladder accumulator', 'precharge', 'isothermal', 'adiabatic'],
         assumptions=['Polytropic gas law P·V^n = const with ABSOLUTE pressures (gauge inputs shift the '
                      'compression ratio and give roughly −4 % volume errors). n = 1 for slow (isothermal) '
                      'cycling, n = 1.4 for fast (adiabatic) cycling of air. Precharge must be below the '
                      'minimum system pressure (P0 < P1 < P2, enforced); the usable volume is V1 − V2.'],
         refs=['Polytropic accumulator sizing; ΔV = V0[(P0/P1)^(1/n) − (P0/P2)^(1/n)]']),

    # ---- pressure drops -----------------------------------------------------------------------------
    Calc('line_pressure_drop', 'Line pressure drop (Darcy–Weisbach)',
         vars=[V('f', '', 'Darcy friction factor (user input; see fluid_dynamics)'),
               V('L', 'm', 'line length'), V('D', 'm', 'line inner diameter'),
               V('rho', 'kg/m^3', 'fluid density', label='ρ'), V('Q', 'm^3/s', 'flow rate'),
               V('v', 'm/s', 'mean flow velocity'), V('dp', 'Pa', 'pressure drop', label='Δp')],
         equations=['v = 4*Q/(pi*D**2)', 'dp = f*(L/D)*rho*v**2/2'],
         outputs=['v', 'dp'],
         example={'f': 0.02, 'L': 10.0, 'D': 0.02, 'rho': 870.0, 'Q': 0.0002},
         keywords=['darcy weisbach', 'friction factor', 'hose', 'tubing', 'line loss'],
         assumptions=['Darcy–Weisbach loss with the Darcy friction factor f (not Fanning, which is f/4). '
                      'Minor (fitting) losses add via K·ρv²/2.'],
         refs=['Darcy–Weisbach: Δp = f (L/D) ρv²/2']),

    # ---- power and heat -----------------------------------------------------------------------------
    Calc('hydraulic_power_heat', 'Hydraulic power and heat generation',
         vars=[V('p', 'Pa', 'pressure'), V('Q', 'm^3/s', 'flow rate'),
               V('eta', '', 'system efficiency', hi=1.0, label='η'),
               V('P_hyd', 'W', 'hydraulic power p·Q'), V('P_heat', 'W', 'heat (power loss)')],
         equations=['P_hyd = p*Q', 'P_heat = P_hyd*(1 - eta)'],
         outputs=['P_hyd', 'P_heat'],
         example={'p': 10e6, 'Q': 0.0006, 'eta': 0.8},
         keywords=['power loss', 'cooling', 'oil heating'],
         assumptions=['Hydraulic power P = p·Q; the inefficiency (1 − η) of that power is dissipated as heat '
                      'into the oil (sizing the cooler).'],
         refs=['P = p·Q; heat = P(1 − η)']),

    # ---- pneumatics ---------------------------------------------------------------------------------
    Calc('pneumatic_consumption', 'Pneumatic cylinder air consumption and free air delivery',
         vars=[V('D', 'm', 'bore diameter'), V('d', 'm', 'rod diameter'),
               V('s', 'm', 'stroke'), V('p_gauge', 'Pa', 'supply gauge pressure'),
               V('p_atm', 'Pa', 'atmospheric pressure (for free-air reference)'),
               V('TR', '', 'temperature ratio T_ref/T_supply (1 = same temperature)', label='TR'),
               V('N', 'Hz', 'cycles per second'),
               V('A_bore', 'm^2', 'piston area'), V('A_ann', 'm^2', 'annulus area'),
               V('V_cycle', 'm^3', 'working volume per full cycle (out + return) at pressure'),
               V('V_free', 'm^3', 'free air delivered per cycle (at p_atm and T_ref)'),
               V('Q_free', 'm^3/s', 'free air delivery (FAD)')],
         equations=['A_bore = pi*D**2/4', 'A_ann = pi*(D**2 - d**2)/4',
                    'V_cycle = (A_bore + A_ann)*s',
                    'V_free = V_cycle*(p_gauge + p_atm)/p_atm*TR',
                    'Q_free = V_free*N'],
         outputs=['A_bore', 'A_ann', 'V_cycle', 'V_free', 'Q_free'],
         example={'D': 0.05, 'd': 0.02, 's': 0.1, 'p_gauge': 6e5, 'p_atm': 101325.0, 'TR': 1.0, 'N': 0.5},
         keywords=['free air delivery', 'FAD', 'air consumption', 'scfm'],
         assumptions=['Isothermal Boyle expansion of the compressed working volume to atmospheric '
                      'pressure; one cycle = one extension plus one retraction. The temperature ratio '
                      'TR = T_ref/T_supply converts the delivered air to a reference temperature (set 1 to '
                      'ignore temperature); FAD is typically quoted at standard reference conditions.'],
         refs=['V_free = V_work·(p_abs/p_atm)·(T_ref/T_supply); FAD = V_free·N']),
]

# DESCRIPTION: Ergonomics and safety section: NIOSH lifting equation and lifting index, ISO 13855 safety
#              distance, stored-energy hazards (spring, compressed gas, flywheel) and lifting-equipment
#              safety factors.
"""Ergonomics and safety calculators.

The revised NIOSH lifting equation uses the metric load constant LC = 23 kg and the closed-form
multipliers HM = 25/H, VM = 1 − 0.003|V − 75|, DM = 0.82 + 4.5/D, AM = 1 − 0.0032·A (H, V, D in cm, A in
degrees) with the NIOSH bounds applied (see ``_niosh``); the frequency FM and coupling CM multipliers are
user-supplied table inputs. The safety distance follows ISO 13855 S = K·T + C with K and C as user inputs,
and the compressed-gas stored energy is one model (isentropic) of several. Spring and flywheel stored
energy are not duplicated here — see ``springs.spring_energy`` and ``kinetics.rotational_kinetic_energy``.
Relations are cited by name, recalled.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('ergonomics', 'Ergonomics and safety', order=480,
                  summary='NIOSH lifting index, ISO 13855 safety distance, stored-pressure-energy hazard and '
                          'lifting-equipment safety factors.')


def _niosh(values):
    """Revised NIOSH lifting equation (metric): RWL = LC·HM·VM·DM·AM·FM·CM, LI = Load/RWL.

    Multiplier bounds (NIOSH safety limits): HM = 25/H capped at 1 for H < 25 cm and 0 for H > 63 cm;
    VM = 1 − 0.003|V − 75|, zero for V > 175 cm; DM = 0.82 + 4.5/D capped at 1 for D < 25 cm (D = 0 is
    allowed) and 0 for D > 175 cm; AM = 1 − 0.0032·A, zero for A > 135° (H, V, D in cm, A in degrees).
    """
    H, V, D, A = values['H'], values['V'], values['D'], values['A']
    FM, CM, Load = values['FM'], values['CM'], values['Load']
    LC = 23.0  # NIOSH load constant, kg
    H_cm, V_cm, D_cm = H * 100.0, V * 100.0, D * 100.0
    A_deg = math.degrees(A)
    if H_cm <= 0:
        raise MechError('H must be positive (horizontal distance from the ankles)')
    if V_cm < 0 or D_cm < 0:
        raise MechError('V and D must be non-negative')
    # HM: cap at 1 below 25 cm; out of reach (zero) above 63 cm.
    HM = 0.0 if H_cm > 63.0 else min(1.0, 25.0 / H_cm)
    # VM: zero beyond 175 cm.
    VM = 0.0 if V_cm > 175.0 else 1.0 - 0.003 * abs(V_cm - 75.0)
    # DM: cap at 1 below 25 cm (D = 0 allowed); zero beyond 175 cm.
    if D_cm > 175.0:
        DM = 0.0
    elif D_cm < 25.0:
        DM = 1.0
    else:
        DM = 0.82 + 4.5 / D_cm
    # AM: zero beyond 135°.
    AM = 0.0 if A_deg > 135.0 else 1.0 - 0.0032 * A_deg
    RWL = LC * HM * VM * DM * AM * FM * CM
    warnings = []
    if H_cm < 25.0:
        warnings.append(f'H = {H_cm:.2f} cm is below 25 cm (HM capped at 1)')
    elif H_cm > 63.0:
        warnings.append(f'H = {H_cm:.2f} cm exceeds 63 cm (HM = 0)')
    if V_cm > 175.0:
        warnings.append(f'V = {V_cm:.2f} cm exceeds 175 cm (VM = 0)')
    if D_cm < 25.0:
        warnings.append(f'D = {D_cm:.2f} cm is below 25 cm (DM capped at 1)')
    elif D_cm > 175.0:
        warnings.append(f'D = {D_cm:.2f} cm exceeds 175 cm (DM = 0)')
    if A_deg > 135.0:
        warnings.append(f'A = {A_deg:.2f}° exceeds 135° (AM = 0)')
    if RWL <= 0:
        out = {'rwl': 0.0, 'hm': HM, 'vm': VM, 'dm': DM, 'am': AM, 'li': float('inf')}
        out['_warnings'] = warnings + ['RWL = 0: the lifting index is unbounded '
                                       '(any nonzero load exceeds the recommended limit)']
        return out
    out = {'li': Load / RWL, 'rwl': RWL, 'hm': HM, 'vm': VM, 'dm': DM, 'am': AM}
    if warnings:
        out['_warnings'] = warnings
    return out


CALCS = [
    Calc('niosh_lifting', 'NIOSH lifting equation: recommended weight limit and lifting index',
         vars=[V('H', 'm', 'horizontal location of the hands from the ankles', lo=0.0),
               V('V', 'm', 'vertical location of the hands at the lift origin', lo=0.0),
               V('D', 'm', 'vertical travel distance (origin to destination)', lo=0.0),
               V('A', 'rad', 'asymmetry angle', lo=0.0, hi=math.pi),
               V('FM', '', 'frequency multiplier (NIOSH FM table, user input)', lo=0.0, hi=1.0),
               V('CM', '', 'coupling multiplier (NIOSH CM table: 1.0 good, 0.95 fair, 0.90 poor)', lo=0.0, hi=1.0),
               V('Load', 'kg', 'actual lifted load'),
               V('li', '', 'lifting index = Load/RWL'), V('rwl', 'kg', 'recommended weight limit'),
               V('hm', '', 'horizontal multiplier 25/H'), V('vm', '', 'vertical multiplier'),
               V('dm', '', 'distance multiplier'), V('am', '', 'asymmetric multiplier')],
         procedure=_niosh, outputs=['li', 'rwl', 'hm', 'vm', 'dm', 'am'],
         example={'H': 0.4, 'V': 0.75, 'D': 0.5, 'A': 0.0, 'FM': 1.0, 'CM': 1.0, 'Load': 20.0},
         keywords=['niosh', 'lifting index', 'rwp', 'ergonomics', 'manual handling'],
         assumptions=['Revised NIOSH equation, metric form: LC = 23 kg, HM = 25/H, VM = 1 − 0.003|V − 75|, '
                      'DM = 0.82 + 4.5/D, AM = 1 − 0.0032·A (H, V, D in cm, A in degrees). FM and CM are table '
                      'inputs (not built in). NIOSH bounds are applied: HM = 1 for H < 25 cm and 0 for H > 63 cm, '
                      'VM = 0 for V > 175 cm, DM = 1 for D < 25 cm (D = 0 allowed) and 0 for D > 175 cm, '
                      'AM = 0 for A > 135°. When RWL = 0 the lifting index is unbounded (a warning is returned '
                      'rather than a finite value). LI > 1 indicates an increased risk of lifting injury.'],
         refs=['Revised NIOSH lifting equation (Waters et al. 1993; closed forms and bounds recalled)']),

    Calc('safety_distance', 'Machine safety distance (ISO 13855: S = K·T + C)',
         vars=[V('S', 'm', 'minimum safety distance'), V('K', 'm/s', 'approach (hand) speed'),
               V('T', 's', 'total stopping/response time'), V('C', 'm', 'intrusion/reach distance')],
         equations=['S = K*T + C'], outputs=['S'],
         example={'K': 2.0, 'T': 0.2, 'C': 0.05},
         keywords=['iso 13855', 'guarding', 'safety distance', 'light curtain', 'safe speed'],
         assumptions=['K (e.g. 2 m/s walking, 1.6 m/s for some applications) and C (e.g. 8(d−14) mm for a '
                      'light curtain) are user inputs from ISO 13855. Solving for K gives the permissible '
                      'approach speed for a given distance.'],
         refs=['ISO 13855 positioning of safeguards: S = K·T + C (recalled)']),

    Calc('stored_pressure_energy', 'Stored energy of a compressed gas (isentropic expansion work)',
         vars=[V('E', 'J', 'isentropic expansion energy'), V('P1', 'Pa', 'absolute vessel pressure'),
               V('P2', 'Pa', 'discharge (atmospheric) pressure'), V('V', 'm^3', 'vessel volume'),
               V('gamma', '', 'ratio of specific heats cp/cv', lo=1.01, hi=2.0, label='γ')],
         equations=['E = P1*V/(gamma - 1)*(1 - (P2/P1)**((gamma - 1)/gamma))'], outputs=['E'],
         example={'P1': 1e6, 'P2': 101325.0, 'V': 0.1, 'gamma': 1.4},
         keywords=['pressure vessel', 'stored energy', 'burst hazard', 'isentropic'],
         assumptions=['One common model of pressure-vessel burst energy: reversible adiabatic (isentropic) '
                      'expansion of an ideal gas from P1 to P2 with P2 ≤ P1 and both pressures absolute. '
                      'Other models exist (Brode, isothermal) and give different values — choose the model '
                      'that matches the release process.'],
         refs=['W = P1·V/(γ−1)·[1 − (P2/P1)^((γ−1)/γ)] (isentropic expansion, recalled; one of several models)']),

    Calc('lifting_safety', 'Working load limit from a lifting-equipment safety factor (n = F_ult/F_wll)',
         vars=[V('n', '', 'design factor from the applicable lifting standard (user input)', lo=1.0),
               V('F_ult', 'N', 'breaking (ultimate) strength'), V('F_wll', 'N', 'working load limit')],
         equations=['n = F_ult/F_wll'], outputs=['F_wll'],
         example={'n': 5.0, 'F_ult': 50000.0},
         keywords=['rigging', 'sling', 'safety factor', 'wll', 'swl', 'lifting'],
         assumptions=['This is the same factor-of-safety relation as design_fund.factor_of_safety. The design '
                      'factor n is a user input selected from the applicable standard (e.g. 5:1 for general '
                      'rigging hardware); for a multi-leg sling the WLL is shared between legs and reduced by the '
                      'leg angle. This relation does not certify equipment — follow the governing code.'],
         refs=['n = F_ult/F_wll (safety factor, recalled)']),
]

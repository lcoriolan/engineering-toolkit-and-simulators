# DESCRIPTION: Belts, chains and ropes section: open-belt length and angle of wrap, belt tensions
#              (tight/slack/centrifugal/initial), power and slip, V-belt rated power and number of belts,
#              roller-chain length/links/velocity and chordal speed variation, and wire-rope working
#              load, bending stress and sheave pressure (p/Su for the fatigue chart).
"""Belts, chains and ropes calculators.

Conventions:
- This section adds what the statics capstan calculators leave out: centrifugal tension, belt geometry
  (length, wrap angle), power, slip, and the chain/rope relations. The bare capstan ratio (no centrifugal
  term) is statics.belt_friction / statics.vbelt_friction.
- For a V-belt the wedge groove raises the effective friction to mu/sin(beta/2) (beta = groove included
  angle): use statics.vbelt_friction rather than the flat-belt belt_tension_ratio below.
- V-belt rated power per belt and the service / wrap / length correction factors are USER INPUTS naming
  the manufacturer catalogue. Wire-rope modulus E_r, wire ultimate strength S_u and the rope factor of
  safety are inputs; the fatigue life is read from the rope maker's p/Su-vs-cycles chart (input data),
  so only p/Su is computed here.
- Speeds are CYCLIC (rev/s = Hz): enter rpm as "1200 1/min" or rpm/60, never "1200 rpm".

Relations are the standard ones in machine-design texts (Budynas & Nisbett, *Shigley's Mechanical
Engineering Design*; ASME/AGMA belt-practice), cited by relation name, recalled.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('belts_chains', 'Belts, chains, and ropes', order=180,
                  summary='Belt geometry and tensions, power and slip, V-belt selection, roller-chain '
                          'length and chordal action, wire-rope load, bending stress and pressure.')

PI = math.pi


def _chain_length(values):
    """Roller-chain length from center distance C_p (pitches), pitch p, sprocket teeth N1, N2.

    L_p = 2 C_p + (N1 + N2)/2 + (N2 - N1)^2 / (4 pi^2 C_p)   (pitches)
    Number of links is rounded up to the nearest EVEN integer; chain length L = L_p p.
    """
    Cp, p = values['C_p'], values['p']
    N1, N2 = values['N1'], values['N2']
    if abs(N1 - round(N1)) > 1e-6 or abs(N2 - round(N2)) > 1e-6:
        raise MechError('N1 and N2 (sprocket teeth) must be integers')
    N1, N2 = int(round(N1)), int(round(N2))
    if Cp <= 0 or p <= 0:
        raise MechError('C_p (center distance in pitches) and p (pitch) must be positive')
    Lp = 2.0 * Cp + (N1 + N2) / 2.0 + (N2 - N1) ** 2 / (4.0 * PI ** 2 * Cp)
    links = int(math.ceil(Lp / 2.0)) * 2
    return {'L': Lp * p, 'L_p': Lp, 'L_links': float(links)}


def _belt_length(values):
    """Exact open-belt length (no small-angle approximation), D2 >= D1.

    alpha = asin((D2 - D1)/(2C)); wrap theta_s = pi - 2 alpha (small), theta_l = pi + 2 alpha (large).
    L = sqrt(4 C^2 - (D2 - D1)^2) + (D2 theta_l + D1 theta_s)/2.
    """
    C, D1, D2 = values['C'], values['D1'], values['D2']
    if C <= 0 or D1 <= 0 or D2 <= 0:
        raise MechError('C, D1, D2 must be positive')
    if 2 * C < abs(D2 - D1):
        raise MechError('center distance too small: need C >= (D2 - D1)/2 for an open belt')
    alpha = math.asin((D2 - D1) / (2.0 * C))
    theta_s = PI - 2.0 * alpha
    theta_l = PI + 2.0 * alpha
    L = math.sqrt(4.0 * C * C - (D2 - D1) ** 2) + (D2 * theta_l + D1 * theta_s) / 2.0
    return {'L': L}


def _belt_tension_ratio(values):
    """Total tight-side tension from the capstan relation with centrifugal tension.

    (T1 - T_c) = (T2 - T_c) e^(mu theta); the slack-side effective tension must be positive.
    """
    T2, T_c, mu, theta = values['T2'], values['T_c'], values['mu'], values['theta']
    if T2 <= T_c:
        raise MechError('T2 (slack-side total tension) must exceed T_c (centrifugal tension) or the '
                        'belt cannot grip the pulley')
    return {'T1': T_c + (T2 - T_c) * math.exp(mu * theta)}


CALCS = [
    # ---- belt geometry -------------------------------------------------------------------------
    Calc('belt_length', 'Exact open-belt length between two pulleys',
         vars=[V('L', 'm', 'belt length'),
               V('C', 'm', 'center distance'),
               V('D1', 'm', 'smaller pulley diameter'),
               V('D2', 'm', 'larger pulley diameter')],
         procedure=_belt_length, outputs=['L'],
         example={'C': 2.0, 'D1': 0.2, 'D2': 0.5},
         keywords=['belt length', 'open belt', 'pulley', 'centre distance'],
         assumptions=['Open (uncrossed) belt on two pulleys, exact: alpha = asin((D2-D1)/(2C)), wrap '
                      'theta_s = pi - 2 alpha (small pulley), theta_l = pi + 2 alpha (large pulley), '
                      'L = sqrt(4 C^2 - (D2-D1)^2) + (D2 theta_l + D1 theta_s)/2. Requires C >= (D2-D1)/2. '
                      'The familiar approximation L = 2C + pi(D1+D2)/2 + (D2-D1)^2/(4C) agrees to within '
                      '~1% for C >> (D2-D1).'],
         refs=['Open-belt length L = sqrt(4C^2 - (D2-D1)^2) + (D2 theta_l + D1 theta_s)/2 (recalled)']),

    Calc('belt_wrap', 'Angle of wrap on each pulley (open belt)',
         vars=[V('theta_s', 'rad', 'wrap angle on the small pulley', hi=PI, label='θs'),
               V('theta_l', 'rad', 'wrap angle on the large pulley', lo=PI, hi=2 * PI, label='θl'),
               V('D1', 'm', 'smaller pulley diameter'),
               V('D2', 'm', 'larger pulley diameter'),
               V('C', 'm', 'center distance')],
         equations=['theta_s = pi - 2*asin((D2 - D1)/(2*C))', 'theta_l = pi + 2*asin((D2 - D1)/(2*C))'],
         outputs=['theta_s', 'theta_l'],
         example={'D1': 0.2, 'D2': 0.5, 'C': 2.0},
         keywords=['wrap angle', 'arc of contact', 'pulley'],
         assumptions=['Open belt, D2 >= D1. The small-pulley wrap angle governs friction capacity: use it '
                      'in the capstan ratio (statics.belt_friction or belt_tension_ratio here). Requires '
                      'C >= (D2 - D1)/2.'],
         refs=['Wrap angle theta = pi ± 2 asin((D2-D1)/(2C)) (recalled)']),

    # ---- belt tensions --------------------------------------------------------------------------
    Calc('belt_centrifugal_tension', 'Centrifugal belt tension (T_c = m v^2)',
         vars=[V('T_c', 'N', 'centrifugal tension', label='Tc'),
               V('m', 'kg/m', 'belt mass per unit length'),
               V('v', 'm/s', 'belt speed')],
         equations=['T_c = m*v**2'], outputs=['T_c'],
         example={'m': 0.5, 'v': 20.0},
         keywords=['centrifugal', 'tension', 'belt mass'],
         assumptions=['Centrifugal tension is uniform around the belt and does not contribute to power '
                      'transmission; it raises the total belt tension and reduces the useful friction.'],
         refs=['Centrifugal tension T_c = m v^2 (recalled)']),

    Calc('belt_tension_ratio', 'Belt tension ratio with centrifugal tension (capstan + T_c)',
         vars=[V('T1', 'N', 'total tight-side tension (including centrifugal)'),
               V('T2', 'N', 'total slack-side tension (including centrifugal)'),
               V('T_c', 'N', 'centrifugal tension', label='Tc'),
               V('mu', '', 'coefficient of friction between belt and pulley', label='μ'),
               V('theta', 'rad', 'wrap angle (use the small-pulley value)', label='θ')],
         procedure=_belt_tension_ratio, outputs=['T1'],
         example={'T2': 100.0, 'T_c': 50.0, 'mu': 0.3, 'theta': PI},
         keywords=['capstan', 'flat belt', 'centrifugal', 'friction', 'tension ratio'],
         assumptions=['Impending slip on the effective tensions (total minus centrifugal): '
                      '(T1 - T_c) = (T2 - T_c) e^(mu theta); a MechError is raised if T2 <= T_c (the '
                      'slack-side effective tension must stay positive). With T_c = 0 this reduces to the '
                      'capstan equation (statics.belt_friction). For a V-belt use the effective friction '
                      'mu/sin(beta/2) (statics.vbelt_friction).'],
         refs=['Capstan equation with centrifugal tension (recalled)']),

    Calc('belt_initial_tension', 'Initial (installation) belt tension from running tensions',
         vars=[V('T_i', 'N', 'initial (installation) tension', label='Ti'),
               V('T1', 'N', 'total tight-side tension (including centrifugal)'),
               V('T2', 'N', 'total slack-side tension (including centrifugal)'),
               V('T_c', 'N', 'centrifugal tension', label='Tc')],
         equations=['T_i = (T1 + T2)/2 - T_c'], outputs=['T_i'],
         example={'T1': 600.0, 'T2': 200.0, 'T_c': 100.0},
         keywords=['initial tension', 'installation', 'preload', 'belt'],
         assumptions=['T1 and T2 are the TOTAL (running) tensions; the installed tension is the mean of '
                      'the effective tensions, T_i = (T1 + T2)/2 - T_c. Tension the belt so the '
                      'effective tensions stay positive (T2 > T_c).'],
         refs=['Initial tension T_i = (T1 + T2)/2 - T_c (recalled)']),

    # ---- power and slip ------------------------------------------------------------------------
    Calc('belt_power', 'Power transmitted by a belt (P = (T1 - T2) v)',
         vars=[V('P', 'W', 'transmitted power'),
               V('T1', 'N', 'total tight-side tension'),
               V('T2', 'N', 'total slack-side tension'),
               V('v', 'm/s', 'belt speed')],
         equations=['P = (T1 - T2)*v'], outputs=['P'],
         example={'T1': 600.0, 'T2': 200.0, 'v': 20.0},
         keywords=['belt power', 'transmitted power', 'kW'],
         assumptions=['Centrifugal tension cancels in the difference: power is carried by (T1 - T2) alone.'],
         refs=['Belt power P = (T1 - T2) v (recalled)']),

    Calc('belt_slip', 'Belt slip fraction from the speed ratio',
         vars=[V('s', '', 'slip fraction (0 = no slip)', hi=1.0),
               V('n1', 'Hz', 'driver pulley speed in rev/s (enter rpm/60)'),
               V('n2', 'Hz', 'driven pulley speed in rev/s (enter rpm/60)'),
               V('D1', 'm', 'driver pulley diameter'),
               V('D2', 'm', 'driven pulley diameter')],
         equations=['s = 1 - (n2/n1)*(D2/D1)'], outputs=['s'],
         example={'n1': 30.0, 'n2': 7.2, 'D1': 0.2, 'D2': 0.8},
         keywords=['slip', 'creep', 'speed ratio'],
         assumptions=['s = 1 - (n2 D2)/(n1 D1) = (v_driver - v_driven)/v_driver. Flat/vee belts creep '
                      '~1-2%; larger s means the belt is slipping and losing power.'],
         refs=['Belt slip s = 1 - (n2 D2)/(n1 D1) (recalled)']),

    # ---- V-belts -------------------------------------------------------------------------------
    Calc('v_belt_selection', 'V-belt design power and number of belts',
         vars=[V('P_d', 'W', 'design power (service factor x transmitted power)'),
               V('N_b', '', 'number of belts (round up to an integer)'),
               V('K_s', '', 'service factor (catalogue/table input)'),
               V('P', 'W', 'transmitted (nameplate) power'),
               V('P_r', 'W', 'rated power per belt (manufacturer catalogue input)'),
               V('C_theta', '', 'angle-of-wrap correction factor (catalogue input)'),
               V('C_L', '', 'belt-length correction factor (catalogue input)')],
         equations=['P_d = K_s*P', 'N_b = P_d/(P_r*C_theta*C_L)'], outputs=['P_d', 'N_b'],
         example={'K_s': 1.3, 'P': 5000.0, 'P_r': 2000.0, 'C_theta': 0.95, 'C_L': 0.98},
         keywords=['v-belt', 'number of belts', 'service factor', 'rated power'],
         assumptions=['P_r, K_s, C_theta and C_L are catalogue inputs. Round N_b up to the next whole '
                      'belt; each belt shares the load only approximately (small drives use 1 belt).'],
         refs=['Number of belts N_b = P_d / (P_r C_theta C_L) (recalled)']),

    # ---- roller chain ---------------------------------------------------------------------------
    Calc('chain_length', 'Roller-chain length, number of links and velocity-independent length',
         vars=[V('C_p', '', 'center distance in pitches (= C/p)'),
               V('p', 'm', 'chain pitch'),
               V('N1', '', 'small sprocket teeth', lo=6.0, integer=True),
               V('N2', '', 'large sprocket teeth', lo=6.0, integer=True),
               V('L', 'm', 'chain length (metres)'),
               V('L_p', '', 'chain length in pitches'),
               V('L_links', '', 'number of links (rounded up to an even integer)', integer=True)],
         procedure=_chain_length, outputs=['L', 'L_p', 'L_links'],
         example={'C_p': 30.0, 'p': 0.0127, 'N1': 17, 'N2': 51},
         keywords=['roller chain', 'links', 'chain length', 'sprocket', 'pitches'],
         assumptions=['L_p = 2 C_p + (N1+N2)/2 + (N2-N1)^2/(4 pi^2 C_p). The number of links must be an '
                      'even integer so the chain can be joined (offset link otherwise); it is rounded up '
                      'here. Prefer ~30-50 pitches of center distance and N1 >= 17 for smooth running.'],
         refs=['Roller-chain length in pitches (ANSI/ASME B29; recalled)']),

    Calc('chain_velocity', 'Roller-chain velocity (v = p N n)',
         vars=[V('v', 'm/s', 'chain speed'),
               V('p', 'm', 'chain pitch'),
               V('N', '', 'sprocket teeth', lo=6.0, integer=True),
               V('n', 'Hz', 'sprocket speed in rev/s (enter rpm/60)')],
         equations=['v = p*N*n'], outputs=['v'],
         example={'p': 0.0127, 'N': 17, 'n': 20.0},
         keywords=['chain velocity', 'chain speed', 'sprocket'],
         assumptions=['Average chain velocity (ignores the chordal oscillation — see chordal_speed).'],
         refs=['Chain velocity v = p N n (recalled)']),

    Calc('chordal_speed', 'Chordal speed variation of a chain drive (polygon effect)',
         vars=[V('delta_v', '', 'chordal speed variation, delta_v/V (fraction of the mean speed V)'),
               V('N', '', 'sprocket teeth', lo=6.0, integer=True)],
         equations=['delta_v = (pi/N)*(1/sin(pi/N) - 1/tan(pi/N))'], outputs=['delta_v'],
         example={'N': 17},
         keywords=['chordal', 'polygon effect', 'speed variation', 'sprocket'],
         assumptions=['The chain rides the sprocket as a polygon, so its speed oscillates about the mean '
                      'V = p N n. Shigley measures the variation relative to that mean speed: '
                      'delta_v/V = (pi/N)(1/sin(pi/N) - 1/tan(pi/N)). N = 17 -> 0.0171 (1.7%), N = 6 -> '
                      '0.140 (14%). More teeth (N >= 17, preferably 19-25) reduce the variation and its '
                      'whip/whine.'],
         refs=['Chordal speed variation delta_v/V = (pi/N)(1/sin(pi/N) - 1/tan(pi/N)) (Shigley; recalled)']),

    # ---- wire rope ------------------------------------------------------------------------------
    Calc('rope_working_load', 'Wire-rope safe working load (F_work = F_break / n)',
         vars=[V('F_work', 'N', 'safe working load'),
               V('F_break', 'N', 'catalogue breaking strength of the rope'),
               V('n', '', 'factor of safety (design-code input; ~5 hoisting, 8-12 elevators)')],
         equations=['F_work = F_break/n'], outputs=['F_work'],
         example={'F_break': 200000.0, 'n': 5.0},
         keywords=['wire rope', 'working load', 'factor of safety', 'breaking strength'],
         assumptions=['F_break is the catalogue (nominal) breaking strength; n is the design-code factor '
                      'of safety. Check the bending and bearing limits too (rope_bending_stress, '
                      'rope_pressure).'],
         refs=['Safe working load = breaking strength / factor of safety (recalled)']),

    Calc('rope_bending_stress', 'Wire-rope bending stress over a sheave (sigma_b = E_r d_w / D)',
         vars=[V('sigma_b', 'Pa', 'bending stress in the outer wires', label='σb'),
               V('E_r', 'Pa', 'rope modulus of elasticity (input; ~80-90 GPa for wire rope)'),
               V('d_w', 'm', 'wire diameter (not the rope diameter)'),
               V('D', 'm', 'sheave (or drum) diameter')],
         equations=['sigma_b = E_r*d_w/D'], outputs=['sigma_b'],
         example={'E_r': 84e9, 'd_w': 0.002, 'D': 0.6},
         keywords=['wire rope', 'bending stress', 'sheave', 'drum'],
         assumptions=['Bending stress is proportional to the wire diameter over the sheave diameter — '
                      'the reason large D/d_w ratios are required for long rope life (D/d >= 40 typical). '
                      'E_r is an input (rope construction dependent).'],
         refs=['Wire-rope bending stress sigma_b = E_r d_w / D (recalled)']),

    Calc('rope_pressure', 'Wire-rope sheave pressure and p/Su ratio (feeds the fatigue chart)',
         vars=[V('p', 'Pa', 'bearing pressure between rope and sheave'),
               V('r', '', 'pressure ratio p/Su (x-axis of the rope fatigue chart)'),
               V('T', 'N', 'rope tension'),
               V('d', 'm', 'rope diameter'),
               V('D', 'm', 'sheave (or drum) diameter'),
               V('Su', 'Pa', 'wire ultimate tensile strength (input; ~1570-1960 MPa)')],
         equations=['p = 2*T/(d*D)', 'r = p/Su'], outputs=['p', 'r'],
         example={'T': 10000.0, 'd': 0.012, 'D': 0.6, 'Su': 1770e6},
         keywords=['wire rope', 'bearing pressure', 'p/Su', 'fatigue', 'sheave pressure'],
         assumptions=['p = 2 T/(d D) is the contact pressure of the rope on the sheave. The ratio p/Su '
                      'is the abscissa of the rope maker\'s fatigue (bending-cycle) chart; the cycles-to-'
                      'failure are read from that chart (input data, not built in).'],
         refs=['Rope/sheave bearing pressure p = 2T/(d D) (recalled)']),
]

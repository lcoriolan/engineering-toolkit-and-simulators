# DESCRIPTION: Gears section: tooth geometry (module/diametral pitch), proportions, ratios, contact ratio
#              and interference, spur/helical/bevel/worm forces and geometry, Lewis and AGMA bending and
#              contact stress with all modifying factors, planetary torques (Willis speeds live in the
#              mechanisms section), gear life cycles and bending/pitting safety factors.
"""Gears calculators.

Conventions:
- Catalogue / standard table values (Lewis form factor Y, AGMA geometry factors J and I, the modifying
  factors Ko/Kv/Ks/Km/KB/KT/KR/Cf and the elastic coefficient Cp, the stress-cycle factors YN/ZN and the
  hardness-ratio factor CH) are USER INPUTS whose ``desc`` names the AGMA 2001/2101 table or figure; none
  are recalled from memory as defaults.
- Closed forms implemented directly: module/circular-pitch/diametral-pitch geometry, standard full-depth
  tooth proportions (addendum m, dedendum 1.25 m, clearance 0.25 m), the Barth velocity factor (multiplying
  form Kv >= 1), minimum teeth to avoid interference (rack limit N_min = 2k/sin^2(phi) and the
  pinion-with-mating-gear form), and the planetary torques that follow from power balance.
- Standard 20 deg full-depth involute system throughout. Pressure and helix/cone angles are in radians
  internally (enter deg). gear_ratio carries magnitudes only; signed (directional) train values live in
  mechanisms.gear_pair / mechanisms.planetary_trains.
- Planetary SPEEDS (the Willis relation) are owned by mechanisms.planetary_trains; this section keeps only
  the planetary TORQUES.

Relations are the standard ones in gear/machine-design texts (Budynas & Nisbett, *Shigley's Mechanical
Engineering Design*; AGMA 2001-C95 / 2101-D04 fundamental rating equations), cited by relation name,
recalled rather than re-read.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('gears', 'Gears', order=170,
                  summary='Tooth geometry, ratios, contact ratio and interference, tooth forces, Lewis '
                          'and AGMA bending/contact, helical/bevel/worm gears, planetary trains and gear '
                          'life.')

PI = math.pi


def _barth(values):
    """Barth velocity factor Kv (multiplying form, Kv >= 1) for the Lewis bending path, V in m/s.

    Kv_cast   = (3.05 + V)/3.05        cast-iron / cast teeth
    Kv_cut    = (6.1 + V)/6.1          cut or milled teeth
    Kv_hobbed = (3.56 + sqrt(V))/3.56  hobbed or shaped teeth
    Kv_shaved = (5.56 + sqrt(V))/5.56  shaved or ground teeth
    The additive constants carry units of m/s (sqrt(V) terms carry sqrt(m/s)), so this is a procedure
    (not a dimensional equation). Kv >= 1 multiplies the static Lewis stress: sigma = Kv Wt/(F m Y).
    """
    V = values['V']
    sv = math.sqrt(V)
    return {'Kv_cast': (3.05 + V) / 3.05,
            'Kv_cut': (6.1 + V) / 6.1,
            'Kv_hobbed': (3.56 + sv) / 3.56,
            'Kv_shaved': (5.56 + sv) / 5.56}


def _gear_life_cycles(values):
    """Number of load cycles at speed n (rev/s) over L_h hours: N = L_h * 3600 * n."""
    return {'N': values['L_h'] * 3600.0 * values['n']}


CALCS = [
    # ---- tooth geometry ------------------------------------------------------------------------
    Calc('spur_geometry', 'Spur gear module, circular pitch and pitch diameter (metric)',
         vars=[V('m', 'm', 'module (pitch diameter / teeth)'),
               V('p', 'm', 'circular pitch (arc length between adjacent teeth on the pitch circle)'),
               V('d', 'm', 'pitch diameter'),
               V('N', '', 'number of teeth', lo=6.0, integer=True)],
         equations=['m = d/N', 'p = pi*m'], outputs=['m', 'p'],
         example={'d': 0.08, 'N': 20},
         keywords=['module', 'circular pitch', 'pitch diameter', 'diametral'],
         assumptions=['Metric (module) system. Circular pitch p = pi m. For the imperial diametral-pitch '
                      'system see diametral_pitch (m in mm = 25.4 / P_d).'],
         refs=['m = d/N; p = pi m (recalled)']),

    Calc('diametral_pitch', 'Diametral pitch (imperial, teeth per unit length)',
         vars=[V('P_d', '1/m', 'diametral pitch (teeth per metre)'),
               V('N', '', 'number of teeth', lo=6.0, integer=True),
               V('d', 'm', 'pitch diameter')],
         equations=['P_d = N/d'], outputs=['P_d'],
         example={'N': 20, 'd': 0.0635},
         keywords=['diametral pitch', 'imperial', 'Pd'],
         assumptions=['P_d = N/d. The conventional diametral pitch in teeth/inch = P_d x 0.0254 (so '
                      'd = 2.5 in and N = 20 give 8 teeth/inch). Module m (mm) = 25.4 / P_d(in^-1).'],
         refs=['Diametral pitch P_d = N/d (recalled)']),

    Calc('tooth_proportions', 'Standard full-depth tooth proportions (addendum, dedendum, clearance)',
         vars=[V('a', 'm', 'addendum (radial tooth height above the pitch circle)'),
               V('b', 'm', 'dedendum (radial depth below the pitch circle)'),
               V('c', 'm', 'clearance (root clearance)'),
               V('h_t', 'm', 'whole tooth depth (a + b)'),
               V('m', 'm', 'module')],
         equations=['a = m', 'b = 1.25*m', 'c = 0.25*m', 'h_t = a + b'], outputs=['a', 'b', 'c', 'h_t'],
         example={'m': 0.004},
         keywords=['addendum', 'dedendum', 'clearance', 'whole depth', 'full depth'],
         assumptions=['Standard 20 deg full-depth involute proportions: addendum = m, dedendum = 1.25 m, '
                      'clearance = 0.25 m. Stub and other systems use different coefficients.'],
         refs=['Standard full-depth tooth proportions (recalled)']),

    Calc('center_distance', 'Center distance and pitch diameters of a gear pair',
         vars=[V('C', 'm', 'center distance'),
               V('d1', 'm', 'pinion pitch diameter'), V('d2', 'm', 'gear pitch diameter'),
               V('m', 'm', 'module'),
               V('N1', '', 'pinion teeth', lo=6.0, integer=True), V('N2', '', 'gear teeth', lo=6.0, integer=True)],
         equations=['d1 = m*N1', 'd2 = m*N2', 'C = (d1 + d2)/2'], outputs=['d1', 'd2', 'C'],
         example={'m': 0.004, 'N1': 20, 'N2': 40},
         keywords=['center distance', 'centre distance', 'pitch'],
         assumptions=['Standard centers (no profile shift): C = (d1 + d2)/2 = m(N1 + N2)/2.'],
         refs=['Center distance C = (d1 + d2)/2 (recalled)']),

    Calc('gear_ratio', 'Gear ratio, speed ratio and torque ratio',
         vars=[V('i', '', 'gear (train) ratio N2/N1 = omega1/omega2 = T2/T1'),
               V('N1', '', 'driving (pinion) teeth', lo=1.0, integer=True),
               V('N2', '', 'driven (gear) teeth', lo=1.0, integer=True),
               V('omega1', 'rad/s', 'driving angular speed', label='ω1'),
               V('omega2', 'rad/s', 'driven angular speed', label='ω2'),
               V('T1', 'N*m', 'driving torque'), V('T2', 'N*m', 'driven torque')],
         equations=['i = N2/N1', 'omega2 = omega1/i', 'T2 = i*T1'], outputs=['i', 'omega2', 'T2'],
         example={'N1': 20, 'N2': 40, 'omega1': 100.0, 'T1': 10.0},
         keywords=['speed ratio', 'torque ratio', 'reduction', 'train value'],
         assumptions=['Ideal (100% efficiency) single mesh: power is conserved, T1 omega1 = T2 omega2, '
                      'so T2 = i T1. Multiply by an efficiency for real losses. i > 1 is a speed reducer. '
                      'i is a MAGNITUDE (|N2/N1|); the signed train value with direction (external vs '
                      'internal mesh) is carried by mechanisms.gear_pair.'],
         refs=['Speed/torque ratio of a gear pair (recalled)']),

    # ---- contact ratio and interference -------------------------------------------------------
    Calc('contact_ratio', 'Contact ratio of a standard spur gear pair',
         vars=[V('m', 'm', 'module'),
               V('phi', 'rad', 'pressure angle (deg input)', hi=PI / 2, label='φ'),
               V('N1', '', 'pinion teeth', lo=6.0, integer=True),
               V('N2', '', 'gear teeth', lo=6.0, integer=True),
               V('r1', 'm', 'pinion pitch radius'), V('r2', 'm', 'gear pitch radius'),
               V('m_c', '', 'contact ratio (mean number of teeth in contact)')],
         equations=['r1 = m*N1/2', 'r2 = m*N2/2',
                    'm_c = (sqrt((r1 + m)**2 - (r1*cos(phi))**2) + sqrt((r2 + m)**2 - (r2*cos(phi))**2) - (r1 + r2)*sin(phi))/(pi*m*cos(phi))'],
         outputs=['r1', 'r2', 'm_c'],
         example={'m': 0.004, 'phi': math.radians(20.0), 'N1': 20, 'N2': 40},
         keywords=['contact ratio', 'transverse contact', 'involute'],
         assumptions=['Standard full-depth 20 deg involute (addendum = m). m_c = path of contact / base '
                      'pitch. Requires m_c > 1.2 for smooth running; addendum modification (profile shift) '
                      'changes it.'],
         refs=['Length of path of contact / base pitch (AGMA geometry; recalled)']),

    Calc('min_teeth_interference', 'Minimum teeth to avoid interference (with a rack / large gear)',
         vars=[V('N_min', '', 'minimum teeth to avoid undercut'),
               V('k', '', 'addendum factor (addendum/module; 1 for standard full depth)'),
               V('phi', 'rad', 'pressure angle', hi=PI / 2, label='φ')],
         equations=['N_min = 2*k/sin(phi)**2'], outputs=['N_min'],
         example={'k': 1.0, 'phi': math.radians(20.0)},
         keywords=['undercut', 'interference', 'minimum teeth', 'pinion'],
         assumptions=['N_min = 2k/sin^2(phi) is the smallest tooth count on a gear meshing with a rack (or '
                      'a very large gear) that avoids undercut. Standard 20 deg full-depth: N_min = 17.1, '
                      'so use >= 17 (18 recommended). Profile shift allows fewer teeth.'],
         refs=['Minimum teeth to avoid interference N_min = 2k/sin^2(phi) (recalled)']),

    Calc('min_teeth_pinion', 'Minimum pinion teeth to avoid interference with a mating gear (ratio m_G)',
         vars=[V('N_P', '', 'minimum pinion teeth to avoid interference with the gear'),
               V('k', '', 'addendum factor (addendum/module; 1 for standard full depth)'),
               V('m_G', '', 'gear ratio N_G/N_P (mating gear teeth / pinion teeth)'),
               V('phi', 'rad', 'pressure angle', hi=PI / 2, label='φ')],
         equations=['N_P = 2*k/((1 + 2*m_G)*sin(phi)**2) * (m_G + sqrt(m_G**2 + (1 + 2*m_G)*sin(phi)**2))'],
         outputs=['N_P'],
         example={'k': 1.0, 'm_G': 2.0, 'phi': math.radians(20.0)},
         keywords=['undercut', 'interference', 'minimum teeth', 'pinion', 'gear ratio'],
         assumptions=['Minimum pinion teeth to avoid interference (undercut) when meshing with a gear of '
                      'ratio m_G = N_G/N_P. Reduces to the rack limit N_min = 2k/sin^2(phi) as m_G -> inf. '
                      'Standard 20 deg full-depth, m_G = 2: N_P = 14.16 (use 15+). Profile shift allows '
                      'fewer teeth.'],
         refs=['Minimum pinion teeth to avoid interference with a mating gear (recalled)']),

    # ---- tooth forces ---------------------------------------------------------------------------
    Calc('spur_tooth_forces', 'Spur gear tooth forces: tangential and radial (separating)',
         vars=[V('W_t', 'N', 'transmitted (tangential) tooth force'),
               V('W_r', 'N', 'radial (separating) tooth force'),
               V('T', 'N*m', 'torque on the gear being analysed'),
               V('d', 'm', 'pitch diameter of that gear'),
               V('phi', 'rad', 'pressure angle', hi=PI / 2, label='φ')],
         equations=['W_t = 2*T/d', 'W_r = W_t*tan(phi)'], outputs=['W_t', 'W_r'],
         example={'T': 200.0, 'd': 0.08, 'phi': math.radians(20.0)},
         keywords=['tangential force', 'separating force', 'tooth load'],
         assumptions=['Spur gears carry no axial load (W_a = 0). W_t is tangential at the pitch point; '
                      'use the pitch diameter (AGMA uses the operating pitch diameter).'],
         refs=['Tangential tooth load W_t = 2T/d; W_r = W_t tan(phi) (recalled)']),

    Calc('helical_pitch', 'Helical gear normal/transverse pitches, modules and pitch diameter',
         vars=[V('p_n', 'm', 'normal circular pitch'),
               V('p_t', 'm', 'transverse circular pitch'),
               V('m_n', 'm', 'normal module (tooling/standard module)'),
               V('m_t', 'm', 'transverse module'),
               V('psi', 'rad', 'helix angle', hi=PI / 2, label='ψ'),
               V('N', '', 'number of teeth', lo=6.0, integer=True),
               V('d', 'm', 'pitch diameter')],
         equations=['p_n = pi*m_n', 'm_t = m_n/cos(psi)', 'p_t = pi*m_t', 'd = m_t*N'],
         outputs=['p_n', 'm_t', 'p_t', 'd'],
         example={'m_n': 0.004, 'psi': math.radians(30.0), 'N': 20},
         keywords=['helical', 'normal pitch', 'transverse pitch', 'helix angle', 'module'],
         assumptions=['Helical gears are cut in the normal plane, so m_n (and p_n) are the standard '
                      'values. Transverse quantities are the axial-projected ones: m_t = m_n/cos(psi), '
                      'p_t = p_n/cos(psi), d = m_t N.'],
         refs=['Helical gear normal/transverse pitch relations (recalled)']),

    Calc('helical_angles', 'Helical gear transverse pressure angle from normal pressure angle',
         vars=[V('phi_t', 'rad', 'transverse pressure angle', hi=PI / 2, label='φt'),
               V('phi_n', 'rad', 'normal pressure angle', hi=PI / 2, label='φn'),
               V('psi', 'rad', 'helix angle', hi=PI / 2, label='ψ')],
         equations=['phi_t = atan(tan(phi_n)/cos(psi))'], outputs=['phi_t'],
         example={'phi_n': math.radians(20.0), 'psi': math.radians(30.0)},
         keywords=['helical', 'pressure angle', 'normal plane'],
         assumptions=['tan(phi_t) = tan(phi_n)/cos(psi). phi_t > phi_n for helical gears.'],
         refs=['Transverse pressure angle of a helical gear (recalled)']),

    Calc('helical_tooth_forces', 'Helical gear tooth forces: tangential, radial and axial thrust',
         vars=[V('W_t', 'N', 'transmitted (tangential) tooth force'),
               V('W_r', 'N', 'radial (separating) tooth force'),
               V('W_a', 'N', 'axial (thrust) tooth force'),
               V('T', 'N*m', 'torque on the gear being analysed'),
               V('d', 'm', 'pitch diameter'),
               V('phi_n', 'rad', 'normal pressure angle', hi=PI / 2, label='φn'),
               V('psi', 'rad', 'helix angle', hi=PI / 2, label='ψ')],
         equations=['W_t = 2*T/d', 'W_r = W_t*tan(phi_n)/cos(psi)', 'W_a = W_t*tan(psi)'],
         outputs=['W_t', 'W_r', 'W_a'],
         example={'T': 200.0, 'd': 0.08, 'phi_n': math.radians(20.0), 'psi': math.radians(30.0)},
         keywords=['helical', 'axial thrust', 'tangential force', 'radial force'],
         assumptions=['W_a = W_t tan(psi) is the axial thrust the helix produces (use thrust bearings). '
                      'W_r uses the normal pressure angle resolved through the helix.'],
         refs=['Helical tooth force components (recalled)']),

    Calc('bevel_angles', 'Bevel gear pitch cone angles (shaft angle Sigma)',
         vars=[V('gamma1', 'rad', 'pinion pitch cone angle', hi=PI / 2, label='γ1'),
               V('gamma2', 'rad', 'gear pitch cone angle', hi=PI / 2, label='γ2'),
               V('N1', '', 'pinion teeth', lo=1.0, integer=True),
               V('N2', '', 'gear teeth', lo=1.0, integer=True),
               V('Sigma', 'rad', 'shaft (included) angle', hi=PI, label='Σ')],
         equations=['gamma1 = atan(sin(Sigma)/(N2/N1 + cos(Sigma)))',
                    'gamma2 = atan(sin(Sigma)/(N1/N2 + cos(Sigma)))'],
         outputs=['gamma1', 'gamma2'],
         example={'N1': 20, 'N2': 40, 'Sigma': math.radians(90.0)},
         keywords=['bevel', 'pitch cone', 'cone angle', 'miter'],
         assumptions=['Straight/spiral bevel gears. For the common right-angle case (Sigma = 90 deg): '
                      'tan(gamma1) = N1/N2, tan(gamma2) = N2/N1 and gamma1 + gamma2 = Sigma.'],
         refs=['Bevel pitch cone angles (recalled)']),

    Calc('bevel_forces', 'Bevel gear tooth forces: tangential, radial and axial',
         vars=[V('W_t', 'N', 'transmitted (tangential) tooth force at the mean radius'),
               V('W_r', 'N', 'radial tooth force'),
               V('W_a', 'N', 'axial (thrust) tooth force'),
               V('T', 'N*m', 'torque on the gear being analysed'),
               V('d_m', 'm', 'mean pitch diameter of that gear'),
               V('phi', 'rad', 'pressure angle', hi=PI / 2, label='φ'),
               V('gamma', 'rad', 'pitch cone angle of that gear', hi=PI / 2, label='γ')],
         equations=['W_t = 2*T/d_m', 'W_r = W_t*tan(phi)*cos(gamma)', 'W_a = W_t*tan(phi)*sin(gamma)'],
         outputs=['W_t', 'W_r', 'W_a'],
         example={'T': 200.0, 'd_m': 0.08, 'phi': math.radians(20.0), 'gamma': math.radians(26.565)},
         keywords=['bevel', 'thrust', 'radial force', 'axial force'],
         assumptions=['Forces at the mean radius of the bevel gear. The radial/axial split depends on the '
                      'pitch cone angle gamma: the axial component is W_t tan(phi) sin(gamma).'],
         refs=['Bevel gear force components (recalled)']),

    # ---- worm gearing ---------------------------------------------------------------------------
    Calc('worm_geometry', 'Worm lead, lead angle and axial pitch',
         vars=[V('L_w', 'm', 'worm lead (advance per revolution)'),
               V('p_x', 'm', 'axial pitch'),
               V('N_w', '', 'number of starts (threads)', lo=1.0, integer=True),
               V('lam', 'rad', 'lead angle', hi=PI / 2, label='λ'),
               V('d_w', 'm', 'worm pitch diameter')],
         equations=['L_w = p_x*N_w', 'lam = atan(L_w/(pi*d_w))'], outputs=['L_w', 'lam'],
         example={'p_x': 0.01, 'N_w': 2, 'd_w': 0.04},
         keywords=['worm', 'lead', 'lead angle', 'starts'],
         assumptions=['Lead L_w = axial pitch x number of starts; tan(lambda) = L_w/(pi d_w).'],
         refs=['Worm lead and lead angle (recalled)']),

    Calc('worm_efficiency', 'Worm gear efficiency (worm driving)',
         vars=[V('eta', '', 'efficiency (worm driving the gear)', hi=1.0, label='η'),
               V('phi_n', 'rad', 'normal pressure angle', hi=PI / 2, label='φn'),
               V('f', '', 'coefficient of friction at the mesh'),
               V('lam', 'rad', 'lead angle', hi=PI / 2, label='λ')],
         equations=['eta = (cos(phi_n) - f*tan(lam))/(cos(phi_n) + f/tan(lam))'], outputs=['eta'],
         example={'phi_n': math.radians(20.0), 'f': 0.05, 'lam': math.atan(0.02 / (PI * 0.04))},
         keywords=['worm', 'efficiency', 'self locking'],
         assumptions=['Worm driving the gear. If the gear tries to drive the worm (reversal) the '
                      'efficiency is (cos(phi_n) - f/tan(lam))/(cos(phi_n) + f*tan(lam)); the worm is '
                      'self-locking when f >= cos(phi_n)*tan(lam) (the normal pressure angle reduces the '
                      'effective lead-angle tangent; the bare f >= tan(lam) overstates the locking limit).'],
         refs=['Worm-gear efficiency (worm driving) (recalled)']),

    Calc('worm_heat', 'Worm gear power loss and heat dissipation capacity',
         vars=[V('H_loss', 'W', 'power lost to heat (1 - eta) P_in'),
               V('H_diss', 'W', 'housing heat-dissipation capacity h A dT'),
               V('P_in', 'W', 'input power'),
               V('eta', '', 'efficiency', hi=1.0, label='η'),
               V('h', 'W/(m^2*K)', 'overall heat-transfer coefficient of the housing (input)'),
               V('A', 'm^2', 'housing surface area'),
               V('dT', 'K', 'allowable temperature rise (delta, K = degC difference)')],
         equations=['H_loss = (1 - eta)*P_in', 'H_diss = h*A*dT'], outputs=['H_loss', 'H_diss'],
         example={'P_in': 2000.0, 'eta': 0.74, 'h': 15.0, 'A': 0.3, 'dT': 40.0},
         keywords=['worm', 'heat', 'oil', 'dissipation', 'fan'],
         assumptions=['Worm drives are usually limited by heat, not strength: require H_diss >= H_loss. '
                      'h is an input (h ~ 10-15 W/m^2/K for still air, higher with a fan on the worm shaft).'],
         refs=['Worm-gear heat balance (recalled)']),

    # ---- planetary (epicyclic) -------------------------------------------------------------------
    Calc('planetary_torques', 'Planetary gear torques (ideal, lossless)',
         vars=[V('R', '', 'ring-to-sun tooth ratio Z_R/Z_S'),
               V('T_S', 'N*m', 'sun torque', signed=True),
               V('T_R', 'N*m', 'ring torque', signed=True),
               V('T_C', 'N*m', 'carrier torque', signed=True),
               V('Z_S', '', 'sun teeth', lo=6.0, integer=True),
               V('Z_R', '', 'ring teeth', lo=6.0, integer=True)],
         equations=['R = Z_R/Z_S', 'T_R = R*T_S', 'T_C = -(1 + R)*T_S'],
         outputs=['R', 'T_R', 'T_C'],
         example={'Z_S': 20, 'Z_R': 60, 'T_S': 10.0},
         keywords=['epicyclic', 'planetary', 'torque', 'reaction'],
         assumptions=['Ideal lossless train: T_S + T_R + T_C = 0, T_R = R T_S, T_C = -(1 + R) T_S. The '
                      'carrier carries the largest (reaction) torque. Multiply by efficiency for losses. '
                      'Speeds (the Willis relation omega_S - omega_C = -(Z_R/Z_S)(omega_R - omega_C)) are '
                      'owned by mechanisms.planetary_trains; this calculator gives only the torques that '
                      'follow from power balance.'],
         refs=['Planetary torque equilibrium (power balance; recalled)']),

    # ---- strength --------------------------------------------------------------------------------
    Calc('lewis_bending', 'Lewis bending stress (sigma = W_t / (F m Y))',
         vars=[V('sigma', 'Pa', 'tooth bending stress', label='σ'),
               V('W_t', 'N', 'transmitted (tangential) tooth force'),
               V('F', 'm', 'face width'),
               V('m', 'm', 'module'),
               V('Y', '', 'Lewis form factor (module/diametral-pitch based; table/input for the tooth count)')],
         equations=['sigma = W_t/(F*m*Y)'], outputs=['sigma'],
         example={'W_t': 5000.0, 'F': 0.04, 'm': 0.004, 'Y': 0.30},
         keywords=['lewis', 'bending', 'form factor'],
         assumptions=['Static Lewis stress. Y is the Lewis form factor for the standard full-depth tooth, '
                      'based on the module (SI) / diametral-pitch (imperial) system and read from a table '
                      'for the tooth count (input). For a moving tooth the dynamic stress MULTIPLIES by the '
                      'Barth velocity factor Kv (>= 1): sigma = Kv Wt/(F m Y), consistent with the AGMA Kv '
                      '(see barth_velocity_factor). Compare with the material allowable bending stress.'],
         refs=['Lewis bending equation sigma = W_t/(F m Y) (recalled)']),

    Calc('barth_velocity_factor', 'Barth velocity (dynamic) factor Kv (multiplying form, Kv >= 1)',
         vars=[V('V', 'm/s', 'pitch-line velocity'),
               V('Kv_cast', '', 'Kv for cast-iron / cast teeth = (3.05 + V)/3.05'),
               V('Kv_cut', '', 'Kv for cut/milled teeth = (6.1 + V)/6.1'),
               V('Kv_hobbed', '', 'Kv for hobbed/shaped teeth = (3.56 + sqrt(V))/3.56'),
               V('Kv_shaved', '', 'Kv for shaved/ground teeth = (5.56 + sqrt(V))/5.56')],
         procedure=_barth, outputs=['Kv_cast', 'Kv_cut', 'Kv_hobbed', 'Kv_shaved'],
         example={'V': 10.0},
         keywords=['velocity factor', 'dynamic factor', 'barth', 'kv'],
         assumptions=['Barth empirical velocity factor in MULTIPLYING form Kv >= 1, V in m/s. The additive '
                      'constants (3.05, 6.1, 3.56, 5.56 m/s) are empirical; the sqrt(V) forms apply to the '
                      'hobbed/shaped and shaved/ground accuracy classes. Use Kv with the static Lewis stress '
                      'as sigma = Kv Wt/(F m Y); the AGMA quality-number Kv (an input to agma_bending) is a '
                      'different dynamic factor.'],
         refs=['Barth velocity factor (multiplying form; recalled)']),

    Calc('agma_bending', 'AGMA bending stress (all modifying factors)',
         vars=[V('sigma_b', 'Pa', 'AGMA bending stress', label='σb'),
               V('W_t', 'N', 'transmitted tangential load'),
               V('Ko', '', 'overload factor (AGMA 2001 table)'),
               V('Kv', '', 'dynamic factor (AGMA figure/table)'),
               V('Ks', '', 'size factor (AGMA table)'),
               V('Km', '', 'load-distribution factor (AGMA table)'),
               V('KB', '', 'rim-thickness factor (AGMA table)'),
               V('F', 'm', 'face width (narrowest member)'),
               V('m', 'm', 'module (transverse, m_t for helical)'),
               V('J', '', 'geometry factor for bending strength (AGMA figure)')],
         equations=['sigma_b = W_t*Ko*Kv*Ks*Km*KB/(F*m*J)'], outputs=['sigma_b'],
         example={'W_t': 5000.0, 'Ko': 1.25, 'Kv': 1.1, 'Ks': 1.0, 'Km': 1.3, 'KB': 1.0,
                  'F': 0.04, 'm': 0.004, 'J': 0.30},
         keywords=['agma', 'bending stress', '2001', '2101', 'factor of safety'],
         assumptions=['AGMA 2001-C95/2101-D04 bending stress (SI form). All modifying factors Ko/Kv/Ks/'
                      'Km/KB and the geometry factor J are INPUTS naming the AGMA tables/figures. Feed the '
                      'result to gear_safety_factors.'],
         refs=['AGMA fundamental bending stress equation (recalled)']),

    Calc('agma_contact', 'AGMA contact (pitting) stress (all modifying factors)',
         vars=[V('sigma_c', 'Pa', 'AGMA contact stress', label='σc'),
               V('W_t', 'N', 'transmitted tangential load'),
               V('Ko', '', 'overload factor (AGMA 2001 table)'),
               V('Kv', '', 'dynamic factor (AGMA figure/table)'),
               V('Ks', '', 'size factor (AGMA table)'),
               V('Km', '', 'load-distribution factor (AGMA table)'),
               V('Cf', '', 'surface-condition factor (AGMA table; = 1 standard)'),
               V('Cp', 'Pa^0.5', 'elastic coefficient (AGMA table; e.g. 191 MPa^0.5 steel/steel)'),
               V('d_p', 'm', 'pinion operating pitch diameter'),
               V('F', 'm', 'face width (narrowest member)'),
               V('I', '', 'geometry factor for pitting resistance (AGMA figure)')],
         equations=['sigma_c = Cp*sqrt(W_t*Ko*Kv*Ks*Km*Cf/(d_p*F*I))'], outputs=['sigma_c'],
         example={'W_t': 4000.0, 'Ko': 1.25, 'Kv': 1.1, 'Ks': 1.0, 'Km': 1.3, 'Cf': 1.0,
                  'Cp': 191000.0, 'd_p': 0.08, 'F': 0.04, 'I': 0.10},
         keywords=['agma', 'contact stress', 'pitting', 'hertz', 'cp'],
         assumptions=['AGMA 2101-D04 contact (pitting) stress (SI form), sigma_c = Cp sqrt(Wt Ko Kv Ks Km '
                      'Cf /(d_p F I)). Cp, Cf, all modifying factors and the geometry factor I are INPUTS '
                      'naming the AGMA tables/figures.'],
         refs=['AGMA fundamental contact stress equation (recalled)']),

    # ---- life and safety -------------------------------------------------------------------------
    Calc('gear_safety_factors', 'Gear safety factors against bending and pitting',
         vars=[V('n_b', '', 'factor of safety against bending fatigue'),
               V('n_c', '', 'factor of safety against pitting'),
               V('sigma_b', 'Pa', 'AGMA bending stress', label='σb'),
               V('sigma_c', 'Pa', 'AGMA contact stress', label='σc'),
               V('S_t', 'Pa', 'allowable bending stress (material, AGMA S-N)'),
               V('S_c', 'Pa', 'allowable contact stress (material, AGMA S-N)'),
               V('YN', '', 'stress-cycle factor for bending (AGMA figure)'),
               V('ZN', '', 'stress-cycle factor for pitting (AGMA figure)'),
               V('CH', '', 'hardness-ratio factor (AGMA table; = 1 for equal hardness)'),
               V('KT', '', 'temperature factor (AGMA table)'),
               V('KR', '', 'reliability factor (AGMA table)')],
         equations=['n_b = S_t*YN/(KT*KR*sigma_b)', 'n_c = S_c*ZN*CH/(KT*KR*sigma_c)'],
         outputs=['n_b', 'n_c'],
         example={'sigma_b': 186.2e6, 'sigma_c': 902.8e6, 'S_t': 300e6, 'S_c': 1200e6,
                  'YN': 1.0, 'ZN': 1.0, 'CH': 1.0, 'KT': 1.0, 'KR': 1.0},
         keywords=['factor of safety', 'agma', 'bending', 'pitting', 'life'],
         assumptions=['S_t, S_c, YN, ZN, CH, KT, KR are INPUTS naming the AGMA material/figures/tables. '
                      'n_b is the bending factor of safety S_F, compared directly (stress-linear). n_c is '
                      'S_H, a strength/stress ratio: pitting is a Hertzian (stress-squared) failure, so the '
                      'pitting factor of safety is S_H^2 = n_c^2 — compare S_F with S_H^2, not with S_H. '
                      'n_b, n_c >= 1 (design codes apply larger minimums).'],
         refs=['AGMA factor of safety (allowable/applied stress) (recalled)']),

    Calc('gear_life_cycles', 'Gear life: number of load cycles from hours at speed',
         vars=[V('N', '', 'number of load (tooth) cycles'),
               V('L_h', 'h', 'design life in hours'),
               V('n', 'Hz', 'shaft speed in rev/s (enter rpm/60 or "1/min")')],
         procedure=_gear_life_cycles, outputs=['N'],
         example={'L_h': 5000.0, 'n': 30.0},
         keywords=['life', 'cycles', 'load cycles', 'durability'],
         assumptions=['N = L_h * 3600 * n revolutions of that shaft; each tooth of that gear sees one load '
                      'application per revolution in a single mesh. For a gear with several meshes per '
                      'revolution (idler, planet) multiply by the number of meshes. Use the pinion speed for '
                      'the pinion tooth and the gear speed for the gear tooth; feed N into the stress-cycle '
                      'factors YN/ZN (inputs).'],
         refs=['Load cycles N = L_h * 3600 * n (recalled)']),
]

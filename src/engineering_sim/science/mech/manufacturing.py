# DESCRIPTION: Manufacturing-processes section: machining (cutting speed, feed, MRR, time, force/power,
#              Merchant circle, Taylor tool life, surface finish), milling and drilling, grinding,
#              rolling, forging, extrusion, drawing, sheet forming (bend, blank/punch, springback, deep
#              draw), casting (shrinkage, Chvorinov, riser, gating), welding, injection molding, powder
#              metallurgy and AM.
"""Manufacturing-process calculators.

Conventions:
- Spindle speed is carried in rad/s (enter rpm; the surface speed is v = ω·D/2, so no hidden 2π) and in
  rev/s (cyclic, unit Hz) where a feed is per revolution. Feed f is always the advance per revolution,
  declared in m with the 'per revolution' understood to be the spindle revolution; milling feed per tooth
  f_z multiplies by the number of teeth z and the rev/s n.
- Material constants (Taylor n/C, specific cutting energy, strength coefficient K and n, shape factors,
  friction) are user inputs whose desc names the source table; none is recalled from memory as normative.
- Merchant circle sign convention: rake angle α positive; cutting force F_c in the cutting direction,
  thrust force F_t normal to it.
- Springback uses the constant neutral-axis length θ_f/θ_i = (R_i + t/2)/(R_f + t/2) together with the
  elastic-plastic radius-ratio R_i/R_f = 1 − 3x + 4x³, x = σ_y R_i/(E t); x ≥ 0.5 means the whole
  thickness stayed elastic (no plastic bend), which is rejected.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section

SECTION = section('manufacturing', 'Manufacturing processes', order=340,
                  summary='Machining, Merchant/Taylor, milling/drilling, grinding, rolling, forging, '
                          'extrusion, drawing, sheet forming, casting, welding, injection molding, powder '
                          'metallurgy, AM.')

PI = math.pi


def _springback(values):
    """Sheet-bending springback from the constant neutral-axis length.

    R_i/R_f = 1 − 3x + 4x³ with x = σ_y R_i/(E t) (elastic-perfectly-plastic radius recovery); the neutral
    axis at mid-thickness keeps its length, θ_i(R_i + t/2) = θ_f(R_f + t/2), so θ_f/θ_i = (R_i + t/2)/(R_f + t/2).
    """
    sigma_y, E, R_i, t, theta_i = values['sigma_y'], values['E'], values['R_i'], values['t'], values['theta_i']
    if E <= 0:
        raise MechError('elastic modulus E must be positive')
    if t <= 0:
        raise MechError('sheet thickness t must be positive')
    if R_i <= 0:
        raise MechError('bend radius R_i must be positive')
    x = sigma_y * R_i / (E * t)
    if x >= 0.5:
        raise MechError('x = σ_y·R_i/(E·t) = %.4g is at or past the fully-elastic limit 0.5: the bend is '
                        'not plastic, so the springback model does not apply' % x)
    ratio = 1.0 - 3.0 * x + 4.0 * x ** 3
    R_f = R_i / ratio
    theta_f = theta_i * (R_i + t / 2.0) / (R_f + t / 2.0)
    return {'ratio': ratio, 'R_f': R_f, 'theta_f': theta_f, 'theta_sb': theta_i - theta_f}


def _injection_cooling_time(values):
    """Injection-molding cooling time: first Fourier term of a slab cooled from both faces.

    profile='mean' uses the mean (thickness-averaged) temperature coefficient 8/π²;
    profile='centreline' uses the mid-plane coefficient 4/π.
    """
    s, alpha = values['s'], values['alpha']
    T_m, T_w, T_e = values['T_m'], values['T_w'], values['T_e']
    if alpha <= 0:
        raise MechError('thermal diffusivity α must be positive')
    if s <= 0:
        raise MechError('part wall thickness s must be positive')
    if T_m <= T_w:
        raise MechError('melt temperature T_m must exceed the mold wall temperature T_w')
    if T_e <= T_w:
        raise MechError('ejection temperature T_e must exceed the mold wall temperature T_w')
    K = 8.0 / PI ** 2 if values['profile'] == 'mean' else 4.0 / PI
    arg = K * (T_m - T_w) / (T_e - T_w)
    if arg <= 1.0:
        raise MechError('ejection temperature is too close to the melt temperature '
                        '(cooling time would be zero or negative)')
    t = s ** 2 / (PI ** 2 * alpha) * math.log(arg)
    return {'t': t}


def _weld_cooling_rate(values):
    """Weld cooling rate (3-D thick-plate Rosenthal): R = 2πk(T_c − T_0)²/H."""
    k, T_c, T_0, H = values['k'], values['T_c'], values['T_0'], values['H']
    if T_c <= T_0:
        raise MechError('temperature of interest T_c must exceed the preheat temperature T_0 '
                        '(at or below preheat there is no cooling)')
    if H <= 0:
        raise MechError('net heat input H must be positive')
    if k <= 0:
        raise MechError('thermal conductivity k must be positive')
    return {'R': 2.0 * PI * k * (T_c - T_0) ** 2 / H}


def _am_layer_count(values):
    """3-D printing layer count, rounded UP to a whole number of layers."""
    H, L_t = values['H'], values['L_t']
    if L_t <= 0:
        raise MechError('layer thickness L_t must be positive')
    if H <= 0:
        raise MechError('build height H must be positive')
    n = math.ceil(H / L_t - 1e-9)
    return {'N': float(n), 'H_built': n * L_t}


CALCS = [
    # ---- machining --------------------------------------------------------------------------------
    Calc('cutting_speed', 'Machining (cutting) speed from spindle speed and diameter',
         vars=[V('v', 'm/s', 'cutting (surface) speed'),
               V('omega', 'rad/s', 'spindle angular speed (enter rpm or rps)', label='ω'),
               V('D', 'm', 'workpiece (turning) or tool (milling) diameter')],
         equations=['v = omega*D/2'], outputs=['v'],
         example={'omega': 1000 * 2 * PI / 60, 'D': 0.05},
         keywords=['surface speed', 'sfpm', 'rpm', 'turning', 'milling'],
         assumptions=['v = ω·D/2 uses ω in rad/s directly; there is no extra 2π for an rpm entry '
                      '(1000 rpm = 104.72 rad/s).'],
         refs=['Surface speed v = πDN = ω·D/2 (recalled)']),

    Calc('feed_rate', 'Machining feed rate from feed per revolution and spindle speed',
         vars=[V('f_m', 'm/s', 'linear feed rate'), V('f', 'm', 'feed per revolution of the spindle'),
               V('n', 'Hz', 'spindle speed in revolutions per second (1500 rpm = 25 1/s)')],
         equations=['f_m = f*n'], outputs=['f_m'],
         example={'f': 0.0002, 'n': 25.0},
         keywords=['feed', 'ipm', 'mm/min', 'turning'],
         assumptions=['n is cyclic (rev/s): enter Hz, 1/s or 1/min, not rpm (pint treats rpm as angular and '
                      'the core refuses the mix-up). Divide rpm by 60 to get rev/s.'],
         refs=['Feed rate = feed per revolution × spindle speed (recalled)']),

    Calc('milling_feed_rate', 'Milling table feed rate from feed per tooth',
         vars=[V('f_m', 'm/s', 'linear table feed rate'),
               V('f_z', 'm', 'feed per tooth (chip load per tooth)'),
               V('z', '', 'number of teeth on the cutter', lo=1.0, integer=True),
               V('n', 'Hz', 'spindle speed in revolutions per second (1500 rpm = 25 1/s)')],
         equations=['f_m = f_z*z*n'], outputs=['f_m'],
         example={'f_z': 0.0001, 'z': 4.0, 'n': 25.0},
         keywords=['milling', 'feed per tooth', 'chip load', 'table feed'],
         assumptions=['Milling feed rate = feed per tooth × number of teeth × spindle rev/s (f_z·z·N). '
                      'Distinct from turning, where the feed is already per revolution.'],
         refs=['Milling feed rate = f_z·z·n (recalled)']),

    Calc('mrr', 'Material removal rate (turning)',
         vars=[V('MRR', 'm^3/s', 'volumetric material removal rate'),
               V('v', 'm/s', 'cutting speed'), V('f', 'm', 'feed per revolution'), V('d', 'm', 'depth of cut')],
         equations=['MRR = v*f*d'], outputs=['MRR'],
         example={'v': 2.618, 'f': 0.0002, 'd': 0.001},
         keywords=['material removal', 'chip', 'productivity'],
         assumptions=['Turning MRR = v·f·d (the swept chip area is f·d).'],
         refs=['MRR = v·f·d (recalled)']),

    Calc('milling_mrr', 'Material removal rate (milling)',
         vars=[V('MRR', 'm^3/s', 'volumetric material removal rate'),
               V('f_m', 'm/s', 'linear table feed rate'),
               V('a_e', 'm', 'radial engagement (width of cut, stepover)'),
               V('a_p', 'm', 'axial depth of cut')],
         equations=['MRR = f_m*a_e*a_p'], outputs=['MRR'],
         example={'f_m': 0.01, 'a_e': 0.02, 'a_p': 0.003},
         keywords=['milling', 'material removal', 'productivity'],
         assumptions=['Milling MRR = table feed rate × radial engagement × axial depth (the swept chip area is '
                      'a_e·a_p).'],
         refs=['Milling MRR = f_m·a_e·a_p (recalled)']),

    Calc('drilling_mrr', 'Material removal rate (drilling)',
         vars=[V('MRR', 'm^3/s', 'volumetric material removal rate'),
               V('D', 'm', 'drill diameter'),
               V('f_m', 'm/s', 'axial feed rate (m/s)')],
         equations=['MRR = pi*D**2/4*f_m'], outputs=['MRR'],
         example={'D': 0.01, 'f_m': 0.001},
         keywords=['drilling', 'material removal', 'productivity'],
         assumptions=['Drilling MRR = hole area (πD²/4) × axial feed rate.'],
         refs=['Drilling MRR = (πD²/4)·f_m (recalled)']),

    Calc('machining_time', 'Machining time for a straight cut',
         vars=[V('t', 's', 'cutting time'), V('L', 'm', 'length of cut'),
               V('f', 'm', 'feed per revolution'), V('n', 'Hz', 'spindle speed (rev/s; 1500 rpm = 25 1/s)')],
         equations=['t = L/(f*n)'], outputs=['t'],
         example={'L': 0.1, 'f': 0.0002, 'n': 25.0},
         keywords=['cycle time', 'pass', 'turning'],
         assumptions=['One pass at constant spindle speed; feed per revolution × rev/s = linear feed rate.'],
         refs=['t = L/(f·n) (recalled)']),

    Calc('cutting_force_power', 'Cutting force and machining power from specific cutting energy',
         vars=[V('F_c', 'N', 'cutting force'), V('P_c', 'W', 'machining (cutting) power'),
               V('u', 'J/m^3', 'specific cutting energy (material constant, e.g. Kienzle)'),
               V('f', 'm', 'feed per revolution'), V('d', 'm', 'depth of cut'), V('v', 'm/s', 'cutting speed')],
         equations=['F_c = u*f*d', 'P_c = F_c*v'], outputs=['F_c', 'P_c'],
         example={'u': 2.5e9, 'f': 0.0002, 'd': 0.001, 'v': 2.618},
         keywords=['specific energy', 'kienzle', 'cutting power', 'spindle power'],
         assumptions=['u is the specific cutting energy (energy per unit volume, a material input); '
                      'F_c = u·(chip area f·d).'],
         refs=['F_c = u·f·d; P_c = F_c·v (recalled)']),

    Calc('chip_thickness_ratio', 'Chip thickness ratio (undeformed / actual chip thickness)',
         vars=[V('r', '', 'chip thickness ratio t0/tc', hi=1.0),
               V('t0', 'm', 'undeformed chip thickness (feed)'), V('tc', 'm', 'actual chip thickness')],
         equations=['r = t0/tc'], outputs=['r'],
         example={'t0': 0.0002, 'tc': 0.0005},
         keywords=['chip', 'cutting ratio', 'orthogonal'],
         assumptions=['r < 1 because the chip is thicker than the undeformed layer (r = 1 only for no shear).'],
         refs=['Chip thickness ratio r = t0/tc (recalled)']),

    Calc('shear_angle_merchant', 'Shear plane angle from the chip thickness ratio (Merchant geometry)',
         vars=[V('phi', 'rad', 'shear plane angle', hi=PI / 2, label='φ'),
               V('r', '', 'chip thickness ratio t0/tc', hi=1.0),
               V('alpha', 'rad', 'rake angle', signed=True, lo=-PI / 3, hi=PI / 3, label='α')],
         equations=['phi = atan(r*cos(alpha)/(1 - r*sin(alpha)))'], outputs=['phi'],
         example={'r': 0.4, 'alpha': math.radians(10)},
         keywords=['merchant', 'shear angle', 'chip ratio', 'orthogonal'],
         assumptions=['Geometric relation between the rake angle and the chip ratio; α positive as drawn.'],
         refs=['tan φ = r cos α / (1 − r sin α) (Merchant, recalled)']),

    Calc('merchant_friction_angle', 'Merchant circle friction angle and coefficient',
         vars=[V('beta', 'rad', 'friction angle on the rake face', hi=PI / 2, label='β'),
               V('mu', '', 'mean friction coefficient on the rake face', label='μ'),
               V('F_c', 'N', 'cutting force'), V('F_t', 'N', 'thrust force'),
               V('alpha', 'rad', 'rake angle', signed=True, lo=-PI / 3, hi=PI / 3, label='α')],
         equations=['beta = atan((F_c*sin(alpha) + F_t*cos(alpha))/(F_c*cos(alpha) - F_t*sin(alpha)))',
                    'mu = tan(beta)'],
         outputs=['beta', 'mu'],
         example={'F_c': 600.0, 'F_t': 250.0, 'alpha': math.radians(10)},
         keywords=['merchant circle', 'friction', 'orthogonal cutting'],
         assumptions=['Forces resolved on the rake face: F = F_c sin α + F_t cos α, N = F_c cos α − F_t sin α, '
                      'μ = F/N. Requires N > 0.'],
         refs=['Merchant force circle, μ = (F_c sin α + F_t cos α)/(F_c cos α − F_t sin α) (recalled)']),

    Calc('taylor_tool_life', 'Taylor tool life equation (V·Tⁿ = constant)',
         vars=[V('V1', 'm/s', 'cutting speed at tool life T1'), V('T1', 's', 'tool life at cutting speed V1'),
               V('V2', 'm/s', 'second cutting speed'), V('T2', 's', 'tool life at cutting speed V2'),
               V('n', '', 'Taylor exponent (material/tool constant)')],
         equations=['V1*T1**n = V2*T2**n'], outputs=['T2'],
         example={'V1': 3.0, 'T1': 300.0, 'V2': 3.5, 'n': 0.2},
         keywords=['tool life', 'taylor', 'wear', 'tool wear'],
         assumptions=['V·Tⁿ = C is constant, so one known (V1, T1) point plus the exponent n predicts T2 at V2 '
                      '(T2 = T1·(V1/V2)^(1/n)). C itself is a material/tool input recovered from the data point.'],
         refs=['Taylor tool-life law V·Tⁿ = C (recalled)']),

    Calc('surface_roughness', 'Ideal turned-surface roughness from feed and nose radius',
         vars=[V('R_a', 'm', 'arithmetic mean roughness (ideal geometry)'),
               V('R_t', 'm', 'peak-to-valley (maximum) roughness (ideal geometry)'),
               V('f', 'm', 'feed per revolution'), V('r_n', 'm', 'tool nose radius')],
         equations=['R_a = f**2/(32*r_n)', 'R_t = f**2/(8*r_n)'], outputs=['R_a', 'R_t'],
         example={'f': 0.0002, 'r_n': 0.0008},
         keywords=['surface finish', 'ra', 'rt', 'turning', 'nose radius'],
         assumptions=['Ideal (geometric) profile only, R_a ≈ f²/(32 r_n) and R_t ≈ f²/(8 r_n); real finish is '
                      'rougher (built-up edge, vibration, feed marks).'],
         refs=['Ideal surface roughness R_a = f²/(32·r_n), R_t = f²/(8·r_n) (recalled)']),

    # ---- grinding ----------------------------------------------------------------------------------
    Calc('grinding', 'Grinding wheel speed and material removal rate',
         vars=[V('v_s', 'm/s', 'wheel surface speed'),
               V('MRR', 'm^3/s', 'volumetric material removal rate'),
               V('omega_s', 'rad/s', 'wheel angular speed (enter rpm)', label='ωs'),
               V('D_s', 'm', 'wheel diameter'),
               V('v_w', 'm/s', 'workpiece (table) speed'),
               V('d', 'm', 'depth of cut (wheel infeed)'), V('b', 'm', 'width of cut (wheel engagement)')],
         equations=['v_s = omega_s*D_s/2', 'MRR = v_w*d*b'], outputs=['v_s', 'MRR'],
         example={'omega_s': 2000 * 2 * PI / 60, 'D_s': 0.15, 'v_w': 0.2, 'd': 0.00002, 'b': 0.01},
         keywords=['surface grinding', 'wheel speed', 'abrasive'],
         assumptions=['MRR = workpiece speed × depth of cut × width; wheel speed v_s = ω_s·D_s/2 (no hidden 2π).'],
         refs=['Grinding MRR = v_w·d·b; wheel speed v_s = ω_s·D_s/2 (recalled)']),

    # ---- rolling -----------------------------------------------------------------------------------
    Calc('rolling_force', 'Flat rolling force, contact length, draft and roll pressure',
         vars=[V('F', 'N', 'rolling force'), V('L', 'm', 'projected contact length'),
               V('dh', 'm', 'draft (thickness reduction h0 − hf)'),
               V('p', 'Pa', 'mean roll pressure F/(w·L) (= average flow stress in this model)'),
               V('w', 'm', 'strip width'), V('sigma_f', 'Pa', 'average flow stress', label='σf'),
               V('R', 'm', 'roll radius'), V('h0', 'm', 'entry thickness'), V('hf', 'm', 'exit thickness')],
         equations=['dh = h0 - hf', 'L = sqrt(R*dh)', 'F = w*sigma_f*L', 'p = F/(w*L)'],
         outputs=['F', 'L', 'dh', 'p'],
         example={'w': 0.3, 'sigma_f': 150e6, 'R': 0.25, 'h0': 0.02, 'hf': 0.018},
         keywords=['cold rolling', 'hot rolling', 'draft', 'contact length'],
         assumptions=['No friction-hill correction (p = σ_f); hf ≤ h0 required.'],
         refs=['Rolling force F = w·σ_f·√(R·Δh); contact length L = √(R·Δh) (recalled)']),

    Calc('rolling_max_draft', 'Maximum draft for roll bite (friction limit)',
         vars=[V('dh_max', 'm', 'maximum draft in one pass'),
               V('mu', '', 'friction coefficient between roll and strip', label='μ'),
               V('R', 'm', 'roll radius')],
         equations=['dh_max = mu**2*R'], outputs=['dh_max'],
         example={'mu': 0.2, 'R': 0.25},
         keywords=['bite', 'friction', 'draft limit', 'rolling'],
         assumptions=['Bite condition tan α ≤ μ with Δh ≈ R·tan²α gives Δh_max = μ²R (small-angle form).'],
         refs=['Roll bite condition Δh_max = μ²R (recalled)']),

    # ---- forging -----------------------------------------------------------------------------------
    Calc('flow_stress', 'Flow stress from the power-law (Hollomon) equation',
         vars=[V('sigma_f', 'Pa', 'flow stress', label='σf'),
               V('K', 'Pa', 'strength coefficient (material input)'),
               V('epsilon', '', 'true (logarithmic) strain', label='ε'),
               V('n', '', 'strain-hardening exponent (material input)')],
         equations=['sigma_f = K*epsilon**n'], outputs=['sigma_f'],
         example={'K': 600e6, 'epsilon': 0.3, 'n': 0.2},
         keywords=['hollomon', 'true stress', 'strain hardening', 'cold work'],
         assumptions=['σ_f = K·εⁿ; K and n are material inputs (cold-working curves).'],
         refs=['Hollomon power-law flow stress (recalled)']),

    Calc('forging_force', 'Forging force from flow stress and projected area',
         vars=[V('F', 'N', 'forging force'), V('K_f', '', 'forging shape (multiplication) factor', lo=1.0),
               V('sigma_f', 'Pa', 'flow stress', label='σf'), V('A', 'm^2', 'projected area of the forging')],
         equations=['F = K_f*sigma_f*A'], outputs=['F'],
         example={'K_f': 5.0, 'sigma_f': 400e6, 'A': 0.005},
         keywords=['open die', 'closed die', 'press force', 'upset'],
         assumptions=['K_f accounts for friction and complex shape (a table input, ~1 for simple upset to ~8 for '
                      'complex closed-die shapes).'],
         refs=['F = K_f·σ_f·A (recalled)']),

    Calc('forging_upset_force', 'Flat-die upsetting force with friction (open-die forging)',
         vars=[V('F', 'N', 'upsetting force'),
               V('sigma_f', 'Pa', 'flow stress of the workpiece (Y_f)', label='σf'),
               V('r', 'm', 'final disc (contact) radius'),
               V('mu', '', 'friction coefficient at the die–workpiece interface', label='μ'),
               V('h', 'm', 'final disc height')],
         equations=['F = sigma_f*pi*r**2*(1 + 2*mu*r/(3*h))'],
         outputs=['F'],
         example={'sigma_f': 400e6, 'r': 0.05, 'mu': 0.2, 'h': 0.03},
         keywords=['upset forging', 'open die', 'friction hill', 'press force'],
         assumptions=['Axisymmetric upsetting between flat dies with Coulomb friction; the bracketed term '
                      '(1 + 2μr/(3h)) is the friction-hill correction to the frictionless force σ_f·πr². '
                      'Requires h > 0.'],
         refs=['Flat-die upsetting force F = Y_f·πr²(1 + 2μr/(3h)) (recalled)']),

    # ---- extrusion ---------------------------------------------------------------------------------
    Calc('extrusion', 'Extrusion ratio and ideal extrusion pressure',
         vars=[V('R', '', 'extrusion ratio A0/Af', lo=1.0),
               V('p', 'Pa', 'ideal extrusion pressure'),
               V('A0', 'm^2', 'initial (billet) cross-sectional area'),
               V('Af', 'm^2', 'final (extruded) cross-sectional area'),
               V('sigma_f', 'Pa', 'flow stress', label='σf')],
         equations=['R = A0/Af', 'p = sigma_f*ln(R)'], outputs=['R', 'p'],
         example={'A0': 1e-3, 'Af': 1e-4, 'sigma_f': 400e6},
         keywords=['extrusion', 'direct', 'indirect', 'ram pressure'],
         assumptions=['Ideal (frictionless, no redundant work) pressure p = σ_f·ln R; real ram pressure is higher.'],
         refs=['Extrusion ratio R = A0/Af; ideal pressure p = σ_f·ln R (recalled)']),

    # ---- drawing -----------------------------------------------------------------------------------
    Calc('drawing', 'Wire/bar drawing force and reduction',
         vars=[V('F', 'N', 'drawing force'), V('r', '', 'area reduction (A0 − Af)/A0', hi=1.0),
               V('A0', 'm^2', 'initial cross-sectional area'), V('Af', 'm^2', 'final cross-sectional area'),
               V('sigma_f', 'Pa', 'flow stress', label='σf')],
         equations=['F = Af*sigma_f*ln(A0/Af)', 'r = 1 - Af/A0'], outputs=['F', 'r'],
         example={'A0': 1e-3, 'Af': 1e-4, 'sigma_f': 400e6},
         keywords=['wire drawing', 'bar drawing', 'reduction', 'die'],
         assumptions=['Ideal (frictionless) drawing stress σ_d = σ_f·ln(A0/Af); F = σ_d·Af. Af ≤ A0 required.'],
         refs=['Drawing force F = A_f·σ_f·ln(A0/Af) (recalled)']),

    Calc('drawing_max_reduction', 'Maximum drawing reduction per pass (die stress limit)',
         vars=[V('r_max', '', 'maximum area reduction per pass', hi=1.0),
               V('sigma_allow', 'Pa', 'allowable die (drawing) stress'),
               V('sigma_f', 'Pa', 'flow stress of the material', label='σf')],
         equations=['r_max = 1 - exp(-sigma_allow/sigma_f)'], outputs=['r_max'],
         example={'sigma_allow': 400e6, 'sigma_f': 400e6},
         keywords=['drawing limit', 'die stress', 'reduction limit'],
         assumptions=['At the limit the drawing stress equals the allowable die stress σ_allow = σ_f·ln(A0/Af). '
                      'With σ_allow = σ_f the ideal limit is r_max = 1 − 1/e ≈ 0.632.'],
         refs=['r_max = 1 − exp(−σ_allow/σ_f) (recalled)']),

    # ---- sheet forming -----------------------------------------------------------------------------
    Calc('bend_allowance', 'Sheet metal bend allowance (arc length at the neutral axis)',
         vars=[V('BA', 'm', 'bend allowance (neutral-axis arc length)'),
               V('alpha', 'rad', 'bend angle (included bend, swept angle)', label='α'),
               V('R', 'm', 'inside bend radius'), V('K', '', 'K-factor (neutral-axis offset / thickness)'),
               V('t', 'm', 'sheet thickness')],
         equations=['BA = alpha*(R + K*t)'], outputs=['BA'],
         example={'alpha': PI / 2, 'R': 0.002, 'K': 0.33, 't': 0.002},
         keywords=['bend deduction', 'k factor', 'neutral axis', 'forming'],
         assumptions=['Neutral axis at K·t from the inside surface (K ≈ 0.33 for common steels; input).'],
         refs=['Bend allowance BA = α(R + Kt) (recalled)']),

    Calc('bend_force', 'V-die bending (air bend) force',
         vars=[V('F', 'N', 'bending force'), V('K_b', '', 'die opening / force factor (≈1.33 for a V-die)'),
               V('L', 'm', 'bend length'), V('t', 'm', 'sheet thickness'),
               V('S_ut', 'Pa', 'tensile strength of the sheet'), V('W', 'm', 'die opening width')],
         equations=['F = K_b*L*t**2*S_ut/W'], outputs=['F'],
         example={'K_b': 1.33, 'L': 0.5, 't': 0.002, 'S_ut': 450e6, 'W': 0.02},
         keywords=['press brake', 'air bend', 'v die', 'forming force'],
         assumptions=['K_b ≈ 1.33 for a V-die with W ≈ 8t (input); force grows as t².'],
         refs=['V-bending force F = K_b·L·t²·S_ut/W (recalled)']),

    Calc('springback', 'Sheet bending springback (constant neutral-axis length)',
         vars=[V('ratio', '', 'radius ratio R_i/R_f (≤ 1; smaller = more springback)', hi=1.0),
               V('R_f', 'm', 'bend radius after unloading'),
               V('theta_f', 'rad', 'bend angle after unloading', label='θf'),
               V('theta_sb', 'rad', 'springback angle (θ_i − θ_f)'),
               V('sigma_y', 'Pa', 'yield strength', label='σy'), V('E', 'Pa', 'elastic modulus'),
               V('R_i', 'm', 'bend radius before unloading'), V('t', 'm', 'sheet thickness'),
               V('theta_i', 'rad', 'bend angle before unloading', label='θi')],
         procedure=_springback,
         outputs=['ratio', 'R_f', 'theta_f', 'theta_sb'],
         example={'sigma_y': 210e6, 'E': 70e9, 'R_i': 0.01, 't': 0.002, 'theta_i': PI / 2},
         keywords=['springback', 'overbend', 'elastic recovery', 'forming'],
         assumptions=['Elastic-perfectly-plastic bending. The radius recovery is R_i/R_f = 1 − 3x + 4x³ with '
                      'x = σ_y·R_i/(E·t), and the neutral axis at mid-thickness keeps its length, '
                      'θ_f/θ_i = (R_i + t/2)/(R_f + t/2). Valid for x < 0.5 (x ≥ 0.5 is a fully elastic bend with '
                      'no permanent set, which is rejected).'],
         refs=['Springback radius ratio R_i/R_f = 1 − 3(σy Ri/Et) + 4(σy Ri/Et)³ and neutral-axis '
               'θ_f/θ_i = (R_i + t/2)/(R_f + t/2) (recalled)']),

    Calc('blanking_punching_force', 'Blanking / punching force',
         vars=[V('F', 'N', 'blanking or punching force'),
               V('P_per', 'm', 'sheared perimeter (πD for a round hole)'),
               V('t', 'm', 'sheet thickness'), V('tau_s', 'Pa', 'shear strength of the sheet')],
         equations=['F = P_per*t*tau_s'], outputs=['F'],
         example={'P_per': PI * 0.02, 't': 0.002, 'tau_s': 300e6},
         keywords=['punching', 'blanking', 'shearing', 'press'],
         assumptions=['τ_s ≈ 0.7·UTS when only the tensile strength is known (input).'],
         refs=['Blanking force F = perimeter × t × shear strength (recalled)']),

    Calc('blanking_clearance', 'Blanking / punching die clearance',
         vars=[V('c', 'm', 'clearance per side'), V('k', '', 'clearance factor (fraction of thickness, e.g. 0.06)'),
               V('t', 'm', 'sheet thickness')],
         equations=['c = k*t'], outputs=['c'],
         example={'k': 0.06, 't': 0.002},
         keywords=['die clearance', 'punch', 'blanking', 'shear'],
         assumptions=['Typical clearance per side is 3–10% of thickness depending on material (input).'],
         refs=['Die clearance c = k·t (recalled)']),

    # ---- deep drawing ------------------------------------------------------------------------------
    Calc('blank_diameter', 'Blank diameter for a cylindrical deep-drawn cup',
         vars=[V('D_b', 'm', 'blank diameter'),
               V('d_p', 'm', 'cup (punch) diameter'), V('h', 'm', 'cup height')],
         equations=['D_b = sqrt(d_p**2 + 4*d_p*h)'], outputs=['D_b'],
         example={'d_p': 0.05, 'h': 0.03},
         keywords=['deep drawing', 'blank', 'cup', 'equal area'],
         assumptions=['Constant wall thickness, equal surface area (no bottom radius or flange added).'],
         refs=['Blank diameter D_b = √(d_p² + 4·d_p·h) (recalled)']),

    Calc('draw_ratio', 'Draw ratio and limiting draw ratio check',
         vars=[V('DR', '', 'draw ratio D_b/d_p', lo=1.0),
               V('margin', '', 'feasibility margin LDR/DR (≥ 1 = feasible in one draw)'),
               V('LDR', '', 'limiting draw ratio (material limit, ≈1.8–2.2 first draw)'),
               V('D_b', 'm', 'blank diameter'), V('d_p', 'm', 'cup (punch) diameter')],
         equations=['DR = D_b/d_p', 'margin = LDR/DR'], outputs=['DR', 'margin'],
         example={'D_b': 0.092195, 'd_p': 0.05, 'LDR': 2.0},
         keywords=['deep drawing', 'ldr', 'drawability', 'wrinkling'],
         assumptions=['Feasible in one draw when DR ≤ LDR (margin ≥ 1), else redraw or anneal. LDR is a '
                      'material input, not computed here.'],
         refs=['Draw ratio DR = D_b/d_p (recalled)']),

    # ---- casting -----------------------------------------------------------------------------------
    Calc('pattern_dimension', 'Casting pattern dimensions from shrinkage allowance',
         vars=[V('L_p', 'm', 'pattern dimension'), V('delta', 'm', 'shrinkage allowance L_c·s'),
               V('L_c', 'm', 'desired casting dimension'), V('s', '', 'shrinkage factor (fraction, e.g. 0.02)')],
         equations=['L_p = L_c*(1 + s)', 'delta = L_c*s'], outputs=['L_p', 'delta'],
         example={'L_c': 0.1, 's': 0.02},
         keywords=['shrinkage', 'pattern', 'allowance', 'casting'],
         assumptions=['The pattern is made LARGER than the casting (L_p = L_c(1+s)) so the metal shrinks to the '
                      'desired size. s is the linear shrinkage as a fraction m/m (0.02 for steel, ~0.01 for cast '
                      'iron; input) — sources quoting mm/m or % must be divided by 1000 or 100 respectively.'],
         refs=['Pattern dimension = casting dimension × (1 + s) (recalled)']),

    Calc('chvorinov', 'Chvorinov solidification time',
         vars=[V('t_s', 's', 'solidification time'), V('C', 's/m^2', 'mold constant (material/mold input)'),
               V('V', 'm^3', 'casting volume'), V('A', 'm^2', 'casting surface area')],
         equations=['t_s = C*(V/A)**2'], outputs=['t_s'],
         example={'C': 5e5, 'V': 1e-4, 'A': 0.02},
         keywords=['chvorinov', 'solidification', 'mold constant', 'modulus'],
         assumptions=['t_s = C(V/A)² with the mold constant C as input (order 4·10⁵ s/m² for steel in sand).'],
         refs=['Chvorinov rule t_s = C·(V/A)² (recalled)']),

    Calc('riser_sizing', 'Riser modulus and adequacy ratio (cylindrical side riser)',
         vars=[V('M_r', 'm', 'riser modulus V/A'), V('ratio', '', 'M_r / M_c (≥ 1.2 to feed the casting)'),
               V('D', 'm', 'riser diameter'), V('H', 'm', 'riser height'),
               V('M_c', 'm', 'casting modulus (V/A of the casting)')],
         equations=['M_r = D*H/(4*H + 2*D)', 'ratio = M_r/M_c'], outputs=['M_r', 'ratio'],
         example={'D': 0.05, 'H': 0.075, 'M_c': 0.005},
         keywords=['riser', 'feeder', 'modulus', 'solidification'],
         assumptions=['Cylindrical side riser (side + both ends cooling): M = DH/(4H + 2D). A top riser '
                      '(side + top cooling) is DH/(4H + D). Keep ratio ≥ 1.2.'],
         refs=['Riser modulus = V/A; cylindrical side riser M = DH/(4H + 2D) (recalled)']),

    Calc('gating_velocity', 'Gating (pouring) velocity from sprue height (Torricelli)',
         vars=[V('v', 'm/s', 'metal velocity at the gate'), V('g', 'm/s^2', 'acceleration of gravity'),
               V('h', 'm', 'sprue height (head of metal above the gate)')],
         equations=['v = sqrt(2*g*h)'], outputs=['v'],
         example={'g': 9.80665, 'h': 0.2},
         keywords=['gating', 'sprue', 'pouring', 'runner', 'torricelli'],
         assumptions=['Ideal (loss-free) flow from the pouring basin; real gates use a velocity < this.'],
         refs=['Gating velocity v = √(2gh) (Torricelli, recalled)']),

    # ---- welding -----------------------------------------------------------------------------------
    Calc('weld_heat_input', 'Arc-welding heat input per unit length',
         vars=[V('H', 'J/m', 'heat input per unit weld length'),
               V('V', 'V', 'arc voltage'), V('I', 'A', 'welding current'),
               V('eta', '', 'arc (process) efficiency', hi=1.0, label='η'),
               V('v', 'm/s', 'travel speed')],
         equations=['H = V*I*eta/v'], outputs=['H'],
         example={'V': 25.0, 'I': 200.0, 'eta': 0.8, 'v': 0.005},
         keywords=['heat input', 'arc welding', 'smaw', 'gmaw', 'j/m'],
         assumptions=['η ≈ 0.7–0.85 for SMAW/GMAW, ~0.9 submerged arc (input).'],
         refs=['Heat input H = V·I·η/v (recalled)']),

    Calc('weld_melting_efficiency', 'Welding melting efficiency',
         vars=[V('eta_m', '', 'melting efficiency (energy to melt / heat input)', hi=1.0, label='ηm'),
               V('A_m', 'm^2', 'weld bead cross-sectional area'),
               V('rho', 'kg/m^3', 'density of the deposited metal', label='ρ'),
               V('v', 'm/s', 'travel speed'),
               V('c', 'J/(kg*K)', 'specific heat of the metal'),
               V('T_m', 'K', 'melting temperature'), V('T_0', 'K', 'initial (ambient) temperature'),
               V('L_f', 'J/kg', 'latent heat of fusion'),
               V('V', 'V', 'arc voltage'), V('I', 'A', 'welding current'),
               V('eta', '', 'arc (process) efficiency', hi=1.0, label='η')],
         equations=['eta_m = A_m*rho*v*(c*(T_m - T_0) + L_f)/(V*I*eta)'],
         outputs=['eta_m'],
         example={'A_m': 3e-5, 'rho': 7800.0, 'v': 0.005, 'c': 500.0, 'T_m': 1800.0, 'T_0': 300.0,
                  'L_f': 270e3, 'V': 25.0, 'I': 200.0, 'eta': 0.8},
         keywords=['melting efficiency', 'fusion', 'weld pool'],
         assumptions=['Energy to melt = A_m·ρ·v·(c·ΔT + L_f) (sensible + latent, no superheat); heat input = V·I·η.'],
         refs=['Melting efficiency = ṁ(c·ΔT + L_f)/(V·I·η) (recalled)']),

    Calc('weld_cooling_rate', 'Weld cooling rate (3-D thick-plate Rosenthal)',
         vars=[V('R', 'K/s', 'cooling rate at temperature T_c'),
               V('k', 'W/(m*K)', 'thermal conductivity of the plate'),
               V('T_c', 'K', 'temperature of interest (e.g. 550 °C for steel)'),
               V('T_0', 'K', 'preheat / initial temperature'),
               V('H', 'J/m', 'net heat input per unit length')],
         procedure=_weld_cooling_rate,
         outputs=['R'],
         example={'k': 40.0, 'T_c': 823.0, 'T_0': 298.0, 'H': 8e5},
         keywords=['cooling rate', 'rosenthal', 'haz', 'preheat', 'martensite'],
         assumptions=['Thick (3-D) plate: R = 2πk(T_c − T_0)²/H. For thin plates the 2-D form '
                      'R = 2πkρc(t/H)²(T_c − T_0)³ applies instead. Requires T_c > T_0 (there is no cooling at '
                      'or below the preheat temperature).'],
         refs=['Rosenthal thick-plate cooling rate (recalled)']),

    # ---- injection molding --------------------------------------------------------------------------
    Calc('injection_clamping_force', 'Injection molding clamping force',
         vars=[V('F', 'N', 'clamping force'), V('P_inj', 'Pa', 'injection (cavity) pressure'),
               V('A_proj', 'm^2', 'projected area of the part (and runner)')],
         equations=['F = P_inj*A_proj'], outputs=['F'],
         example={'P_inj': 35e6, 'A_proj': 0.02},
         keywords=['injection molding', 'clamp', 'tonnage', 'cavity pressure'],
         assumptions=['Cavity pressure (typically 20–60 MPa for thermoplastics) is an input.'],
         refs=['Clamping force F = P_inj·A_proj (recalled)']),

    Calc('injection_cooling_time', 'Injection molding cooling time (slab conduction)',
         vars=[V('t', 's', 'cooling time to the ejection temperature'),
               V('s', 'm', 'part wall thickness'),
               V('alpha', 'm^2/s', 'thermal diffusivity of the polymer', label='α'),
               V('T_m', 'K', 'melt temperature'), V('T_w', 'K', 'mold wall temperature'),
               V('T_e', 'K', 'ejection temperature'),
               V('profile', '', 'temperature basis of T_e', kind='text',
                 choices=['mean', 'centreline'])],
         procedure=_injection_cooling_time,
         outputs=['t'],
         example={'s': 0.002, 'alpha': 1e-7, 'T_m': 503.0, 'T_w': 323.0, 'T_e': 373.0,
                  'profile': 'mean'},
         keywords=['injection molding', 'cycle time', 'cooling', 'thermal diffusivity'],
         assumptions=['First-term (Fourier) solution of a slab cooled from both faces. profile="mean" uses the '
                      'thickness-averaged temperature with the 8/π² coefficient (T_e is then the MEAN ejection '
                      'temperature); profile="centreline" uses the mid-plane temperature with the 4/π coefficient. '
                      'Requires T_m > T_w and T_e > T_w.'],
         refs=['Injection-molding cooling time t = (s²/π²α)·ln[K(T_m − T_w)/(T_e − T_w)] with K = 8/π² (mean) or '
               '4/π (centreline) (recalled)']),

    # ---- powder metallurgy --------------------------------------------------------------------------
    Calc('pm_density', 'Powder-metallurgy green/sintered density',
         vars=[V('rho', 'kg/m^3', 'green (or sintered) density', label='ρ'),
               V('rho_rel', '', 'relative density rho/rho_theory', hi=1.0),
               V('m', 'kg', 'compact mass'), V('V', 'm^3', 'compact volume'),
               V('rho_theory', 'kg/m^3', 'theoretical (pore-free) density')],
         equations=['rho = m/V', 'rho_rel = rho/rho_theory'], outputs=['rho', 'rho_rel'],
         example={'m': 0.05, 'V': 8e-6, 'rho_theory': 8000.0},
         keywords=['powder metallurgy', 'green density', 'sintering', 'porosity'],
         assumptions=['Density from mass and volume; relative density = fraction of the pore-free density.'],
         refs=['Density ρ = m/V (recalled)']),

    Calc('pm_heckel', 'Powder compaction pressure (Heckel equation)',
         vars=[V('rho_rel', '', 'relative density (0..1)', hi=1.0),
               V('A', '', 'Heckel intercept (die-fill parameter)'),
               V('K', '1/Pa', 'Heckel slope parameter (yield-pressure related)'),
               V('P', 'Pa', 'compaction pressure')],
         equations=['rho_rel = 1 - exp(-(A + K*P))'], outputs=['rho_rel'],
         example={'A': 0.6, 'K': 5e-9, 'P': 300e6},
         keywords=['heckel', 'compaction', 'densification', 'green strength'],
         assumptions=['Heckel compaction law ln(1/(1−ρ_rel)) = A + K·P; A and K are material inputs.'],
         refs=['Heckel equation ln(1/(1 − ρ_rel)) = A + K·P (recalled)']),

    # ---- additive manufacturing ---------------------------------------------------------------------
    Calc('am_layer_count', '3D printing layer count (rounded up to a whole layer)',
         vars=[V('N', '', 'number of layers (rounded up to a whole layer)', lo=1.0, integer=True),
               V('H', 'm', 'part build height'), V('L_t', 'm', 'layer thickness'),
               V('H_built', 'm', 'actual built height N·L_t')],
         procedure=_am_layer_count,
         outputs=['N', 'H_built'],
         example={'H': 0.05, 'L_t': 0.0002},
         no_invert={'H': 'ceil(H/L_t) is not one-to-one: any H in ((N−1)·L_t, N·L_t] gives the same N',
                    'L_t': 'ceil(H/L_t) is not one-to-one in L_t either'},
         keywords=['additive', '3d printing', 'layers', 'slicing', 'fused deposition'],
         assumptions=['The layer count is rounded UP to a whole number of layers (N = ceil(H/L_t)); H_built = '
                      'N·L_t is the height actually built, which is ≥ H unless H/L_t is exact.'],
         refs=['N = ceil(H/L_t) (recalled)']),

    Calc('am_build_time', '3D printing build time (extrusion-limited)',
         vars=[V('t', 's', 'printing time (extrusion-limited)'),
               V('V', 'm^3', 'part volume'), V('A_e', 'm^2', 'extruded bead cross-sectional area'),
               V('v', 'm/s', 'print head speed')],
         equations=['t = V/(A_e*v)'], outputs=['t'],
         example={'V': 1e-5, 'A_e': 1.2e-7, 'v': 0.05},
         keywords=['additive', '3d printing', 'build time', 'print speed'],
         assumptions=['Extrusion-limited time = volume/(bead area × head speed); travel, acceleration and '
                      'layer changes add overhead not counted here.'],
         refs=['Build time t = V/(A_e·v) (recalled)']),
]

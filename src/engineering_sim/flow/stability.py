import math
import numpy as np
from . import aero
DEG = math.pi / 180.0
G = 9.80665

def _cg_reference(config):
    """Reference point (metres) at the CG: x = flight.cg_x, y=z=0."""
    cgx = config['flight'].get('cg_x', 0.0) or 0.0
    return np.array([cgx / 1000.0, 0.0, 0.0])

def _reference_quantities(config):
    wing = config['wing']
    return {'S': wing['area_mm2'] / 1000000.0, 'c': wing['mac'] / 1000.0, 'b': wing['span_total'] / 1000.0}

def _coef(config, sys_, ref, alpha=0.0, beta=0.0, rates=None):
    """Coefficients for one velocity field on a pre-built VLMSystem."""
    V = config['flight']['V_ms']
    rho = config['flight']['rho']
    vel = aero.freestream_velocity(V, alpha, beta, rates, sys_.lat.colloc, ref)
    res = sys_.solve(vel, rho, ref)
    return aero.coefficients(sys_.lat, res['F_g'], res['M_g'], rho, V, ref, D_i=res['D_i'])

def _limit(config, key):
    """Acceptance limit `key` from config['limits'] (parametric override, integrator addition 2026-09-15 02:15 MDT | 2026-09-15 04:15 EDT | 2026-09-15 08:15 Zulu), else config['declared']."""
    return tuple((config.get('limits') or {}).get(key, config['declared'][key]))

def morph_deflections(config, delta_deg, differential=False):
    """Deflection list for the wing morph zones.

    ``delta_deg`` > 0 = trailing edge DOWN (more camber, more lift).  If
    ``differential``, the right half deflects -delta (TE up) and the left +delta
    (TE down) — the conventional aileron sense producing a right roll for delta>0.
    """
    defs = []
    for zone in config['morph_zones']:
        half = 1 if zone['half'] == 'R' else -1
        d = delta_deg if not differential else -delta_deg if half == 1 else delta_deg
        defs.append({'surface': 'wing', 'half': half, 'mode': 'morph', 'x_start': zone['x_start_frac'], 'delta_deg': d, 'y0': zone['y0'] / 1000.0, 'y1': zone['y1'] / 1000.0})
    return defs

def flap_deflections(config, delta_deg, differential=False):
    """Deflection list for the conventional hinged ailerons (plain flap) for
    comparison against the morph zones; same differential sense as morph.

    Uses the measured aileron hinge (``chord_frac`` from the baseline control
    surfaces) when present, else the declared 0.65.  The flap spans the union of the
    aileron y-ranges, matching the morph-zone span.
    """
    css = config['wing'].get('control_surfaces', [])
    if css:
        y0 = min((cs['y0'] for cs in css)) / 1000.0
        y1 = max((cs['y1'] for cs in css)) / 1000.0
        fracs = [cs['chord_frac'] for cs in css if cs.get('chord_frac') is not None]
        x_start = sum(fracs) / len(fracs) if fracs else config['declared']['aileron_x_start']
    else:
        y0, y1 = (0.0, 0.0)
        x_start = config['declared']['aileron_x_start']
    defs = []
    for half, sign in ((+1, 1.0), (-1, -1.0)):
        d = delta_deg if not differential else -delta_deg if half == 1 else delta_deg
        defs.append({'surface': 'wing', 'half': half, 'mode': 'flap', 'x_start': x_start, 'delta_deg': d, 'y0': sign * min(y0, y1), 'y1': sign * max(y0, y1)})
    return defs

def ruddervator_deflections(config, delta_deg, differential=False):
    """Deflection list for the V-tail ruddervators (plain flap on the tail)."""
    x_start = config['vtail'].get('ruddervator_x_start_frac', config['declared']['ruddervator_x_start'])
    defs = []
    for half in (+1, -1):
        d = delta_deg if not differential else -delta_deg if half == 1 else delta_deg
        defs.append({'surface': 'tail', 'half': half, 'mode': 'flap', 'x_start': x_start, 'delta_deg': d})
    return defs

def _central(f, h):
    """Central-difference slope of scalar-returning f at 0 with step h."""
    return (f(h) - f(-h)) / (2 * h)

def longitudinal(config, reference=None, spanwise=24, chordwise=6, body_cm_alpha=0.0):
    """CL_alpha, Cm_alpha, CL_q, Cm_q about the CG, plus the neutral point and SM.

    NP: x_np = x_ref - (dCm/dalpha)/(dCL/dalpha) * c, central differences +/-1 deg.
    ``body_cm_alpha`` is the declared Munk body pitch-stiffness term (per rad, from
    :func:`body_corrections`); it is folded into the body-inclusive NP/SM so the
    reported static margin is physical.  Both the VLM-only and body-inclusive NP/SM
    are returned (destabilising: the body term moves the NP forward).
    """
    ref = _cg_reference(config) if reference is None else np.asarray(reference, float)
    rq = _reference_quantities(config)
    c = rq['c']
    V = config['flight']['V_ms']
    sys_ = aero.VLMSystem(config, deflections=None, spanwise=spanwise, chordwise=chordwise)

    def CL(a):
        return _coef(config, sys_, ref, alpha=a)['CL']

    def Cm(a):
        return _coef(config, sys_, ref, alpha=a)['Cm']
    CL_alpha = _central(CL, 1.0 * DEG) / DEG
    Cm_alpha = _central(Cm, 1.0 * DEG) / DEG
    Cm_alpha_body = float(body_cm_alpha)
    Cm_alpha_total = Cm_alpha + Cm_alpha_body
    x_np = ref[0] - Cm_alpha / CL_alpha * c
    sm = (x_np - ref[0]) / c
    x_np_body = ref[0] - Cm_alpha_total / CL_alpha * c
    sm_body = (x_np_body - ref[0]) / c
    hq = 0.01

    def _rate(qhat):
        q = qhat * 2 * V / c
        return _coef(config, sys_, ref, rates=(0.0, q, 0.0))
    CL_q = _central(lambda qh: _rate(qh)['CL'], hq)
    Cm_q = _central(lambda qh: _rate(qh)['Cm'], hq)
    return {'CL_alpha': CL_alpha, 'Cm_alpha': Cm_alpha, 'Cm_alpha_body': Cm_alpha_body, 'Cm_alpha_total': Cm_alpha_total, 'CL_q': CL_q, 'Cm_q': Cm_q, 'x_np': x_np, 'static_margin': sm, 'x_np_body': x_np_body, 'static_margin_body': sm_body, 'reference': ref, 'checks': {'Cm_alpha_negative': Cm_alpha_total < 0, 'SM_in_band': _limit(config, 'SM_accept')[0] <= sm_body <= _limit(config, 'SM_accept')[1]}}

def lateral_directional(config, reference=None, spanwise=24, chordwise=6, body_cn_beta=0.0):
    """Lateral-directional derivatives Cy_beta, Cn_beta, Cl_beta, Cn_r, Cl_r, Cn_p,
    Cl_p from VLM sideslip and body-rate solves, about the CG.

    ``body_cn_beta`` is the declared Munk body yaw-destabilising term (per rad, from
    :func:`body_corrections`); it is folded into the body-inclusive Cn_beta so the
    reported directional stability is physical.  Both VLM-only and body-inclusive
    values are returned (destabilising: the body term reduces Cn_beta).
    """
    ref = _cg_reference(config) if reference is None else np.asarray(reference, float)
    rq = _reference_quantities(config)
    b, V = (rq['b'], config['flight']['V_ms'])
    sys_ = aero.VLMSystem(config, deflections=None, spanwise=spanwise, chordwise=chordwise)

    def coef(beta=0.0, rates=None):
        return _coef(config, sys_, ref, beta=beta, rates=rates)
    hb = 1.0 * DEG
    Cy_beta = _central(lambda x: coef(beta=x)['CY'], hb) / DEG
    Cn_beta = _central(lambda x: coef(beta=x)['Cn'], hb) / DEG
    Cl_beta = _central(lambda x: coef(beta=x)['Cl'], hb) / DEG
    Cn_beta_body = float(body_cn_beta)
    Cn_beta_total = Cn_beta + Cn_beta_body
    hr = 0.01

    def _yawrate(rhat):
        r = rhat * 2 * V / b
        return coef(rates=(0.0, 0.0, r))

    def _rollrate(phat):
        p = phat * 2 * V / b
        return coef(rates=(p, 0.0, 0.0))
    Cn_r = _central(lambda x: _yawrate(x)['Cn'], hr)
    Cl_r = _central(lambda x: _yawrate(x)['Cl'], hr)
    Cn_p = _central(lambda x: _rollrate(x)['Cn'], hr)
    Cl_p = _central(lambda x: _rollrate(x)['Cl'], hr)
    return {'Cy_beta': Cy_beta, 'Cn_beta': Cn_beta, 'Cn_beta_body': Cn_beta_body, 'Cn_beta_total': Cn_beta_total, 'Cl_beta': Cl_beta, 'Cn_r': Cn_r, 'Cl_r': Cl_r, 'Cn_p': Cn_p, 'Cl_p': Cl_p, 'checks': {'Cn_beta_positive': Cn_beta_total > 0, 'Cl_beta_negative': Cl_beta < 0, 'Cl_p_negative': Cl_p < 0}}

def vtail_volumes(config):
    """Horizontal and vertical projected V-tail volume coefficients.

    Projection convention: S_h = S_tail cos^2(Gamma), S_v = S_tail sin^2(Gamma)
    (Raymer's V-tail projection; the squared factors apportion the dihedral tail
    area to its horizontal/vertical roles).  l_t is the tail arm (tail quarter-MAC
    x - wing quarter-MAC x).  V_h = S_h l_t /(S c), V_v = S_v l_t /(S b).
    """
    rq = _reference_quantities(config)
    S, c, b = (rq['S'], rq['c'], rq['b'])
    vt = config['vtail']
    gamma = math.radians(vt.get('dihedral_deg', 0.0))
    S_tail = vt['area_mm2'] / 1000000.0
    wing_qx = (config['wing'].get('mac_le_x', 0.0) + config['wing'].get('mac', 0.0) / 4) / 1000.0
    tail_qx = (vt.get('mac_le_x', 0.0) + vt.get('mac', 0.0) / 4) / 1000.0
    l_t = tail_qx - wing_qx
    Sh = S_tail * math.cos(gamma) ** 2
    Sv = S_tail * math.sin(gamma) ** 2
    Vh = Sh * l_t / (S * c)
    Vv = Sv * l_t / (S * b)
    lo, hi = _limit(config, 'Vh_accept')
    vlo, vhi = _limit(config, 'Vv_accept')
    return {'S_tail': S_tail, 'l_t': l_t, 'Gamma_deg': vt.get('dihedral_deg', 0.0), 'S_h': Sh, 'S_v': Sv, 'V_h': Vh, 'V_v': Vv, 'checks': {'Vh_in_band': lo <= Vh <= hi, 'Vv_in_band': vlo <= Vv <= vhi}}

def control_derivatives(config, reference=None, spanwise=24, chordwise=6):
    """Roll/yaw control from the morph zones, and pitch control from the ruddervator.

    Differential morph: Cl_delta_morph, Cn_delta_morph (per rad), and the ratio of
    morph vs the original plain aileron (Cl only).  Symmetric ruddervator (elevator):
    CL_de and Cm_de (per rad).  Central differences +/-1 deg (six VLM systems total).
    """
    ref = _cg_reference(config) if reference is None else np.asarray(reference, float)
    deg = 1.0

    def coefs(deflections):
        sys_ = aero.VLMSystem(config, deflections=deflections, spanwise=spanwise, chordwise=chordwise)
        return _coef(config, sys_, ref)
    mp = coefs(morph_deflections(config, deg, differential=True))
    mm = coefs(morph_deflections(config, -deg, differential=True))
    Cl_da_morph = (mp['Cl'] - mm['Cl']) / (2.0 * deg * DEG)
    Cn_da_morph = (mp['Cn'] - mm['Cn']) / (2.0 * deg * DEG)
    fp = coefs(flap_deflections(config, deg, differential=True))
    fm = coefs(flap_deflections(config, -deg, differential=True))
    Cl_da_flap = (fp['Cl'] - fm['Cl']) / (2.0 * deg * DEG)
    ratio = Cl_da_morph / Cl_da_flap if abs(Cl_da_flap) > 1e-12 else None
    rudd = None
    if config.get('vtail'):
        rp = coefs(ruddervator_deflections(config, deg, differential=False))
        rm = coefs(ruddervator_deflections(config, -deg, differential=False))
        rudd = {'CL_de': (rp['CL'] - rm['CL']) / (2.0 * deg * DEG), 'Cm_de': (rp['Cm'] - rm['Cm']) / (2.0 * deg * DEG)}
    return {'Cl_delta_morph': Cl_da_morph, 'Cn_delta_morph': Cn_da_morph, 'Cl_delta_aileron': Cl_da_flap, 'ratio_morph_over_aileron': ratio, 'CL_de': rudd['CL_de'] if rudd else None, 'Cm_de': rudd['Cm_de'] if rudd else None}

def roll_authority(config, cd=None, ld=None, reference=None):
    """Steady non-dimensional roll rate p_hat = p b/(2V) at full morph deflection.

    p_hat = - (Cl_delta * delta) / Cl_p  (rolling moment balanced by roll damping).
    Accepts precomputed control/lateral derivatives to avoid re-solving.  Also reports
    the conventional-aileron p_hat at the same deflection for comparison, and, when
    morph p_hat falls below the MIL-F-8785C-style 0.07 target, the deflection that
    would reach it (linear in delta) rather than inventing a new limit.
    """
    ref = _cg_reference(config) if reference is None else np.asarray(reference, float)
    if cd is None:
        cd = control_derivatives(config, reference=ref)
    if ld is None:
        ld = lateral_directional(config, reference=ref)
    Cl_da = cd['Cl_delta_morph']
    Cl_da_ail = cd['Cl_delta_aileron']
    Cl_p = ld['Cl_p']
    limit = max((z['limit_deg'] for z in config['morph_zones']), default=12.0)
    p_hat = -Cl_da * (limit * DEG) / Cl_p if abs(Cl_p) > 1e-12 else None
    p_hat_ail = -Cl_da_ail * (limit * DEG) / Cl_p if abs(Cl_p) > 1e-12 and Cl_da_ail is not None else None
    target = config['declared']['roll_authority']
    delta_needed = None
    if p_hat is not None and abs(p_hat) > 1e-12 and (abs(p_hat) < target):
        delta_needed = limit * target / abs(p_hat)
    ok = p_hat is not None and abs(p_hat) >= target
    return {'Cl_delta_morph': Cl_da, 'Cl_delta_aileron': Cl_da_ail, 'Cl_p': Cl_p, 'delta_full_deg': limit, 'p_hat': p_hat, 'p_hat_aileron': p_hat_ail, 'delta_needed_deg': delta_needed, 'checks': {'roll_authority_ok': ok}}

def trim(config, mode='ruddervator', n_load=None, V=None, limit=None, reference=None, spanwise=24, chordwise=6):
    """Trim the symmetric control deflection for Cm = 0 at CL = n W/(q S).

    Nested bisection: for each deflection delta, solve alpha so CL(alpha,delta) equals
    the required CL, then bisect delta to zero the pitching moment Cm about the CG.
    ``V`` is the airspeed for the case (default cruise V_ms; the n=4 pull-up uses
    V_A).  ``limit`` defaults to the morph +/-12 deg for ``mode="morph"`` and the
    declared ruddervator +/-20 deg for ``mode="ruddervator"``.  Reports the trimmed
    deflection and angle of attack and whether the deflection is inside limits.
    """
    ref = _cg_reference(config) if reference is None else np.asarray(reference, float)
    rq = _reference_quantities(config)
    S = rq['S']
    if V is None:
        V = config['flight']['V_ms']
    rho = config['flight']['rho']
    n = n_load if n_load is not None else config['flight']['load_factor']
    mass = (config['mass'] or {}).get('total_g', 0.0)
    W = mass / 1000.0 * G
    qS = 0.5 * rho * V * V * S
    CL_required = n * W / qS if qS > 0 else 0.0

    def deflect(delta):
        if mode == 'morph':
            return morph_deflections(config, delta)
        return ruddervator_deflections(config, delta)
    if limit is None:
        limit = max((z['limit_deg'] for z in config['morph_zones']), default=12.0) if mode == 'morph' else config['declared']['ruddervator_limit_deg']

    def system(delta):
        return aero.VLMSystem(config, deflections=deflect(delta), spanwise=spanwise, chordwise=chordwise)

    def alpha_for_CL(delta, sys_):
        lo, hi = (-20.0, 20.0)
        for _ in range(40):
            mid = (lo + hi) / 2
            if _coef(config, sys_, ref, alpha=mid)['CL'] < CL_required:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def Cm_trim(delta):
        sys_ = system(delta)
        a = alpha_for_CL(delta, sys_)
        return _coef(config, sys_, ref, alpha=a)['Cm']
    lo, hi = (-limit, limit)
    Cm_lo, Cm_hi = (Cm_trim(lo), Cm_trim(hi))
    if Cm_lo * Cm_hi > 0:
        delta_trim = None
        inside = False
    else:
        for _ in range(40):
            mid = (lo + hi) / 2
            if Cm_trim(mid) * Cm_lo > 0:
                lo = mid
            else:
                hi = mid
        delta_trim = (lo + hi) / 2
        inside = abs(delta_trim) <= limit
    alpha_trim = None if delta_trim is None else alpha_for_CL(delta_trim, system(delta_trim))
    alpha_at_bound = alpha_trim is not None and abs(alpha_trim) >= 19.5
    return {'mode': mode, 'V_ms': V, 'limit_deg': limit, 'CL_required': CL_required, 'delta_trim_deg': delta_trim, 'alpha_trim_deg': alpha_trim, 'alpha_at_bound': alpha_at_bound, 'inside_limits': inside, 'checks': {'trim_inside_limits': inside}}

def body_corrections(config):
    """Declared Munk/Gilruth-White slender-body pitch/yaw destabilising corrections.

    Cm_alpha,body = +2 * Vol_body / (S * c) per rad (pitch, destabilising: nose-up with
    increasing alpha).  Cn_beta,body = -2 * Vol_body / (S * b) per rad (yaw,
    destabilising: reduces weathercock stability, DATCOM-style).  The fuselage volume
    is a prismatic-of-revolution estimate from the fuselage stations when present,
    else 0 (reported as omitted).  These are separate declared corrections folded into
    the NP/SM/Cn_beta outputs (not into the raw VLM coefficients).
    """
    rq = _reference_quantities(config)
    S, c, b = (rq['S'], rq['c'], rq['b'])
    fus = config.get('fuselage') or {}
    vol = 0.0
    sts = fus.get('stations') or []
    wing_st = (config.get('wing') or {}).get('stations') or []
    root = min(wing_st, key=lambda st: st['y']) if wing_st else None
    x_lo, x_hi = (root['le_x'], root['te_x']) if root else (float('inf'), float('-inf'))
    excluded_mm = 0.0
    if len(sts) >= 2:
        for a, z in zip(sts, sts[1:]):
            w = (a.get('width', 0.0) + z.get('width', 0.0)) / 2
            h = (a.get('height', 0.0) + z.get('height', 0.0)) / 2
            x0, x1 = sorted((a['x'], z['x']))
            inside = max(0.0, min(x1, x_hi) - max(x0, x_lo))
            dx = x1 - x0 - inside
            excluded_mm += inside
            vol += math.pi / 4 * w * h * dx
        vol /= 1000000000.0
    Cm_alpha_body = 2 * vol / (S * c) if S * c > 0 else 0.0
    Cn_beta_body = -2 * vol / (S * b) if S * b > 0 else 0.0
    return {'body_volume_m3': vol, 'excluded_wing_root_chord_mm': excluded_mm, 'Cm_alpha_body_per_rad': Cm_alpha_body, 'Cn_beta_body_per_rad': Cn_beta_body, 'basis': config['declared']['munk_body'] + '; ' + config['declared']['body_yaw']}
body_pitch_correction = body_corrections

def maneuver_speed(config, n=None):
    """Stall and manoeuvre (corner) speeds from the declared CL_max.

    V_stall = sqrt(2 W / (rho S CL_max)); V_A = V_stall * sqrt(n).  ``n`` defaults to
    the declared load factor.  CL_max is the declared E205 section value (cited-typical).
    """
    n = n if n is not None else config['flight']['load_factor']
    rq = _reference_quantities(config)
    S = rq['S']
    rho = config['flight']['rho']
    mass = (config['mass'] or {}).get('total_g', 0.0)
    W = mass / 1000.0 * G
    CL_max = (config.get('viscous') or {}).get('CL_max') or config['declared']['CL_max']
    V_stall = math.sqrt(2.0 * W / (rho * S * CL_max)) if S > 0 and CL_max > 0 and (W > 0) else 0.0
    return {'n': n, 'CL_max': CL_max, 'V_stall_ms': V_stall, 'V_A_ms': V_stall * math.sqrt(n)}

def _n4_spanwise_load(config, tr, spanwise, chordwise):
    """Wing spanwise lift per unit span (N/m) at the trimmed n=4 pull-up (V_A).

    Uses the already-computed n=4 ruddervator trim ``tr``; extracts the wing-only
    strip loads (Kutta-Joukowski rho V gamma).  Returns {"y_m": [...],
    "lift_N_per_m": [...]} plus the trimmed state, or an empty load when the trim is
    not achievable.
    """
    a, de = (tr['alpha_trim_deg'], tr['delta_trim_deg'])
    if a is None or de is None:
        return {'y_m': [], 'lift_N_per_m': [], 'trimmed': False, 'alpha_trim_deg': None, 'delta_trim_deg': None}
    V = tr['V_ms']
    ref = _cg_reference(config)
    rho = config['flight']['rho']
    sys_ = aero.VLMSystem(config, deflections=ruddervator_deflections(config, de), spanwise=spanwise, chordwise=chordwise)
    vel = aero.freestream_velocity(V, a, 0.0, None, sys_.lat.colloc, ref)
    res = sys_.solve(vel, rho, ref)
    swl = aero.spanwise_load(sys_.lat, res['gamma'], rho, V, surface_id=0)
    return {'y_m': [float(v) for v in swl['y']], 'lift_N_per_m': [float(v) for v in swl['lift_per_span']], 'trimmed': True, 'alpha_trim_deg': a, 'delta_trim_deg': de}

def _fmt(v, spec='%.1f'):
    """Format a possibly-None value for a finding string."""
    return spec % v if v is not None else 'n/a'

def analyze(config, spanwise=24, chordwise=6):
    """Run the full stability analysis and return a schema-shaped result dict.

    Adds the round-2 outputs Task C consumes: ``values.derivatives`` (flat, per-rad),
    ``values.spanwise_load`` (wing lift per span at the trimmed n=4 pull-up), and
    ``values.reference`` (S, c, b, V_ms, rho, cg_x_mm).  Body pitch/yaw corrections are
    folded into NP/SM/Cn_beta (with/without reported); trims are 1 g cruise (V_ms) and
    n=4 pull-up (V_A) with a +/-20 deg ruddervator, plus the wing-morph trim at n=4.
    """
    body = body_corrections(config)
    longi = longitudinal(config, spanwise=spanwise, chordwise=chordwise, body_cm_alpha=body['Cm_alpha_body_per_rad'])
    latd = lateral_directional(config, spanwise=spanwise, chordwise=chordwise, body_cn_beta=body['Cn_beta_body_per_rad'])
    vv = vtail_volumes(config)
    cd = control_derivatives(config, spanwise=spanwise, chordwise=chordwise)
    ra = roll_authority(config, cd=cd, ld=latd)
    n = config['flight']['load_factor']
    mv = maneuver_speed(config, n)
    has_tail = bool(config.get('vtail'))
    tr_1g = trim(config, mode='ruddervator', n_load=1.0, V=config['flight']['V_ms'], spanwise=spanwise, chordwise=chordwise) if has_tail else None
    tr_n4 = trim(config, mode='ruddervator', n_load=n, V=mv['V_A_ms'], spanwise=spanwise, chordwise=chordwise) if has_tail else None
    tr_morph = trim(config, mode='morph', n_load=n, V=mv['V_A_ms'], spanwise=spanwise, chordwise=chordwise)
    swl = _n4_spanwise_load(config, tr_n4, spanwise, chordwise) if tr_n4 is not None else None
    derivatives = {'CL_a': longi['CL_alpha'], 'CD0': (config.get('viscous') or {}).get('CD0') or config['declared']['CD0'], 'Cm_a': longi['Cm_alpha_total'], 'Cm_q': longi['Cm_q'], 'CL_q': longi['CL_q'], 'CY_b': latd['Cy_beta'], 'Cl_b': latd['Cl_beta'], 'Cn_b': latd['Cn_beta_total'], 'Cl_p': latd['Cl_p'], 'Cn_p': latd['Cn_p'], 'Cl_r': latd['Cl_r'], 'Cn_r': latd['Cn_r'], 'Cl_da': cd['Cl_delta_morph'], 'Cn_da': cd['Cn_delta_morph'], 'Cm_de': cd['Cm_de'], 'CL_de': cd['CL_de']}
    rq = _reference_quantities(config)
    reference = {'S_m2': rq['S'], 'c_m': rq['c'], 'b_m': rq['b'], 'V_ms': config['flight']['V_ms'], 'rho': config['flight']['rho'], 'cg_x_mm': config['flight'].get('cg_x')}
    values = {'longitudinal': longi, 'lateral_directional': latd, 'vtail_volumes': vv, 'control_derivatives': cd, 'roll_authority': ra, 'maneuver': mv, 'trim_1g_cruise': tr_1g, 'trim_n4_pullup': tr_n4, 'trim_morph': tr_morph, 'body_corrections': body, 'derivatives': derivatives, 'spanwise_load': swl, 'reference': reference}
    checks = {}
    for section in ('longitudinal', 'lateral_directional', 'vtail_volumes', 'roll_authority'):
        for k, v in values[section].get('checks', {}).items():
            checks[f'{section}.{k}'] = v
    for name, tr in (('trim_1g', tr_1g), ('trim_n4', tr_n4)):
        if tr is not None:
            checks[f'{name}.inside_limits'] = tr['inside_limits']
    findings = []
    if not checks.get('lateral_directional.Cn_beta_positive'):
        findings.append('Cn_beta (with body) is not positive: directional stability is not met by the V-tail alone.')
    if not checks.get('longitudinal.SM_in_band'):
        findings.append('Static margin (with body) %s is outside the declared 0.05-0.15 band.' % _fmt(longi['static_margin_body'], '%.3f'))
    if not checks.get('roll_authority.roll_authority_ok'):
        if ra['p_hat'] is not None:
            findings.append('Roll authority pb/2V=%s at +/-%s deg is below 0.07; reaching 0.07 needs +/-%s deg (linear).' % (_fmt(ra['p_hat'], '%.3f'), _fmt(ra['delta_full_deg']), _fmt(ra['delta_needed_deg'])))
        else:
            findings.append('Roll authority could not be evaluated (Cl_p ~ 0).')
    for name, tr in (('1 g cruise', tr_1g), ('n=4 pull-up', tr_n4)):
        if tr is not None and (not tr['inside_limits']):
            findings.append('%s ruddervator trim (%s deg) is outside +/-%s deg.' % (name, _fmt(tr['delta_trim_deg']), _fmt(tr['limit_deg'])))
        if tr is not None and tr.get('alpha_at_bound'):
            findings.append('%s: CL_required=%.3f is not reachable within +/-20 deg alpha (trim alpha pinned at %.1f deg).' % (name, tr['CL_required'], tr['alpha_trim_deg']))
    if tr_morph is not None and (not tr_morph['inside_limits']):
        findings.append('Wing morph camber cannot trim Cm=0 within +/-12 deg at n=%.0f (it acts nose-down, the wrong sense for the forward-CG nose-down moment).' % n)
    if config.get('fuselage') is None:
        findings.append('No fuselage geometry supplied; body effects are omitted (see body_corrections).')
    elif body['body_volume_m3'] > 0:
        sts = (config['fuselage'] or {}).get('stations') or []
        if sts:
            wmax = max((s.get('width', 0.0) for s in sts), default=0.0)
            hmax = max((s.get('height', 0.0) for s in sts), default=0.0)
            if hmax > 0 and wmax > 2.5 * hmax:
                findings.append('Fuselage stations are wide/flat (max width %.0f mm vs height %.0f mm), consistent with wing-center-section parts being included; the volume-based Munk body term (%.2f/rad pitch, %.2f/rad yaw) is therefore likely overestimated. Re-measure the fuselage pod (Task A) before trusting the body-inclusive SM/Cn_beta.' % (wmax, hmax, body['Cm_alpha_body_per_rad'], body['Cn_beta_body_per_rad']))
    return aero.plain({'inputs': {'airframe': config.get('airframe'), 'V_ms': config['flight']['V_ms'], 'cg_x_mm': config['flight'].get('cg_x'), 'scale': config['scale']}, 'values': values, 'checks': checks, 'findings': findings, 'method': 'VLM (aero.py) central-difference derivatives about the CG + declared Munk body pitch/yaw terms; trims at 1 g cruise (V_ms) and n=4 pull-up (V_A). Screening linearised estimates, not flight test/wind tunnel/CFD.', 'declared': dict(config['declared'])})

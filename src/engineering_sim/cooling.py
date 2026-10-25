from dataclasses import dataclass, field
import math
from engineering_sim import thermal
ARRANGEMENTS = ('parallel_open', 'parallel_metered', 'series')

@dataclass
class Channel:
    """One family of identical parallel oil channels in one module (SI)."""
    n_parallel: int
    area_m2: float
    perimeter_m: float
    width_m: float
    length_m: float
    one_wall_heated: bool = True
    coil_area_m2: float = 0.0
    iron_area_m2: float = 0.0

@dataclass(kw_only=True)
class Rotor:
    """Dry rotor cooled across the air gap and, optionally, by oil in a hollow shaft (mm, rpm, L/min)."""
    loss_w: float
    speed_rpm: float
    rotor_r: float
    air_gap: float
    gap_area_m2: float
    shaft_r: float
    shaft_len: float
    stack: float
    limit_c: float
    shaft_bore_r: float
    feed_tube_r: float
    shaft_oil_lpm: float
    contact_area_m2: float
    contact_h: float

@dataclass(kw_only=True)
class Duty:
    """What a machine publishes to the cooling designer."""
    copper_loss_w: float
    iron_loss_w: float
    n_modules: int
    winding: object
    iron: Channel
    rotor: Rotor
    oil: dict
    inlet_c: float
    coil_limit_c: float
    iron_limit_c: float
    pump_limit_bar: float
    pump_margin: float
    flow_max_lpm: float
    gap_min_mm: float
    gap_max_mm: float
    gap_tol_mm: float
    name: str

def gap_channels(n_faces, flow_len_mm, face_width_mm, gap_mm, end_area_m2=0.0):
    """Thin gaps along coil faces, heated on the coil wall only: the topology-neutral winding channel.

    Radial machine: n_faces = 2 × slots, flow_len = stack, face_width = radial coil depth.
    Axial-flux coil tray: n_faces = 2 × coils, flow_len = radial coil length, face_width = axial coil thickness.
    `end_area_m2` adds end-turn surface wetted at the same film coefficient (declared, as in `case.hydraulics`).
    """
    g, w, length = (gap_mm * 0.001, face_width_mm * 0.001, flow_len_mm * 0.001)
    return Channel(n_parallel=n_faces, area_m2=g * w, perimeter_m=2 * (g + w), width_m=w, length_m=length, one_wall_heated=True, coil_area_m2=n_faces * length * w + end_area_m2)

def _oil(duty):
    o = duty.oil
    return dict(rho=o['rho'], cp=o['cp'], k=o['k'], mu=o['nu_cst'] * 1e-06 * o['rho'])

def _path(ch, q, n_series, oil):
    return thermal.slot_channel(q, ch.n_parallel, ch.area_m2, ch.perimeter_m, ch.width_m, ch.length_m, n_series, oil, one_wall_heated=ch.one_wall_heated)

def _open_split(w, i, q, n, oil):
    lo, hi = (0.0, 1.0)
    for _ in range(80):
        mid = (lo + hi) / 2
        if _path(w, q * mid, n, oil)['dp_pa'] > _path(i, q * (1 - mid), n, oil)['dp_pa']:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

def _rotor_temp(r, bore_c, inlet_c, n, oil):
    tc = thermal.taylor_couette_h(r.speed_rpm * 2 * math.pi / 60, r.rotor_r * 0.001, r.air_gap * 0.001)
    g_gap = tc['h_w_m2k'] * r.gap_area_m2 + thermal.shaft_conductance(r.shaft_r * 0.001, r.shaft_len * 0.001, 2) / n
    g_cool, sink, shaft_dp = (0.0, inlet_c, 0.0)
    if r.shaft_bore_r > 0 and r.shaft_oil_lpm > 0:
        ro, ri, rt = (r.shaft_r * 0.001, r.shaft_bore_r * 0.001, r.feed_tube_r * 0.001)
        q = r.shaft_oil_lpm / 60000.0
        film = thermal.slot_channel(q, 1, math.pi * (ri ** 2 - rt ** 2), 2 * math.pi * (ri + rt), math.pi * (ri + rt), r.stack * 0.001, n, oil, one_wall_heated=True)
        length = r.stack * 0.001
        g_cool = 1.0 / (1.0 / (r.contact_h * r.contact_area_m2) + math.log(ro / ri) / (2 * math.pi * thermal.STEEL_K_W_MK * length) + 1.0 / (film['h_w_m2k'] * 2 * math.pi * ri * length))
        sink = inlet_c + r.loss_w / (q * oil['rho'] * oil['cp'])
        shaft_dp = film['dp_pa']
    total = g_gap + g_cool
    temp = (r.loss_w / n + g_gap * bore_c + g_cool * sink) / total if total > 0 else float('inf')
    return (temp, shaft_dp)

def evaluate(duty, gap_mm, flow_lpm, arrangement, fraction=None):
    """Temperatures, pressure drop and utilisation of one design point. `fraction` = share of flow through the winding (metered only)."""
    if arrangement not in ARRANGEMENTS:
        raise ValueError('arrangement must be one of ' + ', '.join(ARRANGEMENTS))
    oil, n = (_oil(duty), duty.n_modules)
    w, i = (duty.winding(gap_mm), duty.iron)
    q = max(flow_lpm / 60000.0, 1e-12)
    if i is None:
        arrangement_eff, frac = ('series', 1.0)
    elif arrangement == 'parallel_open':
        arrangement_eff, frac = (arrangement, _open_split(w, i, q, n, oil))
    elif arrangement == 'parallel_metered':
        arrangement_eff, frac = (arrangement, min(max(fraction if fraction is not None else 0.5, 0.0001), 1 - 0.0001))
    else:
        arrangement_eff, frac = ('series', 1.0)
    qw = q * frac
    qi = q if arrangement_eff == 'series' else q * (1 - frac)
    pw = _path(w, qw, n, oil)
    pi = _path(i, qi, n, oil) if i is not None else None
    cap_w = qw * oil['rho'] * oil['cp']
    cap_i = qi * oil['rho'] * oil['cp']
    rotor_loss = duty.rotor.loss_w if duty.rotor else 0.0
    q_fe = duty.iron_loss_w + rotor_loss
    g_w = 0.5 * pw['h_w_m2k'] * w.iron_area_m2 * n
    g_i = pi['h_w_m2k'] * i.iron_area_m2 * n if i is not None else 0.0
    if g_w + g_i <= 0 and q_fe > 0:
        raise ValueError('the iron has no wetted surface: give the winding channel iron_area_m2 or supply an iron channel')
    q_fe_w = q_fe * g_w / (g_w + g_i) if g_w + g_i > 0 else 0.0
    for _ in range(60):
        t_w_out = duty.inlet_c + (duty.copper_loss_w + q_fe_w) / cap_w
        t_i_in = t_w_out if arrangement_eff == 'series' and i is not None else duty.inlet_c
        t_i_out = t_i_in + (q_fe - q_fe_w) / cap_i if i is not None else t_i_in
        t_fe = (q_fe + g_w * t_w_out + g_i * t_i_out) / (g_w + g_i) if g_w + g_i > 0 else t_w_out
        new = g_w * (t_fe - t_w_out)
        if abs(new - q_fe_w) < 1e-09:
            break
        q_fe_w = 0.5 * (q_fe_w + new)
    film = duty.copper_loss_w / n / (pw['h_w_m2k'] * w.coil_area_m2)
    t_coil = t_w_out + film
    tol = min(duty.gap_tol_mm, gap_mm * 0.9)
    tight, loose = ((gap_mm - tol) / gap_mm, (gap_mm + tol) / gap_mm)
    t_coil_tight = duty.inlet_c + (t_w_out - duty.inlet_c) / tight ** 3 + film * tight
    t_coil_loose = duty.inlet_c + (t_w_out - duty.inlet_c) / loose ** 3 + film * loose
    t_coil_worst = max(t_coil, t_coil_tight, t_coil_loose)
    dp = pw['dp_pa'] + (pi['dp_pa'] if arrangement_eff == 'series' and pi else 0.0)
    if arrangement_eff == 'parallel_metered' and pi:
        dp = max(pw['dp_pa'], pi['dp_pa'])
    span_c, span_fe = (duty.coil_limit_c - duty.inlet_c, duty.iron_limit_c - duty.inlet_c)
    use = dict(coil=(t_coil_worst - duty.inlet_c) / span_c, iron=(t_fe - duty.inlet_c) / span_fe)
    out = dict(arrangement=arrangement_eff, gap_mm=gap_mm, flow_lpm=flow_lpm, winding_flow_fraction=frac, winding_h_w_m2k=pw['h_w_m2k'], winding_re=pw['re'], winding_regime=pw['regime'], iron_h_w_m2k=pi['h_w_m2k'] if pi else None, dp_bar=dp / 100000.0, pump_w=dp * q, winding_oil_out_c=t_w_out, iron_oil_out_c=t_i_out, coil_film_rise_k=film, coil_temp_c=t_coil, coil_temp_worst_c=t_coil_worst, coil_temp_tight_c=t_coil_tight, coil_temp_loose_c=t_coil_loose, iron_temp_c=t_fe, mixed_outlet_c=duty.inlet_c + (duty.copper_loss_w + q_fe) / (q * oil['rho'] * oil['cp']), coil_r_k_w=(t_coil_worst - duty.inlet_c) / duty.copper_loss_w if duty.copper_loss_w > 0 else 0.0)
    if duty.rotor:
        t_rotor, shaft_dp = _rotor_temp(duty.rotor, t_fe, duty.inlet_c, n, oil)
        out.update(rotor_temp_c=t_rotor, shaft_dp_bar=shaft_dp / 100000.0)
        use['rotor'] = (t_rotor - duty.inlet_c) / (duty.rotor.limit_c - duty.inlet_c)
    out['utilisation'] = use
    out['limiting'] = max(use, key=use.get)
    out['stator_limiting'] = max(('coil', 'iron'), key=use.get)
    out['loss_headroom_x'] = 1.0 / max(use.values()) if max(use.values()) > 0 else float('inf')
    out['stator_loss_headroom_x'] = 1.0 / max(use['coil'], use['iron']) if max(use['coil'], use['iron']) > 0 else float('inf')
    return out

def _flow_at_budget(duty, gap_mm, arrangement, fraction):
    """Largest jacket flow within the pump budget and the flow cap (Δp rises monotonically with flow)."""
    budget = duty.pump_limit_bar * duty.pump_margin
    if evaluate(duty, gap_mm, duty.flow_max_lpm, arrangement, fraction)['dp_bar'] <= budget:
        return duty.flow_max_lpm
    lo, hi = (0.0001, duty.flow_max_lpm)
    for _ in range(60):
        mid = (lo + hi) / 2
        if evaluate(duty, gap_mm, mid, arrangement, fraction)['dp_bar'] > budget:
            hi = mid
        else:
            lo = mid
    return lo

def _score(point):
    """The jacket is designed against the stator limits it controls; the rotor is reported, not traded."""
    u = point['utilisation']
    return max(u['coil'], u['iron'])

def _best_fraction(duty, gap_mm, flow_of):
    """Metered split minimising the worse of coil and iron utilisation (unimodal: golden section)."""
    phi = (math.sqrt(5) - 1) / 2
    lo, hi = (0.02, 0.98)

    def cost(f):
        return _score(evaluate(duty, gap_mm, flow_of(f), 'parallel_metered', f))
    a, b = (hi - phi * (hi - lo), lo + phi * (hi - lo))
    ca, cb = (cost(a), cost(b))
    for _ in range(40):
        if ca < cb:
            hi, b, cb = (b, a, ca)
            a = hi - phi * (hi - lo)
            ca = cost(a)
        else:
            lo, a, ca = (a, b, cb)
            b = lo + phi * (hi - lo)
            cb = cost(b)
    return (lo + hi) / 2

def design(duty, arrangements=ARRANGEMENTS, gap_steps=51):
    """Choose gap, flow and (for metered) split for each arrangement; rank by stator loss headroom.

    Returns dict(best=…, by_arrangement={…}, as_built=None). Flow is the largest the pump budget and
    the flow cap allow: more oil never heats a laminar design, it only costs pump power, which is reported.
    """
    if duty.gap_min_mm <= 0 or duty.gap_max_mm < duty.gap_min_mm:
        raise ValueError('gap range must satisfy 0 < gap_min_mm <= gap_max_mm')
    gaps = [duty.gap_min_mm + (duty.gap_max_mm - duty.gap_min_mm) * k / (gap_steps - 1) for k in range(gap_steps)] if duty.gap_max_mm > duty.gap_min_mm else [duty.gap_min_mm]
    result = {}
    for arr in arrangements:
        if duty.iron is None and arr != 'series':
            continue
        best = None
        for gap in gaps:
            frac = None
            if arr == 'parallel_metered':
                frac = _best_fraction(duty, gap, lambda f, g=gap: _flow_at_budget(duty, g, 'parallel_metered', f))
            flow = _flow_at_budget(duty, gap, arr, frac)
            point = evaluate(duty, gap, flow, arr, frac)
            if best is None or _score(point) < _score(best):
                best = point
        result[arr] = best
    ranked = sorted(result.values(), key=_score)
    return dict(best=ranked[0], by_arrangement=result, ranking=[p['arrangement'] for p in ranked])

def flow_at_budget(duty, gap_mm, arrangement='parallel_open', fraction=None):
    """Public name of the pump-budget flow rule (L/min)."""
    return _flow_at_budget(duty, gap_mm, arrangement, fraction)

def report(duty, as_built=None, arrangements=ARRANGEMENTS):
    """Plain JSON-able summary: the chosen design, every arrangement's optimum and the as-built point."""
    d = design(duty, arrangements)

    def trim(pt):
        return {k: round(v, 4) if isinstance(v, float) else {a: round(b, 4) for a, b in v.items()} if isinstance(v, dict) else v for k, v in pt.items()}
    out = dict(machine=duty.name, limits=dict(coil_c=duty.coil_limit_c, iron_c=duty.iron_limit_c, inlet_c=duty.inlet_c, pump_budget_bar=duty.pump_limit_bar * duty.pump_margin, flow_max_lpm=duty.flow_max_lpm, gap_mm=[duty.gap_min_mm, duty.gap_max_mm], gap_tol_mm=duty.gap_tol_mm), losses_w=dict(copper=duty.copper_loss_w, iron=duty.iron_loss_w, rotor=duty.rotor.loss_w if duty.rotor else 0.0), best=trim(d['best']), ranking=d['ranking'], by_arrangement={k: trim(v) for k, v in d['by_arrangement'].items()})
    if as_built:
        out['as_built'] = trim(as_built)
        out['stator_headroom_gain_x'] = round(d['best']['stator_loss_headroom_x'] / as_built['stator_loss_headroom_x'], 3)
    return out

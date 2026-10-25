import math
import hashlib
import json
from engineering_sim import correlations
VERSION='0.1.0'

def segment_distance(p1, q1, p2, q2):
    """Closest distance between bounded 3-D line segments, including parallel cases."""
    dot = lambda x, y: sum((a * b for a, b in zip(x, y)))
    u = [b - a for a, b in zip(p1, q1)]
    v = [b - a for a, b in zip(p2, q2)]
    r = [a - b for a, b in zip(p1, p2)]
    a = dot(u, u)
    e = dot(v, v)
    b = dot(u, v)
    c = dot(u, r)
    f = dot(v, r)
    clamp = lambda t: max(0, min(1, t))
    if a < 1e-20 or e < 1e-20:
        raise ValueError('Network clearance requires nonzero segments.')
    denom = a * e - b * b
    s = clamp((b * f - c * e) / denom) if denom > 1e-12 * a * e else 0
    t = (b * s + f) / e
    if t < 0:
        t = 0
        s = clamp(-c / a)
    elif t > 1:
        t = 1
        s = clamp((b - c) / a)
    return math.sqrt(sum(((r[i] + s * u[i] - t * v[i]) ** 2 for i in range(3))))

def _positive(packet, key, missing, allow_zero=False):
    """Never infer material/process numbers from a resin family or source dimensions."""
    value = packet.get(key)
    if type(value) not in (int, float) or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        missing.append(key)
        return None
    return float(value)

def thermal(network, tool, p):
    """Steady plug-flow heat exchange, wall power/warmup, and cold-gate conduction screen.

    Equal simultaneous unit flow requires balancing/meters; it is an imposed
    scenario. Heat transfer is prescribed, not predicted from guessed polymer
    properties. Constant-property laminar pressure uses shared ENGToolkit Darcy
    correlations. No total injection-pressure or cycle-time rating is returned.
    """
    material = p.get('recipe.material', {})
    process = p.get('recipe.process', {})
    missing = []
    grade = p['mold.resin_grade']
    if grade.strip().lower() in ('', 'unassigned', 'unknown'):
        missing.append('mold.resin_grade')
    if material.get('grade') != grade:
        missing.append('recipe.material.grade matching mold.resin_grade')
    if not material.get('reference'):
        missing.append('recipe.material.reference')
    rho = _positive(material, 'density_kg_m3', missing)
    cp = _positive(material, 'heat_capacity_J_kgK', missing)
    k = _positive(material, 'conductivity_W_mK', missing)
    limit = _positive(material, 'minimum_flow_temperature_c', missing)
    max_temp = _positive(material, 'maximum_melt_temperature_c', missing)
    residence_limit = _positive(material, 'maximum_residence_s', missing)
    viscosity = _positive(material, 'apparent_viscosity_Pas', missing)
    keys = ['melt_temperature_c', 'mold_temperature_c', 'ambient_temperature_c', 'flow_per_inlet_cm3_s', 'header_setpoint_c', 'heater_W_m', 'loss_W_mK', 'wall_heat_capacity_J_mK', 'melt_wall_h_W_m2K', 'cycle_interval_s', 'minimum_hot_core_fraction']
    v = {key: _positive(process, key, missing, allow_zero=key == 'heater_W_m') for key in keys}
    identity = hashlib.sha256(json.dumps(dict(network=network, material=material, process=process, tool=tool), sort_keys=True, allow_nan=False).encode()).hexdigest()
    base = dict(version=VERSION, identity=identity, resin_grade=grade, material_reference=material.get('reference'), inlet_temperature_c=process.get('melt_temperature_c'), status='unresolved', missing_inputs=missing, operating_window_qualified=False, assumptions=['Same recipe at every unit; imposed equal simultaneous flow.', 'Constant material properties; no shear heating, phase-change latent heat, crystallization or degradation kinetics.', 'Prescribed heat transfer and insulation losses; steady flow in prefilled heated headers.', 'Wall warmup concerns empty hardware, not pellet melting or cold polymer startup.', 'Cold-gate conduction is a screening scenario, not a gate-seal/packing or safe-time prediction.'])
    if missing:
        return base
    if not v['mold_temperature_c'] < limit < v['melt_temperature_c'] <= max_temp:
        return dict(base, status='invalid_inputs', reason='Require mold temperature < minimum flow temperature < entering melt temperature <= grade maximum.')
    if not v['ambient_temperature_c'] < v['header_setpoint_c'] <= max_temp:
        return dict(base, status='invalid_inputs', reason='Header setpoint must exceed ambient and remain within the grade maximum.')
    if not 0 < v['minimum_hot_core_fraction'] < 1:
        raise ValueError('minimum_hot_core_fraction must be strictly between 0 and 1.')
    count = len(network['units'])
    inlets = len(network['melt_inlets'])
    total_q = v['flow_per_inlet_cm3_s'] * 1e-06 * inlets
    unit_q = total_q / count
    edge_results = {}
    node_t = {root: v['melt_temperature_c'] for root in network['melt_inlets']}
    unit_shot = (tool['cavity_volume_mm3'] + (tool['feed_network']['feed_volume_outside_cavity_mm3'] if tool.get('feed_network') else 0)) * 1e-09
    pending = [e for e in network['edges'] if e['circuit'] == 'melt']
    while pending:
        ready = [e for e in pending if e['start'] in node_t]
        if not ready:
            raise ValueError('Thermal network is cyclic or disconnected.')
        for e in ready:
            length = e['length_mm'] / 1000
            diameter = e['diameter_mm'] / 1000
            q = unit_q * len(e['downstream_units'])
            tin = node_t[e['start']]
            mdot_cp = rho * q * cp
            ua = v['melt_wall_h_W_m2K'] * math.pi * diameter * length
            exchange = mdot_cp * -math.expm1(-ua / mdot_cp)
            loss = v['loss_W_mK'] * length
            available = v['heater_W_m'] * length
            setpoint = v['header_setpoint_c']
            ambient = v['ambient_temperature_c']
            required = loss * (setpoint - ambient) + exchange * (setpoint - tin)
            used = max(0, min(available, required))
            wall = (used + loss * ambient + exchange * tin) / (loss + exchange)
            tout = wall + (tin - wall) * math.exp(-ua / mdot_cp)
            node_t[e['end']] = tout
            velocity = q / (math.pi * diameter ** 2 / 4)
            re = rho * velocity * diameter / viscosity
            pressure = correlations.darcy_laminar(re) * (length / diameter) * rho * velocity ** 2 / 2
            heat_capacity = v['wall_heat_capacity_J_mK'] * length
            warmup = None
            if available > loss * (setpoint - ambient):
                warmup = -heat_capacity / loss * math.log1p(-loss * (setpoint - ambient) / available)
            edge_results[e['id']] = dict(outlet_temperature_c=tout, wall_temperature_c=wall, heater_used_W=used, heater_available_W=available, requested_heater_W=required, setpoint_maintained=0 <= required <= available, empty_wall_warmup_s=warmup, pressure_drop_Pa=pressure, reynolds=re, transit_s=e['volume_mm3'] * 1e-09 / q, flow_cm3_s=q * 1000000.0, thermal_energy_residual_W=used - loss * (wall - ambient) - mdot_cp * (tout - tin))
            pending.remove(e)
    unit_results = []
    branches = tool.get('feed_network', {}).get('branches', [])
    alpha = k / (rho * cp)
    for u in network['units']:
        tin = node_t[u['melt_inlet']]
        time_fill = unit_shot / unit_q
        gate_times = []
        for branch in branches:
            half = branch['gate_height_mm'] / 2000
            x = v['minimum_hot_core_fraction']
            gate_times.append(slab_threshold_time(tin, v['mold_temperature_c'], limit, half, alpha, x))
        deadline = min(gate_times) if gate_times else None
        path = [edge_results[e] for e in u['melt_path']]
        hot_volume_share = sum((next((e['volume_mm3'] for e in network['edges'] if e['id'] == name)) / len(next((e['downstream_units'] for e in network['edges'] if e['id'] == name))) for name in u['melt_path']))
        renewal = hot_volume_share * 1e-09 / unit_shot * v['cycle_interval_s']
        unit_results.append(dict(label=u['label'], mold_inlet_temperature_c=tin, imposed_fill_time_s=time_fill, heated_transit_s=sum((e['transit_s'] for e in path)), mean_shot_renewal_s=renewal, header_drop_pressure_MPa=sum((e['pressure_drop_Pa'] for e in path)) / 1000000.0, cold_gate_hot_core_threshold_s=deadline, fill_to_gate_threshold_ratio=time_fill / deadline if deadline else None, thermal_screen_pass=tin >= limit and tin <= max_temp and (renewal <= residence_limit) and (deadline is not None) and (time_fill < deadline) and (time_fill <= v['cycle_interval_s'])))
    return dict(base, status='screen_complete', missing_inputs=[], total_imposed_flow_cm3_s=total_q * 1000000.0, volume_per_shot_cm3=unit_shot * count * 1000000.0, edges=edge_results, units=unit_results, all_heater_setpoints_maintained=all((e['setpoint_maintained'] for e in edge_results.values())), total_installed_heater_W=sum((e['heater_available_W'] for e in edge_results.values())), maximum_empty_wall_warmup_s=max((e['empty_wall_warmup_s'] for e in edge_results.values())) if all((e['empty_wall_warmup_s'] is not None for e in edge_results.values())) else None, all_unit_screens_pass=all((u['thermal_screen_pass'] for u in unit_results)), pressure_scope='Heated header/drop straight-pipe losses only, imposed balanced flow. Local sprue/runner/gates, cavity and packing pressure excluded.')

def slab_threshold_time(initial, wall, threshold, half_thickness_m, alpha_m2_s, core_fraction):
    """Time for T(x=fraction*half-thickness) to reach threshold in a fixed-wall slab.

    Zero means already below threshold; None means no finite cooling crossing.
    A dimensionless series/bisection avoids unstable time stepping and scales as
    thickness squared / diffusivity. It excludes latent heat and moving melt.
    """
    if initial <= threshold:
        return 0.0
    if wall >= threshold:
        return None
    target = (threshold - wall) / (initial - wall)

    def theta(fo):
        return 4 / math.pi * sum(((-1) ** n / (2 * n + 1) * math.cos((2 * n + 1) * math.pi * core_fraction / 2) * math.exp(-((2 * n + 1) * math.pi / 2) ** 2 * fo) for n in range(120)))
    lo = 0.0
    hi = 1.0
    while theta(hi) > target:
        hi *= 2
    for _ in range(70):
        mid = (lo + hi) / 2
        if theta(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 * half_thickness_m ** 2 / alpha_m2_s

def life_and_timing(network, tool, thermal_result, p, source_identity):
    """Plot measured wear trends against cycles/hours without inventing mold fatigue life.

    Calibration is tied to the complete material/process/tooling/route identity.
    A new part or thermal recipe invalidates it until applicability is reviewed.
    Wear rates are empirical depths per cycle, not Archard coefficients borrowed
    from another contact mechanism. Fracture, corrosion and thermal fatigue need
    their own supported damage models before an overall mold-life rating exists.
    """
    packet = p.get('recipe.life', {})
    process = p.get('recipe.process', {})
    missing = []
    basis = {k: v for k, v in p.items() if k.startswith(('mold.', 'route.', 'study.', 'network.', 'recipe.material', 'recipe.process')) and k != 'study.workers'}
    basis['source_identity'] = source_identity
    identity = hashlib.sha256(json.dumps(basis, sort_keys=True, allow_nan=False).encode()).hexdigest()
    result = dict(status='unresolved', basis_identity=identity, mold_life_qualified=False, scope='Empirical wear-to-limit scenario per insert. Fatigue, cracking, corrosion and maintenance effects are not predicted.', cycles_are_per_mold=True, missing_inputs=missing, timeline=[])
    phases = process.get('phase_seconds', {})
    phase_names = ['close_evacuate', 'pack_hold', 'cool_after_hold', 'open_eject']
    if thermal_result and thermal_result.get('status') == 'screen_complete' and all((type(phases.get(k)) in (int, float) and math.isfinite(phases[k]) and (phases[k] >= 0) for k in phase_names)):
        durations = dict(phases, fill=max((u['imposed_fill_time_s'] for u in thermal_result['units'])))
        elapsed = 0.0
        for name in ['close_evacuate', 'fill', 'pack_hold', 'cool_after_hold', 'open_eject']:
            duration = durations[name]
            result['timeline'].append(dict(phase=name, start_s=elapsed, duration_s=duration))
            elapsed += duration
        interval = process['cycle_interval_s']
        result['cycle_schedule_consistent'] = elapsed <= interval
        if elapsed < interval:
            result['timeline'].append(dict(phase='idle', start_s=elapsed, duration_s=interval - elapsed))
        result['cycle_interval_s'] = interval
        result['specified_phase_total_s'] = elapsed
    else:
        result['timing_missing'] = 'Recipe phase_seconds and a completed thermal/flow scenario are required.'
    if p['mold.tool_material'].strip().lower() in ('', 'unassigned', 'unknown'):
        missing.append('mold.tool_material')
    if packet.get('basis_identity') != identity:
        missing.append('recipe.life.basis_identity matching this exact recipe/source')
    if not packet.get('reference'):
        missing.append('recipe.life.reference: measured/supplier wear data for these conditions')
    low = _positive(packet, 'wear_low_um_per_cycle', missing)
    nominal = _positive(packet, 'wear_nominal_um_per_cycle', missing)
    high = _positive(packet, 'wear_high_um_per_cycle', missing)
    limit = _positive(packet, 'wear_limit_um', missing)
    initial = _positive(packet, 'current_wear_um', missing, True)
    interval = _positive(process, 'cycle_interval_s', missing)
    if missing:
        return result
    if not low <= nominal <= high:
        raise ValueError('Wear rates must satisfy low <= nominal <= high.')
    remaining = max(0, limit - initial)
    cycles = dict(early=remaining / high, nominal=remaining / nominal, late=remaining / low)
    horizon = cycles['late']
    points = []
    for i in range(101):
        n = horizon * i / 100
        points.append(dict(additional_cycles=n, operating_hours=n * interval / 3600, wear_low_um=initial + low * n, wear_nominal_um=initial + nominal * n, wear_high_um=initial + high * n))
    result.update(status='calibrated_wear_scenario', reference=packet['reference'], wear_limit_um=limit, points=points, remaining_cycles_to_wear_limit=cycles, remaining_operating_hours={k: n * interval / 3600 for k, n in cycles.items()}, possible_parts_before_nominal_wear_limit=cycles['nominal'] * network['unit_count'])
    return result

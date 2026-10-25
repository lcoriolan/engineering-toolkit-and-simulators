from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import heapq
import json
import math
import multiprocessing
import os
from pathlib import Path
import socket
import time
from . import mold_routing
VERSION = '0.1.0'
SCORE_TIE_TOLERANCE = 1e-07

def fields(prefix='study.'):
    """Sampling/ranking controls scale with the source; hardware limits stay fixed."""
    group = 'Part-driven routing study'

    def f(key, label, value, kind='number', **extra):
        return dict(key=prefix + key, label=label, type=kind, group=group, **extra)
    return [f('mode', 'Local route selection', 'part_driven', 'choice', choices=['part_driven', 'manual']), f('resolution', 'Samples along longest cavity axis', 22, 'integer', min=12, max=40), f('workers', 'Worker limit · 0 uses available cores', 0, 'integer', min=0, max=128), f('gate_counts', 'Candidate gate counts', [1, 2, 3], 'json'), f('sides', 'Candidate entry sides', ['-X', '+X', '-Y', '+Y'], 'json'), f('keepouts', 'Gate exclusion rectangles · normalized XY', [], 'json'), f('gate_wall_fraction', 'Gate height / sampled wall thickness', 0.25, min=0.05, max=0.8), f('gate_width_ratio', 'Gate width / height', 2, min=1, max=8), f('runner_gate_ratio', 'Runner diameter / gate height', 2, min=1, max=6), f('minimum_runner_mm', 'Minimum local runner bore', 2, min=1, max=16), f('maximum_runner_mm', 'Maximum local runner bore', 12, min=1, max=16), f('land_clearance_mm', 'Gate land beyond runner radius', 0.75, min=0.1, max=3), f('setback_fraction', 'Runner setback / stock margin', 0.64, min=0.2, max=0.8), f('gate_penalty', 'Ranking penalty for each additional gate', 0.18, min=0, max=2), f('waste_weight', 'Ranking weight for feed / cavity volume', 0.2, min=0, max=2), f('balance_weight', 'Ranking weight for gate demand imbalance', 0.15, min=0, max=2), f('max_volume_error', 'Maximum sampled volume error fraction', 0.2, min=0.01, max=0.4)]

def settings(p):
    """Validate nested lists beyond the host schema's scalar validation."""
    s = {f['key'][6:]: deepcopy(p[f['key']]) for f in fields()}
    if not s['gate_counts'] or any((type(n) is not int or not 1 <= n <= 8 for n in s['gate_counts'])):
        raise ValueError('study.gate_counts must contain integers from 1 to 8.')
    if not s['sides'] or any((side not in ('-X', '+X', '-Y', '+Y') for side in s['sides'])):
        raise ValueError('study.sides must contain supported cavity sides.')
    for rect in s['keepouts']:
        if not isinstance(rect, list) or len(rect) != 4 or any((type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1 for v in rect)):
            raise ValueError('Gate keepouts are [xmin,ymin,xmax,ymax] fractions of cavity half-size.')
        if rect[0] >= rect[2] or rect[1] >= rect[3]:
            raise ValueError('Gate keepout bounds must increase.')
    if s['minimum_runner_mm'] > s['maximum_runner_mm']:
        raise ValueError('Minimum runner exceeds maximum runner.')
    return s

def fingerprint(source_hash, mold, study, recipe_context):
    """Changing geometry, compensation, recipe material or process invalidates evidence."""
    payload = dict(version=VERSION, source_sha256=source_hash, mold=mold, study=study, context=recipe_context)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()

def _intervals(cavity, axis, fixed, low, high):
    """Exact line/solid crossings with midpoint classification handle holes and tangency."""
    import FreeCAD as App
    import Part
    start = list(fixed)
    end = list(fixed)
    start[axis] = low - 1
    end[axis] = high + 1
    section = cavity.section(Part.makeLine(App.Vector(*start), App.Vector(*end)))
    values = sorted(set((round(tuple(v.Point)[axis], 8) for v in section.Vertexes)))
    result = []
    for a, b in zip(values, values[1:]):
        midpoint = list(fixed)
        midpoint[axis] = (a + b) / 2
        if b - a > 1e-07 and cavity.isInside(App.Vector(*midpoint), 1e-07, True):
            result.append((a, b))
    return result

def sample(cavity, resolution, max_error):
    """Voxelize interior occupancy; neighbours may connect only within one solid chord.

    Thickness is the minimum axis-aligned material chord at a sample, not normal
    wall thickness. Oblique walls and features below the sample pitch need a finer
    study; disconnected/unresolved graphs are explicitly rejected.
    """
    import FreeCAD as App
    import Part
    b = cavity.optimalBoundingBox(False, False)
    low = [b.XMin, b.YMin, b.ZMin]
    span = [b.XLength, b.YLength, b.ZLength]
    counts = [max(3, math.ceil(resolution * d / max(span))) for d in span]
    pitch = [d / n for d, n in zip(span, counts)]
    coords = [[a + (i + 0.5) * step for i in range(n)] for a, step, n in zip(low, pitch, counts)]
    chords = [{}, {}, {}]
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for i in range(counts[other[0]]):
            for j in range(counts[other[1]]):
                point = [0, 0, 0]
                point[other[0]] = coords[other[0]][i]
                point[other[1]] = coords[other[1]][j]
                chords[axis][i, j] = _intervals(cavity, axis, point, low[axis], low[axis] + span[axis])
    nodes = []
    lookup = {}
    interval_ids = []
    for ix in range(counts[0]):
        for iy in range(counts[1]):
            for iz in range(counts[2]):
                idx = (ix, iy, iz)
                point = [coords[a][idx[a]] for a in range(3)]
                lengths = []
                ids = []
                for axis in range(3):
                    key = tuple((idx[a] for a in range(3) if a != axis))
                    hit = next(((k, v) for k, v in enumerate(chords[axis][key]) if v[0] < point[axis] < v[1]), None)
                    if hit is None:
                        break
                    k, (lo, hi) = hit
                    lengths.append(hi - lo)
                    ids.append(k)
                if len(lengths) == 3:
                    lookup[idx] = len(nodes)
                    nodes.append(dict(xyz=point, thickness_mm=min(lengths)))
                    interval_ids.append(ids)
    if not nodes:
        raise ValueError('Cavity sampling found no interior; increase study resolution.')
    edges = [[] for _ in nodes]
    for idx, i in lookup.items():
        for axis in range(3):
            nxt = list(idx)
            nxt[axis] += 1
            j = lookup.get(tuple(nxt))
            if j is not None and interval_ids[i][axis] == interval_ids[j][axis]:
                length = pitch[axis]
                cost = length / min(nodes[i]['thickness_mm'], nodes[j]['thickness_mm']) ** 3
                edges[i].append((j, cost, length))
                edges[j].append((i, cost, length))
    seen = {0}
    todo = [0]
    while todo:
        for j, _, _ in edges[todo.pop()]:
            if j not in seen:
                seen.add(j)
                todo.append(j)
    if len(seen) != len(nodes):
        raise ValueError('Sample graph disconnected: unresolved thin links or multiple cavity solids; increase resolution or use manual routing.')
    approximation = len(nodes) * math.prod(pitch)
    error = abs(approximation - cavity.Volume) / cavity.Volume
    if error > max_error:
        raise ValueError('Sampled volume error exceeds study.max_volume_error; increase resolution.')
    plane = Part.makePlane(span[0] + 2, span[1] + 2, App.Vector(low[0] - 1, low[1] - 1, 0))
    outline = [[[p.x, p.y] for p in edge.discretize(Deflection=min(pitch) / 4)] for edge in cavity.section(plane).Edges]
    thickness = sorted((n['thickness_mm'] for n in nodes))
    profile = dict(cavity_volume_mm3=cavity.Volume, bounds_mm=[low, [a + d for a, d in zip(low, span)]], sample_count=len(nodes), grid_counts=counts, pitch_mm=pitch, sampled_volume_error_fraction=error, axis_chord_p10_mm=thickness[int(0.1 * (len(thickness) - 1))], axis_chord_min_mm=thickness[0], outline_xy_mm=outline, connected=True, scope='Interior grid; minimum axial chord proxy, not measured normal wall thickness or a filling mesh.')
    return dict(nodes=nodes, edges=edges, profile=profile)

def _seed(cavity, graph, point):
    """Connect the gate to a nearby sample through material, never across a cavity void."""
    import FreeCAD as App
    import Part
    candidates = sorted(range(len(graph['nodes'])), key=lambda i: sum(((a - b) ** 2 for a, b in zip(point, graph['nodes'][i]['xyz']))))
    for i in candidates[:32]:
        line = Part.makeLine(App.Vector(*point), App.Vector(*graph['nodes'][i]['xyz']))
        if line.Length - line.common(cavity).Length < 1e-05:
            return i
    raise ValueError('Gate cannot connect to the sampled interior without crossing a void.')

def paths(graph, seeds):
    """Multi-source shortest-cost paths assign cavity demand and remote samples to gates."""
    n = len(graph['nodes'])
    cost = [math.inf] * n
    distance = [0.0] * n
    owner = [-1] * n
    queue = []
    for gate, i in enumerate(seeds):
        cost[i] = 0.0
        owner[i] = gate
        heapq.heappush(queue, (0.0, gate, i))
    while queue:
        value, gate, i = heapq.heappop(queue)
        if value != cost[i] or gate != owner[i]:
            continue
        for j, weight, length in graph['edges'][i]:
            proposed = value + weight
            if proposed < cost[j]:
                cost[j] = proposed
                distance[j] = distance[i] + length
                owner[j] = gate
                heapq.heappush(queue, (proposed, gate, j))
    if any((i < 0 for i in owner)):
        raise ValueError('Unreachable sampled cavity volume.')
    shares = [owner.count(i) / n for i in range(len(seeds))]
    return dict(max_cost=max(cost), max_path_mm=max(distance), volume_fractions=shares, demand_imbalance=(max(shares) - min(shares)) * len(shares), costs=cost)

def _validate_candidate(payload):
    """Each candidate gets an isolated OCC process; preserve a useful rejection record."""
    import FreeCAD
    import Part
    from .mold_geometry import construct
    source_path, candidate = payload
    try:
        source = Part.Shape()
        source.read(source_path)
        _, report = construct(source, candidate['mold'], inspect=False, routing=candidate['route'])
        candidate.update(accepted=True, feed_volume_mm3=report['feed_network']['feed_volume_outside_cavity_mm3'])
    except Exception as exc:
        candidate.update(accepted=False, rejection=type(exc).__name__ + ': ' + str(exc))
    return candidate

def solve(source_path, mold, p, context=None):
    """Return selected effective tooling settings and source-bound comparative evidence."""
    import FreeCAD
    import Part
    from .mold_geometry import compensated
    started = time.monotonic()
    s = settings(p)
    source_path = Path(source_path)
    source = Part.Shape()
    source.read(str(source_path))
    cavity = compensated(source, mold)
    graph = sample(cavity, s['resolution'], s['max_volume_error'])
    profile = graph['profile']
    height = max(0.2, min(8, profile['axis_chord_p10_mm'] * s['gate_wall_fraction']))
    diameter = max(s['minimum_runner_mm'], height * s['runner_gate_ratio'])
    if diameter > s['maximum_runner_mm']:
        raise ValueError('Part-derived local runner exceeds the configured maximum diameter.')
    bounds = cavity.optimalBoundingBox(False, False)
    candidates = []
    rejections = []
    for side in dict.fromkeys(s['sides']):
        tangent = 0 if side.endswith('Y') else 1
        half = [bounds.XLength / 2, bounds.YLength / 2][tangent]
        values = sorted((node['xyz'][tangent] / half for node in graph['nodes']))
        for count in dict.fromkeys(s['gate_counts']):
            name = side + '_' + str(count)
            effective = deepcopy(mold)
            effective['sprue_diameter'] = max(mold['sprue_diameter'], diameter * 1.2)
            offsets = [max(-0.8, min(0.8, values[min(len(values) - 1, int((i + 0.5) * len(values) / count))])) for i in range(count)]
            route = dict(side=side, gate_offsets=offsets, inlet_offset=sum(offsets) / len(offsets), runner_diameter_mm=diameter, gate_height_mm=height, gate_width_mm=height * s['gate_width_ratio'], gate_length_mm=diameter / 2 + s['land_clearance_mm'], runner_setback_mm=mold['margin'] * s['setback_fraction'], vent_side=('-' if side[0] == '+' else '+') + side[1], vent_offsets=offsets)
            try:
                route = mold_routing.normalize(route)
                seeds = []
                for offset in offsets:
                    point, tan, _, _, _ = mold_routing._edge_point(cavity, side, offset)
                    x, y = (point.x / (bounds.XLength / 2), point.y / (bounds.YLength / 2))
                    dx = abs(tan.x) * route['gate_width_mm'] / bounds.XLength
                    dy = abs(tan.y) * route['gate_width_mm'] / bounds.YLength
                    if any((a - dx <= x <= c + dx and b - dy <= y <= d + dy for a, b, c, d in s['keepouts'])):
                        raise ValueError('Gate intersects a declared keepout.')
                    seeds.append(_seed(cavity, graph, tuple(point)))
                metric = paths(graph, seeds)
                candidates.append(dict(id=name, mold=effective, route=route, metrics={k: v for k, v in metric.items() if k != 'costs'}))
            except ValueError as exc:
                rejections.append(dict(id=name, accepted=False, rejection=str(exc)))
    workers = min(s['workers'] or os.cpu_count() or 1, len(candidates))
    if candidates:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            results = list(pool.map(_validate_candidate, [(str(source_path), c) for c in candidates]))
    else:
        results = []
    valid = [c for c in results if c['accepted']]
    if not valid:
        raise ValueError('No admissible part-driven route: ' + json.dumps(results + rejections))
    reference = max((c['metrics']['max_cost'] for c in valid)) or 1
    for candidate in valid:
        metrics = candidate['metrics']
        gates = len(candidate['route']['gate_offsets'])
        candidate['score'] = metrics['max_cost'] / reference + s['gate_penalty'] * (gates - 1) + s['waste_weight'] * candidate['feed_volume_mm3'] / cavity.Volume + s['balance_weight'] * metrics['demand_imbalance']
    winner = min(valid, key=lambda c: (round(c['score'] / SCORE_TIE_TOLERANCE), c['id']))
    report = dict(version=VERSION, source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(), host=socket.gethostname(), workers=workers, available_cores=os.cpu_count(), parallel_strategy='Independent OCC candidate processes', elapsed_s=time.monotonic() - started, profile=profile, selected_id=winner['id'], candidates=results + rejections, settings=s, selected_mold=winner['mold'], selected_route=winner['route'], flow_balance_verified=False, thermal_process_qualified=False, scope='Best score within the declared edge-runner candidate family; direct sprues remain a manual option. Geometry ranking is not a physical filling solution.')
    report['identity'] = fingerprint(report['source_sha256'], mold, s, context or {})
    return (winner['mold'], winner['route'], report)

import math
import FreeCAD as App
import Part
V = App.Vector

def box(shape):
    """Use true shape extrema rather than spline control-point envelopes."""
    return shape.optimalBoundingBox(False, False)

def solid(shape, label):
    """Reject fragmented, invalid or non-positive manufacturing components."""
    if shape.isNull() or not shape.isValid() or len(shape.Solids) != 1 or (shape.Volume <= 0):
        raise ValueError(label + ': expected one valid connected solid; change parting/feature layout.')
    return shape

def orient(shape, p):
    """Place the exact source rigidly before independent uniform compensation."""
    result = shape.copy()
    axis = {'X': V(1, 0, 0), 'Y': V(0, 1, 0), 'Z': V(0, 0, 1)}[p['pull_axis']]
    result.Placement = App.Placement(V(), App.Rotation(axis, V(0, 0, 1))).multiply(result.Placement)
    b = box(result)
    result.translate(V(-(b.XMin + b.XMax) / 2, -(b.YMin + b.YMax) / 2, -b.ZMin - b.ZLength * p['parting_pct'] / 100))
    return result

def compensated(shape, p):
    """Never modify stored source BREP; use analytic scaling and a checked OCC offset."""
    result = orient(shape, p)
    result.scale(1 / (1 - p['shrink_pct'] / 100))
    if p['cavity_allowance']:
        result = result.makeOffsetShape(p['cavity_allowance'], 1e-06, False, False, 0, 0, False)
    return solid(result, 'Compensated cavity')

def intersections(shape, x, y):
    """Find exact line/solid boundary intersections for gate and ejection datums."""
    b = box(shape)
    edge = Part.makeLine(V(x, y, b.ZMin - 1), V(x, y, b.ZMax + 1))
    return sorted({round(v.Point.z, 9) for v in shape.section(edge).Vertexes})

def samples(shape, limit=180):
    """Sample trimmed faces deterministically, preserving original face IDs for review."""
    points = []
    total = 0
    for index, face in enumerate(shape.Faces, 1):
        u0, u1, v0, v1 = face.ParameterRange
        for fu, fv in ((0.5, 0.5), (0.15, 0.15), (0.15, 0.85), (0.85, 0.15), (0.85, 0.85)):
            u = u0 + (u1 - u0) * fu
            v = v0 + (v1 - v0) * fv
            try:
                point = face.valueAt(u, v)
                if not face.isInside(point, 1e-06, True):
                    continue
                normal = face.normalAt(u, v)
                normal.normalize()
                total += 1
                if len(points) < limit:
                    points.append((index, point, normal))
            except (ValueError, Part.OCCError):
                continue
    return (points, total)

def review(cavity, a, b, p):
    """Bounded withdrawal and ball-tool collision tests report findings, never certification."""
    bound = box(cavity)
    extent = bound.ZLength
    tol = max(1e-06, cavity.Volume * 1e-09)
    findings = []
    withdraw = []
    distances = sorted(set([0.02, *[extent * f for f in (0.01, 0.05, 0.2, 0.5, 1.05)]]))
    for label, insert, direction in [('A', a, 1), ('B', b, -1)]:
        collisions = []
        for distance in distances:
            moved = insert.copy()
            moved.translate(V(0, 0, direction * distance))
            overlap = moved.common(cavity).Volume
            if overlap > tol:
                collisions.append(dict(travel_mm=distance, overlap_mm3=overlap))
        withdraw.append(dict(half=label, collisions=collisions))
        if collisions:
            findings.append(dict(severity='blocked', code='undercut_' + label, message=label + ' insert intersects the part during withdrawal. Change pull/parting, split the part, or engineer a side action.'))
    points, total = samples(cavity)
    low_draft = set()
    tool_collisions = set()
    r = p['tool_diameter'] / 2
    for index, point, normal in points:
        draft = math.degrees(math.asin(min(1, abs(normal.z))))
        if draft < p['minimum_draft'] - 1e-06:
            low_draft.add(index)
        if abs(point.z) < 1e-06:
            continue
        center = point - normal * r
        direction = -1 if point.z > 0 else 1
        insert = a if direction < 0 else b
        travel = abs(center.z) + r + 1
        cutter = Part.makeSphere(r, center).fuse(Part.makeCylinder(r, travel, center, V(0, 0, direction)))
        if cutter.common(insert).Volume > max(1e-05, r ** 3 * 1e-05):
            tool_collisions.add(index)
    if low_draft:
        findings.append(dict(severity='review', code='draft', faces=sorted(low_draft), message='Sampled faces have less than the requested draft. Adjust the parametric part before cutting tooling.'))
    if tool_collisions:
        findings.append(dict(severity='blocked', code='cutter_access', faces=sorted(tool_collisions), message='Selected ball cutter/shank collides with insert material at sampled cavity points. Use a smaller tool, different split or removable insert.'))
    depths = {'A': max(0, bound.ZMax), 'B': max(0, -bound.ZMin)}
    stock = box(a)
    stock_b = box(b)
    if stock.XLength > p['machine_x'] or stock.YLength > p['machine_y'] or max(stock.ZLength, stock_b.ZLength) > p['machine_z']:
        findings.append(dict(severity='blocked', code='machine_envelope', message='Insert stock exceeds the entered machine envelope; split tooling into modules or use a larger machine.'))
    if max(depths.values()) > p['tool_reach']:
        findings.append(dict(severity='blocked', code='tool_reach', message='Cavity depth exceeds the selected cutter reach.'))
    if total > len(points):
        findings.append(dict(severity='review', code='sampling_limit', message='Complex surface sampling reached its limit; untested faces remain.'))
    if str(p['resin_grade']).strip().lower() in ('', 'unassigned'):
        findings.append(dict(severity='review', code='resin', message='Assign a part material grade and its measured process shrinkage. Zero is an uncompensated design, not a material recommendation.'))
    length = max(bound.XLength, bound.YLength, bound.ZLength)
    stack = 2 * p['machining_tolerance'] * (1 - p['shrink_pct'] / 100) + length * p['shrink_uncertainty'] / 100
    if stack > p['part_tolerance']:
        findings.append(dict(severity='review', code='part_tolerance', message='Worst-case machining/shrinkage stack exceeds the requested finished-part tolerance.'))
    return dict(status='blocked' if any((f['severity'] == 'blocked' for f in findings)) else 'review_required', production_ready=False, findings=findings, sampled_surface_points=len(points), available_sample_points=total, withdrawal=withdraw, withdrawal_distances_mm=distances, depth_mm=depths, estimated_finished_dimension_error_mm=stack, requested_part_tolerance_mm=p['part_tolerance'], assumptions=['Discrete pull tests and sampled ball/shank access do not certify continuous demolding or every CNC surface.', 'Sharp corners, wall uniformity, holder/fixture access, filling, temperature and clamping require review.'], references=[])
from .molding import calculate
from .molding import calculate
def construct(source, p, inspect=True, routing=None):
    """Build matched inserts, feed, cooling, guides and ejectors around one exact solid."""
    casting = p.get('process', 'injection') == 'casting'
    from . import mold_routing
    route = mold_routing.normalize(routing) if routing is not None else None
    if casting and route is not None:
        raise ValueError('Optional runner routing currently supports injection tooling only.')
    original = solid(source, 'Source component')
    cavity = compensated(original, p)
    b = box(cavity)
    if min(b.XLength, b.YLength, b.ZLength) < 2 or max(b.XLength, b.YLength, b.ZLength) > 500:
        raise ValueError('Supported compensated component envelope is 2–500 mm on each axis.')
    m = p['margin']
    wall = p['back_wall']
    x0 = b.XMin - m
    x1 = b.XMax + m
    y0 = b.YMin - m
    y1 = b.YMax + m
    z0 = min(-wall, b.ZMin - wall)
    z1 = max(wall, b.ZMax + wall)
    lower = Part.makeBox(x1 - x0, y1 - y0, -z0, V(x0, y0, z0))
    upper = Part.makeBox(x1 - x0, y1 - y0, z1, V(x0, y0, 0))
    a = upper.cut(cavity)
    bb = lower.cut(cavity)
    records = []
    bom = []

    def item(name, shape, color, kind, reference=False):
        """All exported manufacturing pieces use the same assembled tooling datum."""
        solid(shape, name)
        records.append(dict(name=name, shape=shape, color=color, reference=reference, visible=not reference))
        if not reference:
            bom.append(dict(item=name, quantity=1, process=kind, volume_mm3=shape.Volume))
    feed_network = None
    if route and route['mode'] == 'parting_runner':
        sprue, feed_network = mold_routing.construct_feed(cavity, p, route, (x0, x1, y0, y1, z0, z1))
        a = a.cut(sprue)
        bb = bb.cut(sprue)
        gate_record = dict(mode='parting_runner', inlet_mm=feed_network['inlet_mm'], branches=feed_network['branches'])
    else:
        gx = p['gate_x'] * b.XLength / 2
        gy = p['gate_y'] * b.YLength / 2
        hits = intersections(cavity, gx, gy)
        if not hits or hits[-1] < -1e-06:
            raise ValueError('Direct sprue misses the A-side cavity. Adjust gate X/Y or parting plane.')
        gz = hits[-1]
        sprue = Part.makeCone(p['gate_diameter'] / 2, p['sprue_diameter'] / 2, z1 - gz + 0.1, V(gx, gy, gz - 0.1))
        if sprue.common(cavity).Volume <= 1e-07:
            raise ValueError('Gate does not connect to the cavity.')
        a = a.cut(sprue)
        gate_record = dict(x_mm=gx, y_mm=gy, z_mm=gz, inlet_z_mm=z1, diameter_mm=p['gate_diameter'], inlet_diameter_mm=p['sprue_diameter'])
    guide = p['guide_diameter']
    clearance = p['guide_clearance']
    web = p['minimum_web']
    gx0 = x0 + guide
    gx1 = x1 - guide
    gy0 = y0 + guide
    gy1 = y1 - guide
    positions = [(gx0, gy0), (gx1, gy0), (gx0, gy1), (gx1 - guide * 0.5, gy1)]
    for i, (x, y) in enumerate(positions, 1):
        hole = Part.makeCylinder((guide + clearance) / 2, z1 - z0, V(x, y, z0))
        if hole.distToShape(cavity)[0] < web:
            raise ValueError('Guide bore violates cavity steel web.')
        if route and hole.distToShape(sprue)[0] < web:
            raise ValueError('Melt feed violates guide-bore web.')
        a = a.cut(hole)
        bb = bb.cut(hole)
        item('guide_post_' + str(i), Part.makeCylinder(guide / 2, -z0 + min(12, z1 * 0.5), V(x, y, z0)), '#d4b67f', 'purchase / cut to length')
    casting_vents = []
    vent_tools = []
    if casting:
        for fx, fy in p['casting_vents']:
            x = fx * b.XLength / 2
            y = fy * b.YLength / 2
            hits = intersections(cavity, x, y)
            if not hits or hits[-1] < 0:
                raise ValueError('Casting vent misses the A-side cavity; adjust its XY position or parting plane.')
            tip = hits[-1]
            tool = Part.makeCylinder(p['casting_vent_diameter'] / 2, z1 - tip + 0.1, V(x, y, tip - 0.1))
            if tool.common(cavity).Volume <= 1e-07:
                raise ValueError('Casting vent does not reach the cavity.')
            if tool.common(sprue).Volume > 1e-07 or any((tool.common(other).Volume > 1e-07 for other in vent_tools)):
                raise ValueError('Casting vents must remain separate from the pour channel and one another.')
            for xg, yg in positions:
                if math.hypot(x - xg, y - yg) < (p['casting_vent_diameter'] + guide + clearance) / 2 + web:
                    raise ValueError('Casting vent violates guide-bore web.')
            a = a.cut(tool)
            vent_tools.append(tool)
            casting_vents.append(dict(x_mm=x, y_mm=y, cavity_z_mm=tip, exit_z_mm=z1, diameter_mm=p['casting_vent_diameter']))
    cooling = []
    if not casting and p['cooling']:
        for label, z in [('A', b.ZMax + wall / 2), ('B', b.ZMin - wall / 2)]:
            for y in (b.YMin - m * 0.2, b.YMax + m * 0.2):
                drill = Part.makeCylinder(p['cooling_diameter'] / 2, x1 - x0, V(x0, y, z), V(1, 0, 0))
                if drill.distToShape(cavity)[0] < web:
                    raise ValueError('Cooling channel violates cavity steel web.')
                if drill.common(sprue).Volume > 1e-07:
                    raise ValueError('Cooling intersects sprue.')
                if route and drill.distToShape(sprue)[0] < web:
                    raise ValueError('Melt feed violates cooling-channel web.')
                for xg, yg in positions:
                    if abs(y - yg) < (guide + clearance + p['cooling_diameter']) / 2 + web:
                        raise ValueError('Cooling channel too close to guide bore; increase stock margin.')
                if label == 'A':
                    a = a.cut(drill)
                else:
                    bb = bb.cut(drill)
                cooling.append(dict(half=label, axis='X', y_mm=y, z_mm=z, diameter_mm=p['cooling_diameter'], length_mm=x1 - x0))
    ejectors = []
    ejector_shapes = []
    for i, (fx, fy) in enumerate([] if casting else p['ejectors'], 1):
        x = fx * b.XLength / 2
        y = fy * b.YLength / 2
        hits = intersections(cavity, x, y)
        if not hits or hits[0] >= 0:
            raise ValueError('Ejector ' + str(i) + ' misses the B-side part; adjust XY positions or parting plane.')
        tip = hits[0]
        radius = p['ejector_diameter'] / 2
        start = z0 - 8
        bore = Part.makeCylinder((p['ejector_diameter'] + p['ejector_clearance']) / 2, tip - z0 + radius, V(x, y, z0))
        if route and bore.distToShape(sprue)[0] < web:
            raise ValueError('Melt feed violates ejector-bore web.')
        pin = Part.makeCylinder(radius, tip - start + radius, V(x, y, start)).cut(cavity)
        if len(pin.Solids) != 1:
            raise ValueError('Ejector tip crosses a thin wall; choose another position or smaller pin.')
        if any((pin.distToShape(other)[0] < web for other in ejector_shapes)):
            raise ValueError('Ejectors are too close together.')
        for channel in cooling:
            if channel['half'] == 'B' and abs(y - channel['y_mm']) < radius + p['cooling_diameter'] / 2 + web:
                raise ValueError('Ejector collides with cooling web.')
        bb = bb.cut(bore)
        item('ejector_pin_' + str(i), pin, '#cfba8c', 'purchase / finish tip')
        ejector_shapes.append(pin)
        ejectors.append(dict(x_mm=x, y_mm=y, tip_z_mm=tip, diameter_mm=p['ejector_diameter'], bore_mm=p['ejector_diameter'] + p['ejector_clearance']))
    vent = None
    edge_vents = []
    if route and route['vent_mode'] == 'edge_ports':
        tools, edge_vents = mold_routing.construct_vents(cavity, p, route, (x0, x1, y0, y1, z0, z1), sprue)
        for tool in tools:
            for xg, yg in positions:
                hole = Part.makeCylinder((guide + clearance) / 2, z1 - z0, V(xg, yg, z0))
                if tool.distToShape(hole)[0] < web:
                    raise ValueError('Vent violates guide-bore web.')
            a = a.cut(tool)
    elif not casting and p['vent_depth']:
        strip = Part.makeBox(p['vent_width'], y1, p['vent_depth'], V(-p['vent_width'] / 2, 0, 0))
        contact = strip.common(cavity)
        if contact.Volume <= 1e-07:
            raise ValueError('Parting vent misses material. Change parting or disable the vent for manual layout.')
        cb = box(contact)
        start = max(0, cb.YMax - 0.1)
        vent = Part.makeBox(p['vent_width'], y1 - start, p['vent_depth'], V(-p['vent_width'] / 2, start, 0))
        a = a.cut(vent)
    item('insert_A', a, '#729fc4', 'CNC cavity + pour/vent drills' if casting else 'CNC from parting face + back sprue + side drills')
    item('insert_B', bb, '#77b6a4', 'CNC cavity + guide bores' if casting else 'CNC from parting face + ejector reaming + side drills')
    if route:
        item('melt_feed_reference', sprue, '#dc9143', 'reference', True)
        records[-1]['visible'] = True
    item('source_component', orient(original, p), '#db9175', 'reference', True)
    item('compensated_cavity', cavity, '#e0af78', 'reference', True)
    tolerance = max(1e-06, cavity.Volume * 1e-09)
    if max(a.common(cavity).Volume, bb.common(cavity).Volume, a.common(bb).Volume) > tolerance:
        raise ValueError('Mold/cavity material interference.')
    hardware = [row for row in records if not row['reference']]
    for i, row in enumerate(hardware):
        for other in hardware[i + 1:]:
            if row['shape'].common(other['shape']).Volume > tolerance:
                raise ValueError('Tooling components overlap: ' + row['name'] + ' / ' + other['name'])
    report = dict(geometry_valid=True, tooling_process='casting' if casting else 'injection', casting_vents=casting_vents, source_volume_mm3=original.Volume, cavity_volume_mm3=cavity.Volume, original_preserved=p['shrink_pct'] == 0 and p['cavity_allowance'] == 0, stock_mm=dict(width=x1 - x0, length=y1 - y0, A_height=z1, B_height=-z0), datum='Parting Z=0; cavity centered XY', tooling_bom=bom, tool_material=p['tool_material'], guide_bores=[dict(x_mm=x, y_mm=y, bore_diameter_mm=guide + clearance, post_diameter_mm=guide, z_min_mm=z0, z_max_mm=z1) for x, y in positions], stock_origin_mm=dict(A=[x0, y0, 0], B=[x0, y0, z0]), cooling_drills=cooling, ejectors=ejectors, vent=dict(width_mm=p['vent_width'], depth_mm=p['vent_depth'], start_y_mm=start, end_y_mm=y1) if vent is not None else None, gate=gate_record, tolerance=calculate(p), clamp_force_bound_kn=None if casting else (b.XLength * b.YLength + math.pi * (p['sprue_diameter'] / 2) ** 2) * p['cavity_pressure'] * p['clamp_factor'] / 1000, clamp_basis='Casting clamp requirement unassigned; no injection pressure assumed.' if casting else 'Component projected bounding rectangle plus sprue circle × assumed cavity pressure × factor; kN', machining_setups=[dict(part='insert_A', setups=['Parting face −Z approach: cavity, guide bores, vent', 'Back face +Z: tapered direct sprue', 'Side X: straight cooling drills']), dict(part='insert_B', setups=['Parting face +Z approach: core/cavity, guide and ejector bores', 'Side X: straight cooling drills'])])
    if casting:
        report['machining_setups'] = [dict(part='insert_A', setups=['Parting face −Z approach: cavity and guide bores', 'Back face +Z: pour channel and separate vents']), dict(part='insert_B', setups=['Parting face +Z approach: cavity and guide bores'])]
    if route:
        report.update(routing_version=mold_routing.VERSION, feed_network=feed_network, edge_vents=edge_vents, flow_balance_verified=False, vacuum_hardware_qualified=False)
        if feed_network:
            fb = box(sprue)
            area = (max(b.XMax, fb.XMax) - min(b.XMin, fb.XMin)) * (max(b.YMax, fb.YMax) - min(b.YMin, fb.YMin))
            report.update(projected_area_bound_mm2=area, clamp_force_bound_kn=area * p['cavity_pressure'] * p['clamp_factor'] / 1000, clamp_basis='Combined cavity and feed projected bounding rectangle × assumed pressure × factor; geometric screen, not a press rating.')
            report['machining_setups'][0]['setups'][0] += '; half-round runners and edge gates'
            report['machining_setups'][0]['setups'][1] = 'Back face +Z: tapered sprue to parting runner'
            report['machining_setups'][1]['setups'][0] += '; matching half-round runners and edge gates'
    if inspect:
        report['moldability'] = review(cavity, a, bb, p)
    records[-2]['validation'] = report
    return (records, report)

def check(parts, p):
    """Read geometry checks retained with cached reference parts, avoiding repeated analysis."""
    result = next((x.get('validation') for x in parts if x['name'] == 'source_component'), None)
    if not result:
        raise ValueError('Tooling construction report missing.')
    return result

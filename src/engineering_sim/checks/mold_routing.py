from copy import deepcopy
import math
VERSION = '0.1.0'
CAD_TOL_MM = 1e-06
JOIN_OVERLAP_MM = 0.05

def fields(prefix='route.'):
    """Return optional controls; position fractions scale with the cavity, stock sizes do not."""

    def number(key, label, value, low, high, unit='mm'):
        return dict(key=prefix + key, label=label, min=low, max=high, type='number', unit=unit, group='Mold routing')
    return [dict(key=prefix + 'mode', label='Melt feed', type='choice', choices=['direct', 'parting_runner'], unit='', group='Mold routing'), dict(key=prefix + 'side', label='Runner entry side', type='choice', choices=['-X', '+X', '-Y', '+Y'], unit='', group='Mold routing'), dict(key=prefix + 'gate_offsets', label='Gate offsets along cavity edge', type='json', unit='fraction of half-width', group='Mold routing'), number('inlet_offset', 'Sprue offset along edge', 0, -0.85, 0.85, 'fraction of half-width'), number('runner_diameter_mm', 'Common runner diameter', 4, 1, 16), dict(key=prefix + 'branch_diameters_mm', label='Branch diameters (empty = common diameter)', type='json', unit='mm', group='Mold routing'), number('runner_setback_mm', 'Runner centre outside cavity bounds', 16, 4, 80), number('gate_width_mm', 'Edge gate width', 2, 0.3, 20), number('gate_height_mm', 'Edge gate height', 1, 0.2, 8), number('gate_length_mm', 'Edge gate land length', 3, 1, 12), dict(key=prefix + 'vent_mode', label='Parting vent layout', type='choice', choices=['legacy', 'edge_ports'], unit='', group='Mold routing'), dict(key=prefix + 'vent_side', label='Vent outlet side', type='choice', choices=['-X', '+X', '-Y', '+Y'], unit='', group='Mold routing'), dict(key=prefix + 'vent_offsets', label='Vent offsets along cavity edge', type='json', unit='fraction of half-width', group='Mold routing')]

def normalize(raw=None):
    """Validate bounded finite routing data before kernel work; reject ambiguous layouts."""
    raw = {} if raw is None else deepcopy(raw)
    if not isinstance(raw, dict):
        raise ValueError('Routing must be an object.')
    schema = fields(prefix='')
    known = {row['key'] for row in schema}
    if set(raw) - known:
        raise ValueError('Unknown routing controls: ' + ', '.join(sorted(set(raw) - known)))
    values = {}
    for row in schema:
        value = raw[row['key']]
        if row['type'] == 'number':
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(row['key'] + ': expected a finite number.')
            if not row['min'] <= value <= row['max']:
                raise ValueError(row['key'] + ': outside supported range.')
            value = float(value)
        elif row['type'] == 'choice':
            if value not in row['choices']:
                raise ValueError(row['key'] + ': unknown choice.')
        elif not isinstance(value, list):
            raise ValueError(row['key'] + ': expected a list.')
        values[row['key']] = value
    for key in ('gate_offsets', 'vent_offsets'):
        points = values[key]
        if not 1 <= len(points) <= 8 or any((type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 0.85 for v in points)):
            raise ValueError(key + ': use one to eight finite offsets between -0.85 and 0.85.')
        if len(set(points)) != len(points):
            raise ValueError(key + ': duplicate offsets are not permitted.')
    diameters = values['branch_diameters_mm']
    if diameters and len(diameters) != len(values['gate_offsets']):
        raise ValueError('Specify one branch diameter per gate, or leave the list empty.')
    if any((type(d) not in (int, float) or not math.isfinite(d) or (not 1 <= d <= 16) for d in diameters)):
        raise ValueError('Branch diameters must be finite, between 1 and 16 mm.')
    maximum_d = max([values['runner_diameter_mm'], *diameters])
    if values['gate_length_mm'] <= maximum_d / 2 + JOIN_OVERLAP_MM:
        raise ValueError('Gate land must extend beyond the runner radius.')
    if values['runner_setback_mm'] <= values['gate_length_mm'] + maximum_d / 2:
        raise ValueError('Increase runner setback beyond the gate land and runner radius.')
    return values

def _basis(bounds, side):
    """Return in-plane tangent/outward vectors and centred cavity half-extents."""
    import FreeCAD as App
    if side.endswith('Y'):
        return (App.Vector(1, 0, 0), App.Vector(0, 1 if side[0] == '+' else -1, 0), bounds.XLength / 2, bounds.YLength / 2)
    return (App.Vector(0, 1, 0), App.Vector(1 if side[0] == '+' else -1, 0, 0), bounds.YLength / 2, bounds.XLength / 2)

def _edge_point(cavity, side, offset):
    """Find the actual parting-plane boundary; never substitute the bounding rectangle."""
    import Part
    bounds = cavity.optimalBoundingBox(False, False)
    tangent, normal, half_t, half_n = _basis(bounds, side)
    centre = tangent * (offset * half_t)
    line = Part.makeLine(centre - normal * (half_n + 1), centre + normal * (half_n + 1))
    points = [vertex.Point for vertex in cavity.section(line).Vertexes]
    if not points:
        raise ValueError('Selected edge position misses the cavity at the parting plane.')
    return (max(points, key=lambda p: p.dot(normal)), tangent, normal, half_t, half_n)

def _rectangle(start, end, tangent, width, zmin, height):
    """Extrude a rectangular planar gate/vent, preserving arbitrary X/Y side choice."""
    import FreeCAD as App
    import Part
    points = [start - tangent * width / 2, end - tangent * width / 2, end + tangent * width / 2, start + tangent * width / 2]
    vertices = [App.Vector(p.x, p.y, zmin) for p in points]
    return Part.Face(Part.makePolygon(vertices + [vertices[0]])).extrude(App.Vector(0, 0, height))

def _tube(start, end, diameter):
    """Create one full-round cold runner; zero-length shared-trunk pieces are omitted."""
    import Part
    delta = end - start
    if delta.Length <= CAD_TOL_MM:
        return None
    return Part.makeCylinder(diameter / 2, delta.Length, start, delta / delta.Length)

def construct_feed(cavity, p, routing, stock_bounds):
    """Build a connected sprue/tree/gate tool with measured path geometry, not fill predictions."""
    import FreeCAD as App
    import Part
    r = normalize(routing)
    x0, x1, y0, y1, z0, z1 = stock_bounds
    bounds = cavity.optimalBoundingBox(False, False)
    tangent, normal, half_t, half_n = _basis(bounds, r['side'])
    inlet = tangent * (r['inlet_offset'] * half_t) + normal * (half_n + r['runner_setback_mm'])
    diameter = r['runner_diameter_mm']
    if p['sprue_diameter'] < diameter:
        raise ValueError('Sprue inlet diameter must cover the common runner diameter.')
    sprue = Part.makeCone(diameter / 2, p['sprue_diameter'] / 2, z1 + JOIN_OVERLAP_MM, inlet - App.Vector(0, 0, JOIN_OVERLAP_MM))
    tools = [sprue]
    gate_tools, runner_tools, branches = ([], [], [])
    for index, offset in enumerate(r['gate_offsets']):
        entry, tangent, normal, _, _ = _edge_point(cavity, r['side'], offset)
        branch_d = (r['branch_diameters_mm'] or [diameter] * len(r['gate_offsets']))[index]
        outer = entry + normal * r['gate_length_mm']
        elbow = tangent * (offset * half_t) + normal * (half_n + r['runner_setback_mm'])
        gate = _rectangle(entry - normal * JOIN_OVERLAP_MM, outer + normal * JOIN_OVERLAP_MM, tangent, r['gate_width_mm'], -r['gate_height_mm'] / 2, r['gate_height_mm'])
        if gate.common(cavity).Volume <= CAD_TOL_MM:
            raise ValueError('Edge gate does not intersect the cavity; change its position or height.')
        pieces = [tool for tool in (_tube(inlet, elbow, diameter), _tube(elbow, outer - normal * JOIN_OVERLAP_MM, branch_d)) if tool is not None]
        pieces.append(Part.makeSphere(max(diameter, branch_d) / 2, elbow))
        for tool in pieces:
            if tool.common(cavity).Volume > CAD_TOL_MM:
                raise ValueError('Runner enters the cavity outside its gate; increase gate land/setback.')
        runner_tools.extend(pieces)
        gate_tools.append(gate)
        tools.extend([*pieces, gate])
        branches.append(dict(id='gate_' + str(index + 1), offset=offset, entry_mm=[entry.x, entry.y, entry.z], runner_diameter_mm=branch_d, runner_length_mm=(elbow - inlet).Length + (outer - elbow).Length, gate_width_mm=r['gate_width_mm'], gate_height_mm=r['gate_height_mm'], gate_length_mm=r['gate_length_mm']))
    for i, gate in enumerate(gate_tools):
        if any((gate.common(other).Volume > CAD_TOL_MM for other in gate_tools[i + 1:])):
            raise ValueError('Edge gates overlap; increase gate spacing or reduce gate width.')
    feed = tools[0]
    for tool in tools[1:]:
        feed = feed.fuse(tool)
    feed = feed.removeSplitter()
    if not feed.isValid() or len(feed.Solids) != 1:
        raise ValueError('Feed network must be one valid connected solid.')
    fb = feed.optimalBoundingBox(False, False)
    web = p['minimum_web']
    if min(fb.XMin - x0, x1 - fb.XMax, fb.YMin - y0, y1 - fb.YMax) < web:
        raise ValueError('Runner/sprue violates the outer stock web; increase stock margin.')
    if fb.ZMin - z0 < web:
        raise ValueError('Runner violates the back wall of the B insert.')
    lengths = [branch['runner_length_mm'] + branch['gate_length_mm'] for branch in branches]
    mean = sum(lengths) / len(lengths)
    report = dict(version=VERSION, mode='parting_runner', side=r['side'], inlet_mm=[inlet.x, inlet.y, z1], branches=branches, path_length_imbalance_fraction=(max(lengths) - min(lengths)) / mean if mean else 0, feed_volume_outside_cavity_mm3=feed.cut(cavity).Volume, projected_bounds_mm=[fb.XMin, fb.YMin, fb.XMax, fb.YMax], flow_balance_verified=False, scope='Connected routing geometry. Path symmetry is not a filling, packing or thermal result.')
    return (feed, report)

def construct_vents(cavity, p, routing, stock_bounds, feed):
    """Cut specified shallow edge vents; outlet coordinates identify a future vacuum interface."""
    r = normalize(routing)
    if p['vent_depth'] == 0:
        return ([], [])
    x0, x1, y0, y1, _, _ = stock_bounds
    tools, records = ([], [])
    for index, offset in enumerate(r['vent_offsets']):
        entry, tangent, normal, _, _ = _edge_point(cavity, r['vent_side'], offset)
        outer_n = {'+X': x1, '-X': -x0, '+Y': y1, '-Y': -y0}[r['vent_side']]
        end = entry + normal * (outer_n - entry.dot(normal))
        tool = _rectangle(entry - normal * JOIN_OVERLAP_MM, end + normal * JOIN_OVERLAP_MM, tangent, p['vent_width'], 0, p['vent_depth'])
        if tool.common(cavity).Volume <= 1e-09:
            raise ValueError('Selected vent does not connect to the cavity.')
        if tool.common(feed).Volume > 1e-09:
            raise ValueError('Vent intersects the melt feed network; select another outlet.')
        if any((tool.common(other).Volume > 1e-09 for other in tools)):
            raise ValueError('Selected vent channels overlap.')
        tools.append(tool)
        records.append(dict(id='vent_' + str(index + 1), side=r['vent_side'], offset=offset, entry_mm=[entry.x, entry.y, 0], outlet_mm=[end.x, end.y, 0], width_mm=p['vent_width'], depth_mm=p['vent_depth'], vacuum_hardware_qualified=False))
    return (tools, records)

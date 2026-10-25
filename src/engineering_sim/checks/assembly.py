# DESCRIPTION: Deterministic assembly-sequence check: each declared step sweeps the moving parts' exact faces back along their insertion path and intersects the sweep with everything already installed; fastener tool envelopes are swept to free space the same way.
"""Sample exact solids along straight insertion paths and check tool access.
The step is 1 mm; thinner obstacles can be missed. Only the tested translations
and declared tool envelopes are checked. Radial separation uses exact booleans.
Parts and allowed-contact patterns are supplied by the caller; no design is bundled.
"""
import fnmatch
import math

VOLUME_TOL = 1e-2          # mm³: coincident-face numerical noise
STEP_MM = 1.0              # path sampling step; obstacles thinner than this along the path can be missed (declared)
RADIAL_MARGIN_MM = 0.1     # proposed separation cylinder sits this far beyond the obstacle's extreme vertex radius


def _match(names, patterns):
    """Sorted part names matching any glob pattern."""
    return sorted({n for n in names for pat in patterns if fnmatch.fnmatchcase(n, pat)})


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    if n <= 0:
        raise ValueError('Assembly direction must be non-zero.')
    return [c / n for c in v]


def _boxes_near(a, b, pad=0.05):
    """Axis-aligned bounding boxes overlap (with a small pad)."""
    return not (a.XMax + pad < b.XMin or b.XMax + pad < a.XMin or a.YMax + pad < b.YMin or
                b.YMax + pad < a.YMin or a.ZMax + pad < b.ZMin or b.ZMax + pad < a.ZMin)


def _radii(shape, d, cache):
    """(min, max) vertex radius of `shape` about the origin line along d (cached by shape identity and direction)."""
    key = (id(shape), tuple(round(c, 9) for c in d), 'radii')
    if key not in cache:
        rs = []
        for v in shape.Vertexes:
            p = (v.X, v.Y, v.Z)
            along = p[0] * d[0] + p[1] * d[1] + p[2] * d[2]
            rs.append(math.sqrt(max(0.0, p[0] ** 2 + p[1] ** 2 + p[2] ** 2 - along ** 2)))
        cache[key] = (min(rs), max(rs)) if rs else (0.0, 0.0)
    return cache[key]


def _within(shape, d, radius, inside, span, cache):
    """Exact proof that `shape` lies inside (inside=True) or outside a cylinder of `radius` about the origin line along d.

    Inside: nothing of the shape beyond the cylinder (common with a large tube R..R_big ≤ VOLUME_TOL). Outside: common with
    the cylinder ≤ VOLUME_TOL. `span` is a half-length covering both shapes' extents along d. Cached.
    """
    import FreeCAD as App
    import Part
    key = (id(shape), tuple(round(c, 9) for c in d), round(radius, 6), inside)
    if key in cache:
        return cache[key]
    base = App.Vector(*[-c * span for c in d])
    axis = App.Vector(*d)
    cyl = Part.makeCylinder(radius, 2.0 * span, base, axis)
    if inside:
        big = Part.makeCylinder(radius + 4.0 * span, 2.0 * span, base, axis)
        vol = shape.common(big.cut(cyl)).Volume
    else:
        vol = shape.common(cyl).Volume
    cache[key] = vol <= VOLUME_TOL
    return cache[key]


def radially_separated(moving, obstacle, d, travel, cache):
    """True when exact booleans prove a cylinder about the origin line along d separates the two parts (see module doc)."""
    m0, m1 = _radii(moving, d, cache)
    o0, o1 = _radii(obstacle, d, cache)
    boxes = [moving.BoundBox, obstacle.BoundBox]
    span = travel + max(max(abs(b.XMin), abs(b.XMax), abs(b.YMin), abs(b.YMax), abs(b.ZMin), abs(b.ZMax)) for b in boxes) + 10.0
    if m0 > o1 + 2.0 * RADIAL_MARGIN_MM:                         # moving part outboard of the obstacle
        r = o1 + RADIAL_MARGIN_MM
        return _within(obstacle, d, r, True, span, cache) and _within(moving, d, r, False, span, cache)
    if m1 + 2.0 * RADIAL_MARGIN_MM < o0:                         # moving part inboard of the obstacle
        r = o0 - RADIAL_MARGIN_MM
        return _within(obstacle, d, r, False, span, cache) and _within(moving, d, r, True, span, cache)
    return False


def sweep_blocks(shape, direction, travel, installed):
    """Installed parts hit while `shape` travels into place: {name: dict(volume_mm3, at_mm)} (first hit per part).

    The solid is translated back along −direction in fixed STEP_MM increments from STEP_MM to `travel`; each copy is
    intersected with installed solids whose bounding boxes it touches. Booleans are between valid solids only.
    (Face-prism sweeps were tried first and rejected: prisms of imported spline faces gave false hits, e.g. a 10 mm³
    prism reporting 197 mm³ common and hits at radii the motor never reaches.)
    Limit: an obstacle thinner than STEP_MM along the path can be missed.
    """
    import FreeCAD as App
    blocks = {}
    pending = list(installed)
    n = int(math.ceil(travel / STEP_MM))
    for i in range(1, n + 1):
        t = min(i * STEP_MM, travel)
        moved = shape.copy()
        moved.translate(App.Vector(*[-c * t for c in direction]))
        mb = moved.BoundBox
        still = []
        for name, other in pending:
            if _boxes_near(mb, other.BoundBox, pad=0.0):
                vol = moved.common(other).Volume
                if vol > VOLUME_TOL:
                    blocks[name] = dict(volume_mm3=round(vol, 3), at_mm=round(t, 3))
                    continue                                   # first hit recorded; stop testing this obstacle
            still.append((name, other))
        pending = still
        if not pending:
            break
        # Early exit (exact, deterministic): the copy moves toward −direction. Once the far end of its box along
        # `direction` lies below the near end of every remaining obstacle's box, no later copy can touch them.
        if max(_proj(mb, direction)) < min(min(_proj(o.BoundBox, direction)) for _, o in pending) - 1e-6:
            break
    return blocks


def _proj(box, d):
    """Projections of a bounding box's 8 corners onto direction d."""
    return [x * d[0] + y * d[1] + z * d[2] for x in (box.XMin, box.XMax) for y in (box.YMin, box.YMax) for z in (box.ZMin, box.ZMax)]


def tool_blocks(tool, installed):
    """Installed parts intersecting a straight tool envelope from the fastener head outward: {name: volume}."""
    import FreeCAD as App
    import Part
    out = _unit(tool['outward'])
    body = Part.makeCylinder(tool['diameter_mm'] / 2.0, tool['length_mm'], App.Vector(*tool['head']), App.Vector(*out))
    blocks = {}
    for name, other in installed:
        if not _boxes_near(body.BoundBox, other.BoundBox):
            continue
        vol = body.common(other).Volume
        if vol > VOLUME_TOL:
            blocks[name] = round(vol, 3)
    return blocks


def check(parts, steps):
    """Run the declared sequence against `parts` ({name: shape}); returns a JSON record with per-step results.

    Does not raise: the caller decides whether a blocked step fails the build (models normally raise).
    """
    names = list(parts)
    used = set()
    resolved = []
    for step in steps:
        moving = _match(names, step['parts'])
        if not moving and not step.get('tools'):          # tool-only steps (e.g. a fastener not modelled as a solid) are allowed
            raise ValueError(f"Assembly step '{step['step']}' matches no parts: {step['parts']}")
        overlap = used & set(moving)
        if overlap:
            raise ValueError(f"Assembly step '{step['step']}' re-uses parts already placed: {sorted(overlap)[:3]}")
        used |= set(moving)
        resolved.append((step, moving))
    installed = [n for n in names if n not in used]              # the base: everything not placed by a step
    record = dict(method=f'translated exact solids every {STEP_MM} mm along the path + tool envelopes; exact radial-separation prefilter',
                  step_mm=STEP_MM, volume_tolerance_mm3=VOLUME_TOL, base=sorted(installed), steps=[], blocked_steps=0)
    cache = {}
    import time
    for step, moving in resolved:
        started = time.perf_counter()                            # wall time per step (metadata; not part of the verdict)
        allowed = set(_match(names, step.get('allowed', [])))
        others = [(n, parts[n]) for n in installed if n not in allowed]
        direction = _unit(step['direction'])
        blocks = {}
        separated = 0
        for name in moving:
            # Pairs proven radially separated cannot meet on this axial move; sweep only the rest.
            near = [(n, o) for n, o in others if not radially_separated(parts[name], o, direction, step['travel_mm'], cache)]
            separated += len(others) - len(near)
            for hit, info in sweep_blocks(parts[name], direction, step['travel_mm'], near).items():
                blocks.setdefault(name, {})[hit] = info
        tools = []
        for tool in step.get('tools', []):
            hits = tool_blocks(tool, [(n, parts[n]) for n in installed + moving if n not in moving])
            tools.append(dict(name=tool['name'], head=[round(c, 3) for c in tool['head']], blocked_by=hits, ok=not hits))
        ok = not blocks and all(t['ok'] for t in tools)
        record['steps'].append(dict(step=step['step'], moving=moving, direction=direction, travel_mm=step['travel_mm'],
                                    allowed=sorted(allowed), blocked=blocks, tools=tools, ok=ok, pairs_radially_separated=separated,
                                    seconds=round(time.perf_counter() - started, 1)))
        record['blocked_steps'] += 0 if ok else 1
        installed += moving
    record['ok'] = record['blocked_steps'] == 0
    return record

# DESCRIPTION: Deterministic mounted-fit check between two built designs: derives the mounting transform from declared interface frames (origin, axis, feet/gravity), then MEASURES face contact, pilot coaxiality, bolt-pattern match, feet orientation and support gap, shaft tip and coaxiality, key clocking and clashes on the placed exact solids.
"""Mounted fit check (FreeCAD worker only; FreeCAD is imported inside functions).


Transform
  Each side declares an orthonormal mounting frame: `origin` (face centre), `axis` (unit vector from the mounting face toward
  the driven machine) and `down` (unit vector ⟂ axis: the kit's foot direction; the engine's gravity). The kit is moved so
  origin→origin, axis→axis, down→down: R = B_dst · B_srcᵀ with B = [axis, down, axis × down] as columns; p' = R(p − o_src) + o_dst.
  The frames are proper (right-handed) on both sides, so R is a rotation; det(R) is checked.
Measurements (tolerances are module constants)
  face contact   exact min distance between the two mounting-face parts = 0 and common volume ≤ VOL_TOL
  pilot          axes of the kit spigot OD cylinder and the engine pilot bore cylinder (matched by radius): offset ≤ AXIS_TOL
  bolts          every engine hole axis (cylinders of the declared hole radius, parallel to the axis) has a kit tapped-hole axis
                 within BOLT_TOL in the face plane (and the counts match)
  feet           the kit foot pads' planar faces whose normal ≈ `down` after placement: angle to gravity ≤ ANGLE_TOL; height of
                 their lowest point above the support top is reported (a finding when > 0, a failure when < 0)
  shaft          kit stub tip vs the engine's expected tip along the axis ≤ TIP_TOL; stub axis offset from the coupler bore axis
  key            rotor group turned about the axis in 1° steps: the clear band where the key does not intersect the coupler
                 (a finding if empty); the rotor is left at the band centre for the clash check
  clashes        exact common volume between every kit and engine solid (bounding-box prefilter), excluding declared placeholders
Deterministic: inputs are saved BREPs from hash-recorded builds; parts are processed in sorted name order.
"""
import math

VOL_TOL = 1e-2        # mm³ numerical noise on coincident faces
AXIS_TOL = 0.01       # mm coaxial offset for pilots
BOLT_TOL = 0.05       # mm in-plane hole-centre match
TIP_TOL = 0.05        # mm along the axis
ANGLE_TOL = 0.5       # deg feet normal to gravity


def _vec(v):
    import FreeCAD as App
    return App.Vector(*v)


def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mounting_matrix(src, dst):
    """FreeCAD Matrix moving frame `src` onto frame `dst` (dicts with origin, axis, down). Raises if a frame is not orthonormal."""
    import FreeCAD as App
    R, t = frame_rotation(src, dst)
    m = App.Matrix(R[0][0], R[0][1], R[0][2], t[0], R[1][0], R[1][1], R[1][2], t[1], R[2][0], R[2][1], R[2][2], t[2], 0, 0, 0, 1)
    return m, R, t


def frame_rotation(src, dst):
    """Pure (R, t) with p' = R·p + t moving frame `src` onto `dst` (see module docstring)."""
    bases = []
    for f in (src, dst):
        a, d = _unit(f['axis']), _unit(f['down'])
        if abs(_dot(a, d)) > 1e-6:
            raise ValueError(f'Mounting frame axis and down are not perpendicular: {f}')
        bases.append((a, d, _cross(a, d)))
    (a1, d1, c1), (a2, d2, c2) = bases
    # R = B2 · B1ᵀ (columns of B are axis, down, axis × down)
    R = [[a2[i] * a1[j] + d2[i] * d1[j] + c2[i] * c1[j] for j in range(3)] for i in range(3)]
    det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
           + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    if abs(det - 1.0) > 1e-6:
        raise ValueError(f'Mounting transform is not a proper rotation (det {det:.6f}).')
    o1, o2 = src['origin'], dst['origin']
    t = [o2[i] - sum(R[i][j] * o1[j] for j in range(3)) for i in range(3)]
    return R, t


def place(shapes, matrix):
    """Apply a rigid matrix to each shape in place (transformShape keeps exact geometry for rotations)."""
    for s in shapes:
        s.transformShape(matrix, False, False)


def cylinder_axes(shape, radius, axis, rtol=0.02, atol=1e-4, band=None):
    """[(point on axis, radius)] for cylindrical faces of `shape` with the given radius and axis parallel to `axis` (deduplicated).

    band = (origin, lo, hi): keep only faces whose centre of mass lies lo..hi mm along `axis` from `origin` (e.g. the holes of one
    flange of a part that has hole patterns at both ends).
    """
    import Part
    out = []
    for f in shape.Faces:
        if not isinstance(f.Surface, Part.Cylinder):
            continue
        if band is not None:
            cm = f.CenterOfMass
            along = _dot([cm.x - band[0][0], cm.y - band[0][1], cm.z - band[0][2]], axis)
            if not band[1] <= along <= band[2]:
                continue
        cyl = f.Surface
        if abs(cyl.Radius - radius) > rtol or abs(abs(_dot(_unit([cyl.Axis.x, cyl.Axis.y, cyl.Axis.z]), axis)) - 1.0) > atol:
            continue
        c = [cyl.Center.x, cyl.Center.y, cyl.Center.z]
        if all(_dist_in_plane(c, q, axis) > 1e-6 for q, _ in out):
            out.append((c, cyl.Radius))
    return out


def _dist_in_plane(p, q, axis):
    """Distance between two points after removing the component along `axis` (i.e. between two parallel axis lines)."""
    d = [p[i] - q[i] for i in range(3)]
    along = _dot(d, axis)
    return math.sqrt(max(0.0, _dot(d, d) - along * along))


def check_mount(kit, engine, spec):
    """Run the mounted fit check. kit/engine: {name: shape} ALREADY PLACED (kit transformed). spec: declared interface data.

    spec keys: axis (dst), down (dst), kit_face_part, engine_face_part, pilot=dict(kit_part, kit_radius, engine_part, engine_radius),
    bolts=dict(kit_part, kit_radius, engine_part, engine_radius, count), feet=dict(parts_prefix, support_part),
    shaft=dict(kit_part, radius, engine_tip_along_axis, engine_bore_part, engine_bore_radius), key=dict(kit_part, rotor_parts, engine_part),
    placeholders (engine parts ignored for clashes). Returns (record, failures, findings).
    """
    import FreeCAD as App
    axis, down = _unit(spec['axis']), _unit(spec['down'])
    fails, findings, rec = [], [], {}
    # ---- face contact ----
    kf, ef = kit[spec['kit_face_part']], engine[spec['engine_face_part']]
    d = kf.distToShape(ef)[0]
    v = kf.common(ef).Volume
    rec['face_contact'] = dict(distance_mm=round(d, 5), overlap_mm3=round(v, 4), ok=d <= 1e-4 and v <= VOL_TOL)
    if not rec['face_contact']['ok']:
        fails.append(f'mounting faces not seated: gap {d:.4f} mm, overlap {v:.4f} mm³')
    # ---- pilot coaxiality ----
    p = spec['pilot']
    ka = cylinder_axes(kit[p['kit_part']], p['kit_radius'], axis)
    ea = cylinder_axes(engine[p['engine_part']], p['engine_radius'], axis)
    if not ka or not ea:
        fails.append(f'pilot cylinders not found (kit {len(ka)}, engine {len(ea)})')
        rec['pilot'] = dict(ok=False)
    else:
        off = min(_dist_in_plane(a[0], b[0], axis) for a in ka for b in ea)
        rec['pilot'] = dict(kit_radius=round(ka[0][1], 4), engine_radius=round(ea[0][1], 4), axis_offset_mm=round(off, 5),
                            diametral_clearance_mm=round(2 * (ea[0][1] - ka[0][1]), 4), ok=off <= AXIS_TOL and ea[0][1] >= ka[0][1])
        if not rec['pilot']['ok']:
            fails.append(f'pilot not coaxial or interfering: offset {off:.4f} mm, radii kit {ka[0][1]:.3f} engine {ea[0][1]:.3f}')
    # ---- bolt pattern ----
    b = spec['bolts']
    bb = b.get('band')
    kh = cylinder_axes(kit[b['kit_part']], b['kit_radius'], axis, band=bb)
    eh = cylinder_axes(engine[b['engine_part']], b['engine_radius'], axis, band=b.get('engine_band'))
    centre = ea[0][0] if ea else spec['origin']
    matches = []
    for e_pt, _ in sorted(eh, key=lambda h: _clock(h[0], centre, axis, down)):
        best = min((_dist_in_plane(e_pt, k_pt, axis) for k_pt, _ in kh), default=float('inf'))
        matches.append(dict(engine_hole_clock_deg=round(_clock(e_pt, centre, axis, down), 2), nearest_kit_hole_mm=round(best, 4), ok=best <= BOLT_TOL))
    kit_clocks = sorted(round(_clock(k_pt, centre, axis, down), 2) for k_pt, _ in kh)
    ok_b = len(kh) == len(eh) == b['count'] and all(m['ok'] for m in matches)
    rec['bolts'] = dict(engine_holes=len(eh), kit_holes=len(kh), kit_hole_clocks_deg=kit_clocks, matches=matches, ok=ok_b,
                        clock_reference='deg about the mounting axis from the DOWN direction (gravity) toward axis × down')
    if not eh:
        fails.append(f"engine part {b['engine_part']} has no Ø{2 * b['engine_radius']:g} holes parallel to the axis at the interface (kit has {len(kh)})")
    elif not ok_b:
        need = [m['engine_hole_clock_deg'] for m in matches]
        fails.append(f'bolt patterns do not match with the feet down: engine holes at {need} deg, kit holes at {kit_clocks} deg '
                     f'(from down) — change the kit bolt clock by {_clock_delta(need, kit_clocks, b["count"]):.2f} deg')
    # ---- feet orientation and support gap ----
    fdef = spec['feet']
    feet_faces = []
    for name, index in fdef['faces']:                      # foot pad faces identified in the kit's own frame before placement
        f = kit[name].Faces[index]
        n = f.normalAt(*f.Surface.parameter(f.CenterOfMass))
        feet_faces.append((name, f, _unit([n.x, n.y, n.z])))
    if not feet_faces:
        fails.append('kit declares no foot pad faces')
        rec['feet'] = dict(ok=False)
    else:
        worst = max(math.degrees(math.acos(max(-1.0, min(1.0, _dot(nn, down))))) for _, _, nn in feet_faces)
        low = max(_dot([vv.X, vv.Y, vv.Z], down) for _, f, _ in feet_faces for vv in f.Vertexes)       # lowest = furthest along down
        support = engine[fdef['support_part']]
        top = min(_dot([vv.X, vv.Y, vv.Z], down) for vv in support.Vertexes)                      # support top = least along down
        gap = top - low
        rec['feet'] = dict(faces=len(feet_faces), worst_normal_angle_deg=round(worst, 4), gap_to_support_mm=round(gap, 3),
                           ok=worst <= ANGLE_TOL and gap >= -1e-6)
        if worst > ANGLE_TOL:
            fails.append(f'case feet are {worst:.2f} deg off gravity')
        if gap < -1e-6:
            fails.append(f'case feet penetrate the support by {-gap:.3f} mm')
        elif gap > 1e-3:
            msg = f'case feet stand {gap:.3f} mm above the support ({fdef["support_part"]})'
            # require_contact: the design declares that its feet stand on this support, so any gap is a failure, not a note.
            (fails if fdef.get('require_contact') else findings).append(msg + ('' if fdef.get('require_contact') else ': add a riser or pedestal to carry the case'))
        rec['feet']['require_contact'] = bool(fdef.get('require_contact'))
        rec['feet']['ok'] = rec['feet']['ok'] and (not fdef.get('require_contact') or abs(gap) <= 1e-3)
    # ---- shaft tip and coaxiality ----
    sh = spec['shaft']
    shaft = kit[sh['kit_part']]
    tip = max(_dot([vv.X, vv.Y, vv.Z], axis) for vv in shaft.Vertexes)
    sa = cylinder_axes(shaft, sh['radius'], axis)
    ba = cylinder_axes(engine[sh['engine_bore_part']], sh['engine_bore_radius'], axis, rtol=sh.get('bore_rtol', 0.3))
    off_s = min((_dist_in_plane(a[0], c[0], axis) for a in sa for c in ba), default=float('inf'))
    rec['shaft'] = dict(tip_along_axis_mm=round(tip, 4), engine_expected_mm=sh['engine_tip_along_axis'], tip_error_mm=round(tip - sh['engine_tip_along_axis'], 4),
                        axis_offset_to_bore_mm=round(off_s, 5), bore_radius_mm=round(ba[0][1], 4) if ba else None,
                        ok=abs(tip - sh['engine_tip_along_axis']) <= TIP_TOL and off_s <= AXIS_TOL)
    if not rec['shaft']['ok']:
        fails.append(f"stub tip error {tip - sh['engine_tip_along_axis']:.3f} mm or axis offset {off_s:.4f} mm to the coupler bore")
    # ---- key clocking (rotor free to turn) ----
    k = spec['key']
    key, coupler = kit[k['kit_part']], engine[k['engine_part']]
    pivot = _vec(sa[0][0]) if sa else _vec(spec['origin'])
    clear = []
    for a in range(360):
        t = key.copy()
        t.rotate(pivot, _vec(axis), float(a))
        if t.common(coupler).Volume <= VOL_TOL:
            clear.append(a)
    band = _largest_band(clear)
    turn = float(band[len(band) // 2] % 360) if band else 0.0
    rec['key'] = dict(clear_band_deg=[band[0] % 360, band[-1] % 360] if band else None, rotor_turn_deg=turn, ok=bool(band))
    if not band:
        fails.append('no rotor clocking seats the key in the coupler keyway')
    for name in sorted(n for n in kit if n in k['rotor_parts'] or any(n.startswith(pfx) for pfx in k.get('rotor_prefixes', ()))):
        kit[name].rotate(pivot, _vec(axis), turn)
    # ---- clashes ----
    clashes = []
    for kn in sorted(kit):
        kb = kit[kn].BoundBox
        for en in sorted(engine):
            if en in spec.get('placeholders', ()):
                continue
            if not kb.intersect(engine[en].BoundBox):
                continue
            vol = kit[kn].common(engine[en]).Volume
            if vol > VOL_TOL:
                clashes.append(dict(kit=kn, engine=en, mm3=round(vol, 3)))
    rec['clashes'] = clashes
    if clashes:
        fails.append(f'{len(clashes)} clash(es): ' + ', '.join(f"{c['kit']}×{c['engine']} {c['mm3']} mm³" for c in clashes[:5]))
    rec['ok'] = not fails
    return rec, fails, findings


def _clock(p, centre, axis, down):
    """Angle (deg, 0..360) of point p about the axis through `centre`, measured from `down` toward axis × down."""
    d = [p[i] - centre[i] for i in range(3)]
    side = _cross(axis, down)
    return math.degrees(math.atan2(_dot(d, side), _dot(d, down))) % 360.0


def _clock_delta(need, have, count):
    """Smallest rotation (deg) taking the kit pattern onto the engine pattern, modulo the pattern pitch."""
    if not need or not have:
        return float('nan')
    pitch = 360.0 / count
    delta = (need[0] - have[0]) % pitch
    return delta if delta <= pitch / 2 else delta - pitch


def _largest_band(values):
    """Largest contiguous run in a set of integer degrees on a circle (returns the run, possibly wrapping past 360)."""
    s = set(values)
    if len(s) == 360:
        return list(range(360))
    best, cur = [], []
    for a in range(720):
        if a % 360 in s:
            cur.append(a)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    return best

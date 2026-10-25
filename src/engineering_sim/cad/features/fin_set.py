def apply(shape, p, ctx):
    """Fuse `count` trapezoidal fins radially around the tail of the body."""
    import Part
    from FreeCAD import Vector
    if shape is None:
        return shape
    pub = ctx.get('published', {})
    r = float(pub['body_dia']) / 2.0
    n = int(p['count'])
    rc = float(p['root_chord'])
    tc = float(p['tip_chord'])
    span = float(p['span'])
    sweep = float(p['sweep'])
    t = float(p['thickness'])
    z = float(p.get('z_pos', 0.0))
    x_in = r - 1.0
    x_out = r + span
    pts = [Vector(x_in, -t / 2.0, z), Vector(x_in, -t / 2.0, z + rc), Vector(x_out, -t / 2.0, z + rc - sweep), Vector(x_out, -t / 2.0, z + rc - sweep - tc)]
    try:
        fin0 = Part.Face(Part.makePolygon(pts + [pts[0]])).extrude(Vector(0, t, 0))
    except Exception:
        return shape
    for k in range(n):
        f = fin0.copy()
        f.rotate(Vector(0, 0, 0), Vector(0, 0, 1), k * 360.0 / n)
        try:
            shape = shape.fuse(f)
        except Exception:
            pass
    return shape.removeSplitter()

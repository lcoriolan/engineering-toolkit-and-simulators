def apply(shape, p, ctx):
    """Fuse a central hub + N radial struts spanning to the duct inner wall."""
    import Part
    from FreeCAD import Vector
    if shape is None:
        return shape
    pub = ctx.get('published', {})
    duct_id = float(pub['duct_id'])
    ri = duct_id / 2.0
    hub_r = float(p['hub_dia']) / 2.0
    n = int(p['strut_count'])
    w = float(p['strut_width'])
    depth = float(p['depth'])
    z = float(p['z_pos'])
    hub = Part.makeCylinder(hub_r, depth, Vector(0, 0, z)).cut(Part.makeCylinder(min(3.0, hub_r * 0.5), depth + 2.0, Vector(0, 0, z - 1.0)))
    out = shape.fuse(hub)
    span = max(ri - hub_r + 1.0, 1.0)
    for k in range(n):
        strut = Part.makeBox(span, w, depth, Vector(hub_r - 0.5, -w / 2.0, z))
        strut.rotate(Vector(0, 0, 0), Vector(0, 0, 1), k * 360.0 / n)
        try:
            out = out.fuse(strut)
        except Exception:
            pass
    return out.removeSplitter()

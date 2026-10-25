def apply(shape, p, ctx):
    """Fuse a concentric inner duct tube + a forward bulkhead ring onto the body."""
    import Part
    from FreeCAD import Vector
    if shape is None:
        return shape
    pub = ctx.get('published', {})
    body_dia = float(pub['body_dia'])
    body_wall = float(pub['body_wall'])
    skin_ri = body_dia / 2.0 - body_wall
    od = float(p['duct_od'])
    wall = float(p['wall'])
    L = float(p['length'])
    ro = od / 2.0
    ri = max(ro - wall, 1.0)
    pub['duct_id'] = 2.0 * ri
    tube = Part.makeCylinder(ro, L).cut(Part.makeCylinder(ri, L + 2.0, Vector(0, 0, -1.0)))
    parts = shape.fuse(tube)
    skin_ro = body_dia / 2.0
    ties = int(p.get('ties', 0))
    tie_z = float(p['tie_z'])
    tw = float(p['tie_width'])
    r_in = ro - 1.0
    span = skin_ro - r_in
    for k in range(max(ties, 0)):
        ang = 360.0 * k / max(ties, 1)
        web = Part.makeBox(span, tw, tw, Vector(r_in, -tw / 2.0, tie_z - tw / 2.0))
        web.rotate(Vector(0, 0, 0), Vector(0, 0, 1), ang)
        parts = parts.fuse(web)
    return parts.removeSplitter()

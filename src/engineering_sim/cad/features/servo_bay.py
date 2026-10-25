def apply(shape, p, ctx):
    """Fuse `count` servo-body blocks into the annulus at z_pos, clocked to the fins."""
    import Part
    from FreeCAD import Vector
    if shape is None:
        return shape
    pub = ctx.get('published', {})
    body_dia = float(pub['body_dia'])
    body_wall = float(pub['body_wall'])
    duct_od = float(pub['duct_od'])
    skin_ri = body_dia / 2.0 - body_wall
    duct_ro = duct_od / 2.0
    n = int(p['count'])
    rad = float(p['servo_radial'])
    tan = float(p['servo_tangential'])
    axi = float(p['servo_axial'])
    z = float(p['z_pos'])
    clock = float(p['clock_deg'])
    gap = float(p['gap'])
    x0 = duct_ro + gap
    for k in range(n):
        ang = clock + k * 360.0 / n
        try:
            block = Part.makeBox(rad, tan, axi, Vector(x0, -tan / 2.0, z))
            block.rotate(Vector(0, 0, 0), Vector(0, 0, 1), ang)
            shape = shape.fuse(block)
        except Exception:
            pass
    return shape.removeSplitter()

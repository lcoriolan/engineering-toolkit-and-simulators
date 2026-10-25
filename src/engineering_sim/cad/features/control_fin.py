def apply(shape, p, ctx):
    """Build N pivoting fins (each with a radial hinge stub), deflected about the 25% hinge line."""
    import Part
    from FreeCAD import Vector
    if shape is None:
        return shape
    pub = ctx.get('published', {})
    R = float(pub['body_dia']) / 2.0
    duct_od = float(pub['duct_od'])
    stub_inner = duct_od / 2.0 + 3.0
    n = int(p['count'])
    rc = float(p['root_chord'])
    tc = float(p['tip_chord'])
    span = float(p['span'])
    sweep = float(p['sweep'])
    t = float(p['thickness'])
    z = float(p['z_pos'])
    hf = float(p['hinge_frac'])
    defl = float(p['deflection'])
    clock = float(p['clock_deg'])
    stub_r = float(p['stub_dia']) / 2.0
    x_in = R - 1.0
    x_out = R + span
    z_hinge = z + rc * (1.0 - hf)
    for k in range(n):
        ang = clock + k * 360.0 / n
        try:
            pts = [Vector(x_in, -t / 2.0, z), Vector(x_in, -t / 2.0, z + rc), Vector(x_out, -t / 2.0, z + rc - sweep), Vector(x_out, -t / 2.0, z + rc - sweep - tc)]
            fin = Part.Face(Part.makePolygon(pts + [pts[0]])).extrude(Vector(0, t, 0))
            stub = Part.makeCylinder(stub_r, x_in + 3.0 - stub_inner)
            stub.rotate(Vector(0, 0, 0), Vector(0, 1, 0), 90)
            stub.translate(Vector(stub_inner, 0, z_hinge))
            fin = fin.fuse(stub)
            if abs(defl) > 1e-06:
                fin.rotate(Vector(0, 0, z_hinge), Vector(1, 0, 0), defl)
            fin.rotate(Vector(0, 0, 0), Vector(0, 0, 1), ang)
            shape = shape.fuse(fin)
        except Exception:
            pass
    return shape.removeSplitter()

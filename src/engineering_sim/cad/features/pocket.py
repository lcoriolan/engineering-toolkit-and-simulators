def apply(shape, p, ctx):
    """Cut a rectangular (optionally round-cornered) pocket into the base top face."""
    import Part
    from FreeCAD import Vector
    if shape is None:
        return shape
    pub = ctx.get('published', {})
    L = float(pub['base_len'])
    W = float(pub['base_wid'])
    t = float(pub['base_thickness'])
    flange_t = float(pub.get('flange_thickness', 0.0))
    depth = min(float(p['depth']), t - 0.5)
    m = float(p['margin'])
    x0 = max(m, flange_t + m)
    x1 = L - m
    y0, y1 = (m, W - m)
    if x1 <= x0 or y1 <= y0 or depth <= 0:
        return shape
    cutter = Part.makeBox(x1 - x0, y1 - y0, depth + 1.0, Vector(x0, y0, t - depth))
    r = float(p.get('corner_radius', 0))
    if r > 1e-06:
        try:
            vert = [e for e in cutter.Edges if abs(e.BoundBox.ZLength - (depth + 1.0)) < 1e-06]
            cutter = cutter.makeFillet(min(r, (x1 - x0) / 2 - 0.1, (y1 - y0) / 2 - 0.1), vert)
        except Exception:
            pass
    return shape.cut(cutter)

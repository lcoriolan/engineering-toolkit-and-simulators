def _stations(lo, hi, n):
    if n <= 1:
        return [(lo + hi) / 2.0]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]

def apply(shape, p, ctx):
    """Cut `count` X-running obround slots through the base plate."""
    import Part
    from FreeCAD import Vector
    if shape is None:
        return shape
    pub = ctx.get('published', {})
    L = float(pub['base_len'])
    W = float(pub['base_wid'])
    t = float(pub['base_thickness'])
    flange_t = float(pub.get('flange_thickness', 0.0))
    w = float(p['width'])
    r = w / 2.0
    length = float(p['length'])
    m = float(p['margin'])
    x_start = max(m + r, flange_t + m + r)
    x_end = L - m - r
    run = max(0.0, x_end - x_start)
    ys = _stations(m + r, W - m - r, int(p['count']))
    for y in ys:
        try:
            body = Part.makeBox(run if run > 0 else 0.001, w, t + 2.0, Vector(x_start, y - r, -1.0))
            c1 = Part.makeCylinder(r, t + 2.0, Vector(x_start, y, -1.0))
            c2 = Part.makeCylinder(r, t + 2.0, Vector(x_start + run, y, -1.0))
            slot = body.fuse(c1).fuse(c2)
            shape = shape.cut(slot)
        except Exception:
            pass
    return shape

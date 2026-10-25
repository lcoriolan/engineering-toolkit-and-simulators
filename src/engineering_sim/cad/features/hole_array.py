def _grid(lo, hi, n):
    """n evenly spaced coordinates in [lo, hi] (single point -> midpoint)."""
    if n <= 1:
        return [(lo + hi) / 2.0]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]

def apply(shape, p, ctx):
    """Subtract a grid of through-holes from the base plate."""
    import Part
    from FreeCAD import Vector
    if shape is None:
        return shape
    pub = ctx.get('published', {})
    L = float(pub['base_len'])
    W = float(pub['base_wid'])
    t = float(pub['base_thickness'])
    flange_t = float(pub.get('flange_thickness', 0.0))
    dia = float(p['dia'])
    r = dia / 2.0
    m = float(p['margin'])
    xs = _grid(m, L - m, int(p['cols']))
    ys = _grid(m, W - m, int(p['rows']))
    for x in xs:
        for y in ys:
            if x < flange_t + r:
                continue
            cyl = Part.makeCylinder(r, t + 2.0, Vector(x, y, -1.0))
            shape = shape.cut(cyl)
    return shape

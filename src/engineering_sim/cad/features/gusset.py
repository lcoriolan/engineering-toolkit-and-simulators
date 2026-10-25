def _stations(lo, hi, n):
    """n evenly spaced start positions in [lo, hi] (single -> midpoint)."""
    if n <= 1:
        return [(lo + hi) / 2.0]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]

def apply(shape, p, ctx):
    """Fuse triangular corner ribs onto the base/flange seam."""
    import Part
    from FreeCAD import Vector
    if shape is None:
        return shape
    pub = ctx.get('published', {})
    W = float(pub['base_wid'])
    base_t = float(pub['base_thickness'])
    flange_t = float(pub['flange_thickness'])
    flange_h = float(pub['flange_height'])
    base_len = float(pub['base_len'])
    ry = float(p['thickness'])
    size = min(float(p['size']), (base_len - flange_t) * 0.95, flange_h * 0.95)
    if size <= 0:
        return shape
    n = int(p['count'])
    ys = _stations(0.0, max(0.0, W - ry), n)
    for y in ys:
        p1 = Vector(flange_t, y, base_t)
        p2 = Vector(flange_t + size, y, base_t)
        p3 = Vector(flange_t, y, base_t + size)
        try:
            face = Part.Face(Part.makePolygon([p1, p2, p3, p1]))
            rib = face.extrude(Vector(0, ry, 0))
            shape = shape.fuse(rib)
        except Exception:
            pass
    return shape.removeSplitter()

def _inner_edges(shape, x_at, z_at):
    """Edges running along Y at (x==x_at, z==z_at) - the concave base/flange seam."""
    picks = []
    for e in shape.Edges:
        bb = e.BoundBox
        thin_x = abs(bb.XMax - bb.XMin) < 0.001 and abs(bb.XMin - x_at) < 0.01
        thin_z = abs(bb.ZMax - bb.ZMin) < 0.001 and abs(bb.ZMin - z_at) < 0.01
        if thin_x and thin_z and (bb.YMax - bb.YMin > 1e-06):
            picks.append(e)
    return picks

def apply(shape, p, ctx):
    """Fillet the inner seam edges by radius. No-op if radius ~0 or no shape yet."""
    import Part
    r = float(p.get('radius', 0))
    if shape is None or r <= 1e-06:
        return shape
    pub = ctx.get('published', {})
    base_t = float(pub['base_thickness'])
    flange_t = float(pub['flange_thickness'])
    try:
        edges = _inner_edges(shape, x_at=flange_t, z_at=base_t)
        if edges:
            fh = float(pub['flange_height'])
            bl = float(pub['base_len'])
            r = min(r, fh * 0.9, (bl - flange_t) * 0.9)
            if r > 1e-06:
                return shape.makeFillet(r, edges)
    except Exception:
        pass
    return shape

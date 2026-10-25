def apply(shape, p, ctx):
    """Chamfer the topmost horizontal edges by `size`. No-op if size ~0."""
    import Part
    size = float(p.get('size', 0))
    if shape is None or size <= 1e-06:
        return shape
    zmax = shape.BoundBox.ZMax
    top_edges = [e for e in shape.Edges if abs(e.BoundBox.ZMax - zmax) < 0.001 and abs(e.BoundBox.ZLength) < 0.001]
    if not top_edges:
        return shape
    try:
        return shape.makeChamfer(size, top_edges)
    except Exception:
        return shape

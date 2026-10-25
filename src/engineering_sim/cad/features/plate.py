def apply(shape, p, ctx):
    """Create the plate (base). shape is None for a base feature."""
    import Part
    box = Part.makeBox(float(p['length']), float(p['width']), float(p['thickness']))
    return box if shape is None else shape.fuse(box)

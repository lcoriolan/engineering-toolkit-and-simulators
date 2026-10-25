def apply(shape, p, ctx):
    """Create the base plate. shape is None for the base feature; we return the box."""
    import Part
    box = Part.makeBox(float(p['length']), float(p['width']), float(p['thickness']))
    return box if shape is None else shape.fuse(box)

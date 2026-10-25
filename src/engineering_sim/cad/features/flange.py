def apply(shape, p, ctx):
    """Fuse an upright wall onto the base at x=0. Width/base thickness come from published globals
    (fallback to the current shape's bounding box if the pad didn't publish them)."""
    import Part
    pub = ctx.get('published', {})
    base_wid = float(pub.get('base_wid', shape.BoundBox.YLength if shape is not None else 60.0))
    base_t = float(pub['base_thickness'])
    t = float(p['thickness'])
    h = float(p['height'])
    wall = Part.makeBox(t, base_wid, base_t + h)
    return wall if shape is None else shape.fuse(wall).removeSplitter()

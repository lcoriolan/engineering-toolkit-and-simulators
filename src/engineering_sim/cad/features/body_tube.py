def apply(shape, p, ctx):
    """Create the cylindrical body on +Z. Hollow it if wall>0 (leaving a solid tail cap out)."""
    import Part
    from FreeCAD import Vector
    d = float(p['diameter'])
    r = d / 2.0
    L = float(p['length'])
    wall = float(p.get('wall', 0.0))
    body = Part.makeCylinder(r, L)
    if wall > 1e-06 and wall < r:
        bore = Part.makeCylinder(r - wall, L + 2.0, Vector(0, 0, -1.0))
        body = body.cut(bore)
    return body if shape is None else shape.fuse(body)

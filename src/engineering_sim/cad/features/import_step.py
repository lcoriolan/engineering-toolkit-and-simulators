def apply(shape, p, ctx):
    """Read the STEP file named by params['path'] and return its shape (optionally scaled)."""
    import Part
    from FreeCAD import Matrix
    path = p.get('path')
    if not path:
        raise ValueError("import_step requires a 'path' param pointing to a .step/.stp file")
    s = Part.Shape()
    s.read(path)
    scale = float(p['scale'])
    if abs(scale - 1.0) > 1e-09:
        m = Matrix()
        m.scale(scale, scale, scale)
        s = s.transformGeometry(m)
    return s if shape is None else shape.fuse(s)

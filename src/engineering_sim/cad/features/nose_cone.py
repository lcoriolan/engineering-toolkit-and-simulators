def apply(shape, p, ctx):
    """Fuse a cone or tangent-ogive nose onto the body top. Ogive falls back to cone on failure."""
    import Part
    from FreeCAD import Vector
    pub = ctx.get('published', {})
    r = float(pub['body_dia']) / 2.0
    z0 = float(pub['body_len'])
    L = float(p['length'])
    tip_r = float(p.get('tip_dia', 0.0)) / 2.0
    use_ogive = int(p['ogive']) == 1
    nose = None
    if use_ogive:
        try:
            base = Vector(r, 0, z0)
            tip = Vector(max(tip_r, 0.001), 0, z0 + L)
            mx = (r + tip.x) / 2.0 + 0.12 * L
            mid = Vector(mx, 0, z0 + L * 0.5)
            arc = Part.Arc(base, mid, tip).toShape()
            axis_top = Vector(0, 0, z0 + L)
            axis_bot = Vector(0, 0, z0)
            wire = Part.Wire([arc, Part.makeLine(tip, axis_top), Part.makeLine(axis_top, axis_bot), Part.makeLine(axis_bot, base)])
            face = Part.Face(wire)
            nose = face.revolve(Vector(0, 0, z0), Vector(0, 0, 1), 360)
        except Exception:
            nose = None
    if nose is None:
        nose = Part.makeCone(r, max(tip_r, 0.0), L, Vector(0, 0, z0))
    return nose if shape is None else shape.fuse(nose)

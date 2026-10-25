def apply(shape, p, ctx):
    """Fuse a blunt clear dome (scaled hemisphere) + a camera-mount ring onto the body front."""
    import Part
    from FreeCAD import Vector, Matrix
    pub = ctx.get('published', {})
    R = float(pub['body_dia']) / 2.0
    z0 = float(pub['body_len'])
    L = float(p['length'])
    cam = float(p.get('cam_bore', 0.0))
    import math
    n = 28
    prof = []
    for i in range(n + 1):
        t = math.pi / 2.0 * (i / float(n))
        r = R * math.cos(t)
        z = z0 + L * math.sin(t)
        prof.append(Vector(max(r, 0.001), 0, z))
    try:
        spline = Part.BSplineCurve()
        spline.interpolate(prof)
        wire = Part.Wire([spline.toShape(), Part.makeLine(prof[-1], Vector(0, 0, z0)), Part.makeLine(Vector(0, 0, z0), prof[0])])
        dome = Part.Face(wire).revolve(Vector(0, 0, z0), Vector(0, 0, 1), 360)
    except Exception:
        try:
            wire = Part.makePolygon(prof + [Vector(0, 0, z0), prof[0]])
            dome = Part.Face(wire).revolve(Vector(0, 0, z0), Vector(0, 0, 1), 360)
        except Exception:
            dome = Part.makeSphere(R, Vector(0, 0, z0)).cut(Part.makeBox(4 * R, 4 * R, R + 2, Vector(-2 * R, -2 * R, z0 - R - 2)))
    out = dome if shape is None else shape.fuse(dome)
    if cam > 1e-06 and cam < 2 * R - 4:
        try:
            ring = Part.makeCylinder(cam / 2.0 + 3.0, 6.0, Vector(0, 0, z0 - 3.0)).cut(Part.makeCylinder(cam / 2.0, 8.0, Vector(0, 0, z0 - 4.0)))
            out = out.fuse(ring)
        except Exception:
            pass
    return out.removeSplitter()

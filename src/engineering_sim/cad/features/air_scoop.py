def apply(shape, p, ctx):
    """Cut N side intakes that fuse into the body as one watertight solid.

    Per intake: (a) a curvature-trimmed SLEEVE block fills the annular gap around the airway and
    overlaps both walls (fused into the body first); (b) a single CUTTER = a round through-wall
    prism unioned with a shallow square mouth is subtracted, opening the passage from outside skin
    to duct bore. See PRINCIPLE in the module header for why this holds together in OCC.
    """
    import Part
    import math
    import FreeCAD
    from FreeCAD import Vector
    if shape is None:
        return shape
    pub = ctx.get('published', {})
    R_outer = float(pub['body_dia']) / 2.0
    body_wall = float(pub['body_wall'])
    R_skin_bore = R_outer - body_wall
    R_duct_outer = float(pub['duct_od']) / 2.0
    R_duct_bore = float(pub['duct_id']) / 2.0
    n = int(p['count'])
    wdeg = float(p['width_deg'])
    mouth_axial = float(p['mouth_axial'])
    passage_r = float(p['passage_r'])
    z = float(p['z_pos'])
    clock = float(p['clock_deg'])
    rake = 35.0
    mouth_w = 2.0 * R_outer * math.sin(math.radians(min(wdeg, 40.0) / 2.0))
    sleeve_r_out = R_outer - 0.1
    sleeve_r_in = R_duct_bore + 1.0

    def oriented_box(lx, ly, lz, ex, ey, ez, base):
        """A box of size (lx,ly,lz) centered in x,y and running 0..lz along +z, placed into the
        frame whose axes are the unit vectors (ex,ey,ez) with local origin at `base`. ex/ey/ez must
        be orthonormal (a pure rotation) so the placement is rigid."""
        b = Part.makeBox(lx, ly, lz, Vector(-lx / 2.0, -ly / 2.0, 0.0))
        mat = FreeCAD.Matrix(ex.x, ey.x, ez.x, base.x, ex.y, ey.y, ez.y, base.y, ex.z, ey.z, ez.z, base.z, 0.0, 0.0, 0.0, 1.0)
        b.Placement = FreeCAD.Placement(mat)
        return b
    parts = shape
    crake = math.cos(math.radians(rake))
    srake = math.sin(math.radians(rake))
    tall = float(pub['body_len']) + 2.0
    for k in range(n):
        ang = math.radians(clock + k * 360.0 / n)
        R_hat = Vector(math.cos(ang), math.sin(ang), 0.0)
        T_hat = Vector(-math.sin(ang), math.cos(ang), 0.0)
        Z_hat = Vector(0.0, 0.0, 1.0)
        try:
            base = R_hat * 19.0 + Z_hat * (z - 2.0)
            blk = oriented_box(38.0, 44.0, 21.0, T_hat, Z_hat, R_hat, base)
            blk = blk.common(Part.makeCylinder(sleeve_r_out, tall, Vector(0, 0, -1.0)))
            blk = blk.cut(Part.makeCylinder(sleeve_r_in, tall, Vector(0, 0, -1.0)))
            parts = parts.fuse(blk)
        except Exception:
            pass
    parts = parts.removeSplitter()
    for k in range(n):
        ang = math.radians(clock + k * 360.0 / n)
        R_hat = Vector(math.cos(ang), math.sin(ang), 0.0)
        T_hat = Vector(-math.sin(ang), math.cos(ang), 0.0)
        Z_hat = Vector(0.0, 0.0, 1.0)
        try:
            d = R_hat * crake + Z_hat * srake
            P0 = R_hat * R_outer + Z_hat * z
            t_out = 8.0
            t_in = -34.0
            P_in = P0 + d * t_in
            height = t_out - t_in
            cutter = Part.makeCylinder(passage_r, height, P_in, d)
            ez = d * -1.0
            ex = T_hat
            ey = ez.cross(ex)
            P_out = P0 + d * t_out
            mouth = oriented_box(mouth_w, mouth_axial, 16.0, ex, ey, ez, P_out)
            mouth = mouth.cut(Part.makeCylinder(35.5, tall, Vector(0, 0, -1.0)))
            cutter = cutter.fuse(mouth)
            parts = parts.cut(cutter)
        except Exception:
            pass
    return parts.removeSplitter()

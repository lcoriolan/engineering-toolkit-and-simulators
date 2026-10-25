import math
from typing import Dict, Any, Optional

def _unit_vector(v):
    """Return unit vector of v (2‑tuple)."""
    length = math.hypot(v[0], v[1])
    if length == 0:
        return (0.0, 0.0)
    return (v[0] / length, v[1] / length)

def _angle_between(u, v):
    """Angle (radians) between two vectors in the plane, in [0, pi]."""
    dot = u[0] * v[0] + u[1] * v[1]
    return math.acos(max(-1.0, min(1.0, dot)))

def _fillet_points(p0, u1, u2, r):
    """
    Compute tangent points P1 on line p0 + t*u1 and P2 on p0 + t*u2
    for a circular fillet of radius r tangent to both lines.
    Returns (P1, P2) as (x,y) tuples.
    """
    theta = _angle_between(u1, u2)
    d = r * (1.0 / math.tan(theta / 2.0)) if theta > 1e-09 else 0.0
    P1 = (p0[0] + d * u1[0], p0[1] + d * u1[1])
    P2 = (p0[0] + d * u2[0], p0[1] + d * u2[1])
    return (P1, P2)

def _arc_center_and_angles(P1, P2, u1, u2, r):
    """
    Compute the center of the fillet arc and the start/end angles (in radians)
    for an arc from P1 to P2 with radius r, tangent to lines with directions u1 and u2.
    Returns (center_x, center_y, angle1, angle2) where angles are in the plane's
    coordinate system (x = r, y = z).
    """
    n1 = (u1[1], -u1[0])
    n2 = (-u2[1], u2[0])
    C1 = (P1[0] + r * n1[0], P1[1] + r * n1[1])
    C2 = (P2[0] + r * n2[0], P2[1] + r * n2[1])
    C = ((C1[0] + C2[0]) / 2.0, (C1[1] + C2[1]) / 2.0)
    dx1, dy1 = (P1[0] - C[0], P1[1] - C[1])
    dx2, dy2 = (P2[0] - C[0], P2[1] - C[1])
    ang1 = math.atan2(dy1, dx1)
    ang2 = math.atan2(dy2, dx2)
    diff = ang2 - ang1
    if diff > math.pi:
        ang2 -= 2 * math.pi
    elif diff < -math.pi:
        ang2 += 2 * math.pi
    return (C, ang1, ang2)

def apply(shape, p: Dict[str, Any], ctx: Dict[str, Any]):
    """
    Generate an EDF duct as a single revolved solid.

    Parameters
    ----------
    shape : FreeCAD Part::Shape or None
        Current shape in the design stack (None if BASE).
    p : dict
        Resolved parameter dict (keys match SCHEMA names).
    ctx : dict
        Context containing 'published' dict for cross‑feature globals.

    Returns
    -------
    FreeCAD Part::Shape
        The new duct solid (isValid() == True, one Solid).
    """
    import Part
    from FreeCAD import Vector
    fan_dia = p['fan_dia']
    tip_clearance = p['tip_clearance']
    r_lip = p['r_lip']
    inlet_contraction = p['inlet_contraction']
    inlet_len = p['inlet_len']
    fan_len = p['fan_len']
    nozzle_len = p['nozzle_len']
    fsr = p['nozzle_area_ratio']
    wall = p['wall']
    fan_radius = fan_dia / 2.0 + tip_clearance
    r_highlight = math.sqrt(inlet_contraction) * fan_radius
    r_exit = math.sqrt(fsr) * fan_radius
    if r_highlight <= fan_radius:
        raise ValueError('inlet_contraction must be > 1.0 to provide a lip.')
    if r_exit <= 0:
        raise ValueError('nozzle_area_ratio too small; exit radius would be zero.')
    if wall <= 0:
        raise ValueError('wall thickness must be positive.')
    total_len = inlet_len + fan_len + nozzle_len
    v_inner = (fan_radius - r_highlight, inlet_len)
    len_inner = math.hypot(v_inner[0], v_inner[1])
    if len_inner < 1e-09:
        raise ValueError('inlet_len too small; inner line length zero.')
    u_inner = (v_inner[0] / len_inner, v_inner[1] / len_inner)
    v_outer = (fan_radius + wall - r_highlight, inlet_len)
    len_outer = math.hypot(v_outer[0], v_outer[1])
    if len_outer < 1e-09:
        raise ValueError('inlet_len too small; outer line length zero.')
    u_outer = (v_outer[0] / len_outer, v_outer[1] / len_outer)
    P1, P2 = _fillet_points((r_highlight, 0.0), u_inner, u_outer, r_lip)
    C, ang1, ang2 = _arc_center_and_angles(P1, P2, u_inner, u_outer, r_lip)
    edges = []
    p1_vec = Vector(P1[0], 0.0, P1[1])
    fan_face = Vector(fan_radius, 0.0, inlet_len)
    edges.append(Part.makeLine(p1_vec, fan_face))
    fan_end = Vector(fan_radius, 0.0, inlet_len + fan_len)
    edges.append(Part.makeLine(fan_face, fan_end))
    exit_pt = Vector(r_exit, 0.0, total_len)
    edges.append(Part.makeLine(fan_end, exit_pt))
    outer_exit = Vector(r_exit + wall, 0.0, total_len)
    edges.append(Part.makeLine(exit_pt, outer_exit))
    outer_fan_end = Vector(fan_radius + wall, 0.0, inlet_len + fan_len)
    outer_fan_face = Vector(fan_radius + wall, 0.0, inlet_len)
    edges.append(Part.makeLine(outer_exit, outer_fan_end))
    edges.append(Part.makeLine(outer_fan_end, outer_fan_face))
    p2_vec = Vector(P2[0], 0.0, P2[1])
    edges.append(Part.makeLine(outer_fan_face, p2_vec))
    center_vec = Vector(C[0], 0.0, C[1])
    arc = Part.ArcOfCircle(center_vec, Vector(0, 1, 0), r_lip, ang2, ang1)
    edges.append(arc.toShape())
    wire = Part.Wire(edges)
    if not wire.isClosed():
        raise RuntimeError('Profile wire is not closed. Check geometry construction.')
    face = Part.Face(wire)
    if face.isNull():
        raise RuntimeError('Failed to create face from profile wire.')
    solid = face.revolve(Vector(0, 0, 0), Vector(0, 0, 1), 360.0)
    if solid.isNull():
        raise RuntimeError('Revolve operation produced a null shape.')
    solid = solid.removeSplitter()
    if not solid.isValid():
        raise RuntimeError('Revolved solid is invalid. Check profile for self‑intersection.')
    if len(solid.Solids) != 1:
        raise RuntimeError(f'Expected 1 solid, got {len(solid.Solids)}.')
    ctx['published']['duct_inner_diameter'] = 2.0 * fan_radius
    ctx['published']['overall_length'] = total_len
    return solid

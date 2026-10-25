# DESCRIPTION: Independent joint compatibility, ideal mobility, native bore and finite-body contact checks.
"""Check a connected mechanism without treating an animation as closure proof.

Lengths are millimetres. Body poses and joint frames are homogeneous SE(3)
matrices. A designed joint frame must come from its actual CAD datum: deriving
both frames from the final pose would make the closure check a tautology.
Native functions accept FreeCAD shapes, but the algebra needs only NumPy.
"""

import math
import numpy as np

from .transforms import transform, unit


def skew(vector):
    """Return the cross-product matrix: skew(p) @ v equals p cross v."""
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def residual(actual, expected):
    """Measure positional and angular disagreement between independently built frames."""
    delta = np.linalg.inv(expected) @ actual
    rotation = delta[:3, :3]
    # atan2 keeps tiny errors measurable without acos cancellation near identity.
    sine = np.linalg.norm([rotation[2, 1]-rotation[1, 2], rotation[0, 2]-rotation[2, 0],
                           rotation[1, 0]-rotation[0, 1]])/2
    cosine = (np.trace(rotation)-1)/2
    return dict(position_mm=float(np.linalg.norm(delta[:3, 3])),
                angle_radians=float(math.atan2(sine, cosine)),
                homogeneous_error=float(np.max(np.abs(delta[3]-[0, 0, 0, 1]))),
                # A scaling/reflection matrix can have zero atan2 residual.
                # Check both input rotations too: equal affine distortions in
                # actual/expected must not cancel and masquerade as SE(3).
                rotation_orthogonality_error=float(max(np.max(np.abs(frame[:3, :3].T @ frame[:3, :3]-np.eye(3)))
                                                     for frame in (actual, expected))),
                rotation_determinant_error=float(max(abs(np.linalg.det(frame[:3, :3])-1.) for frame in (actual, expected))))


def joint_residual(parent_pose, child_pose, parent_frame, child_frame, angle, local_axis=(0, 1, 0)):
    """Check all six frame coordinates for one commanded revolute joint.

    Frames align at the declared zero position. Each lives in its owning body's
    deployed coordinates; offset origins are retained, not discarded in a 3x3
    rotation-only calculation. The permitted relative motion is one rotation.
    """
    expected = parent_pose @ parent_frame @ transform(local_axis, [0, 0, 0], angle)
    actual = child_pose @ child_frame
    return residual(actual, expected)


def loop_residual(independent_edge_transforms):
    """Evaluate an actual designed loop of joint/lock transforms against I4.

    A tree has no independent loop. Do not fabricate one by multiplying body
    pose differences: that product telescopes to identity even for a bad design.
    """
    product = np.eye(4)
    for edge in independent_edge_transforms:
        product = product @ np.asarray(edge, dtype=float)
    return residual(product, np.eye(4))


def mobility(owners, joints, fixed_owner='core', locked=False, characteristic_length=100.):
    """Return first-order mobility of the declared ideal rigid-body joint graph.

    Each body has three translations and three rotations. A revolute joint
    equates velocity at its pivot (three rows) and the two transverse angular
    velocities (two rows). A positive angular lock adds the axial row. Scaling
    translations by a characteristic length avoids a units-driven rank result.
    Nonzero infinitesimal mobility does NOT prove a finite folding branch; zero
    locked mobility is an ideal kinematic result, not stiffness/preload testing.
    """
    if fixed_owner not in owners or characteristic_length <= 0:
        raise ValueError('A fixed owner and positive length scale are required')
    index = {name: number for number, name in enumerate(owners)}
    rows = []
    root = np.zeros((6, 6*len(owners)))
    root[:, 6*index[fixed_owner]:6*index[fixed_owner]+6] = np.eye(6)
    rows.extend(root)
    for joint in joints:
        axis = unit(joint['axis'])
        # Pick a stable direction independent of the axis, then span its normal
        # plane. Each row states a real constraint rather than a DOF heuristic.
        seed = np.eye(3)[int(np.argmin(np.abs(axis)))]
        u = unit(np.cross(axis, seed)); v = np.cross(axis, u)
        point = np.asarray(joint['pivot'], dtype=float)/characteristic_length
        angular = np.stack([u, v, axis] if locked else [u, v])
        block = np.zeros((3+len(angular), 6))
        block[:3, :3] = np.eye(3); block[:3, 3:] = -skew(point)
        block[3:, 3:] = angular
        constraint = np.zeros((len(block), 6*len(owners)))
        for name, sign in ((joint['parent'], -1), (joint['child'], 1)):
            at = 6*index[name]
            constraint[:, at:at+6] = sign*block
        rows.extend(constraint)
    matrix = np.asarray(rows)
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = max(matrix.shape)*np.finfo(float).eps*singular[0]*100
    rank = int(np.count_nonzero(singular > tolerance))
    return dict(rank=rank, variables=6*len(owners), nullity=6*len(owners)-rank,
                rank_tolerance=float(tolerance), singular_values=singular.tolist(),
                locked=locked, interpretation='Ideal first-order constraints; no stiffness or finite-motion proof')


def bore_datum(shape, point, axis, radius, position_tolerance=1e-5, angle_tolerance=1e-7):
    """Find native cylindrical faces matching an independently declared pin bore.

    A cylinder's origin can slide along its axis, so compare perpendicular line
    distance and unoriented parallelism. Face count and axial span are reported;
    this does not infer that an isolated cylindrical face is a complete bearing.
    """
    direction = unit(axis); origin = np.asarray(point, dtype=float)
    matches = []
    for number, face in enumerate(shape.Faces):
        surface = face.Surface
        if type(surface).__name__ != 'Cylinder':
            continue
        location = np.array([surface.Center.x, surface.Center.y, surface.Center.z])
        vector = unit([surface.Axis.x, surface.Axis.y, surface.Axis.z])
        line_error = float(np.linalg.norm(np.cross(location-origin, direction)))
        angle_error = float(np.linalg.norm(np.cross(vector, direction)))
        radius_error = abs(surface.Radius-radius)
        # A convex outside cylinder with the same dimensions is not a bore.

        u0, u1, v0, v1 = face.ParameterRange
        u, v = (u0+u1)/2, (v0+v1)/2
        sample = face.valueAt(u, v); normal = face.normalAt(u, v)
        radial = np.array([sample.x, sample.y, sample.z])-location
        radial -= vector*np.dot(radial, vector)
        inward = np.dot(radial, [normal.x, normal.y, normal.z]) < 0
        if inward and line_error <= position_tolerance and angle_error <= angle_tolerance and radius_error <= position_tolerance:
            axial = [float(np.dot(np.array([p.Point.x, p.Point.y, p.Point.z])-origin, direction)) for p in face.Vertexes]
            matches.append(dict(face=number, line_error_mm=line_error, direction_error=angle_error,
                radius_error_mm=radius_error, axial_vertex_range_mm=[min(axial), max(axial)] if axial else None))
    return dict(matched=bool(matches), cylindrical_faces=matches)


def bounds(shape):
    """Use native optimal bounds; trimmed Bezier control boxes can be much larger."""
    b = shape.optimalBoundingBox(False, False)
    return np.array([b.XMin, b.YMin, b.ZMin]), np.array([b.XMax, b.YMax, b.ZMax])


def near(a, b, margin):
    """Conservatively reject far pairs by their world-axis bounding boxes."""
    return bool(np.all(a[0] <= b[1]+margin) and np.all(b[0] <= a[1]+margin))


def remove_regions(shape, regions):
    """Cut declared contact zones sequentially, avoiding overlapping compounds."""
    remaining = shape.copy()
    for region in regions:
        if remaining.isNull():
            break
        remaining = remaining.cut(region)
        if not remaining.isNull() and not remaining.isValid():
            raise ValueError('Invalid geometry after removing a declared contact zone')
    return remaining


def empty_shape(shape):
    """Recognize OCC's empty non-null Compound as well as a null TopoShape.

    Boolean subtraction can leave a valid compound with no geometric contents.
    Asking its distance raises an OCC exception; this does not mean that a
    contact region failed or that a residual body still occupies space.
    """
    return shape.isNull() or not (shape.Vertexes or shape.Edges or shape.Faces or shape.Solids)


def pair_check(a, b, clearance=.2, contact_regions=(), overlap_tolerance=1e-5, distance_tolerance=1e-7):
    """Check finite native bodies; deliberate stops never excuse positive overlap.

    Away from declared stop/seat zones require the full clearance. At a stop,
    permit zero-volume contact only inside explicit small CAD regions. Trimming
    those regions and checking the remaining bodies prevents an intended stop
    from hiding an unrelated close contact elsewhere on the same owner pair.
    """
    distance = float(a.distToShape(b)[0])
    # OCC may return 0.199999999999 for a nominal 0.2 mm gap. This numerical
    # tolerance is one ten-millionth of a millimetre, not a fit allowance.
    if distance+distance_tolerance >= clearance:
        return dict(passed=True, distance_mm=distance, overlap_mm3=0., contact=False)
    intersection = a.common(b)
    if not intersection.isNull() and not intersection.isValid():
        return dict(passed=False, reason='invalid_native_intersection', distance_mm=distance)
    volume = 0. if intersection.isNull() else abs(intersection.Volume)
    if volume > overlap_tolerance:
        return dict(passed=False, reason='positive_overlap', distance_mm=distance, overlap_mm3=volume)
    if not contact_regions:
        return dict(passed=False, reason='insufficient_clearance', distance_mm=distance, overlap_mm3=volume)
    # Check the actual zero-volume intersection too, including residual edges
    # or isolated points, so accidental contact outside the stop cannot pass.
    leftover = remove_regions(intersection, contact_regions) if not intersection.isNull() else intersection
    if not leftover.isNull() and (leftover.Area > 1e-7 or leftover.Length > 1e-7 or len(leftover.Vertexes)):
        return dict(passed=False, reason='contact_outside_declared_stop', distance_mm=distance, overlap_mm3=volume)
    clean_a = remove_regions(a, contact_regions); clean_b = remove_regions(b, contact_regions)
    empty = empty_shape(clean_a) or empty_shape(clean_b)
    separated = None if empty else float(clean_a.distToShape(clean_b)[0])
    passed = empty or separated+distance_tolerance >= clearance
    return dict(passed=passed, reason='declared_contact' if passed else 'clearance_away_from_stop',
                distance_mm=distance, noncontact_distance_mm=separated, overlap_mm3=volume, contact=True)

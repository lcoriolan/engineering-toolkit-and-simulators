import struct
from collections import defaultdict
import numpy as np

def read_stl(path):
    """Read an STL file, auto-detecting binary vs ASCII.

    Binary STL layout (per the CAD authoring format): an 80-byte header, a
    uint32 triangle count, then count records of 12 float32 (normal + 3 vertices)
    plus a uint16 attribute byte count = 50 bytes per triangle.

    Returns ``(vertices, faces)`` where ``vertices`` is an ``(N, 3)`` float64
    array of unique vertex coordinates and ``faces`` is an ``(M, 3)`` int64
    array of vertex indices (0-based).
    """
    with open(path, 'rb') as fh:
        head = fh.read(5)
    if head[:5].lower() == b'solid':
        size = _file_size(path)
        try:
            with open(path, 'rb') as fh:
                fh.read(80)
                count, = struct.unpack('<I', fh.read(4))
            if 0 < count <= size // 50 and 84 + count * 50 <= size:
                return read_stl_binary(path)
        except struct.error:
            pass
        return read_stl_ascii(path)
    return read_stl_binary(path)

def _file_size(path):
    import os
    return os.path.getsize(path)

def read_stl_binary(path):
    """Read a binary STL into a deduplicated ``(vertices, faces)`` pair."""
    with open(path, 'rb') as fh:
        fh.read(80)
        count, = struct.unpack('<I', fh.read(4))
        buf = np.frombuffer(fh.read(count * 50), dtype=np.uint8)
    if buf.size < count * 50:
        raise ValueError(f'truncated binary STL: {path}')
    floats = np.ndarray((count, 12), dtype='<f4', buffer=buf, strides=(50, 4))
    tris = floats[:, 3:12].reshape(count * 3, 3).astype(np.float64)
    uniq, faces = np.unique(tris, axis=0, return_inverse=True)
    faces = faces.reshape(count, 3).astype(np.int64)
    return (uniq, faces)

def read_stl_ascii(path):
    """Read an ASCII STL (``solid``/``facet normal``/``outer loop``/``vertex``)."""
    verts = []
    faces = []
    with open(path, 'r', errors='replace') as fh:
        buf = []
        for line in fh:
            buf.append(line)
            if line.lstrip().lower().startswith('vertex'):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(verts) % 3 == 0:
                    faces.append([len(verts) - 3, len(verts) - 2, len(verts) - 1])
    if not verts:
        raise ValueError(f'no vertices parsed from ASCII STL: {path}')
    v = np.array(verts, dtype=np.float64)
    f = np.array(faces, dtype=np.int64)
    uniq, fmap = np.unique(v, axis=0, return_inverse=True)
    f = fmap[f].astype(np.int64)
    return (uniq, f)

def mesh_volume(vertices, faces):
    """Signed volume of a closed triangle mesh.

    Formula (divergence theorem): for a closed surface with outward normals,
        V = (1/6) * sum_over_faces det(v0, v1, v2)
    where each face contributes the signed volume of the tetrahedron formed by
    that face and the origin. Units are cubic mm for mm inputs.

    The sign is + for outward-oriented meshes; use ``abs`` for a physical volume.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    det = np.einsum('ij,ij->i', v0, np.cross(v1, v2))
    return det.sum() / 6.0

def mesh_centroid(vertices, faces):
    """Centroid of a closed mesh (volume-weighted average of tetra centroids).

    The centroid of the tetrahedron ``(0, v0, v1, v2)`` is ``(v0+v1+v2)/4``;
    summing each tetrahedron's signed volume times its centroid and normalising
    by the total signed volume gives the solid's centroid. Units of mm.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    det = np.einsum('ij,ij->i', v0, np.cross(v1, v2))
    vol = det.sum() / 6.0
    tet_c = (v0 + v1 + v2) / 4.0
    c = np.einsum('i,ij->j', det, tet_c) / (det.sum() or 1.0)
    return c

def mesh_inertia(vertices, faces):
    """Inertia tensor (3x3) of a closed mesh about its centroid, unit density.

    Principle: decompose the solid into tetrahedra ``(0, v0, v1, v2)``, one per
    face. For a tetrahedron with vertices ``{p0..p3}`` (here ``p0 = 0``),
        J = integral of (x (x^T)) dV = (V/20) * [ sum_k p_k p_k^T + S S^T ],
        S = sum_k p_k,
    which follows from integrating barycentric monomials over the simplex
    (integral lambda_i lambda_j = (1/120) for i!=j, (1/60) for i==j, with the
    simplex-volume factor). Each face contributes its *signed* J.

    The inertia tensor about the origin is then ``I0 = tr(J) * I - J``, and the
    parallel-axis theorem gives the tensor about the centroid
        ``Ic = I0 - V * (|c|^2 I - c c^T)``.
    Units: mm^5 (a mass density of 1 g/mm^3 is implied; multiply by density
    for kg·mm²). The returned value is ``(I, volume, centroid)``.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    det = np.einsum('ij,ij->i', v0, np.cross(v1, v2))
    vol = det.sum() / 6.0
    pts = [v0, v1, v2]
    S = v0 + v1 + v2
    pp = sum((np.einsum('ni,nj->nij', p, p) for p in pts))
    ss = np.einsum('ni,nj->nij', S, S)
    j = np.einsum('i,ijk->jk', det, pp + ss) / 120.0
    I0 = np.trace(j) * np.eye(3) - j
    c = mesh_centroid(vertices, faces)
    r2 = float(c @ c)
    I_c = I0 - vol * (r2 * np.eye(3) - np.outer(c, c))
    return (I_c, vol, c)

def slice_plane(vertices, faces, axis, value, stitch_tol=None):
    """Slice a mesh with a plane ``coord[axis] == value`` into polylines.

    Returns ``(closed_loops, open_chains)``, each a list of ``(N, 2)`` float64
    arrays in the *two remaining* source coordinates, in source order. For the
    DroneENG frame (axes: 0 = X_src span, 1 = Y_src thickness, 2 = Z_src chord),
    slicing ``axis=0`` yields (Y_src, Z_src) points.

    Method: intersect every triangle with the plane (linear edge interpolation),
    collect segments, then stitch endpoints within ``stitch_tol`` (default
    0.01 mm — about an order of magnitude above float32 rounding at metre scale)
    into closed loops and open chains. Coplanar faces (all three vertices on the
    plane) are skipped: their boundary edges are contributed by neighbouring
    triangles, avoiding duplicate segments.
    """
    if stitch_tol is None:
        scale = float(np.max(np.abs(vertices))) if vertices.size else 1.0
        stitch_tol = max(0.01, scale * 1e-05)
    keep = [a for a in (0, 1, 2) if a != axis]
    d = vertices[:, axis] - value
    eps = 1e-09 * max(1.0, float(np.max(np.abs(vertices))) if vertices.size else 1.0)
    sign = np.where(np.abs(d) < eps, 0, np.where(d > 0, 1, -1))
    s = sign[faces]
    segments = []
    for tri, sv in zip(vertices[faces], s):
        zeros = np.flatnonzero(sv == 0)
        if zeros.size == 0:
            pairs = [(0, 1), (1, 2), (2, 0)]
            pts = []
            for i, j in pairs:
                if sv[i] * sv[j] < 0:
                    pts.append(_edge_point(tri[i], tri[j], axis, value))
            if len(pts) == 2:
                segments.append((pts[0], pts[1]))
        elif zeros.size == 1:
            z = zeros[0]
            others = [k for k in range(3) if k != z]
            if sv[others[0]] * sv[others[1]] < 0:
                p = _edge_point(tri[others[0]], tri[others[1]], axis, value)
                segments.append((tri[z][keep].copy(), p))
        elif zeros.size == 2:
            z = [k for k in range(3) if sv[k] == 0]
            segments.append((tri[z[0]][keep].copy(), tri[z[1]][keep].copy()))
    return _stitch(segments, stitch_tol)

def _edge_point(a, b, axis, value):
    da = a[axis] - value
    db = b[axis] - value
    t = da / (da - db)
    p = a + t * (b - a)
    keep = [k for k in (0, 1, 2) if k != axis]
    return p[keep]

def _stitch(segments, tol):
    """Stitch loose segments into closed loops and open chains.

    Endpoints closer than ``tol`` are merged (coordinates averaged). Walks the
    resulting graph, treating degree-2 vertices as polyline interiors; vertices
    of degree != 2 terminate a chain. Greedy: for clean slices every vertex has
    degree 2 and loops are recovered exactly.
    """
    canon = {}
    pts_by_key = defaultdict(list)
    adj = defaultdict(set)

    def key(p):
        return (int(round(p[0] / tol)), int(round(p[1] / tol)))
    for p1, p2 in segments:
        k1, k2 = (key(p1), key(p2))
        if k1 == k2:
            continue
        pts_by_key[k1].append(p1)
        pts_by_key[k2].append(p2)
        adj[k1].add(k2)
        adj[k2].add(k1)
    canon = {k: np.mean(np.asarray(ps), axis=0) for k, ps in pts_by_key.items()}
    closed = []
    open_ = []
    used = set()
    for start in list(adj):
        for nb in list(adj[start]):
            e = tuple(sorted((start, nb)))
            if e in used:
                continue
            used.add(e)
            chain = [start, nb]
            prev, cur = (start, nb)
            while True:
                nxts = [k for k in adj[cur] if k != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                e = tuple(sorted((cur, nxt)))
                if e in used:
                    break
                used.add(e)
                chain.append(nxt)
                prev, cur = (cur, nxt)
            pts = np.asarray([canon[k] for k in chain])
            if chain[0] == chain[-1]:
                closed.append(pts)
            else:
                open_.append(pts)
    return (closed, open_)

def polygon_area(pts):
    """Signed area of a closed polygon by the shoelace formula.

        A = (1/2) * sum_i (x_i y_{i+1} - x_{i+1} y_i)

    Positive for counter-clockwise ordering (in a right-handed 2D frame).
    Units are mm² for mm inputs.
    """
    pts = np.asarray(pts, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)

def polyline_length(pts):
    """Total length of a polyline (sum of segment norms), mm."""
    pts = np.asarray(pts, dtype=np.float64)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

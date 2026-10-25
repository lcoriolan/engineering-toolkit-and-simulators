import math
from pathlib import Path
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from . import vlm as vlm_kernel
DEG = math.pi / 180.0

def plain(value):
    """Convert a result tree to strict JSON-serialisable types.

    numpy scalars -> python scalars, arrays -> lists, non-finite floats -> None
    (JSON has no NaN/Infinity).  Rejects only genuinely unserialisable objects.
    """
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and (not math.isfinite(value)):
        return None
    return value

class LatticeSurface:
    """Adapter exposing a mean-surface sampler as kernel-style quads.

    Mirrors the source LatticeSurface API: ``mesh_thin_surface('quad', chord_panels,
    cosine, add_camber)`` returns (points, faces) with faces ordered
    [FL, BL, BR, FR] (front = leading edge, back = trailing edge, left/right =
    spanwise), the exact convention the vendored kernel's ``solve`` consumes.
    ``reverse`` flips span ordering so a mirrored half still winds with +z normals.
    """

    def __init__(self, name, sampler, span_count, reverse=False):
        self.name = name
        self.sampler = sampler
        self.span_count = span_count
        self.reverse = reverse

    def mesh_thin_surface(self, method, chordwise_resolution, chordwise_spacing_function, add_camber):
        n = chordwise_resolution
        spans = (1 - np.cos(np.linspace(0, math.pi, self.span_count + 1))) / 2
        if self.reverse:
            spans = spans[::-1]
        chords = np.asarray(chordwise_spacing_function(0, 1, n + 1))
        points = np.array([self.sampler(s, x) for s in spans for x in chords])
        faces = []
        for i in range(self.span_count):
            for j in range(n):
                a = i * (n + 1) + j
                faces.append([a, a + 1, a + n + 2, a + n + 1])
        return (points, np.asarray(faces))

def _interp_stations(stations):
    """Return (y, le_x, chord, z_le, twist) interpolators of span fraction s in [0,1].

    Stations are ascending-y dicts with keys y (mm), le_x (mm), chord (mm),
    z_chord (mm, leading-edge z of the mean chord line), twist_deg.  All geometry
    is converted to metres here; angles to radians.  Linear interpolation in the
    normalised span coordinate (root=0, tip=1).
    """
    ys = np.array([s['y'] for s in stations], float) / 1000.0
    le = np.array([s['le_x'] for s in stations], float) / 1000.0
    ch = np.array([s['chord'] for s in stations], float) / 1000.0
    zl = np.array([s.get('z_chord', 0.0) or 0.0 for s in stations], float) / 1000.0
    tw = np.array([s.get('twist_deg', 0.0) or 0.0 for s in stations], float) * DEG
    span = ys[-1] - ys[0]
    frac = (ys - ys[0]) / span if span > 0 else np.zeros_like(ys)

    def interp(s):
        return (np.interp(s, frac, ys), np.interp(s, frac, le), np.interp(s, frac, ch), np.interp(s, frac, zl), np.interp(s, frac, tw))
    return interp

def surface_sampler(stations, mirror=1.0):
    """Return f(s, x) -> (x_m, y_m, z_m) for a mean surface from its stations.

    s = span fraction (0 root -> 1 tip), x = chord fraction (0 LE -> 1 TE).
    The chord line tilts with the local twist about the leading edge:
    z = z_le - chord * x * tan(twist).  ``mirror`` = -1 mirrors y (port half).
    """
    interp = _interp_stations(stations)

    def sampler(s, x):
        y, le, ch, zl, tw = interp(s)
        return [le + ch * x, mirror * y, zl - ch * x * math.tan(tw)]
    return sampler

def camber_slope(airfoil_coords, n=201):
    """Mean camber-line slope dzc/dx of an airfoil section, as a callable of x/c.

    ``airfoil_coords`` are [x/c, z/c] pairs ordered TE -> upper -> LE -> lower -> TE
    (the contract airfoil format).  The camber line is the upper/lower mean at each
    x/c, and the slope is its central difference on a uniform chord grid.  For a flat
    or symmetric section this returns 0 everywhere (no camber boundary condition).

    Thin-airfoil principle: a cambered section behaves like a flat plate rotated by
    the camber-line slope, so the boundary condition is n_eff = normalize(n - s t)
    with s = dzc/dx and t the chord axis.  Units: dimensionless (d(z/c)/d(x/c)).
    Assumes the coords are normalised and closed at the TE; omits thickness (the VLM
    mean surface is the camber line, not the skin).
    """
    pts = np.asarray(airfoil_coords, float)
    x, z = (pts[:, 0], pts[:, 1])
    ile = int(np.argmin(x))
    xu, zu = (x[:ile + 1], z[:ile + 1])
    xl, zl = (x[ile:], z[ile:])
    xu, zu = (xu[np.argsort(xu)], zu[np.argsort(xu)])
    xl, zl = (xl[np.argsort(xl)], zl[np.argsort(xl)])
    grid = np.linspace(0.0, 1.0, n)
    zu_i = np.interp(grid, xu, zu)
    zl_i = np.interp(grid, xl, zl)
    zc = 0.5 * (zu_i + zl_i)
    slope = np.gradient(zc, grid)

    def f(x_frac):
        return float(np.interp(x_frac, grid, slope))
    return f

def camber_from_config(config):
    """Return {surface_name: slope-callable} for the airfoil camber in ``config``.

    The wing airfoil comes from ``config['airfoil']['coords']`` (selected by
    configure.py).  The V-tail has no measured airfoil and is left flat (declared
    small effect on CL0/Cm0/trim).  Returns {} when no airfoil is supplied, so all
    synthetic/flat tests keep the flat-plate behaviour.
    """
    af = config.get('airfoil')
    if not af or not af.get('coords'):
        return {}
    return {'wing': camber_slope(af['coords'])}

class Lattice:
    """Concatenated vortex-lattice geometry with per-panel metadata.

    Attributes (all SI metres / radians):
      quads   (N,4,3) FL,BL,BR,FR vertices
      left,right (N,3) bound-vortex endpoints (quarter-chord, panel span edges)
      centre  (N,3) bound-vortex midpoints
      colloc  (N,3) collocation points (3/4 chord, mid-span)
      normals (N,3) unit geometric normals
      chord_axis (N,3) unit chord direction (LE->TE)
      span_axis (N,3) unit bound-vortex direction
      area    (N,)  panel area
      x_frac  (N,)  chord fraction of the panel centre
      y_abs   (N,)  absolute spanwise coordinate of the panel centre
      surface (N,)  int 0=wing, 1=tail
      half    (N,)  int +1 right, -1 left
      strip   (N,)  spanwise strip index (for grouped load distribution)
    """

    def __init__(self):
        self.quads = None

def _append_surface(lat, quads, strip0, surface_id, half, chordwise):
    """Append one surface's quads to the lattice and compute its panel metadata.

    Quads are ordered spanwise-outer / chordwise-inner (i*chordwise + j), so the
    spanwise strip index of panel p is p // chordwise.
    """
    fl, bl, br, fr = [quads[:, i, :] for i in range(4)]
    left = 0.75 * fl + 0.25 * bl
    right = 0.75 * fr + 0.25 * br
    centre = (left + right) / 2
    colloc = (0.25 * (fl + fr) + 0.75 * (bl + br)) / 2
    crs = np.cross(fr - bl, fl - br)
    area2 = np.linalg.norm(crs, axis=1)
    if np.any(area2 <= 0):
        raise ValueError('Degenerate lattice panel')
    normals = crs / area2[:, None]
    span_vec = right - left
    span_axis = span_vec / np.linalg.norm(span_vec, axis=1)[:, None]
    chord_axis = (bl + br) / 2 - (fl + fr) / 2
    chord_axis /= np.linalg.norm(chord_axis, axis=1)[:, None]
    le_mid = (fl + fr) / 2
    te_mid = (bl + br) / 2
    n = len(quads)
    le_x_strip = fl[::chordwise, 0]
    te_x_strip = bl[chordwise - 1::chordwise, 0]
    le_x = np.repeat(le_x_strip, chordwise)
    te_x = np.repeat(te_x_strip, chordwise)
    proj = (colloc[:, 0] - le_x) / np.maximum(te_x - le_x, 1e-12)
    area = np.linalg.norm(crs, axis=1) / 2
    strips = strip0 + np.arange(n) // chordwise
    if lat.quads is None:
        lat.quads = quads
        lat.left = left
        lat.right = right
        lat.centre = centre
        lat.colloc = colloc
        lat.normals = normals
        lat.chord_axis = chord_axis
        lat.span_axis = span_axis
        lat.area = area
        lat.x_frac = proj
        lat.y_abs = centre[:, 1].copy()
        lat.surface = np.full(n, surface_id, int)
        lat.half = np.full(n, half, int)
        lat.strip = strips
    else:
        lat.quads = np.concatenate([lat.quads, quads])
        lat.left = np.concatenate([lat.left, left])
        lat.right = np.concatenate([lat.right, right])
        lat.centre = np.concatenate([lat.centre, centre])
        lat.colloc = np.concatenate([lat.colloc, colloc])
        lat.normals = np.concatenate([lat.normals, normals])
        lat.chord_axis = np.concatenate([lat.chord_axis, chord_axis])
        lat.span_axis = np.concatenate([lat.span_axis, span_axis])
        lat.area = np.concatenate([lat.area, area])
        lat.x_frac = np.concatenate([lat.x_frac, proj])
        lat.y_abs = np.concatenate([lat.y_abs, centre[:, 1]])
        lat.surface = np.concatenate([lat.surface, np.full(n, surface_id, int)])
        lat.half = np.concatenate([lat.half, np.full(n, half, int)])
        lat.strip = np.concatenate([lat.strip, strips])

def build_lattice(config, spanwise=24, chordwise=6):
    """Build the full aircraft lattice (wing both halves + V-tail both halves).

    Cosine spacing in span (per the source adapter) and in chord (per the kernel).
    The right half is built root->tip; the left half is the mirror with reversed
    span ordering so its panels keep +z normals and +y bound vortices.
    """
    lat = Lattice()
    lat.quads = None

    def _surface_points(stations, reverse, mirror=1.0):
        sampler = surface_sampler(stations, mirror=mirror)
        n = chordwise
        spans = (1 - np.cos(np.linspace(0, math.pi, spanwise + 1))) / 2
        if reverse:
            spans = spans[::-1]
        chords = (1 - np.cos(np.linspace(0, math.pi, n + 1))) / 2
        pts = np.array([sampler(s, x) for s in spans for x in chords])
        faces = []
        for i in range(spanwise):
            for j in range(n):
                a = i * (n + 1) + j
                faces.append([a, a + 1, a + n + 2, a + n + 1])
        return np.asarray(pts)[np.asarray(faces)]
    wing = config['wing']
    strip = 0
    right = _surface_points(wing['stations'], reverse=False, mirror=+1.0)
    _append_surface(lat, right, strip, 0, +1, chordwise)
    strip += spanwise
    left = _surface_points(wing['stations'], reverse=True, mirror=-1.0)
    _append_surface(lat, left, strip, 0, -1, chordwise)
    strip += spanwise
    if config.get('vtail') and config['vtail'].get('stations'):
        vt = config['vtail']
        tright = _surface_points(vt['stations'], reverse=False, mirror=+1.0)
        _append_surface(lat, tright, strip, 1, +1, chordwise)
        strip += spanwise
        tleft = _surface_points(vt['stations'], reverse=True, mirror=-1.0)
        _append_surface(lat, tleft, strip, 1, -1, chordwise)
    lat.n = len(lat.quads)
    lat.wing_area = wing['area_mm2'] / 1000000.0
    lat.span = wing['span_total'] / 1000.0
    lat.mac = wing['mac'] / 1000.0
    return lat

def effective_normals(lat, deflections, camber=None):
    """Per-panel boundary-condition normals after camber and control deflections.

    ``camber`` is an optional dict ``{surface_name: slope_fn}`` (see
    :func:`camber_from_config`); for each panel of that surface the geometric normal
    is rotated in the (chord, normal) plane by the section's mean camber-line slope
    s = dzc/dx, i.e. ``n_eff = normalise(n - s * chord_axis)``.

    ``deflections`` is a list of dicts::

        {"surface": "wing"|"tail", "half": +1|-1|0 (0 = both),
         "mode": "morph"|"flap",
         "x_start": float (chord fraction where the deflection begins),
         "delta_deg": float (positive = trailing edge DOWN),
         "y0", "y1": optional absolute-y span limits (metres) restricting the zone}

    For a panel with chord fraction x beyond x_start, the local slope increment is
      s = -tan(delta) * ramp,   ramp = 1 (flap) or (x - x_start)/(1 - x_start)
    (morph).  The normal is rotated in the (chord, normal) plane:
      n_eff = normalise(n - s * chord_axis),
    i.e. about the local span axis, so the mesh geometry is never moved.  Camber is
    applied first, then deflections; both are small rotations about the span axis and
    compose additively to first order.
    """
    n = lat.normals.copy()
    if camber:
        for surf_name, slope_fn in camber.items():
            sid = 0 if surf_name == 'wing' else 1
            sel = lat.surface == sid
            if slope_fn is None or not sel.any():
                continue
            s_c = np.array([slope_fn(float(x)) for x in lat.x_frac[sel]])
            n_eff = n[sel] - s_c[:, None] * lat.chord_axis[sel]
            n_eff /= np.linalg.norm(n_eff, axis=1)[:, None]
            n[sel] = n_eff
    for d in deflections:
        surf = d['surface']
        half = d.get('half', 0)
        mode = d.get('mode', 'flap')
        x_start = float(d.get('x_start', 0.0))
        delta = float(d.get('delta_deg', 0.0))
        sel = lat.surface == (0 if surf == 'wing' else 1)
        if half != 0:
            sel &= lat.half == half
        if 'y0' in d or 'y1' in d:
            y0 = float(d.get('y0', -np.inf))
            y1 = float(d.get('y1', np.inf))
            lo, hi = (min(y0, y1), max(y0, y1))
            sel &= (lat.y_abs >= lo) & (lat.y_abs <= hi)
        if not sel.any():
            continue
        tan_d = math.tan(delta * DEG)
        x = lat.x_frac[sel]
        aft = x >= x_start
        s = np.zeros_like(x)
        if mode == 'morph':
            if x_start < 1.0:
                s[aft] = -tan_d * (x[aft] - x_start) / (1.0 - x_start)
        else:
            s[aft] = -tan_d
        n_eff = lat.normals[sel] - s[:, None] * lat.chord_axis[sel]
        n_eff /= np.linalg.norm(n_eff, axis=1)[:, None]
        n[sel] = n_eff
    return n

def solve_lattice(lat, velocity, density, reference, normals=None, core_m=1e-07, chunk=128):
    """Solve the linear vortex-lattice system with the vendored Biot-Savart core.

    ``velocity`` is the freestream at each panel, either a constant (3,) or a
    per-panel (N,3) array (used for body-rate boundary conditions).  ``normals``
    overrides the geometric panel normals (morph/flap boundary condition).  Returns
    dict with F_g (3,), M_g (3,) about ``reference``, gamma (N,), panel_forces (N,3),
    centres (N,3), and the boundary residual.

    Formula (Kutta-Joukowski at each bound vortex): F_i = rho * V_i x (gamma_i * span),
    M_i = (centre_i - reference) x F_i.  The influence matrix is A_ij = n_i . v_ij
    with v_ij the unit-circulation induced velocity of horseshoe j at collocation i;
    the boundary condition is A gamma = -n . V_inf.
    """
    left, right = (lat.left, lat.right)
    colloc, centre = (lat.colloc, lat.centre)
    if normals is None:
        normals = lat.normals
    velocity = np.asarray(velocity, float)
    if velocity.ndim == 1:
        velocity = np.broadcast_to(velocity, (lat.n, 3)).copy()
    n = lat.n
    matrix = np.empty((n, n))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        induced = vlm_kernel.horseshoe(colloc[start:stop], left, right, core_m)
        matrix[start:stop] = np.einsum('ijk,ik->ij', induced, normals[start:stop])
    rhs = -np.einsum('ij,ij->i', normals, velocity)
    factors, pivots = lu_factor(matrix)
    gamma = lu_solve((factors, pivots), rhs)
    residual = np.linalg.norm(matrix @ gamma - rhs, np.inf) / max(np.linalg.norm(rhs, np.inf), 1.0)
    if not np.all(np.isfinite(gamma)):
        raise ValueError('Influence solve failed')
    velocities = np.empty((n, 3))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        velocities[start:stop] = velocity[start:stop] + np.einsum('ijk,j->ik', vlm_kernel.horseshoe(centre[start:stop], left, right, core_m), gamma)
    forces = density * np.cross(velocities, right - left) * gamma[:, None]
    moments = np.cross(centre - np.asarray(reference, float), forces)
    return dict(F_g=forces.sum(axis=0), M_g=moments.sum(axis=0), gamma=gamma, panel_forces=forces, centres=centre, relative_boundary_residual=residual)

class VLMSystem:
    """Cached linear system for many solves on one lattice + deflection state.

    The influence matrix (and its LU factorisation) depends only on the lattice
    geometry and the effective panel normals (i.e. the deflection state), not on the
    freestream.  Building it once and reusing the factors for many right-hand sides
    (angles of attack, sideslips, body rates) makes the derivative sweeps cheap.
    """

    def __init__(self, config, deflections=None, spanwise=24, chordwise=6, core_m=1e-07, chunk=128):
        self.config = config
        self.lat = build_lattice(config, spanwise, chordwise)
        self.normals = effective_normals(self.lat, deflections or [], camber=camber_from_config(config))
        self.core_m = core_m
        self.chunk = chunk
        left, right = (self.lat.left, self.lat.right)
        colloc = self.lat.colloc
        n = self.lat.n
        matrix = np.empty((n, n))
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            induced = vlm_kernel.horseshoe(colloc[start:stop], left, right, core_m)
            matrix[start:stop] = np.einsum('ijk,ik->ij', induced, self.normals[start:stop])
        self.matrix = matrix
        self.factors, self.pivots = lu_factor(matrix)
        B = np.empty((n, n, 3))
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            B[start:stop] = vlm_kernel.horseshoe(self.lat.centre[start:stop], left, right, core_m)
        self.B = B

    def solve(self, velocity, density, reference):
        """Solve for one velocity field (3,) or (N,3); return F, M, gamma, drag."""
        velocity = np.asarray(velocity, float)
        if velocity.ndim == 1:
            velocity = np.broadcast_to(velocity, (self.lat.n, 3)).copy()
        rhs = -np.einsum('ij,ij->i', self.normals, velocity)
        gamma = lu_solve((self.factors, self.pivots), rhs)
        if not np.all(np.isfinite(gamma)):
            raise ValueError('Influence solve failed')
        residual = np.linalg.norm(self.matrix @ gamma - rhs, np.inf) / max(np.linalg.norm(rhs, np.inf), 1.0)
        velocities = velocity + np.einsum('ijk,j->ik', self.B, gamma)
        forces = density * np.cross(velocities, self.lat.right - self.lat.left) * gamma[:, None]
        moments = np.cross(self.lat.centre - np.asarray(reference, float), forces)
        D_i, _, _ = trefftz_induced_drag(self.lat, gamma, density)
        return dict(F_g=forces.sum(axis=0), M_g=moments.sum(axis=0), gamma=gamma, panel_forces=forces, D_i=D_i, relative_boundary_residual=residual)

def wake_velocity(field, left, right, direction=(1.0, 0.0, 0.0), core_m=1e-07):
    """Unit-circulation velocity at M points from the two semi-infinite trailing
    filaments of N horseshoes (the wake-only part of the kernel's ``horseshoe``).

    Reproduces the kernel's wake terms exactly: for filaments starting at ``left``
    and ``right`` extending along ``direction``, the induced velocity is
    (-d x a * wa + d x b * wb) / (4 pi), with wa = (1 + a.d/|a|)/(|d x a|^2 + core^2).
    Reference implementation kept for the kernel's wake decomposition; induced drag
    is computed by the lifting-line ``trefftz_induced_drag``.  Not a viscous wake.
    """
    field = np.asarray(field, float)
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    direction = np.asarray(direction, float)
    direction /= np.linalg.norm(direction)
    a = field[:, None, :] - left[None, :, :]
    b = field[:, None, :] - right[None, :, :]
    na = np.sqrt(np.sum(a * a, axis=2) + core_m ** 2)
    nb = np.sqrt(np.sum(b * b, axis=2) + core_m ** 2)
    ua = np.cross(direction, a)
    ub = np.cross(direction, b)
    da = np.sum(ua * ua, axis=2) + core_m ** 2
    db = np.sum(ub * ub, axis=2) + core_m ** 2
    wa = (1 + np.sum(a * direction, axis=2) / na) / da
    wb = (1 + np.sum(b * direction, axis=2) / nb) / db
    return (-ua * wa[:, :, None] + ub * wb[:, :, None]) / (4 * math.pi)

def strip_circulation(lat, gamma, surface_id=None):
    """Total bound circulation per spanwise strip, sorted by ascending span y.

    ``surface_id`` restricts to one surface (0=wing, 1=tail); None uses all panels.
    Returns (y_c, Gamma_c, dy_c): strip centres, strip circulations (sum of the
    chordwise panel circulations) and strip spanwise widths.
    """
    sel = np.ones(lat.n, bool) if surface_id is None else lat.surface == surface_id
    st = lat.strip[sel]
    g = gamma[sel]
    strips = np.unique(st)
    G = np.array([g[st == k].sum() for k in strips])
    yc = np.array([lat.centre[sel, 1][st == k].mean() for k in strips])
    dy = np.array([np.abs(lat.right[sel, 1][st == k] - lat.left[sel, 1][st == k]).mean() for k in strips])
    order = np.argsort(yc)
    return (yc[order], G[order], dy[order])

def trefftz_induced_drag(lat, gamma, density):
    """Induced drag from the Trefftz-plane trailing vortex sheet (lifting-line form).

    Uses the WING panels only: the wing is planar (z=0) so the lifting-line sheet
    discretisation applies; the V-tail's own induced drag is omitted (declared small
    — the tail is a lightly loaded, low-aspect surface and its induced drag is a
    second-order term on total drag).  Each wing strip's total circulation Gamma(y)
    forms a trailing vortex sheet whose shed vorticity at each strip edge is the drop
    in circulation.  The downwash at each strip centre is the sum of the edge-vortex
    contributions (evaluating at strip centres avoids the self-singularity):
      w_k = sum_j shed_j / (4 pi (y_k - y_edge_j)),
      D_i = rho sum_k Gamma_k w_k dy_k.
    For elliptic loading this recovers CDi = CL^2/(pi AR) (span efficiency ~1).
    Approximate: lifting-line discretisation, not the exact far Trefftz plane.
    """
    yc, G, dy = strip_circulation(lat, gamma, surface_id=0)
    K = len(G)
    y_edge = np.empty(K + 1)
    y_edge[0] = yc[0] - dy[0] / 2
    y_edge[1:K] = (yc[:-1] + yc[1:]) / 2
    y_edge[K] = yc[-1] + dy[-1] / 2
    Gext = np.concatenate([[0.0], G, [0.0]])
    shed = Gext[1:] - Gext[:-1]
    w = np.zeros(K)
    for k in range(K):
        w[k] = np.sum(shed / (4 * math.pi * (yc[k] - y_edge)))
    D = density * float(np.sum(G * w * dy))
    return (D, w, dy)

def coefficients(lat, F_g, M_g, rho, V, reference, D_i=None):
    """Convert solver force/moment (solver frame) into stability-axis coefficients."""
    qS = 0.5 * rho * V * V * lat.wing_area
    qSc = qS * lat.mac
    qSb = qS * lat.span
    out = {'CL': float(F_g[2] / qS), 'CY': float(F_g[1] / qS), 'Cx': float(F_g[0] / qS), 'Cm': float(M_g[1] / qSc), 'Cl': float(-M_g[0] / qSb), 'Cn': float(-M_g[2] / qSb)}
    if D_i is not None:
        out['CDi'] = float(D_i / qS)
        out['CDi_Fx'] = float(F_g[0] / qS)
    return out

def spanwise_load(lat, gamma, rho, V, surface_id=None):
    """Spanwise lift distribution: per-spanwise-strip circulation and lift.

    ``surface_id`` restricts to one surface (0=wing, 1=tail); None uses all panels.
    Strips are the spanwise panel indices; circulation is summed over the chordwise
    panels of each strip, and the strip lift is the Kutta-Joukowski form
    rho V gamma_strip dy.  ``lift_per_span`` is rho V gamma_strip (N/m, the lift per
    unit span that structure tasks consume); ``loading`` is the dimensionless span
    loading 2 gamma_strip / (V * mac).  Returns dicts of arrays plus a combined list.
    """
    sel = np.ones(lat.n, bool) if surface_id is None else lat.surface == surface_id
    strips = np.unique(lat.strip[sel])
    y = np.array([lat.centre[sel][lat.strip[sel] == k, 1].mean() for k in strips])
    gam = np.array([gamma[sel][lat.strip[sel] == k].sum() for k in strips])
    dy = np.array([np.abs(lat.right[sel][lat.strip[sel] == k, 1] - lat.left[sel][lat.strip[sel] == k, 1]).mean() for k in strips])
    lift = rho * V * gam * dy
    lift_per_span = rho * V * gam
    loading = 2.0 * gam / (V * lat.mac)
    return {'y': y, 'gamma': gam, 'dy': dy, 'lift': lift, 'lift_per_span': lift_per_span, 'loading': loading}

def solve(config, alpha_deg=0.0, beta_deg=0.0, rates=None, deflections=None, reference=None, rho=None, V_ms=None, spanwise=24, chordwise=6, core_m=1e-07):
    """Solve one flight condition and return the full result dict.

    ``rates`` = (p, q, r) in the standard body convention (rad/s, about the
    standard x-forward/z-down axes) — internally mapped to the solver frame.
    ``reference`` defaults to the CG x from ``config['flight']`` at y=z=0.
    """
    rho = rho if rho is not None else config['flight']['rho']
    V = V_ms if V_ms is not None else config['flight']['V_ms']
    if reference is None:
        cgx = config['flight'].get('cg_x', 0.0) / 1000.0 if config['flight'].get('cg_x') else 0.0
        reference = np.array([cgx, 0.0, 0.0])
    else:
        reference = np.asarray(reference, float)
    sys_ = VLMSystem(config, deflections=deflections, spanwise=spanwise, chordwise=chordwise, core_m=core_m)
    velocity = freestream_velocity(V, alpha_deg, beta_deg, rates, sys_.lat.colloc, reference)
    res = sys_.solve(velocity, rho, reference)
    coef = coefficients(sys_.lat, res['F_g'], res['M_g'], rho, V, reference, D_i=res['D_i'])
    load = spanwise_load(sys_.lat, res['gamma'], rho, V)
    return plain({'coefficients': coef, 'forces': {'F_g': res['F_g'], 'M_g': res['M_g'], 'D_i': res['D_i']}, 'circulation': res['gamma'], 'spanwise': load, 'residual': res['relative_boundary_residual'], 'reference': reference, 'V': V, 'rho': rho})

def freestream_velocity(V, alpha_deg, beta_deg, rates=None, positions=None, rotation_center=None):
    """Freestream velocity (3,) or (N,3) in the solver frame, including body rates.

    V_inf = V (cos a cos b, -sin b, sin a cos b).  Body rates ``(p, q, r)`` are the
    standard convention (about x-forward/y-right/z-down); mapped to the solver frame
    the angular velocity is omega = (-p, q, -r) about the rotation centre.
    """
    a = alpha_deg * DEG
    b = beta_deg * DEG
    v = V * np.array([math.cos(a) * math.cos(b), -math.sin(b), math.sin(a) * math.cos(b)])
    if rates is None:
        return v
    p, q, r = rates
    omega = np.array([-p, q, -r], float)
    pos = np.asarray(positions, float)
    centre = np.asarray(rotation_center, float)
    return v[None, :] - np.cross(omega, pos - centre)

def convergence(config, alpha_deg, base=(24, 6), factor=2, **kw):
    """Convergence helper: solve at base and refined panel counts, report the change.

    Returns the CL_alpha computed at each level and the relative change between the
    base and refined meshes.
    """
    results = []
    for mult in (1, factor):
        ns = base[0] * mult
        nc = base[1] * mult
        lo = solve(config, alpha_deg=alpha_deg - 1.0, spanwise=ns, chordwise=nc, **kw)
        hi = solve(config, alpha_deg=alpha_deg + 1.0, spanwise=ns, chordwise=nc, **kw)
        cla = (hi['coefficients']['CL'] - lo['coefficients']['CL']) / (2.0 * DEG)
        results.append({'spanwise': ns, 'chordwise': nc, 'CL_alpha': cla, 'CL': hi['coefficients']['CL']})
    rel = abs(results[1]['CL_alpha'] - results[0]['CL_alpha']) / abs(results[0]['CL_alpha'])
    return {'levels': results, 'relative_change_CL_alpha': rel}

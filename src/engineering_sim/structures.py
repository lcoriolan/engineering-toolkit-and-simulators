from scipy.integrate import quad
import math
import numpy as np
_trapezoid=np.trapezoid if hasattr(np,"trapezoid") else np.trapz
def tube_area_moment(D, d):
    """Second moment of area of a hollow circular section (m^4).

    I = pi (D^4 - d^4) / 64. D = outer diameter, d = inner diameter (m).
    """
    return math.pi * (D ** 4 - d ** 4) / 64.0

def tube_polar_moment(D, d):
    """Polar (torsion) second moment of area of a hollow circular section (m^4).

    J = pi (D^4 - d^4) / 32 = 2 * I.
    """
    return math.pi * (D ** 4 - d ** 4) / 32.0

def rect_area_moment(w, h):
    """Second moment of area of a solid rectangle about its neutral axis (m^4).

    I = w h^3 / 12.
    """
    return w * h ** 3 / 12.0

def rect_tube_area_moment(w, h, t):
    """Second moment of area of a thin-walled rectangular tube (m^4).

    I = (w h^3 - (w - 2t)(h - 2t)^3) / 12, outer w x h, wall t.
    """
    return (w * h ** 3 - (w - 2.0 * t) * (h - 2.0 * t) ** 3) / 12.0

def bredt_batho_J(Am, perimeter, t):
    """Bredt-Batho torsion constant of a thin-walled closed section (m^4).

    J = 4 A_m^2 t / perimeter, where A_m is the area enclosed by the wall mid-line and
    perimeter the mid-line perimeter (circumferential length), t the (uniform) wall
    thickness. Cited: Megson, "Aircraft Structures for Engineering Students" (Bredt-Batho
    formula for a thin-walled single closed cell). For a circular tube of mean radius r
    this reduces to 2 pi r^3 t; for a rectangular tube it is exact for a uniform-thickness
    box section.
    """
    return 4.0 * Am * Am * t / perimeter

def thin_tube_polar_moment(r, t):
    """Torsion constant of a thin circular tube, J = 2 pi r^3 t (m^4).

    r is the mean radius (mid-wall), t the wall thickness; thin-wall limit of
    ``tube_polar_moment(2r+t, 2r-t)``. Independent analytic result used by the tests.
    """
    return 2.0 * math.pi * r ** 3 * t

def rect_tube_polar_moment(w, h, t):
    """Bredt-Batho torsion constant of a thin-walled rectangular box section (m^4).

    A_m = (w - t)(h - t) (mid-line enclosure), perimeter = 2[(w - t) + (h - t)], so
    J = 4 A_m^2 t / perimeter. w, h are the outer dimensions, t the wall.
    """
    Am = (w - t) * (h - t)
    perimeter = 2.0 * ((w - t) + (h - t))
    return bredt_batho_J(Am, perimeter, t)

def polygon_area_perimeter(pts):
    """Shoelace area and perimeter of a 2-D polygon (pts: Nx2 array), area in |units|^2.

    Assumes an ordered (simple, closed) polygon; the returned area is signed (positive
    for counter-clockwise) and the perimeter is the closed-loop length.
    """
    x = pts[:, 0]
    z = pts[:, 1]
    area = 0.5 * (np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))
    d = np.diff(np.vstack([pts, pts[0]]), axis=0)
    perimeter = float(np.sqrt((d[:, 0] ** 2 + d[:, 1] ** 2)).sum())
    return area, perimeter

def offset_polygon_inward(pts, d):
    """Offset a counter-clockwise simple polygon inward by distance ``d`` (m).

    Each edge is shifted along its inward (left) normal by ``d`` and consecutive shifted
    edges are intersected to form the offset vertices — the exact straight-edge polygon
    offset. Returns the offset polygon (same orientation); used to place the wall
    mid-line at t/2 inside the printed outer skin for the Bredt-Batho section.
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    # Normalise to counter-clockwise so the left normal points inward.
    area, _ = polygon_area_perimeter(pts)
    if area < 0:
        pts = pts[::-1]
    out = np.zeros_like(pts)
    for i in range(n):
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        # Inward (left) normals of the two edges meeting at vertex i.
        def normal(a, b):
            e = b - a
            L = np.hypot(e[0], e[1])
            return np.array([-e[1], e[0]]) / L if L > 0 else np.array([0.0, 0.0])
        n_prev = normal(p0, p1)
        n_next = normal(p1, p2)
        # Offset the two edge lines: line k is (vertex + d*n_k) parallel to the edge.
        a_prev = p0 + d * n_prev
        d_prev = p1 - p0
        a_next = p1 + d * n_next
        d_next = p2 - p1
        # Intersection of the two offset lines (2x2 solve).
        M = np.array([[d_prev[0], -d_next[0]], [d_prev[1], -d_next[1]]])
        rhs = a_next - a_prev
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if abs(det) < 1e-14:
            out[i] = p1 + d * (n_prev + n_next) / 2.0  # parallel fallback
        else:
            lam = (rhs[0] * M[1, 1] - rhs[1] * M[0, 1]) / det
            out[i] = a_prev + lam * d_prev
    return out

def airfoil_shell_geometry(coords, chord_m, x_start_frac, t_m):
    """Closed-section mid-line geometry of a printed wing skin (m, m).

    ``coords`` is the baseline airfoil (x/c, z/c) as a TE -> upper -> LE -> lower -> TE
    loop. It is scaled by ``chord_m``; the part aft of ``x_start_frac`` (a morph-zone
    hinge, which is an open section and carries no torsion) is cut off and closed with a
    straight edge at the hinge. The polygon is then offset inward by t/2 (the wall
    mid-line) and its enclosed area A_m and perimeter are returned (with the shoelace
    area before offset for reference). Returns dict(Am, perimeter, area_outer, chord_m).
    """
    pts = np.asarray(coords, dtype=float) * chord_m  # x, z in metres
    # Split the loop at the leading edge (min x) into upper (TE->LE) and lower (LE->TE).
    i_le = int(np.argmin(pts[:, 0]))
    upper = pts[: i_le + 1]          # TE -> LE (x decreasing)
    lower = pts[i_le + 1:]           # LE -> TE (x increasing)
    if x_start_frac >= 1.0 or x_start_frac <= 0.0:
        poly = np.vstack([upper, lower])
        truncated = False
    else:
        xh = x_start_frac * chord_m
        # Keep only the points at or forward of the hinge on each surface.
        up = upper[upper[:, 0] <= xh + 1e-12]
        lo = lower[lower[:, 0] <= xh + 1e-12]
        z_uh = _interp_z(up, xh)
        z_lh = _interp_z(lo, xh)
        poly = np.vstack([[xh, z_uh], up, lo, [xh, z_lh]])
        truncated = True
    area_outer, _ = polygon_area_perimeter(poly)
    mid = offset_polygon_inward(poly, t_m / 2.0)
    Am, perimeter = polygon_area_perimeter(mid)
    return {"Am": abs(float(Am)), "perimeter": float(perimeter),
            "area_outer": abs(float(area_outer)), "chord_m": float(chord_m),
            "truncated": truncated}

def _interp_z(pts, xh):
    """Linearly interpolate z at x=xh on an x-ordered upper/lower surface point list."""
    if len(pts) == 0:
        return 0.0
    # pts is ordered along x; find the bracketing segment nearest xh.
    if xh <= pts[0, 0]:
        return float(pts[0, 1])
    if xh >= pts[-1, 0]:
        return float(pts[-1, 1])
    i = int(np.searchsorted(pts[:, 0], xh))
    a, b = pts[i - 1], pts[i]
    if abs(b[0] - a[0]) < 1e-15:
        return float(a[1])
    return float(a[1] + (b[1] - a[1]) * (xh - a[0]) / (b[0] - a[0]))

def schrenk_semi_span_load(wing, n, weight_n):
    """Semi-span lift per unit span by the Schrenk approximation (N/m).

    The lift distribution is the average of the elliptic (ideal) distribution and the
    planform (chord-proportional) distribution, each normalised to the same semi-span
    total n*W/2:

        L'_ell(y)     = (4 n W / (pi b)) * sqrt(1 - (2y/b)^2)
        L'_plan(y)    = (n W / S) * c(y)
        L'(y)         = 0.5 * (L'_ell + L'_plan)

    so that int_0^{b/2} L'(y) dy = n W / 2 (the elliptic peak 4 n W/(pi b) gives a
    semi-span integral of n W/2, and the planform term integrates to n W/2 exactly).
    c(y) is a linear taper between root_chord_m and tip_chord_m (or a station table
    if ``wing["stations"]`` is given). Returns ``(y, load)`` arrays (metres, N/m).
    """
    b = wing["span_m"]; S = wing["area_m2"]
    cr = wing["root_chord_m"]; ct = wing["tip_chord_m"]
    y = np.linspace(0.0, b / 2.0, 501)
    if "stations" in wing and wing["stations"]:
        y = np.array([s["y_m"] for s in wing["stations"]], dtype=float)
        c = np.array([s["chord_m"] for s in wing["stations"]], dtype=float)
    else:
        c = cr + (ct - cr) * (2.0 * y / b)
    ell = (4.0 * n * weight_n / (math.pi * b)) * np.sqrt(np.clip(1.0 - (2.0 * y / b) ** 2, 0.0, None))
    plan = (n * weight_n / S) * c
    load = 0.5 * (ell + plan)
    return y, load

def shear_moment_from_load(y, load):
    """Integrate semi-span lift to shear force and bending moment (root -> tip).

    With y from root (0) to tip (b/2), the shear at station y is the load outboard of y
    and the moment is that load times its lever arm:

        V(y) = int_y^{tip} L'(eta) d eta
        M(y) = int_y^{tip} L'(eta) (eta - y) d eta

    Trapezoidal integration on the supplied grid. Returns (V, M) arrays (N, N m).
    """
    y = np.asarray(y, dtype=float); load = np.asarray(load, dtype=float)
    n = len(y)
    V = np.zeros(n); M = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            V[i] = 0.0; M[i] = 0.0
        else:
            dy = y[i + 1] - y[i]
            V[i] = V[i + 1] + 0.5 * (load[i] + load[i + 1]) * dy
            M[i] = M[i + 1] + V[i + 1] * dy + (load[i + 1] * 0.5 + load[i] / 6.0) * dy ** 2
    return V, M

def cantilever_deflection(y, M, EI):
    """Tip deflection of a cantilever from the bending-moment curve (double integration).

    With the root at y=0 clamped (zero slope and deflection at the root) and M(y) the
    bending moment, the curvature is M/EI, and

        deflection(y) = int_0^y int_0^{s} M(t)/EI dt ds

    EI may be a scalar or an array along y. Trapezoidal integration. Returns the full
    deflection array; the tip value is ``deflection[-1]``.
    """
    y = np.asarray(y, dtype=float); M = np.asarray(M, dtype=float)
    EI = np.full(len(y), float(EI), dtype=float) if np.isscalar(EI) else np.asarray(EI, dtype=float)
    # Guard the outboard region where no spar spans (EI = 0): zero bending stiffness means
    # zero curvature there, so the outboard wing extends as a rigid continuation of the
    # last supported station (declared simplification; the region is small and lightly loaded).
    curvature = np.zeros(len(y), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = np.where(EI > 0.0, M / EI, 0.0)
    slope = np.zeros(len(y))
    for i in range(1, len(y)):
        slope[i] = slope[i - 1] + 0.5 * (curvature[i - 1] + curvature[i]) * (y[i] - y[i - 1])
    deflection = np.zeros(len(y))
    for i in range(1, len(y)):
        deflection[i] = deflection[i - 1] + 0.5 * (slope[i - 1] + slope[i]) * (y[i] - y[i - 1])
    return deflection

def tip_deflection_uniform(w, L, EI):
    """Closed-form cantilever tip deflection under uniform load w: w L^4 / (8 EI).

    Independent analytic result used by the tests to validate :func:`cantilever_deflection`.
    """
    return w * L ** 4 / (8.0 * EI)

def flap_coefficients(x_start, x_ac=0.25, n_terms=4000):
    """Thin-airfoil (Glauert / Abbott & von Doenhoff) coefficients for a plain flap
    hinged at x_start (fraction of chord from the leading edge).

    With theta from the leading edge, x/c = (1 - cos theta)/2, the flap occupies
    theta in [theta_h, pi] where theta_h = arccos(1 - 2 x_start). A flap deflection
    delta changes the camber slope by -delta over the flap; the bound vorticity is

        gamma(theta) = 2 U [ A0 (1 + cos theta)/sin theta + sum_n A_n sin(n theta) ],
        A0 = alpha + delta (pi - theta_h)/pi,  A_n = 2 delta sin(n theta_h)/(n pi).

    Lift:  C_L = 2 pi A0 + pi A_1, giving  dC_L/d_delta = 2 (pi - theta_h + sin theta_h)
    (analytic). The hinge moment about x_start and the pitching moment about x_ac are
    integrated from the same loading; the hinge-moment coefficient is referenced to the
    flap chord c_f = c (1 - x_start), with

        C_halpha = -(c/c_f)^2 I0,   I0 = int_{theta_h}^pi (cos theta_h - cos theta)(1+cos theta) dtheta,
        C_hdelta = -(c/c_f)^2 [ (pi - theta_h)/pi I0 + (2/pi) J ],
        J = int_{theta_h}^pi (cos theta_h - cos theta) (1/2) ln[ sin((theta+theta_h)/2)/sin((theta-theta_h)/2) ] sin theta dtheta.

    Sign convention: hinge moment positive trailing-edge down, so a plain flap is
    restoring (C_halpha < 0, C_hdelta < 0); Cm_ac_delta < 0 (nose-down for positive
    deflection); C_L_delta > 0. Limiting case x_start -> 0 gives C_L_delta -> 2 pi and
    C_halpha = C_hdelta -> -pi/2 (the leading-edge moment derivative of a symmetric
    airfoil). Screening only: thin-airfoil theory over-predicts flap effectiveness
    (no separation) and ignores the finite-span/3-D effects.
    """
    from scipy.integrate import quad
    th_h = math.acos(1.0 - 2.0 * x_start)
    cfc2 = 1.0 / (1.0 - x_start) ** 2  # (c / c_f)^2

    def I0():
        return (math.cos(th_h) * (math.pi - th_h) - math.cos(th_h) * math.sin(th_h)
                + math.sin(th_h) - math.pi / 2.0 + th_h / 2.0 + math.sin(2.0 * th_h) / 4.0)

    Ch_alpha = -cfc2 * I0()
    # delta hinge-moment term (logarithmic removable singularity at theta_h)
    def integrand(th):
        return ((math.cos(th_h) - math.cos(th)) * 0.5
                * math.log(math.sin((th + th_h) / 2.0) / math.sin((th - th_h) / 2.0))
                * math.sin(th))
    J, _ = quad(integrand, th_h + 1e-9, math.pi - 1e-9, limit=500)
    Ch_delta = -cfc2 * ((math.pi - th_h) / math.pi * I0() + (2.0 / math.pi) * J)
    CL_delta = 2.0 * (math.pi - th_h + math.sin(th_h))
    # Pitching moment about the quarter chord due to delta, closed form (Glauert; Anderson, Fundamentals of
    # Aerodynamics §4.8): C_m,c/4 = (pi/4)(A2 - A1) with A1 = 2 delta sin(theta_h)/pi, A2 = delta sin(2 theta_h)/pi
    #   => dC_m,c/4 / d delta = -(1/2) sin(theta_h) (1 - cos(theta_h)).
    # Integrator fix (2026-09-15 01:54 MDT): round 2 integrated the series loading only over the flap
    # (theta_h..pi), but the flap-induced loading acts over the whole chord; that under-stated the moment
    # (-0.358 vs -0.647 at a 0.72c hinge). Verified against an independent discrete-vortex solution
    # (tests: Aeroelastic.test_flap_coefficients_vs_discrete_vortex). x_ac must be 0.25 for this form.
    if abs(x_ac - 0.25) > 1e-12:
        raise ValueError('flap_coefficients: the closed-form moment is about the quarter chord (x_ac = 0.25).')
    Cm_ac_delta = -0.5 * math.sin(th_h) * (1.0 - math.cos(th_h))
    return {"CL_delta": CL_delta, "Cm_ac_delta": Cm_ac_delta,
            "Ch_alpha": Ch_alpha, "Ch_delta": Ch_delta, "theta_h": th_h}

def effective_gj(y, gj, weight):
    """Compliance-weighted (harmonic) mean of GJ(y) for the rigid-section screens.

    The torsional compliance 1/GJ is averaged over the semi-span, weighted by the local
    aeroelastic divergence loading. With e(y) the elastic-axis to aerodynamic-centre
    offset (a declared fraction e_frac of the local chord) the strip load is
    w(y) = e(y) c(y) = e_frac c(y)^2; the constant e_frac cancels in the ratio, so the
    caller passes w(y) = c(y)^2 (see :func:`_effective_gj_for_wing`):

        GJ_eff = (int_0^{b/2} w dy) / (int_0^{b/2} w / GJ(y) dy).

    For constant GJ this returns exactly GJ, so the round-1 lumped divergence/reversal
    formulae are recovered. ``weight`` is a per-station array aligned with ``gj``.
    """
    y = np.asarray(y, dtype=float)
    gj = np.asarray(gj, dtype=float)
    weight = np.asarray(weight, dtype=float)
    mask = gj > 0.0
    if int(mask.sum()) < 2:
        return 0.0  # no torsional stiffness over the span -> divergence at zero speed
    y, gj, weight = y[mask], gj[mask], weight[mask]
    return float(_trapezoid(weight, y) / _trapezoid(weight / gj, y))


def divergence_pressure(GJ, elastic_axis_offset, chord, lift_slope, semi_span):
    """Uniform clamped-wing strip screen: q=(pi/2)^2 GJ/(e c a L^2), SI; not flutter."""
    return (math.pi/2)**2*GJ/(elastic_axis_offset*chord*lift_slope*semi_span**2)

def reversal_pressure(GJ, lift_control_derivative, moment_control_derivative, chord, lift_slope, semi_span):
    """Uniform full-span-control strip screen, SI; negative moment derivative required."""
    return (math.pi/2)**2*GJ*lift_control_derivative/(-moment_control_derivative*chord**2*lift_slope*semi_span**2)

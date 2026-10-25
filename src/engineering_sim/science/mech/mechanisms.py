# DESCRIPTION: Mechanisms and machines section: degrees of freedom (Grübler-Kutzbach), Grashof
#              classification, four-bar and slider-crank position/velocity/acceleration, mechanical
#              advantage, cam motion laws and pressure angle, gear trains, Geneva and quick-return
#              mechanisms, static/dynamic linkage force analysis and rotating-mass balancing.
"""Mechanisms and machines calculators.

Conventions (read before using these relations):
- Planar mechanisms live in the x-y plane; links are numbered a (input crank, pinned to ground), b
  (coupler), c (output rocker/follower, pinned to ground), d (ground link between the two pivots). The
  ground pivot O2 is the crank pivot, O4 the rocker pivot, A the crank tip and B the coupler-rocker
  joint. Crank angle theta2 is measured CCW from the ground line O2O4; angles are radians internally.
- A four-bar ``branch`` selects the assembly mode entered as the text choice 'open' or 'crossed'
  (Norton half-angle form): the open solution is theta4 = phi - beta and the crossed solution
  theta4 = phi + beta, where phi = atan2(a sin(theta2), a cos(theta2) - d) and beta is the angle in the
  A-O4-B triangle from the law of cosines.
- Angular speeds are rad/s (enter rpm and it converts); cyclic speeds belong to the gears/vibrations
  sections where noted.
- Transmission angle mu is the angle between coupler and rocker at joint B; mu = 90 deg is ideal and
  mu near 0/180 deg is a toggle (poor force transmission).

Reference relations are the standard ones in mechanism and machine-theory texts (e.g. Norton, *Design of
Machinery*; Uicker, Pennock & Shigley, *Theory of Machines and Mechanisms*; Shigley & Uicker). They are
cited by relation name, recalled rather than re-read, so no page numbers are given.
"""
import math

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts

SECTION = section('mechanisms', 'Mechanisms and machines', order=70,
                  summary='Degrees of freedom, Grashof, four-bar and slider-crank kinematics, cams, '
                          'gear trains, Geneva/quick-return, linkage forces and balancing.')

PI = math.pi
TWOPI = 2.0 * PI


def _theta3(a, b, c, d, th2, br=1.0):
    """Coupler angle theta3 (rad) for a four-bar at crank angle th2.

    ``br`` selects the assembly: +1 gives theta4 = phi + beta (crossed) and -1 gives phi - beta (open);
    theta3 is the coupler angle of the same branch.
    """
    s = math.sqrt(a * a + d * d - 2.0 * a * d * math.cos(th2))
    phi = math.atan2(a * math.sin(th2), a * math.cos(th2) - d)
    beta = math.acos((s * s + c * c - b * b) / (2.0 * s * c))
    th4 = phi + br * beta
    Ax, Ay = a * math.cos(th2), a * math.sin(th2)
    Bx, By = d + c * math.cos(th4), c * math.sin(th4)
    return math.atan2(By - Ay, Bx - Ax), th4, s


# ---------------------------------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------------------------------
def _grashof(values):
    """Grashof classification of a four-bar from its four link lengths (json [a, b, c, d])."""
    links = values['links']
    if not isinstance(links, list) or len(links) != 4:
        raise MechError('links: give [a, b, c, d] (crank, coupler, rocker, ground) in m')
    a, b, c, d = (float(x) for x in links)
    if min(a, b, c, d) <= 0.0:
        raise MechError('link lengths must be positive')
    s, l = min(a, b, c, d), max(a, b, c, d)
    rest = sorted((a, b, c, d))
    p, q = rest[1], rest[2]
    s_plus_l = s + l
    p_plus_q = p + q
    if s_plus_l > p_plus_q + 1e-12:
        isg, typ = 0.0, 'non-Grashof (double-rocker)'
    elif abs(s_plus_l - p_plus_q) <= 1e-12:
        isg, typ = 1.0, 'change-point'
    else:
        isg = 1.0
        if s == a:
            typ = 'crank-rocker'
        elif s == c:
            typ = 'rocker-crank'
        elif s == b:
            typ = 'double-rocker (Grashof)'
        else:
            typ = 'double-crank'
    return {'s': s, 'l': l, 'p': p, 'q': q, 's_plus_l': s_plus_l, 'p_plus_q': p_plus_q,
            'is_grashof': isg, 'type': typ}


def _fourbar_position(values):
    """Four-bar vector-loop closure: coupler/rocker angles, transmission angle and diagonal at theta2."""
    links = values['links']
    if not isinstance(links, list) or len(links) != 4:
        raise MechError('links: give [a, b, c, d] (crank, coupler, rocker, ground) in m')
    a, b, c, d = (float(x) for x in links)
    th2 = values['theta2']
    br = 1.0 if values['branch'] == 'crossed' else -1.0
    s = math.sqrt(a * a + d * d - 2.0 * a * d * math.cos(th2))
    if s <= 1e-300:
        raise MechError('degenerate four-bar (crank and rocker pivots coincide at this angle)')
    phi = math.atan2(a * math.sin(th2), a * math.cos(th2) - d)
    cosb = (s * s + c * c - b * b) / (2.0 * s * c)
    if cosb > 1.0 + 1e-12 or cosb < -1.0 - 1e-12:
        raise MechError('four-bar cannot assemble at this crank angle (coupler/rocker length mismatch)')
    cosb = max(-1.0, min(1.0, cosb))
    beta = math.acos(cosb)
    th4 = phi + br * beta
    Ax, Ay = a * math.cos(th2), a * math.sin(th2)
    Bx, By = d + c * math.cos(th4), c * math.sin(th4)
    th3 = math.atan2(By - Ay, Bx - Ax)
    mu = math.acos(max(-1.0, min(1.0, (b * b + c * c - s * s) / (2.0 * b * c))))
    return {'theta4': th4, 'theta3': th3, 'mu': mu, 's': s}


def _fourbar_toggle(values):
    """Toggle (dead-centre) crank angles of a crank-rocker and the rocker's extreme angles.

    The rocker reverses when the crank and coupler are collinear (sin(theta2 - theta3) = 0); the two
    toggle crank angles are found by scanning a full crank revolution and refining each sign change.
    The rocker extreme angles theta4_1/theta4_2 are reported for the crossed branch (theta4 = phi + beta).
    Both assembly branches reach their rocker extremes at the SAME toggle crank angles; for the open branch
    (theta4 = phi - beta) the extremes are the mirror values 2*pi - theta4 in this ground-frame convention.
    """
    links = values['links']
    if not isinstance(links, list) or len(links) != 4:
        raise MechError('links: give [a, b, c, d] (crank, coupler, rocker, ground) in m')
    a, b, c, d = (float(x) for x in links)
    s, l = min(a, b, c, d), max(a, b, c, d)
    rest = sorted((a, b, c, d))
    p, q = rest[1], rest[2]
    if s + l >= p + q - 1e-12 or not (a < b and a < c and a < d):
        raise MechError('fourbar_toggle needs a crank-rocker linkage (Grashof, with the input crank a '
                        'strictly the shortest link); this linkage has no two dead-centre toggle positions')
    import scipy.optimize as opt

    def f(t):
        th3, _, _ = _theta3(a, b, c, d, t)
        return math.sin(t - th3)

    prev, prev_t = None, 0.0
    roots = []
    for i in range(1, 721):
        t = TWOPI * i / 720.0
        y = f(t)
        if prev is not None and y * prev < 0.0:
            roots.append(opt.brentq(f, prev_t, t))
        prev, prev_t = y, t
    if len(roots) != 2:
        raise MechError('toggle scan found %d positions; this linkage is not a crank-rocker with two '
                        'dead-centre positions (check the Grashof class)' % len(roots))
    t1, t2 = sorted(roots)
    th4_1 = _theta3(a, b, c, d, t1)[1] % TWOPI
    th4_2 = _theta3(a, b, c, d, t2)[1] % TWOPI
    arc = t2 - t1
    time_ratio = max(arc, TWOPI - arc) / min(arc, TWOPI - arc)
    return {'theta_t1': t1, 'theta_t2': t2, 'theta4_1': th4_1, 'theta4_2': th4_2,
            'time_ratio': time_ratio}


def _fourbar_accel(values):
    """Four-bar coupler and rocker angular accelerations alpha3, alpha4 (vector-loop acceleration).

    Differentiating the four-bar velocity loop once more and separating real/imaginary parts gives the
    linear system
        b cos(theta3) alpha3 - c cos(theta4) alpha4 = A
        b sin(theta3) alpha3 - c sin(theta4) alpha4 = B
    with
        A = -a alpha2 cos(theta2) + a omega2^2 sin(theta2) + b omega3^2 sin(theta3) - c omega4^2 sin(theta4)
        B = -a alpha2 sin(theta2) - a omega2^2 cos(theta2) - b omega3^2 cos(theta3) + c omega4^2 cos(theta4)
    solved for
        alpha3 = (B cos(theta4) - A sin(theta4))/(b sin(theta3 - theta4))
        alpha4 = (B cos(theta3) - A sin(theta3))/(c sin(theta3 - theta4)).
    """
    a, b, c = values['a'], values['b'], values['c']
    th2, th3, th4 = values['theta2'], values['theta3'], values['theta4']
    w2, w3, w4 = values['omega2'], values['omega3'], values['omega4']
    al2 = values['alpha2']
    A = (-a * al2 * math.cos(th2) + a * w2 * w2 * math.sin(th2)
         + b * w3 * w3 * math.sin(th3) - c * w4 * w4 * math.sin(th4))
    B = (-a * al2 * math.sin(th2) - a * w2 * w2 * math.cos(th2)
         - b * w3 * w3 * math.cos(th3) + c * w4 * w4 * math.cos(th4))
    den = math.sin(th3 - th4)
    if abs(den) < 1e-12:
        raise MechError('toggle position: sin(theta3 - theta4) = 0, the acceleration is singular')
    al3 = (B * math.cos(th4) - A * math.sin(th4)) / (b * den)
    al4 = (B * math.cos(th3) - A * math.sin(th3)) / (c * den)
    return {'alpha3': al3, 'alpha4': al4}


def _slider_crank(values):
    """In-line/offset slider-crank position, velocity and acceleration (constant crank speed).

    geometry = [r, L, e]: crank radius r, connecting-rod length L, offset e (crank pivot to slider
    axis; 0 for in-line). g = r sin(theta2) - e; h = L cos(phi) = +/-sqrt(L^2 - g^2), signed by the
    branch (open: h > 0, crossed: h < 0); phi = atan2(g, h) so phi agrees with its omega_phi/alpha_phi.
    """
    geom = values['geometry']
    if not isinstance(geom, list) or len(geom) != 3:
        raise MechError('geometry: give [r, L, e] (crank radius, conrod length, offset) in m')
    r, L, e = (float(x) for x in geom)
    th = values['theta2']
    w = values['omega']
    br = 1.0 if values['branch'] == 'open' else -1.0
    g = r * math.sin(th) - e
    if abs(g) > L:
        raise MechError('slider-crank cannot assemble: |r sin(theta2) - e| = %.4g > L = %.4g' % (abs(g), L))
    st = math.sqrt(L * L - g * g)
    h = br * st                      # L cos(phi), signed by the branch
    x = r * math.cos(th) + h
    v = -r * w * math.sin(th) - r * w * math.cos(th) * g / h
    a = (-r * w * w * math.cos(th) + r * w * w * math.sin(th) * g / h
         - r * r * w * w * math.cos(th) ** 2 * L * L / (h ** 3))
    phi = math.atan2(g, h)
    wphi = r * w * math.cos(th) / h
    aphi = r * w * w * (-math.sin(th) / h + r * g * math.cos(th) ** 2 / (h ** 3))
    return {'x': x, 'v': v, 'a': a, 'phi': phi, 'omega_phi': wphi, 'alpha_phi': aphi}


def _cam_profile(values):
    """Follower displacement/velocity/acceleration/jerk for a standard rise motion law.

    law: 'uniform' (constant velocity), 'shm' (simple harmonic), 'cycloidal' or 'poly345' (3-4-5
    polynomial). All laws act on the rise interval theta in [0, beta]; dwells are outside (not modelled).
    """
    h, beta, th, w = values['h'], values['beta'], values['theta'], values['omega']
    law = values['law']
    if law not in ('uniform', 'shm', 'cycloidal', 'poly345'):
        raise MechError('law must be uniform, shm (simple harmonic), cycloidal or poly345 (3-4-5 polynomial)')
    k = {'uniform': 1, 'shm': 2, 'cycloidal': 3, 'poly345': 4}[law]
    if th < -1e-12 or th > beta + 1e-12:
        raise MechError('theta must lie within the rise interval [0, beta]')
    y = th / beta
    if k == 1:
        s = h * y
        v = h * w / beta
        a = 0.0
        j = 0.0
    elif k == 2:
        s = h / 2.0 * (1.0 - math.cos(PI * y))
        v = PI * h * w / (2.0 * beta) * math.sin(PI * y)
        a = PI ** 2 * h * w * w / (2.0 * beta ** 2) * math.cos(PI * y)
        j = -PI ** 3 * h * w ** 3 / (2.0 * beta ** 3) * math.sin(PI * y)
    elif k == 3:
        s = h * (y - math.sin(TWOPI * y) / TWOPI)
        v = h * w / beta * (1.0 - math.cos(TWOPI * y))
        a = TWOPI * h * w * w / (beta ** 2) * math.sin(TWOPI * y)
        j = 4.0 * PI ** 2 * h * w ** 3 / (beta ** 3) * math.cos(TWOPI * y)
    else:
        s = h * (10.0 * y ** 3 - 15.0 * y ** 4 + 6.0 * y ** 5)
        v = h * w / beta * (30.0 * y ** 2 - 60.0 * y ** 3 + 30.0 * y ** 4)
        a = h * w * w / (beta ** 2) * (60.0 * y - 180.0 * y ** 2 + 120.0 * y ** 3)
        j = h * w ** 3 / (beta ** 3) * (60.0 - 360.0 * y + 360.0 * y ** 2)
    return {'s': s, 'v': v, 'a': a, 'j': j}


def _gear_train_value(values):
    """Signed train value of a simple or compound gear train (Willis-free; planetary uses its own calc)."""
    drivers = values['drivers']
    driven = values['driven']
    ext = int(values['external_meshes'])
    if not isinstance(drivers, list) or not drivers or not isinstance(driven, list) or not driven:
        raise MechError('drivers and driven need a non-empty list of tooth counts each')
    num, den = 1.0, 1.0
    for z in drivers:
        num *= float(z)
    for z in driven:
        den *= float(z)
    sign = -1.0 if ext % 2 else 1.0
    return {'train_value': sign * num / den}


def _balance_one_plane(values):
    """Resultant unbalance (m*r and angle) of rotating masses in one plane."""
    masses = values['masses']
    if not isinstance(masses, list) or not masses:
        raise MechError('masses: give [[m (kg), r (m), theta (deg)], ...]')
    Fx = Fy = 0.0
    for row in masses:
        if len(row) != 3:
            raise MechError('each mass row must be [m_kg, r_m, theta_deg]')
        m, r, th = float(row[0]), float(row[1]), math.radians(float(row[2]))
        Fx += m * r * math.cos(th)
        Fy += m * r * math.sin(th)
    return {'U': math.hypot(Fx, Fy), 'theta_U': math.atan2(Fy, Fx) % TWOPI}


def _balance_two_plane(values):
    """Two-plane (dynamic) balancing correction masses.

    masses = [[m (kg), r (m), theta (deg), z (m)], ...] with z the axial position; planes = [z_A, z_B].
    Solves force balance (sum u + u_A + u_B = 0) and moment balance (sum z*u + z_A*u_A + z_B*u_B = 0)
    in complex form for the correction m*r products u_A, u_B.
    """
    masses = values['masses']
    planes = values['planes']
    if not isinstance(masses, list) or not masses or not isinstance(planes, list) or len(planes) != 2:
        raise MechError('masses [[m,r,theta_deg,z], ...] and planes [z_A, z_B] required')
    zA, zB = float(planes[0]), float(planes[1])
    if abs(zB - zA) < 1e-12:
        raise MechError('correction planes must have distinct axial positions')
    F = 0j
    M = 0j
    for row in masses:
        if len(row) != 4:
            raise MechError('each mass row must be [m_kg, r_m, theta_deg, z_m]')
        m, r, th, z = float(row[0]), float(row[1]), math.radians(float(row[2])), float(row[3])
        u = m * r * complex(math.cos(th), math.sin(th))
        F += u
        M += z * u
    uA = (M - zB * F) / (zB - zA)
    uB = -F - uA
    return {'uA_mag': abs(uA), 'uA_ang': math.atan2(uA.imag, uA.real) % TWOPI,
            'uB_mag': abs(uB), 'uB_ang': math.atan2(uB.imag, uB.real) % TWOPI}


def _slider_crank_dynamic(values):
    """Dynamic (inertia + gas) crank torque and pin/bearing forces of a slider-crank.

    geometry = [r, L, e] (m); omega (rad/s, signed); F_gas (N) is the gas force on the piston, positive
    pushing the piston TOWARD the crank (so the quasi-static limit matches ``crank_effort``); m_piston,
    m_rod (kg) with its CG at l_cg (m) from the crank pin, and m_crankpin (kg) the rotating mass lumped
    at the crank radius. The rod is replaced by two lumped end masses that preserve its total mass and CG
    (piston side m_rod l_cg/L reciprocates, crank side m_rod(1 - l_cg/L) rotates). Piston net force in +x
    is F_gas + m_recip*a toward the crank (a positive in +x, away from the crank) with the exact slider
    acceleration a, and the crank torque follows from the power
    balance T*omega = (net piston force)*(piston velocity), i.e. T = F_rod r (g cos(theta2) + h sin(theta2))/L.
    """
    geom = values['geometry']
    if not isinstance(geom, list) or len(geom) != 3:
        raise MechError('geometry: give [r, L, e] (crank radius, conrod length, offset) in m')
    r, L, e = (float(x) for x in geom)
    th = values['theta2']
    w = values['omega']
    F_gas = values['F_gas']
    m_piston = values['m_piston']
    m_rod = values['m_rod']
    l_cg = values['l_cg']
    m_crankpin = values['m_crankpin']
    if not (0.0 <= l_cg <= L):
        raise MechError('l_cg (rod CG distance from the crank pin) must lie in [0, L]')
    g = r * math.sin(th) - e
    if abs(g) > L:
        raise MechError('slider-crank cannot assemble: |r sin(theta2) - e| > L')
    h = math.sqrt(L * L - g * g)               # open branch (h = L cos(phi) > 0)
    a = (-r * w * w * math.cos(th) + r * w * w * math.sin(th) * g / h
         - r * r * w * w * math.cos(th) ** 2 * L * L / (h ** 3))
    m_recip = m_piston + m_rod * (l_cg / L)
    m_rot = m_crankpin + m_rod * (1.0 - l_cg / L)
    F_rod = (m_recip * a + F_gas) * L / h      # rod axial force, + = compression
    F_wall = F_rod * g / L                     # cylinder wall side thrust on the piston (+y)
    O2x = F_rod * h / L - m_rot * r * w * w * math.cos(th)
    O2y = -F_rod * g / L - m_rot * r * w * w * math.sin(th)
    T = F_rod * r * (g * math.cos(th) + h * math.sin(th)) / L
    return {'T': T, 'F_rod': F_rod, 'F_wall': F_wall, 'O2x': O2x, 'O2y': O2y,
            'm_recip': m_recip, 'm_rot': m_rot}


def _slider_dynamic_chart(values):
    """Crank torque and pin/bearing forces vs crank angle over one revolution (example defaults)."""
    geom = values['geometry']
    r, L, e = (float(x) for x in geom)
    w = values['omega']
    F_gas = values['F_gas']
    m_piston = values['m_piston']
    m_rod = values['m_rod']
    l_cg = values['l_cg']
    m_crankpin = values['m_crankpin']
    xs = [TWOPI * i / 180.0 for i in range(181)]
    xdeg = [math.degrees(t) for t in xs]
    TT, Fr, Fw, O2x, O2y = [], [], [], [], []
    for t in xs:
        g = r * math.sin(t) - e
        h = math.sqrt(L * L - g * g)  # assembles over the full revolution for r + |e| <= L
        a = (-r * w * w * math.cos(t) + r * w * w * math.sin(t) * g / h
             - r * r * w * w * math.cos(t) ** 2 * L * L / (h ** 3))
        m_recip = m_piston + m_rod * (l_cg / L)
        m_rot = m_crankpin + m_rod * (1.0 - l_cg / L)
        F_rod = (m_recip * a + F_gas) * L / h
        TT.append(F_rod * r * (g * math.cos(t) + h * math.sin(t)) / L)
        Fr.append(F_rod)
        Fw.append(F_rod * g / L)
        O2x.append(F_rod * h / L - m_rot * r * w * w * math.cos(t))
        O2y.append(-F_rod * g / L - m_rot * r * w * w * math.sin(t))
    return [charts.chart('Crank torque (dynamic) vs crank angle', 'crank theta2 [deg]', 'T [N*m]',
                         [charts.series('crank torque T', xdeg, TT)]),
            charts.chart('Pin and bearing forces vs crank angle', 'crank theta2 [deg]', 'force [N]',
                         [charts.series('rod force F_rod', xdeg, Fr),
                          charts.series('wall thrust F_wall', xdeg, Fw),
                          charts.series('bearing O2x', xdeg, O2x),
                          charts.series('bearing O2y', xdeg, O2y)])]


# ---------------------------------------------------------------------------------------------------
# Chart builders (sweep over a crank revolution / cam rise)
# ---------------------------------------------------------------------------------------------------
def _fourbar_chart(values):
    links = values['links']
    a, b, c, d = (float(x) for x in links)
    xs = [TWOPI * i / 180.0 for i in range(181)]
    xdeg = [math.degrees(t) for t in xs]
    th4o, th4c, mus = [], [], []
    for t in xs:
        th3, th4o_t, s = _theta3(a, b, c, d, t, -1.0)  # open: theta4 = phi - beta
        th4o.append(math.degrees(th4o_t) % 360.0)
        th4c.append(math.degrees(_theta3(a, b, c, d, t, 1.0)[1]) % 360.0)  # crossed: phi + beta
        mus.append(math.degrees(math.acos(max(-1.0, min(1.0, (b * b + c * c - s * s) / (2.0 * b * c))))))
    return charts.chart('Four-bar output and transmission angle', 'crank theta2 [deg]', 'angle [deg]',
                        [charts.series('theta4 open', xdeg, th4o),
                         charts.series('theta4 crossed', xdeg, th4c),
                         charts.series('transmission mu', xdeg, mus)])


def _slider_chart(values):
    geom = values['geometry']
    r, L, e = (float(x) for x in geom)
    br = 1.0 if values['branch'] == 'open' else -1.0
    w = values['omega']
    xs = [TWOPI * i / 180.0 for i in range(181)]
    xdeg = [math.degrees(t) for t in xs]
    xx, vv, aa = [], [], []
    for t in xs:
        g = r * math.sin(t) - e
        h = br * math.sqrt(L * L - g * g)
        xx.append(r * math.cos(t) + h)
        vv.append(-r * w * math.sin(t) - r * w * math.cos(t) * g / h)
        aa.append(-r * w * w * math.cos(t) + r * w * w * math.sin(t) * g / h
                  - r * r * w * w * math.cos(t) ** 2 * L * L / (h ** 3))
    return charts.chart('Slider-crank motion over one revolution', 'crank theta2 [deg]',
                        'x [m], v [m/s], a [m/s^2]',
                        [charts.series('position x', xdeg, xx),
                         charts.series('velocity v', xdeg, vv),
                         charts.series('acceleration a', xdeg, aa)])


def _cam_chart(values):
    h, beta, w = values['h'], values['beta'], values['omega']
    k = {'uniform': 1, 'shm': 2, 'cycloidal': 3, 'poly345': 4}[values['law']]
    xs = [beta * i / 180.0 for i in range(181)]
    xdeg = [math.degrees(t) for t in xs]
    ss, vv, aa, jj = [], [], [], []
    for t in xs:
        y = t / beta
        if k == 1:
            ss.append(h * y); vv.append(h * w / beta); aa.append(0.0); jj.append(0.0)
        elif k == 2:
            ss.append(h / 2.0 * (1.0 - math.cos(PI * y)))
            vv.append(PI * h * w / (2.0 * beta) * math.sin(PI * y))
            aa.append(PI ** 2 * h * w * w / (2.0 * beta ** 2) * math.cos(PI * y))
            jj.append(-PI ** 3 * h * w ** 3 / (2.0 * beta ** 3) * math.sin(PI * y))
        elif k == 3:
            ss.append(h * (y - math.sin(TWOPI * y) / TWOPI))
            vv.append(h * w / beta * (1.0 - math.cos(TWOPI * y)))
            aa.append(TWOPI * h * w * w / (beta ** 2) * math.sin(TWOPI * y))
            jj.append(4.0 * PI ** 2 * h * w ** 3 / (beta ** 3) * math.cos(TWOPI * y))
        else:
            ss.append(h * (10.0 * y ** 3 - 15.0 * y ** 4 + 6.0 * y ** 5))
            vv.append(h * w / beta * (30.0 * y ** 2 - 60.0 * y ** 3 + 30.0 * y ** 4))
            aa.append(h * w * w / (beta ** 2) * (60.0 * y - 180.0 * y ** 2 + 120.0 * y ** 3))
            jj.append(h * w ** 3 / (beta ** 3) * (60.0 - 360.0 * y + 360.0 * y ** 2))
    return charts.chart('Cam follower motion over the rise', 'cam angle theta [deg]',
                        's [m], v [m/s], a [m/s^2], j [m/s^3]',
                        [charts.series('s', xdeg, ss), charts.series('v', xdeg, vv),
                         charts.series('a', xdeg, aa), charts.series('j', xdeg, jj)])


CALCS = [
    # ---- degrees of freedom ----------------------------------------------------------------------
    Calc('dof_planar', 'Degrees of freedom of a planar mechanism (Grübler)',
         vars=[V('F', '', 'degrees of freedom', signed=True), V('n', '', 'number of links (incl. ground)', integer=True),
               V('j1', '', 'number of full (1-DOF) joints', integer=True),
               V('j2', '', 'number of half (2-DOF) joints', integer=True)],
         equations=['F = 3*(n - 1) - 2*j1 - j2'], outputs=['F'],
         example={'n': 4, 'j1': 4, 'j2': 0},
         keywords=['mobility', 'gruebler', 'kutzbach'],
         assumptions=['Planar Grübler equation. F = 1 is a constrained mechanism, F = 0 a structure, '
                      'F < 0 over-constrained. Does not detect passive (redundant) DOF.'],
         refs=['Grübler mobility criterion for planar linkages (F = 3(n-1) - 2 j1 - j2)']),

    Calc('dof_spatial', 'Degrees of freedom of a spatial mechanism (Kutzbach)',
         vars=[V('F', '', 'degrees of freedom', signed=True), V('n', '', 'number of links (incl. ground)', integer=True),
               V('j1', '', 'joints with 1 DOF', integer=True), V('j2', '', 'joints with 2 DOF', integer=True),
               V('j3', '', 'joints with 3 DOF', integer=True), V('j4', '', 'joints with 4 DOF', integer=True),
               V('j5', '', 'joints with 5 DOF', integer=True)],
         equations=['F = 6*(n - 1) - 5*j1 - 4*j2 - 3*j3 - 2*j4 - j5'], outputs=['F'],
         example={'n': 4, 'j1': 3, 'j2': 1, 'j3': 0, 'j4': 0, 'j5': 0},
         keywords=['mobility', 'spatial', 'gruebler', 'kutzbach'],
         assumptions=['Spatial (Kutzbach) mobility. A j1 joint is a 1-DOF joint (5 constraints), j5 a '
                      '5-DOF joint (1 constraint). Planar mechanisms give 0 here; use dof_planar.'],
         refs=['Kutzbach (Grübler) mobility criterion for spatial linkages']),

    # ---- Grashof, four-bar -----------------------------------------------------------------------
    Calc('grashof', 'Grashof condition and four-bar classification',
         vars=[V('links', '', '[a, b, c, d] link lengths in m (crank, coupler, rocker, ground)', kind='json'),
               V('s', 'm', 'shortest link length'), V('l', 'm', 'longest link length'),
               V('p', 'm', 'one middle link length'), V('q', 'm', 'other middle link length'),
               V('s_plus_l', 'm', 'shortest + longest'), V('p_plus_q', 'm', 'sum of the two middle links'),
               V('is_grashof', '', '1 if s + l <= p + q, else 0'),
               V('type', '', 'linkage class (string)', kind='json')],
         procedure=_grashof, outputs=['s', 'l', 'p', 'q', 's_plus_l', 'p_plus_q', 'is_grashof', 'type'],
         example={'links': [2.0, 7.0, 9.0, 6.0]},
         keywords=['crank-rocker', 'double-crank', 'double-rocker', 'drag-link'],
         assumptions=['Class names use the convention a = input crank, b = coupler, c = output rocker, '
                      'd = ground. Shortest link next to ground gives a crank-rocker (a) or rocker-crank '
                      '(c); shortest ground gives double-crank; shortest coupler a Grashof double-rocker. '
                      'Equal-length ties are ambiguous.'],
         refs=['Grashof condition s + l <= p + q and the resulting four-bar classes (recalled)']),

    Calc('fourbar_transmission', 'Four-bar transmission angle and coupler-ground diagonal',
         vars=[V('s', 'm', 'diagonal distance A-O4'), V('mu', 'rad', 'transmission angle (coupler vs rocker)', hi=PI, label='μ'),
               V('a', 'm', 'crank length'), V('b', 'm', 'coupler length'), V('c', 'm', 'rocker length'),
               V('d', 'm', 'ground length'), V('theta2', 'rad', 'crank angle', lo=0.0, hi=TWOPI, label='θ2')],
         equations=['s = sqrt(a**2 + d**2 - 2*a*d*cos(theta2))',
                    'mu = acos((b**2 + c**2 - s**2)/(2*b*c))'],
         outputs=['s', 'mu'], example={'a': 2.0, 'b': 7.0, 'c': 9.0, 'd': 6.0, 'theta2': math.radians(60)},
         keywords=['transmission angle', 'quality', 'toggle'],
         assumptions=['mu is the angle between coupler and rocker at joint B; 90 deg is ideal. A value '
                      'near 0 or 180 deg is a toggle (poor transmission). Requires the four-bar to assemble '
                      '(cos(mu) within [-1, 1]).'],
         refs=['Law of cosines on the coupler-rocker triangle: cos(mu) = (b^2 + c^2 - s^2)/(2 b c)']),

    Calc('fourbar_toggle', 'Four-bar toggle (dead-centre) crank angles and rocker extremes',
         vars=[V('links', '', '[a, b, c, d] link lengths in m (crank, coupler, rocker, ground)', kind='json'),
               V('theta_t1', 'rad', 'first toggle crank angle (rad)', lo=0.0, hi=TWOPI, label='θt1'),
               V('theta_t2', 'rad', 'second toggle crank angle (rad)', lo=0.0, hi=TWOPI, label='θt2'),
               V('theta4_1', 'rad', 'rocker angle at toggle 1 (rad, [0, 2pi))', lo=0.0, hi=TWOPI, label='θ4,1'),
               V('theta4_2', 'rad', 'rocker angle at toggle 2 (rad, [0, 2pi))', lo=0.0, hi=TWOPI, label='θ4,2'),
               V('time_ratio', '', 'quick-return time ratio (>= 1)', lo=1.0)],
         procedure=_fourbar_toggle, outputs=['theta_t1', 'theta_t2', 'theta4_1', 'theta4_2', 'time_ratio'],
         example={'links': [2.0, 7.0, 9.0, 6.0]},
         keywords=['toggle', 'dead centre', 'quick return', 'limiting positions'],
         assumptions=['For a crank-rocker: the rocker reverses where the crank and coupler are collinear. '
                      'The two toggle crank angles are found numerically; time_ratio = larger crank arc / '
                      'smaller arc is the ratio of stroke times at constant crank speed.'],
         refs=['Toggle positions where the crank and coupler are collinear (found by root-finding sin(theta2 - theta3) = 0)']),

    Calc('fourbar_position', 'Four-bar position (vector-loop closure), open or crossed branch',
         vars=[V('links', '', '[a, b, c, d] link lengths in m (crank, coupler, rocker, ground)', kind='json'),
               V('theta2', 'rad', 'crank angle', lo=0.0, hi=TWOPI, label='θ2'),
               V('branch', '', 'assembly branch (open or crossed)', kind='text', choices=['open', 'crossed']),
               V('theta4', 'rad', 'rocker angle', signed=True, label='θ4'),
               V('theta3', 'rad', 'coupler angle', signed=True, label='θ3'),
               V('mu', 'rad', 'transmission angle', hi=PI, label='μ'),
               V('s', 'm', 'diagonal distance A-O4')],
         procedure=_fourbar_position, outputs=['theta4', 'theta3', 'mu', 's'],
         example={'links': [2.0, 7.0, 9.0, 6.0], 'theta2': math.radians(60), 'branch': 'open'},
         plots=[{'id': 'position', 'title': 'Output and transmission angle vs crank angle', 'fn': _fourbar_chart}],
         keywords=['four bar', 'vector loop', 'open', 'crossed', 'coupler'],
         assumptions=['Planar vector loop; branch "open" gives theta4 = phi - beta and "crossed" gives '
                      'theta4 = phi + beta (Norton half-angle form). theta4/theta3 are signed angles '
                      '(CCW from the ground line). Coupler-point trajectories are not computed.'],
         refs=['Four-bar vector-loop closure solved via the coupler-rocker triangle (law of cosines)']),

    Calc('fourbar_velocity', 'Four-bar angular velocities of coupler and rocker',
         vars=[V('omega3', 'rad/s', 'coupler angular velocity', signed=True, label='ω3'),
               V('omega4', 'rad/s', 'rocker angular velocity', signed=True, label='ω4'),
               V('a', 'm', 'crank length'), V('b', 'm', 'coupler length'), V('c', 'm', 'rocker length'),
               V('theta2', 'rad', 'crank angle', lo=0.0, hi=TWOPI, label='θ2'),
               V('theta3', 'rad', 'coupler angle', signed=True, label='θ3'),
               V('theta4', 'rad', 'rocker angle', signed=True, label='θ4'),
               V('omega2', 'rad/s', 'crank angular velocity (enter rad/s or rpm)', signed=True, label='ω2')],
         equations=['omega3 = a*omega2*sin(theta4 - theta2)/(b*sin(theta3 - theta4))',
                    'omega4 = a*omega2*sin(theta2 - theta3)/(c*sin(theta4 - theta3))'],
         outputs=['omega3', 'omega4'],
         example={'a': 2.0, 'b': 7.0, 'c': 9.0, 'theta2': math.radians(60),
                  'theta3': 1.291344350, 'theta4': 1.918800123, 'omega2': 30.0},
         keywords=['angular velocity', 'velocity analysis'],
         assumptions=['theta3 and theta4 must be consistent with the position solution at theta2 (see '
                      'fourbar_position); the example uses the open branch (theta4 = phi - beta). Velocities '
                      'are signed; a rocker velocity of zero is a toggle.'],
         refs=['Four-bar velocity relations from differentiating the vector loop']),

    Calc('fourbar_acceleration', 'Four-bar angular accelerations of coupler and rocker',
         vars=[V('alpha3', 'rad/s^2', 'coupler angular acceleration', signed=True, label='α3'),
               V('alpha4', 'rad/s^2', 'rocker angular acceleration', signed=True, label='α4'),
               V('a', 'm', 'crank length'), V('b', 'm', 'coupler length'), V('c', 'm', 'rocker length'),
               V('theta2', 'rad', 'crank angle', lo=0.0, hi=TWOPI, label='θ2'),
               V('theta3', 'rad', 'coupler angle', signed=True, label='θ3'),
               V('theta4', 'rad', 'rocker angle', signed=True, label='θ4'),
               V('omega2', 'rad/s', 'crank angular velocity', signed=True, label='ω2'),
               V('omega3', 'rad/s', 'coupler angular velocity', signed=True, label='ω3'),
               V('omega4', 'rad/s', 'rocker angular velocity', signed=True, label='ω4'),
               V('alpha2', 'rad/s^2', 'crank angular acceleration (0 for constant crank speed)', signed=True, label='α2')],
         procedure=_fourbar_accel, outputs=['alpha3', 'alpha4'],
         example={'a': 2.0, 'b': 7.0, 'c': 9.0, 'theta2': math.radians(60),
                  'theta3': 1.291344350340649, 'theta4': 1.9188001232638396,
                  'omega2': 30.0, 'omega3': -11.17422032596634, 'omega4': -2.7449480217310938,
                  'alpha2': 0.0},
         keywords=['angular acceleration', 'acceleration analysis'],
         assumptions=['Position and velocity inputs must be mutually consistent (from fourbar_position and '
                      'fourbar_velocity); the example uses the open branch. alpha2 is the crank angular '
                      'acceleration (0 for constant crank speed). Singular at a toggle '
                      '(sin(theta3 - theta4) = 0).'],
         refs=['Four-bar acceleration relations from differentiating the vector loop twice']),

    Calc('fourbar_ma', 'Four-bar mechanical advantage and velocity ratio (ideal)',
         vars=[V('MA', '', 'mechanical advantage T4/T2', signed=True),
               V('VR', '', 'velocity ratio omega2/omega4', signed=True),
               V('a', 'm', 'crank length'), V('c', 'm', 'rocker length'),
               V('theta2', 'rad', 'crank angle', lo=0.0, hi=TWOPI, label='θ2'),
               V('theta3', 'rad', 'coupler angle', signed=True, label='θ3'),
               V('theta4', 'rad', 'rocker angle', signed=True, label='θ4')],
         equations=['VR = c*sin(theta4 - theta3)/(a*sin(theta2 - theta3))',
                    'MA = -c*sin(theta4 - theta3)/(a*sin(theta2 - theta3))'],
         outputs=['MA', 'VR'], example={'a': 2.0, 'c': 9.0, 'theta2': math.radians(60),
                                        'theta3': 1.291344350, 'theta4': 1.918800123},
         keywords=['mechanical advantage', 'velocity ratio', 'transmission'],
         assumptions=['Ideal (lossless) linkage: MA = T4/T2 = -omega2/omega4. A magnitude |MA| > 1 means '
                      'the linkage multiplies torque; the sign only gives the direction (output torque '
                      'opposing or aiding). MA diverges near a toggle. Angles must match the position '
                      'solution (the example uses the open branch).'],
         refs=['Power balance T2 omega2 + T4 omega4 = 0 with the velocity relation']),

    # ---- slider-crank ----------------------------------------------------------------------------
    Calc('slider_crank', 'Slider-crank position, velocity and acceleration',
         vars=[V('geometry', '', '[r, L, e] in m: crank radius, connecting-rod length, offset (0 = in-line)', kind='json'),
               V('theta2', 'rad', 'crank angle', lo=0.0, hi=TWOPI, label='θ2'),
               V('omega', 'rad/s', 'crank angular velocity (enter rad/s or rpm)', signed=True, label='ω'),
               V('branch', '', 'assembly branch (open or crossed)', kind='text', choices=['open', 'crossed']),
               V('x', 'm', 'slider position from the crank pivot', signed=True),
               V('v', 'm/s', 'slider velocity', signed=True),
               V('a', 'm/s^2', 'slider acceleration (constant crank speed)', signed=True),
               V('phi', 'rad', 'connecting-rod angle', signed=True, label='φ'),
               V('omega_phi', 'rad/s', 'connecting-rod angular velocity', signed=True, label='ωφ'),
               V('alpha_phi', 'rad/s^2', 'connecting-rod angular acceleration', signed=True, label='αφ')],
         procedure=_slider_crank, outputs=['x', 'v', 'a', 'phi', 'omega_phi', 'alpha_phi'],
         example={'geometry': [0.05, 0.2, 0.0], 'theta2': math.radians(30), 'omega': 100.0, 'branch': 'open'},
         plots=[{'id': 'motion', 'title': 'Slider motion over one revolution', 'fn': _slider_chart}],
         keywords=['piston', 'crank', 'connecting rod', 'reciprocating', 'offset'],
         assumptions=['Constant crank speed (crank angular acceleration = 0). x is measured along the '
                      'slider axis from the crank pivot; a positive v/a follow +x. branch "open" gives the '
                      'rod leaning with h = L cos(phi) > 0, "crossed" the mirror assembly (h < 0); the rod '
                      'angle is phi = atan2(g, h). Requires |r sin(theta2) - e| <= L.'],
         refs=['Exact slider-crank kinematics from the vector loop (position, velocity, acceleration)']),

    # ---- cams ------------------------------------------------------------------------------------
    Calc('cam_profile', 'Cam follower motion: uniform, SHM, cycloidal or 3-4-5 polynomial rise',
         vars=[V('h', 'm', 'follower lift over the rise'), V('beta', 'rad', 'rise angle (cam rotation)', lo=0.01, hi=TWOPI, label='β'),
               V('theta', 'rad', 'cam angle within the rise (0..beta)', lo=0.0, hi=TWOPI, label='θ'),
               V('omega', 'rad/s', 'cam angular velocity (enter rad/s or rpm)', lo=0.0, label='ω'),
               V('law', '', 'motion law: uniform, shm, cycloidal or poly345 (3-4-5 polynomial)', kind='text',
                 choices=['uniform', 'shm', 'cycloidal', 'poly345']),
               V('s', 'm', 'follower displacement'), V('v', 'm/s', 'follower velocity'),
               V('a', 'm/s^2', 'follower acceleration', signed=True), V('j', 'm/s^3', 'follower jerk', signed=True)],
         procedure=_cam_profile, outputs=['s', 'v', 'a', 'j'],
         example={'h': 0.01, 'beta': math.radians(60), 'theta': math.radians(20), 'omega': 50.0, 'law': 'shm'},
         plots=[{'id': 'profile', 'title': 'Follower s, v, a, j over the rise', 'fn': _cam_chart}],
         keywords=['cam', 'follower', 'cycloidal', 'simple harmonic', '3-4-5', 'jerk'],
         assumptions=['A rise from 0 to h over cam angle [0, beta]; dwells outside that interval are not '
                      'modelled. The uniform law has an infinite acceleration/jerk impulse at the ends and '
                      'is included for comparison, not for high-speed design. The 3-4-5 polynomial has '
                      'finite (but non-zero at the ends) jerk; the cycloidal law has zero end acceleration.'],
         refs=['Standard cam motion laws: uniform velocity, SHM, cycloidal, 3-4-5 polynomial (recalled)']),

    Calc('cam_pressure_angle', 'Cam pressure angle for a translating roller follower',
         vars=[V('alpha', 'rad', 'pressure angle (0 is ideal, limit ~30 deg)', hi=PI / 2, label='α'),
               V('ds_dtheta', 'm', 'slope ds/d(theta) at the contact point (m per rad)'),
               V('R_b', 'm', 'base circle radius'), V('R_r', 'm', 'roller radius'),
               V('s', 'm', 'follower displacement at the contact point')],
         equations=['alpha = atan(ds_dtheta/(R_b + R_r + s))'], outputs=['alpha'],
         example={'ds_dtheta': 0.008, 'R_b': 0.02, 'R_r': 0.01, 's': 0.004},
         keywords=['pressure angle', 'cam', 'roller follower', 'undercut'],
         assumptions=['Translating roller follower with no offset. alpha = atan((ds/dtheta)/(R_b + R_r + s)). '
                      'Keep alpha below about 30 deg to avoid jamming and excessive side thrust.'],
         refs=['tan(alpha) = (ds/dtheta)/(R_b + R_r + s) for a radial translating roller follower']),

    Calc('cam_base_circle', 'Minimum cam base-circle radius for a pressure-angle limit',
         vars=[V('R_b', 'm', 'minimum base circle radius (negative = no base circle needed)', signed=True),
               V('ds_dtheta', 'm', 'slope ds/d(theta) at the worst point (m per rad)'),
               V('alpha_max', 'rad', 'maximum allowed pressure angle', hi=PI / 2, label='αmax'),
               V('R_r', 'm', 'roller radius'), V('s', 'm', 'follower displacement at the worst point')],
         equations=['R_b = ds_dtheta/tan(alpha_max) - R_r - s'], outputs=['R_b'],
         example={'ds_dtheta': 0.02, 'alpha_max': math.radians(30), 'R_r': 0.01, 's': 0.004},
         keywords=['base circle', 'cam size', 'pressure angle limit'],
         assumptions=['Evaluate at the point of maximum ds/d(theta) for the worst case. A negative R_b means '
                      'the chosen roller/slope already satisfies the limit with a zero base circle.'],
         refs=['R_b = (ds/dtheta)/tan(alpha_max) - R_r - s, from the pressure-angle relation']),

    # ---- gear trains, Geneva, quick return -------------------------------------------------------
    Calc('gear_pair', 'Simple gear pair speed ratio',
         vars=[V('omega_out', 'rad/s', 'driven gear angular velocity', signed=True, label='ωout'),
               V('omega_in', 'rad/s', 'driving gear angular velocity', signed=True, label='ωin'),
               V('T_in', '', 'driving gear teeth', integer=True), V('T_out', '', 'driven gear teeth', integer=True)],
         equations=['omega_out = -omega_in*T_in/T_out'], outputs=['omega_out'],
         example={'omega_in': 1500 * 2 * PI / 60, 'T_in': 20, 'T_out': 40},
         keywords=['gear ratio', 'spur gear', 'speed ratio'],
         assumptions=['External gear pair (reverses direction). Internal mesh drops the minus sign. '
                      'Tooth geometry and stress are in the gears section.'],
         refs=['omega_out/omega_in = -T_in/T_out for an external gear pair']),

    Calc('gear_train_value', 'Gear train value of a simple or compound train',
         vars=[V('drivers', '', 'tooth counts of the driving gears', kind='json'),
               V('driven', '', 'tooth counts of the driven gears', kind='json'),
               V('external_meshes', '', 'number of external (reversing) meshes', kind='text',
                 choices=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']),
               V('train_value', '', 'signed train value e = omega_out/omega_in', signed=True)],
         procedure=_gear_train_value, outputs=['train_value'],
         example={'drivers': [20, 30], 'driven': [40, 50], 'external_meshes': '2'},
         keywords=['compound gear train', 'train value', 'speed ratio', 'reverted'],
         assumptions=['Train value e = (product of driver teeth)/(product of driven teeth), sign = (-1)^(number '
                      'of external meshes). An internal mesh does not reverse. Planetary (epicyclic) trains use '
                      'planetary_trains (Willis).'],
         refs=['Gear train value e = prod(Z_driver)/prod(Z_driven) * (-1)^(external meshes)']),

    Calc('planetary_trains', 'Planetary (epicyclic) gear train (Willis equation)',
         vars=[V('omega_S', 'rad/s', 'sun angular velocity', signed=True, label='ωS'),
               V('omega_R', 'rad/s', 'ring angular velocity', signed=True, label='ωR'),
               V('omega_C', 'rad/s', 'carrier (arm) angular velocity', signed=True, label='ωC'),
               V('Z_S', '', 'sun gear teeth', integer=True), V('Z_R', '', 'ring gear teeth', integer=True)],
         equations=['omega_S - omega_C = -(Z_R/Z_S)*(omega_R - omega_C)'], outputs=['omega_S'],
         example={'omega_R': 0.0, 'omega_C': 0.0, 'Z_S': 20, 'Z_R': 60},
         keywords=['epicyclic', 'planetary', 'sun', 'ring', 'carrier', 'willis'],
         assumptions=['Willis equation for a sun-ring-carrier train (one internal ring mesh, one external '
                      'sun-planet mesh). Hold one member fixed (set its speed to 0) to get the two remaining '
                      'speeds. Sign convention: sun and ring counter-rotate relative to the carrier.'],
         refs=['Willis equation (omega_S - omega_C)/(omega_R - omega_C) = -Z_R/Z_S (recalled)']),

    Calc('geneva', 'Geneva mechanism indexing geometry',
         vars=[V('n', '', 'number of slots in the Geneva wheel', lo=3.0, hi=18.0, integer=True),
               V('theta_index', 'rad', 'wheel index angle per cycle', hi=PI, label='θidx'),
               V('crank_angle', 'rad', 'crank rotation during indexing', hi=PI, label='φidx'),
               V('dwell_angle', 'rad', 'crank rotation during dwell', lo=PI, hi=TWOPI, label='φdwell'),
               V('time_ratio', '', 'fraction of each cycle spent indexing')],
         equations=['theta_index = 2*pi/n', 'crank_angle = pi - 2*pi/n',
                    'dwell_angle = 2*pi - crank_angle', 'time_ratio = crank_angle/(2*pi)'],
         outputs=['theta_index', 'crank_angle', 'dwell_angle', 'time_ratio'],
         example={'n': 4},
         keywords=['geneva', 'indexing', 'intermittent', 'slot'],
         assumptions=['External Geneva wheel. The wheel indexes 2*pi/n per cycle while the crank turns '
                      'pi - 2*pi/n; the rest is dwell. n >= 3 slots (4 and 6 are common).'],
         refs=['Geneva indexing geometry: crank angle of action = pi - 2*pi/n']),

    Calc('quick_return', 'Quick-return time ratio',
         vars=[V('Q', '', 'quick-return ratio (cutting/return time, >= 1)', lo=1.0),
               V('phi_cut', 'rad', 'crank angle swept during the cutting (slow) stroke', hi=TWOPI, label='φcut'),
               V('phi_return', 'rad', 'crank angle swept during the return (fast) stroke', hi=TWOPI, label='φret')],
         equations=['Q = phi_cut/phi_return', 'phi_cut + phi_return = 2*pi'],
         outputs=['Q', 'phi_return'], example={'phi_cut': math.radians(220)},
         keywords=['quick return', 'time ratio', 'shaper', 'whitworth'],
         assumptions=['Constant crank speed: Q = phi_cut/phi_return with phi_cut + phi_return = 2*pi. For a '
                      'slotted-lever (Whitworth) mechanism feed the two crank arcs from fourbar_toggle.'],
         refs=['Quick-return time ratio Q = phi_cut/phi_return']),

    # ---- forces and balancing --------------------------------------------------------------------
    Calc('fourbar_static_force', 'Four-bar static input torque from output torque (virtual work)',
         vars=[V('T2', 'N*m', 'input (crank) torque', signed=True),
               V('T4', 'N*m', 'output (rocker) torque', signed=True),
               V('a', 'm', 'crank length'), V('c', 'm', 'rocker length'),
               V('theta2', 'rad', 'crank angle', lo=0.0, hi=TWOPI, label='θ2'),
               V('theta3', 'rad', 'coupler angle', signed=True, label='θ3'),
               V('theta4', 'rad', 'rocker angle', signed=True, label='θ4')],
         equations=['T2 = -T4*a*sin(theta2 - theta3)/(c*sin(theta4 - theta3))'], outputs=['T2'],
         example={'T4': 20.0, 'a': 2.0, 'c': 9.0, 'theta2': math.radians(60),
                  'theta3': 1.291344350, 'theta4': 1.918800123},
         keywords=['static force', 'virtual work', 'input torque', 'transmission angle'],
         assumptions=['Static equilibrium by virtual work (T2·ω2 + T4·ω4 = 0), no friction or inertia, so '
                      'T2 = -T4·(ω4/ω2) = -T4·a·sin(θ2 - θ3)/(c·sin(θ4 - θ3)). Consistent with fourbar_ma '
                      '(T2 = T4/MA). Angles must match the position solution (the example uses the open '
                      'branch).'],
         refs=['Virtual work: T2·ω2 + T4·ω4 = 0, i.e. T2 = -T4·(ω4/ω2) with the four-bar velocity relation']),

    Calc('balance_one_plane', 'Single-plane balancing: resultant unbalance of rotating masses',
         vars=[V('masses', '', 'rows [m (kg), r (m), theta (deg)] of each unbalance', kind='json'),
               V('U', 'kg*m', 'resultant unbalance m*r'), V('theta_U', 'rad', 'resultant unbalance angle (rad, [0, 2pi))', label='θU')],
         procedure=_balance_one_plane, outputs=['U', 'theta_U'],
         example={'masses': [[0.4, 0.1, 0], [0.4, 0.1, 120]]},
         keywords=['balancing', 'static balance', 'counterweight'],
         assumptions=['All masses in one plane: add a counterweight with m_b r_b = U at theta_U + 180 deg to '
                      'achieve static balance. Angle convention CCW from the reference. The centrifugal force '
                      'of a rotating unbalance is in kinetics.centripetal_force.'],
         refs=['Static balance: vector sum of m*r terms equals zero']),

    Calc('balance_two_plane', 'Two-plane (dynamic) balancing corrections',
         vars=[V('masses', '', 'rows [m (kg), r (m), theta (deg), z (m)] of each unbalance', kind='json'),
               V('planes', '', 'axial positions [z_A, z_B] of the two correction planes (m)', kind='json'),
               V('uA_mag', 'kg*m', 'correction m*r magnitude in plane A'),
               V('uA_ang', 'rad', 'correction angle in plane A (rad, [0, 2pi))', label='θA'),
               V('uB_mag', 'kg*m', 'correction m*r magnitude in plane B'),
               V('uB_ang', 'rad', 'correction angle in plane B (rad, [0, 2pi))', label='θB')],
         procedure=_balance_two_plane, outputs=['uA_mag', 'uA_ang', 'uB_mag', 'uB_ang'],
         example={'masses': [[0.3, 0.1, 0, 0.0], [0.3, 0.1, 90, 0.3]], 'planes': [0.1, 0.2]},
         keywords=['dynamic balance', 'two plane', 'balancing', 'rotor'],
         assumptions=['Corrects both force and moment unbalance with two correction planes at axial positions '
                      'z_A and z_B. Results are m*r products; pick a correction radius to get the mass. The '
                      'centrifugal force of a rotating unbalance is in kinetics.centripetal_force.'],
         refs=['Dynamic balance: force and moment equilibrium in complex form (sum u = 0, sum z u = 0)']),

    Calc('reciprocating_unbalance', 'Shaking force of a reciprocating mass (slider-crank)',
         vars=[V('F_x', 'N', 'inertia (shaking) force along the slider axis', signed=True),
               V('m', 'kg', 'reciprocating mass'), V('r', 'm', 'crank radius'), V('L', 'm', 'connecting-rod length'),
               V('omega', 'rad/s', 'crank angular velocity (enter rad/s or rpm)', label='ω'),
               V('theta', 'rad', 'crank angle', lo=0.0, hi=TWOPI, label='θ')],
         equations=['F_x = m*r*omega**2*(cos(theta) + (r/L)*cos(2*theta))'], outputs=['F_x'],
         example={'m': 0.5, 'r': 0.05, 'L': 0.2, 'omega': 3000 * 2 * PI / 60, 'theta': math.radians(30)},
         keywords=['shaking force', 'primary', 'secondary', 'reciprocating unbalance'],
         assumptions=['Approximate (small r/L) primary + secondary inertia force of the reciprocating mass. '
                      'The exact expression uses the rod angle; the r/L cos(2 theta) term is the first-order '
                      'secondary correction.'],
         refs=['Primary and secondary reciprocating unbalance force (recalled small r/L form)']),

    Calc('crank_effort', 'Engine crank effort (turning moment on the crankshaft)',
         vars=[V('T', 'N*m', 'turning moment (crank effort)', signed=True),
               V('F_p', 'N', 'piston (gas) force along the cylinder axis'),
               V('r', 'm', 'crank radius'), V('L', 'm', 'connecting-rod length'),
               V('theta', 'rad', 'crank angle', lo=0.0, hi=TWOPI, label='θ')],
         equations=['T = F_p*r*sin(theta)*(1 + r*cos(theta)/sqrt(L**2 - r**2*sin(theta)**2))'],
         outputs=['T'], example={'F_p': 5000.0, 'r': 0.05, 'L': 0.2, 'theta': math.radians(30)},
         plots=[{'id': 'diagram', 'title': 'Turning-moment diagram vs crank angle',
                 'x': 'theta', 'y': ['T'], 'from': 0.0, 'to': TWOPI, 'n': 180}],
         keywords=['crank effort', 'turning moment', 'engine torque', 'piston force'],
         assumptions=['Exact turning moment for an in-line slider-crank from the tangential connecting-rod '
                      'force at the crank pin; T is positive when the gas force drives the crank in the '
                      'direction of increasing theta (the power-stroke half-revolution). The turning-moment '
                      'diagram is T vs theta over 2*pi.'],
         refs=['Crank effort T = F_p r sin(theta) (1 + r cos(theta)/sqrt(L^2 - r^2 sin^2(theta)))']),

    Calc('slider_crank_dynamic', 'Slider-crank dynamic crank torque and pin/bearing forces',
         vars=[V('geometry', '', '[r, L, e] in m: crank radius, connecting-rod length, offset (0 = in-line)', kind='json'),
               V('theta2', 'rad', 'crank angle', lo=0.0, hi=TWOPI, label='θ2'),
               V('omega', 'rad/s', 'crank angular velocity (enter rad/s or rpm)', signed=True, label='ω'),
               V('F_gas', 'N', 'gas force on the piston, positive pushing toward the crank', signed=True),
               V('m_piston', 'kg', 'piston (reciprocating) mass'),
               V('m_rod', 'kg', 'connecting-rod mass'),
               V('l_cg', 'm', 'rod centre-of-mass distance from the crank pin (toward the piston)', lo=0.0),
               V('m_crankpin', 'kg', 'rotating mass at the crank pin radius (crank unbalance + pin)'),
               V('T', 'N*m', 'turning moment delivered to the crank by the rod (same sign as crank_effort; shaft load torque = −T)', signed=True),
               V('F_rod', 'N', 'connecting-rod axial force (+ = compression)', signed=True),
               V('F_wall', 'N', 'cylinder wall side thrust on the piston (+y)', signed=True),
               V('O2x', 'N', 'crank-pivot bearing reaction, x component', signed=True),
               V('O2y', 'N', 'crank-pivot bearing reaction, y component', signed=True),
               V('m_recip', 'kg', 'total reciprocating mass (piston + rod CG share)'),
               V('m_rot', 'kg', 'total rotating mass at the crank pin (crankpin + rod CG share)')],
         procedure=_slider_crank_dynamic,
         outputs=['T', 'F_rod', 'F_wall', 'O2x', 'O2y', 'm_recip', 'm_rot'],
         example={'geometry': [0.05, 0.2, 0.0], 'theta2': math.radians(30), 'omega': 100.0,
                  'F_gas': 5000.0, 'm_piston': 0.5, 'm_rod': 0.4, 'l_cg': 0.1, 'm_crankpin': 0.2},
         plots=[{'id': 'dynamic', 'title': 'Dynamic crank torque and pin/bearing forces vs crank angle',
                 'fn': _slider_dynamic_chart}],
         no_invert={'m_crankpin': 'appears only in the bearing reactions (the radial crank-pin inertia '
                                  'force has no moment about the crank pivot), so it cannot be recovered '
                                  'from the crank torque T'},
         keywords=['dynamic force analysis', 'inertia', 'gas force', 'piston', 'crank torque', 'bearing'],
         assumptions=['Constant crank speed; the connecting rod is replaced by two lumped end masses that '
                      'preserve its total mass and CG position (piston side m_rod l_cg/L reciprocates, crank '
                      'side m_rod(1 - l_cg/L) rotates). Effective force toward the crank F = F_gas + m_recip·a (a positive in +x, away from the crank) uses the exact '
                      'slider acceleration; the crank torque is T = F dx/d(theta2) from the power balance. '
                      'Quasi-static limit (masses -> 0) reproduces crank_effort. Open (h > 0) branch; '
                      'requires r + |e| <= L to assemble. Bearing reactions include the crank-pin centrifugal '
                      'force m_rot r omega^2.'],
         refs=['Dynamic force analysis of a slider-crank: two-mass rod model and power balance '
               '(T = F dx/d(theta2); recalled)']),
]

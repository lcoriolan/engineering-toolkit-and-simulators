# DESCRIPTION: Translational accelerations in the inertial frame, vectorised over N vehicles (r, v [N,3], float64):
#                central body + J2, J3, J4 zonal harmonics (Vallado 4th ed. eq. 8-30; signs checked against the
#                  finite-difference gradient of U = mu/r [1 - sum Jn (R/r)^n Pn(sin phi)] in tests/test_space.py)
#                Sun + Moon third body (direct + indirect term), analytic ephemeris (ephem.py)
#                cannonball solar radiation pressure, cylindrical Earth shadow (ASSUMED; penumbra ignored)
#                drag in an atmosphere co-rotating with the Earth, density from density.AirColumn (ONE air model) or
#                  thermosphere.Thermosphere (solar/geomagnetic activity; uses its refresh-slot table like the GPU)
#              The GPU kernel in propagate.py implements the same terms; tests hold the two to 1e-12 relative.
#              Per-vehicle params: cd_am = Cd*A/m [m^2/kg], cr_am = Cr*A/m [m^2/kg].
import numpy as np

from .constants import MU_EARTH, R_GRAV, J2, J3, J4, MU_SUN, MU_MOON, AU, P_SRP_1AU, OMEGA_EARTH, WGS84_A
from .frames import geodetic_alt_fast

TERMS_ALL = ("j2", "j3", "j4", "sun", "moon", "srp", "drag")


def accel_gravity(r, terms=TERMS_ALL):
    r = np.atleast_2d(r)
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    r2 = np.sum(r * r, 1)
    rn = np.sqrt(r2)
    a = -MU_EARTH * r / (rn ** 3)[:, None]
    zr2 = z * z / r2
    if "j2" in terms:
        k = -1.5 * J2 * MU_EARTH * R_GRAV ** 2 / rn ** 5
        a = a + np.stack([k * x * (1 - 5 * zr2), k * y * (1 - 5 * zr2), k * z * (3 - 5 * zr2)], 1)
    if "j3" in terms:
        k = -2.5 * J3 * MU_EARTH * R_GRAV ** 3 / rn ** 7
        f = 3 * z - 7 * z ** 3 / r2
        a = a + np.stack([k * x * f, k * y * f, k * (6 * z * z - 7 * z ** 4 / r2 - 0.6 * r2)], 1)
    if "j4" in terms:
        k = 1.875 * J4 * MU_EARTH * R_GRAV ** 4 / rn ** 7
        f = 1 - 14 * zr2 + 21 * zr2 * zr2
        a = a + np.stack([k * x * f, k * y * f, k * z * (5 - 70.0 / 3.0 * zr2 + 21 * zr2 * zr2)], 1)
    return a


def potential_zonal(r, terms=TERMS_ALL):
    """Zonal part of the potential, -mu/r sum Jn (R/r)^n Pn(sin phi) [m^2/s^2] (no cancellation against mu/r)."""
    r = np.atleast_2d(r)
    rn = np.linalg.norm(r, axis=1)
    s = r[:, 2] / rn
    P = {2: 0.5 * (3 * s ** 2 - 1), 3: 0.5 * (5 * s ** 3 - 3 * s), 4: (35 * s ** 4 - 30 * s ** 2 + 3) / 8.0}
    J = {2: J2, 3: J3, 4: J4}
    u = np.zeros_like(rn)
    for n in (2, 3, 4):
        if "j%d" % n in terms:
            u = u - J[n] * (R_GRAV / rn) ** n * P[n]
    return MU_EARTH / rn * u


def potential(r, terms=TERMS_ALL):
    """Gravity potential U [m^2/s^2] (a = +grad U) for the same zonal set — used by tests and energy checks."""
    r = np.atleast_2d(r)
    return MU_EARTH / np.linalg.norm(r, axis=1) + potential_zonal(r, terms)


def accel_third_body(r, s_body, mu):
    d = s_body[None, :] - r
    return mu * (d / np.linalg.norm(d, axis=1)[:, None] ** 3 - s_body / np.linalg.norm(s_body) ** 3)


def shadow_factor(r, r_sun):
    """1 in sunlight, 0 in the cylindrical umbra."""
    u = r_sun / np.linalg.norm(r_sun)
    proj = r @ u
    perp = np.linalg.norm(r - proj[:, None] * u[None, :], axis=1)
    return np.where((proj < 0) & (perp < WGS84_A), 0.0, 1.0)


def accel_srp(r, r_sun, cr_am):
    d = r - r_sun[None, :]                       # Sun -> vehicle
    dn = np.linalg.norm(d, axis=1)
    return (shadow_factor(r, r_sun) * P_SRP_1AU * cr_am * (AU / dn) ** 2)[:, None] * d / dn[:, None]


def v_rel_air(r, v):
    w = np.array([0.0, 0.0, OMEGA_EARTH])
    return v - np.cross(w[None, :], r)


def accel_drag(r, v, cd_am, air, jd=None):
    if hasattr(air, "rho_eci") and jd is not None:      # thermosphere: time/position-resolved (same table as GPU)
        rho = air.rho_eci(r, jd)
    else:
        h = geodetic_alt_fast(r)
        rho = air.rho(np.maximum(h, -500.0))
    vr = v_rel_air(r, v)
    return -0.5 * (rho * cd_am * np.linalg.norm(vr, axis=1))[:, None] * vr


def accel(r, v, jd, cd_am, cr_am, air=None, terms=TERMS_ALL, sun=None, moon=None):
    from .ephem import sun_eci, moon_eci
    a = accel_gravity(r, terms)
    if "sun" in terms or "srp" in terms:
        sun = sun_eci(jd) if sun is None else sun
    if "sun" in terms:
        a = a + accel_third_body(r, sun, MU_SUN)
    if "moon" in terms:
        a = a + accel_third_body(r, moon_eci(jd) if moon is None else moon, MU_MOON)
    if "srp" in terms:
        a = a + accel_srp(r, sun, np.broadcast_to(cr_am, (len(r),)))
    if "drag" in terms and air is not None:
        a = a + accel_drag(r, v, np.broadcast_to(cd_am, (len(r),)), air, jd)
    return a


def rk4_host(r, v, jd0, h, n_steps, cd_am=0.0, cr_am=0.0, air=None, terms=TERMS_ALL, every=1, cb=None):
    """Fixed-step RK4 (float64). Returns times [s], r_hist, v_hist sampled every `every` steps."""
    r, v = np.array(r, float, ndmin=2), np.array(v, float, ndmin=2)
    ts, rs, vs = [0.0], [r.copy()], [v.copy()]
    for k in range(n_steps):
        t = k * h
        jd = lambda tt: jd0 + tt / 86400.0  # noqa: E731
        f = lambda rr, vv, tt: accel(rr, vv, jd(tt), cd_am, cr_am, air, terms)  # noqa: E731
        k1r, k1v = v, f(r, v, t)
        k2r, k2v = v + 0.5 * h * k1v, f(r + 0.5 * h * k1r, v + 0.5 * h * k1v, t + 0.5 * h)
        k3r, k3v = v + 0.5 * h * k2v, f(r + 0.5 * h * k2r, v + 0.5 * h * k2v, t + 0.5 * h)
        k4r, k4v = v + h * k3v, f(r + h * k3r, v + h * k3v, t + h)
        r = r + h / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        if (k + 1) % every == 0:
            ts.append((k + 1) * h)
            rs.append(r.copy())
            vs.append(v.copy())
            if cb:
                cb(ts[-1], r, v)
    return np.array(ts), np.array(rs), np.array(vs)

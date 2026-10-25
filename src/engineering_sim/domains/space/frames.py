# DESCRIPTION: Time (JD, GMST), ECI <-> ECEF <-> LLA (WGS-84) <-> ENU, Kepler <-> Cartesian. float64 numpy.
#
#              ECI here = TEME/true-of-date-like inertial frame rotated to ECEF by GMST only (IAU-82 GMST, Vallado
#              eq. 3-47). Precession/nutation/polar motion are NOT modelled (ASSUMED: <~1 km position error at LEO
#              after frame conversion, irrelevant to dynamics, which are integrated in the inertial frame). This is
#              the frame SGP4 outputs (TEME), so SGP4 cross-checks need no further rotation.
import numpy as np

from .constants import WGS84_A, WGS84_E2, OMEGA_EARTH, MU_EARTH, J2000_JD, SEC_PER_DAY


def jd_from_utc(year, month, day, hour=0, minute=0, sec=0.0):
    """Julian date (UTC treated as UT1; ASSUMED |UT1-UTC| < 0.9 s)."""
    if month <= 2:
        year, month = year - 1, month + 12
    A = year // 100
    B = 2 - A + A // 4
    jd0 = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    return jd0 + (hour + minute / 60.0 + sec / 3600.0) / 24.0


def gmst(jd_ut1):
    """Greenwich mean sidereal angle [rad] (IAU-82, Vallado eq. 3-47)."""
    T = (np.asarray(jd_ut1, float) - J2000_JD) / 36525.0
    s = 67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * T + 0.093104 * T ** 2 - 6.2e-6 * T ** 3
    return np.mod(np.deg2rad(s / 240.0), 2.0 * np.pi)


def rot_z(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def eci_to_ecef(r_eci, jd, v_eci=None):
    """r [...,3]; returns r_ecef (and v_ecef with Earth-rotation term if v given)."""
    R = rot_z(-gmst(jd))
    r = np.asarray(r_eci, float) @ R.T
    if v_eci is None:
        return r
    w = np.array([0.0, 0.0, OMEGA_EARTH])
    v = np.asarray(v_eci, float) @ R.T - np.cross(w, r)
    return r, v


def ecef_to_eci(r_ecef, jd, v_ecef=None):
    R = rot_z(gmst(jd))
    w = np.array([0.0, 0.0, OMEGA_EARTH])
    r_ecef = np.asarray(r_ecef, float)
    r = r_ecef @ R.T
    if v_ecef is None:
        return r
    v = (np.asarray(v_ecef, float) + np.cross(w, r_ecef)) @ R.T
    return r, v


def lla_to_ecef(lat_rad, lon_rad, h_m):
    sl = np.sin(lat_rad)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
    return np.stack([(N + h_m) * np.cos(lat_rad) * np.cos(lon_rad),
                     (N + h_m) * np.cos(lat_rad) * np.sin(lon_rad),
                     (N * (1.0 - WGS84_E2) + h_m) * sl], axis=-1)


def ecef_to_lla(r_ecef, iters=6):
    """Geodetic latitude/longitude [rad] and ellipsoidal height [m] (fixed-point, converges to <1 mm in 5 iters)."""
    r = np.asarray(r_ecef, float)
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    h = np.zeros_like(p)
    for _ in range(iters):
        sl = np.sin(lat)
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sl * sl)
        h = np.where(np.abs(lat) < np.deg2rad(80), p / np.maximum(np.cos(lat), 1e-12) - N,
                     z / np.where(sl == 0, 1e-12, sl) - N * (1.0 - WGS84_E2))
        lat = np.arctan2(z, p * (1.0 - WGS84_E2 * N / (N + h)))
    return lat, lon, h


def enu_matrix(lat_rad, lon_rad):
    """Rows = E, N, U unit vectors in ECEF."""
    sl, cl, so, co = np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)
    return np.array([[-so, co, 0.0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]])


def geodetic_alt_fast(r_eci):
    """Ellipsoidal height approximation used inside force models: h = |r| - a(1 - f sin^2(phi_gc)) (error < ~20 m
    vs exact; density scale heights are > 5 km so the effect on drag is < 0.4 %)."""
    r = np.asarray(r_eci, float)
    rn = np.linalg.norm(r, axis=-1)
    s2 = (r[..., 2] / rn) ** 2
    from .constants import WGS84_F
    return rn - WGS84_A * (1.0 - WGS84_F * s2)


# ---------------------------------------------------------------- Kepler elements
def kepler_to_rv(a, e, i, raan, argp, nu, mu=MU_EARTH):
    p = a * (1.0 - e * e)
    r_pf = np.array([p * np.cos(nu) / (1 + e * np.cos(nu)), p * np.sin(nu) / (1 + e * np.cos(nu)), 0.0])
    v_pf = np.array([-np.sqrt(mu / p) * np.sin(nu), np.sqrt(mu / p) * (e + np.cos(nu)), 0.0])
    cO, sO, ci, si, cw, sw = np.cos(raan), np.sin(raan), np.cos(i), np.sin(i), np.cos(argp), np.sin(argp)
    Q = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                  [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                  [sw * si, cw * si, ci]])
    return Q @ r_pf, Q @ v_pf


def rv_to_kepler(r, v, mu=MU_EARTH):
    """Returns dict(a, e, i, raan, argp, nu) for r, v [...,3] (vectorised)."""
    r, v = np.asarray(r, float), np.asarray(v, float)
    rn, vn = np.linalg.norm(r, axis=-1), np.linalg.norm(v, axis=-1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=-1)
    n = np.cross(np.array([0.0, 0.0, 1.0]), h)
    nn = np.linalg.norm(n, axis=-1)
    ev = ((vn ** 2 - mu / rn)[..., None] * r - np.sum(r * v, -1)[..., None] * v) / mu
    e = np.linalg.norm(ev, axis=-1)
    a = 1.0 / (2.0 / rn - vn ** 2 / mu)
    i = np.arccos(np.clip(h[..., 2] / hn, -1, 1))
    raan = np.mod(np.arctan2(n[..., 1], n[..., 0]), 2 * np.pi)
    argp = np.arccos(np.clip(np.sum(n * ev, -1) / np.maximum(nn * e, 1e-300), -1, 1))
    argp = np.where(ev[..., 2] < 0, 2 * np.pi - argp, argp)
    nu = np.arccos(np.clip(np.sum(ev * r, -1) / np.maximum(e * rn, 1e-300), -1, 1))
    nu = np.where(np.sum(r * v, -1) < 0, 2 * np.pi - nu, nu)
    return dict(a=a, e=e, i=i, raan=raan, argp=argp, nu=nu)

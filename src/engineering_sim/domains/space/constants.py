# DESCRIPTION: Geodesy / gravity / ephemeris constants for the space domain. Sources inline. Standard gravity comes
#              from engineering_sim.core.constants (fork CORE) when present so the whole sim has ONE g0.
try:
    from engineering_sim.core.constants import G0  # noqa: F401   (core v0.2)
except Exception:                  # core v0.1 fallback: same ISA value as core/atmos.py
    G0 = 9.80665

# WGS-84 (NIMA TR8350.2, 3rd ed.)
WGS84_A = 6378137.0                 # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563       # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
OMEGA_EARTH = 7.292115e-5           # rotation rate [rad/s]

# Gravity: EGM96 / WGS-84 GM; unnormalised zonal harmonics (Vallado 4th ed., Table D-1 / EGM96)
MU_EARTH = 3.986004418e14           # [m^3/s^2]
R_GRAV = 6378136.3                  # EGM96 reference radius [m]
J2 = 1.082626683e-3
J3 = -2.532656e-6
J4 = -1.619621e-6

# Third bodies (IAU 2009 / DE430)
MU_SUN = 1.32712440018e20
MU_MOON = 4.9028000e12
AU = 1.495978707e11
R_SUN = 6.957e8                     # nominal solar radius [m] (IAU 2015 B3)

# Solar radiation pressure at 1 AU: total solar irradiance 1361 W/m^2 (Kopp & Lean 2011) / c
P_SRP_1AU = 1361.0 / 299792458.0    # [N/m^2] = 4.54e-6

# Geomagnetic tilted dipole (IGRF-13 2025 dipole terms, rounded; ASSUMED adequate for attitude-level torque studies)
B0_DIPOLE = 2.98e-5                 # equatorial surface field strength g10..h11 magnitude [T]
DIPOLE_POLE_LAT_DEG = 80.8          # north geomagnetic pole latitude [deg]
DIPOLE_POLE_LON_DEG = -72.6         # north geomagnetic pole longitude [deg]

J2000_JD = 2451545.0
SEC_PER_DAY = 86400.0

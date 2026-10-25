#!/usr/bin/env python3
# DESCRIPTION: Every core module and every domain plug-in takes its constants from here. Values are exact by
#              definition or CODATA 2018 / ISO 2533 unless stated. A port that must reproduce a legacy constant
#              (e.g. g = 9.81) keeps it ONLY inside its explicit "legacy" parity mode and names it there.
#              Each name is a wp.constant (usable inside kernels) and also a plain float on the host.
#              Importable WITHOUT warp (CPU-only containers such as legacy service): then the names are plain floats.
import math

try:
    import warp as _wp
except ImportError:          # CPU-only container: host floats only
    _wp = None


def _c(x):
    """wp.constant(x) when real warp is present (returns x itself); plain x without warp or under an inert warp
    placeholder (domains/env_bridge/site_air.py imports the core that way in legacy service)."""
    if _wp is None:
        return x
    try:
        v = _wp.constant(x)
    except Exception:
        return x
    return v if isinstance(v, (int, float)) else x

G0 = _c(9.80665)               # standard gravity [m/s^2] (CGPM 1901; ISO 2533)
OMEGA_EARTH = _c(7.2921159e-5) # Earth rotation rate [rad/s] (IERS)
R_UNIVERSAL = _c(8.314462618)  # molar gas constant [J/(mol K)] (CODATA 2018, exact)
M_DRY = _c(0.0289644)          # molar mass of dry air [kg/mol] (ISO 2533)
M_H2O = _c(0.01801528)         # molar mass of water [kg/mol]
RD = _c(287.05287)             # specific gas constant, dry air [J/(kg K)] (ISO 2533)
RV = _c(461.495)               # specific gas constant, water vapour [J/(kg K)]
GAMMA_DRY = _c(1.4)            # ratio of specific heats, dry air (ideal diatomic)
KAPPA_VK = _c(0.4)             # von Karman constant (same as legacy sensor env_atmos.KAPPA)
# Sutherland's law for air (White, "Viscous Fluid Flow", 3rd ed., Table 1-2): mu_ref at T_ref, constant S
MU_REF = _c(1.716e-5)          # [Pa s]
T_MU_REF = _c(273.15)          # [K]
S_SUTH = _c(110.4)             # [K]
# ICAO / ISO 2533 standard atmosphere
T0_K = _c(288.15)                       # sea-level temperature [K]
P0_PA = _c(101325.0)                    # sea-level pressure [Pa]
LAPSE_K_PER_M = _c(-0.0065)             # tropospheric lapse rate [K/m]
H_TROPO_M = _c(11000.0)                 # tropopause [m]
C_REF_MS = _c(331.3)                    # dry-air sound speed at 0 C (matches legacy sensor env_atmos.SOUND_SPEED_REF_MS)
RHO_WATER_FRESH = 999.97                # [kg/m^3] at 4 C (host only)
RHO_SEAWATER = 1025.0                   # [kg/m^3] typical surface seawater (ASSUMED nominal; host only)


def coriolis_f(lat_deg):
    """Coriolis parameter f = 2 Omega sin(lat) [1/s]."""
    return 2.0 * OMEGA_EARTH * math.sin(math.radians(lat_deg))


def gravity_wgs84(lat_deg, h_m=0.0):
    """Normal gravity [m/s^2]: Somigliana (WGS-84, NIMA TR8350.2 eq. 4-1) + 2nd-order free-air correction
    (TR8350.2 eq. 4-3). Host helper for domains that want local g (rocket, ballistics); the core default is G0."""
    s2 = math.sin(math.radians(lat_deg)) ** 2
    g = 9.7803253359 * (1 + 0.00193185265241 * s2) / math.sqrt(1 - 0.00669437999013 * s2)
    a, f, m = 6378137.0, 1 / 298.257223563, 0.00344978650684
    return g * (1 - 2 / a * (1 + f + m - 2 * f * s2) * h_m + 3 / a ** 2 * h_m ** 2)

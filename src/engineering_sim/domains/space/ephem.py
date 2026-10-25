# DESCRIPTION: Low-precision analytic Sun and Moon positions in the inertial frame [m] (Vallado 4th ed. Alg. 29 "SUN"
#              and Alg. 31 "MOON", from the Astronomical Almanac). Accuracy ~0.01 deg (Sun) and ~0.3 deg / ~1e3 km
#              (Moon) — ASSUMED adequate for third-body and SRP/eclipse forces (perturbation error << 1 %).
import numpy as np

from .constants import AU, J2000_JD

D = np.pi / 180.0


def sun_eci(jd):
    T = (np.asarray(jd, float) - J2000_JD) / 36525.0
    lam_m = 280.460 + 36000.771 * T
    M = (357.5291092 + 35999.05034 * T) * D
    lam = (lam_m + 1.914666471 * np.sin(M) + 0.019994643 * np.sin(2 * M)) * D
    r = (1.000140612 - 0.016708617 * np.cos(M) - 0.000139589 * np.cos(2 * M)) * AU
    eps = (23.439291 - 0.0130042 * T) * D
    return np.stack([r * np.cos(lam), r * np.cos(eps) * np.sin(lam), r * np.sin(eps) * np.sin(lam)], axis=-1)


def moon_eci(jd):
    T = (np.asarray(jd, float) - J2000_JD) / 36525.0
    lam = (218.32 + 481267.8813 * T + 6.29 * np.sin((134.9 + 477198.85 * T) * D)
           - 1.27 * np.sin((259.2 - 413335.38 * T) * D) + 0.66 * np.sin((235.7 + 890534.23 * T) * D)
           + 0.21 * np.sin((269.9 + 954397.70 * T) * D) - 0.19 * np.sin((357.5 + 35999.05 * T) * D)
           - 0.11 * np.sin((186.6 + 966404.05 * T) * D)) * D
    beta = (5.13 * np.sin((93.3 + 483202.03 * T) * D) + 0.28 * np.sin((228.2 + 960400.87 * T) * D)
            - 0.28 * np.sin((318.3 + 6003.18 * T) * D) - 0.17 * np.sin((217.6 - 407332.20 * T) * D)) * D
    par = (0.9508 + 0.0518 * np.cos((134.9 + 477198.85 * T) * D) + 0.0095 * np.cos((259.2 - 413335.38 * T) * D)
           + 0.0078 * np.cos((235.7 + 890534.23 * T) * D) + 0.0028 * np.cos((269.9 + 954397.70 * T) * D)) * D
    eps = (23.439291 - 0.0130042 * T) * D
    r = 6378137.0 / np.sin(par)
    cb, sb, cl, sl = np.cos(beta), np.sin(beta), np.cos(lam), np.sin(lam)
    return np.stack([r * cb * cl, r * (np.cos(eps) * cb * sl - np.sin(eps) * sb),
                     r * (np.sin(eps) * cb * sl + np.cos(eps) * sb)], axis=-1)

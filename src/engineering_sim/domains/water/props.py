#!/usr/bin/env python3
# DESCRIPTION: Density and kinematic viscosity of fresh / brackish / sea water at the surface (atmospheric pressure;
#              no altitude dependence — water is ~incompressible, < 0.01 % over any lake elevation).
#                density:   pure water Tanaka et al. 2001 (Metrologia 38:301) + UNESCO EOS-80 salinity terms
#                           (Millero & Poisson 1981) at p = 0 dbar.  15 C / 35 g/kg -> 1025.97 kg/m^3.
#                viscosity: ITTC 7.5-02-01-03 (fresh water fit nu(T)) x a salinity factor
#                           [mu(T,S)/mu(T,0)] (Sharqawy et al. 2010, Desal. Water Treat. 16:354, eq. 22)
#                           x rho(T,0)/rho(T,S). At S = 35 this reproduces the ITTC seawater table within 0.1 %.
#              Presets (ASSUMED salinities): fresh 0, brackish 10, sea 35 g/kg.
#              Temperature: explicit T_c, else the site air temperature at the ground (clamped 0.5-35 C, ASSUMED:
#              a lake's surface water tracks the air; a cold-water or reservoir case must set T_c), else 15 C.
#              Lookups, not a solver; valid 0-40 C, 0-42 g/kg. Values are MODELED (published fits).
import math

PRESETS = dict(fresh=0.0, brackish=10.0, sea=35.0)
T_DEFAULT_C = 15.0


def density(T_c=15.0, S_gkg=35.0):
    T, S = float(T_c), float(S_gkg)
    rho_w = 999.974950 * (1 - (T - 3.983035) ** 2 * (T + 301.797) / (522528.9 * (T + 69.34881)))
    A = 8.24493e-1 - 4.0899e-3 * T + 7.6438e-5 * T ** 2 - 8.2467e-7 * T ** 3 + 5.3875e-9 * T ** 4
    B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T ** 2
    C = 4.8314e-4
    return rho_w + A * S + B * S ** 1.5 + C * S ** 2


def _salinity_mu_factor(T_c, S_gkg):
    """Sharqawy 2010 eq. 22: mu_sw / mu_w."""
    T, s = float(T_c), float(S_gkg) / 1000.0
    a = 1.541 + 1.998e-2 * T - 9.52e-5 * T ** 2
    b = 7.974 - 7.561e-2 * T + 4.724e-4 * T ** 2
    return 1 + a * s + b * s * s


def nu_fresh_ittc(T_c):
    """ITTC 7.5-02-01-03 fresh-water kinematic viscosity [m^2/s]."""
    d = float(T_c) - 12.0
    return ((0.000585 * d - 0.03361) * d + 1.2350) * 1e-6


def nu_sea_ittc(T_c):
    """ITTC 7.5-02-01-03 seawater (35 g/kg) kinematic viscosity [m^2/s] — reference only (tests)."""
    d = float(T_c) - 1.0
    return ((0.000659 * d - 0.05076) * d + 1.7688) * 1e-6


def kinematic_viscosity(T_c=15.0, S_gkg=35.0):
    return nu_fresh_ittc(T_c) * _salinity_mu_factor(T_c, S_gkg) * density(T_c, 0.0) / density(T_c, S_gkg)


def dynamic_viscosity(T_c=15.0, S_gkg=35.0):
    return kinematic_viscosity(T_c, S_gkg) * density(T_c, S_gkg)


def water(preset=None, S_gkg=None, T_c=None, air_T_c=None, rho=None, nu=None):
    """The water body as a parameter dict: preset 'fresh'|'brackish'|'sea' (or S_gkg), T_c (else site air at the
    ground, clamped 0.5-35 C, else 15 C). rho / nu override the fits (legacy parity). Returns dict(T_c, S_gkg, rho,
    nu, preset, T_source, rho_source, nu_source)."""
    if S_gkg is None:
        S_gkg = PRESETS[preset or "sea"]
    if T_c is not None:
        T, Tsrc = float(T_c), "explicit"
    elif air_T_c is not None:
        T, Tsrc = min(max(float(air_T_c), 0.5), 35.0), "site air at ground (clamped 0.5-35 C), ASSUMED"
    else:
        T, Tsrc = T_DEFAULT_C, "default 15 C, ASSUMED"
    return dict(preset=preset, S_gkg=float(S_gkg), T_c=T, T_source=Tsrc,
                rho=float(rho) if rho else density(T, S_gkg), nu=float(nu) if nu else kinematic_viscosity(T, S_gkg),
                rho_source="override" if rho else "Tanaka 2001 + EOS-80 salinity terms",
                nu_source="override" if nu else "ITTC 7.5-02-01-03 fresh fit x Sharqawy 2010 salinity factor")


def describe(T_c=15.0, S_gkg=35.0, rho_override=None):
    w = water(S_gkg=S_gkg, T_c=T_c, rho=rho_override)
    w["rho_source"] = "override (legacy parity)" if rho_override else w["rho_source"]
    return w


def ittc57(Re):
    """ITTC-1957 model-ship correlation line."""
    return 0.075 / (math.log10(max(Re, 1e5)) - 2.0) ** 2

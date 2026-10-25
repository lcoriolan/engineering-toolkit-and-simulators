# DESCRIPTION: Standard-library single source for closed-form flow correlations (Darcy friction factor, Dittus–Boelter)
#              shared by operating-point calculations and the mechanical calculator library
#              (engineering_sim/science/mech/fluid_dynamics.py, heat_transfer.py). Replaces four inline copies.
"""Flow correlations with no third-party imports, so both interpreters can use them.

Conventions: Re is the Reynolds number on the diameter (or hydraulic diameter); rel = ε/D relative roughness;
every friction factor is the **Darcy** factor (4 × Fanning). Callers own regime selection and validity warnings —
these functions evaluate the relation only.

Numerical gotcha: expressions keep the exact operation order of the copies they replaced (`(rel / 3.7) ** 1.11`,
`0.023 * Re ** 0.8 * Pr ** n`), so consolidation left every output bit-identical
(proof/mechcalc_02/friction_regression.py). Do not "simplify" them without re-running that proof.

Relations recalled (not re-read): Hagen–Poiseuille 64/Re; Colebrook–White (1939); Haaland (1983); Swamee–Jain (1976);
Blasius smooth-pipe 0.316 Re^-0.25 (≈ 4000 < Re < 1e5); Dittus–Boelter (1930) Nu = 0.023 Re^0.8 Pr^n; Shah & London
(1978) rectangular-duct laminar fits; Petukhov (1970) smooth-tube f; Gnielinski (1976) Nu.
"""
import math

# Specific gas constant of dry air, derived from U.S. Standard Atmosphere 1976 (NOAA-S/T 76-1562): R* = 8.31432e3 J/(kmol·K)
# (§1.1) and M0 = 28.9644 kg/kmol (§1.2) → 287.0531 J/(kg·K). The ICAO value 287.05287 was seen only in a search snippet.
R_AIR_USSA1976 = 8.31432e3 / 28.9644


def darcy_laminar(Re):
    """Fully developed laminar pipe flow: f = 64/Re (valid Re < ~2300)."""
    return 64.0 / Re


def darcy_haaland(Re, rel):
    """Haaland explicit fit to Colebrook: 1/√f = −1.8 log10((ε/D/3.7)^1.11 + 6.9/Re); within ~2 % for Re > 4000."""
    return (-1.8 * math.log10((rel / 3.7) ** 1.11 + 6.9 / Re)) ** -2


def darcy_swamee_jain(Re, rel):
    """Swamee–Jain explicit fit: f = 0.25 / log10(ε/D/3.7 + 5.74/Re^0.9)²; stated range 5000 < Re < 1e8, 1e-6 < ε/D < 0.01."""
    return 0.25 / (math.log10(rel / 3.7 + 5.74 / Re ** 0.9)) ** 2


def darcy_colebrook(Re, rel):
    """Colebrook–White, iterated: 1/√f = −2 log10(ε/(3.7 D) + 2.51/(Re √f)); fixed point seeded with Haaland.

    Raises ValueError when the log argument is non-positive (no turbulent solution at that Re).
    """
    x = -1.8 * math.log10((rel / 3.7) ** 1.11 + 6.9 / Re)  # Haaland seed
    if x <= 0:
        raise ValueError('no positive friction factor at this Re/roughness')
    for _ in range(200):
        arg = rel / 3.7 + 2.51 / Re * x
        if arg <= 0:
            raise ValueError('Colebrook log argument not positive')
        nx = -2.0 * math.log10(arg)
        if abs(nx - x) < 1e-14 * max(1.0, abs(nx)):
            x = nx
            break
        x = nx
    return 1.0 / (x * x)


def darcy_blasius(Re):
    """Blasius smooth-pipe turbulent friction factor: f = 0.316 Re^-0.25 (≈ 4000 < Re < 1e5)."""
    return 0.316 * Re ** -0.25


def nusselt_dittus_boelter(Re, Pr, n=0.4):
    """Dittus–Boelter turbulent internal flow: Nu = 0.023 Re^0.8 Pr^n, n = 0.4 heating / 0.3 cooling (Re > 1e4, 0.6 < Pr < 160)."""
    return 0.023 * Re ** 0.8 * Pr ** n


def f_re_rect_duct_laminar(alpha):
    """Shah & London fully developed laminar f·Re (Darcy) for a rectangular duct of aspect alpha = short/long (0 < alpha ≤ 1).

    96 for parallel plates (alpha → 0), ≈ 56.9 for a square duct."""
    return 96.0 * (1 - 1.3553 * alpha + 1.9467 * alpha ** 2 - 1.7012 * alpha ** 3 + 0.9564 * alpha ** 4 - 0.2537 * alpha ** 5)


def nusselt_rect_duct_laminar_h1(alpha):
    """Shah & London fully developed laminar Nu_H1 (uniform axial flux, peripherally uniform wall T), all walls heated.

    8.235 for parallel plates, ≈ 3.61 for a square duct."""
    return 8.235 * (1 - 2.0421 * alpha + 3.0853 * alpha ** 2 - 2.4765 * alpha ** 3 + 1.0578 * alpha ** 4 - 0.1861 * alpha ** 5)


def darcy_petukhov(Re):
    """Petukhov smooth-tube Darcy friction factor f = (0.790 ln Re − 1.64)⁻² (≈ 3000 < Re < 5e6)."""
    return (0.790 * math.log(Re) - 1.64) ** -2


def nusselt_gnielinski(Re, Pr, f):
    """Gnielinski Nu = (f/8)(Re − 1000)Pr / (1 + 12.7 √(f/8)(Pr^⅔ − 1)) with Darcy f (≈ 3000 < Re < 5e6, 0.5 < Pr < 2000)."""
    return (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * math.sqrt(f / 8) * (Pr ** (2 / 3) - 1))

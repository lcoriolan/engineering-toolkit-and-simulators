#!/usr/bin/env python3
# DESCRIPTION: Locally reacting, semi-infinite porous ground (rigid frame), time convention e^{-i w t}, p ~ e^{ikR}/R.
#              Normalised surface impedance Z (re rho c) from the effective flow resistivity sigma:
#                Miki (1990)            Z = 1 + 5.50 (f/s)^-0.632 + i 8.43 (f/s)^-0.632       (default; well behaved
#                                                                                              below 0.01 f/s)
#                Delany & Bazley (1970) Z = 1 + 9.08 (f/s)^-0.75  + i 11.9 (f/s)^-0.73
#                (s = sigma in kPa s m^-2; Attenborough, Li & Horoshenkov, "Predicting Outdoor Sound", 2007, ch. 5)
#              Plane-wave reflection   Rp = (Z sin(psi) - 1) / (Z sin(psi) + 1)      psi = grazing angle
#              Spherical-wave reflection (Weyl - van der Pol; Chien & Soroka 1975):
#                Q = Rp + (1 - Rp) F(w),  F(w) = 1 + i sqrt(pi) w W(w),  w = sqrt(i k R2 / 2) (sin(psi) + 1/Z)
#                W = Faddeeva function, W(z) = exp(-z^2) erfc(-i z) (Weideman 1994, N = 32; reflection formula for
#                Im z < 0), R2 = image-path length.
#              Flow resistivities per core terrain surface: Embleton, Piercy & Daigle (1983) ranges, ASSUMED values.
import math

import numpy as np

# effective flow resistivity [kPa s m^-2] per core/terrain.py SURFACES name (ASSUMED within the Embleton 1983 ranges)
SIGMA_KPA = {
    "dry_tarmac": 20000.0,     # asphalt, sealed: 20 000 - 30 000 (acoustically ~rigid)
    "wet_tarmac": 20000.0,
    "snow":       20.0,        # new snow 10 - 30
    "ice":        200000.0,    # rigid
    "gravel":     2000.0,      # limestone chips 1 500 - 4 000
    "grass":      200.0,       # grass / pasture 150 - 300
    "dry_soil":   2000.0,      # old dirt roadway 2 000 - 4 000 (firm)
    "sand":       500.0,       # roadside dirt / loose sandy soil 300 - 800
    "mud":        5000.0,      # earth, rain-packed 4 000 - 8 000
    "water":      1.0e7,       # free water surface: rigid for airborne sound
    "rigid":      1.0e9,
}


def impedance(f, sigma_kpa, model="miki"):
    """Normalised surface impedance Z(f) (complex, e^{-iwt}). f [Hz], sigma [kPa s m^-2] (broadcast)."""
    x = np.asarray(f, float) / np.asarray(sigma_kpa, float)
    if model == "miki":
        a = x ** -0.632
        return 1.0 + 5.50 * a + 8.43j * a
    if model in ("db", "delany_bazley"):
        return 1.0 + 9.08 * x ** -0.75 + 11.9j * x ** -0.73
    raise ValueError("impedance model must be 'miki' or 'db'")


def plane_wave_R(sin_psi, Z):
    zs = Z * sin_psi
    return (zs - 1.0) / (zs + 1.0)


# ------------------------------------------------------------------------------------------- Faddeeva function
_N = 32
_M = 2 * _N
_L = math.sqrt(_N / math.sqrt(2.0))
_k = np.arange(-_M + 1, _M)
_t = _L * np.tan(_k * math.pi / (2 * _M))
_f = np.concatenate([[0.0], np.exp(-_t ** 2) * (_L ** 2 + _t ** 2)])
_a = np.real(np.fft.fft(np.fft.fftshift(_f))) / (2 * _M)
_A = np.flipud(_a[1:_N + 1])            # polynomial coefficients, highest power first


def _w_upper(z):
    Zp = (_L + 1j * z) / (_L - 1j * z)
    p = np.polyval(_A, Zp)
    return 2.0 * p / (_L - 1j * z) ** 2 + (1.0 / math.sqrt(math.pi)) / (_L - 1j * z)


def faddeeva(z):
    """W(z) = exp(-z^2) erfc(-i z) for complex z (array). Weideman (1994) rational approximation (|err| ~1e-12 in
    the upper half plane); Im z < 0 by W(z) = 2 exp(-z^2) - W(-z)."""
    z = np.asarray(z, complex)
    out = np.empty_like(z)
    up = z.imag >= 0
    out[up] = _w_upper(z[up])
    zl = z[~up]
    with np.errstate(over="ignore", invalid="ignore"):
        out[~up] = 2.0 * np.exp(-zl ** 2) - _w_upper(-zl)
    return out


def ground_wave_F(w):
    return 1.0 + 1j * math.sqrt(math.pi) * w * faddeeva(w)


def spherical_Q(f, sin_psi, R2, Z, c):
    """Spherical-wave reflection coefficient Q (complex). f [Hz], sin_psi of the grazing angle, R2 image-path
    length [m], Z normalised impedance, c sound speed [m/s] (all broadcast)."""
    k = 2.0 * math.pi * np.asarray(f, float) / c
    Rp = plane_wave_R(sin_psi, Z)
    w = np.sqrt(0.5j * k * R2) * (sin_psi + 1.0 / Z)
    F = ground_wave_F(w)
    return Rp + (1.0 - Rp) * F


def two_ray_pressure(f, r, hs, hr, Z, c, alpha_db_per_m=0.0, spherical=True):
    """Closed form (homogeneous air over flat impedance ground): p = e^{ikR1}/R1 + Q e^{ikR2}/R2, normalised so the
    free field at 1 m is 1. Air absorption on each path if alpha_db_per_m is given."""
    f = np.asarray(f, float)
    k = 2.0 * math.pi * f / c
    R1 = math.hypot(r, hs - hr)
    R2 = math.hypot(r, hs + hr)
    sp = (hs + hr) / R2
    Q = spherical_Q(f, sp, R2, Z, c) if spherical else plane_wave_R(sp, Z)
    a1 = 10.0 ** (-np.asarray(alpha_db_per_m) * R1 / 20.0)
    a2 = 10.0 ** (-np.asarray(alpha_db_per_m) * R2 / 20.0)
    return a1 * np.exp(1j * k * R1) / R1 + Q * a2 * np.exp(1j * k * R2) / R2

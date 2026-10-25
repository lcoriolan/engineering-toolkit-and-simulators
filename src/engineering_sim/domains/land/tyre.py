#!/usr/bin/env python3
# DESCRIPTION: Pure-slip Magic Formula (Pacejka), MF(s) = D sin(C atan(B s - E (B s - atan(B s)))), with the
#              (B, C, D, E) of the surface under the contact patch (core/terrain.SURFACES). Force = MF * Fz, so D is
#              the peak friction coefficient.
#              Combined slip: "normalised slip" similarity method (Pacejka, Tire and Vehicle Dynamics, 3rd ed.,
#              §4.3.2 simple combined-slip model): sigma = (kappa, tan alpha), |F| = Fz MF(|sigma|), F along sigma.
#              Gives a friction circle by construction. The SAME coefficients are used laterally (ASSUMED — the
#              reference set is longitudinal; real tyres have different lateral B/C).
#              Slip (SAE at steady state) is TRANSIENT, computed in vehicles.py from the contact-patch deflection u
#              with relaxation length sigma (Pacejka ch. 7 single-contact-point transient model):
#                  du/dt = v_slip - |Vx| u / sigma,   kappa' = u_x / sigma,   tan alpha' = u_y / sigma
#              -> kappa = (omega R - Vx)/|Vx| when steady; a stiff spring at standstill (no creep on a slope; parked
#              vehicles hold exactly up to tan(th) = D). Below v_low a deflection-rate damper (~0.7 critical with
#              the wheel's share of the mass) is blended in, the usual low-speed fix.
#              Roller wheels (mecanum gamma = +-45 deg, omni gamma = 0: rollers free across a):
#              force only along the roller axis a = cos(g) f + sin(g) l; slip measured along a.
import math

import numpy as np
import warp as wp


@wp.func
def mf(B: float, C: float, D: float, E: float, s: float):
    bs = B * s
    return D * wp.sin(C * wp.atan(bs - E * (bs - wp.atan(bs))))


@wp.func
def combined(coef: wp.vec4, kappa: float, tan_a: float):
    """-> vec2 (Fx/Fz, Fy/Fz); Fy opposes positive lateral slip velocity."""
    sig = wp.sqrt(kappa * kappa + tan_a * tan_a)
    if sig < 1.0e-9:
        return wp.vec2(0.0, 0.0)
    f = mf(coef[0], coef[1], coef[2], coef[3], sig)
    return wp.vec2(f * kappa / sig, -f * tan_a / sig)


def mf_np(B, C, D, E, s):
    s = np.asarray(s, float)
    return D * np.sin(C * np.arctan(B * s - E * (B * s - np.arctan(B * s))))


def mf_peak(B, C, D, E):
    s = np.linspace(0, 1.5, 150001)
    f = mf_np(B, C, D, E, s)
    k = int(np.argmax(f))
    return float(s[k]), float(f[k])

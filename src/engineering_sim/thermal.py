# DESCRIPTION: Oil-jacket slot-channel hydraulics and heat transfer, dry-rotor gap convection and slot-wedge bending for compliant cases.
"""Screening physics for an oil jacket that flows axially through stator slots.

Scope and honesty
-----------------
* Slot channels are treated as rectangular ducts (short side / long side = alpha) in fully
  developed flow. Laminar values use Shah & London (1978) duct correlations for fRe and Nu_H1
  (constant axial heat flux, peripherally uniform temperature). Turbulent values use the
  Gnielinski correlation with the Petukhov friction factor. Developing-length enhancement is
  ignored, which is conservative for heat transfer.
* The dry rotor sheds its core loss across the air gap. The model uses the Taylor–Couette
  correlation of Becker & Kaye for 1.7e3 < Ta < 1e7 (Nu = 0.128 Ta^0.367 up to 1e4, then
  Nu = 0.409 Ta^0.241). Salient rotor poles pump more air than a smooth cylinder, so the smooth
  value is a lower bound. The air-gap convection path is combined with conduction through the
  shaft and bearings.
* Slot wedges are checked as simply supported plates between stack ends under the oil gauge
  pressure. Bonding is ignored, which is conservative.
Every function returns plain numbers in SI (W, K, Pa, m/s) unless its name says otherwise.
"""
import math

AIR_K_W_MK = 0.028          # dry air near 60 °C
AIR_NU_M2_S = 1.9e-5        # kinematic viscosity of air near 60 °C
STEEL_K_W_MK = 45.0         # carbon steel shaft


def duct_alpha(a, b):
    """Aspect ratio short/long side (0 < alpha <= 1) of a rectangular duct."""
    lo, hi = sorted((a, b))
    return lo / hi if hi > 0 else 1.0


def f_re_laminar(alpha):
    """Shah & London fully developed laminar f·Re (Darcy f) for a rectangular duct; 96 for parallel plates."""
    return 96.0 * (1 - 1.3553 * alpha + 1.9467 * alpha ** 2 - 1.7012 * alpha ** 3 + 0.9564 * alpha ** 4 - 0.2537 * alpha ** 5)


def nu_laminar_h1(alpha):
    """Shah & London fully developed laminar Nu_H1 for a rectangular duct heated on all walls; 8.235 for plates."""
    return 8.235 * (1 - 2.0421 * alpha + 3.0853 * alpha ** 2 - 2.4765 * alpha ** 3 + 1.0578 * alpha ** 4 - 0.1861 * alpha ** 5)


def slot_channel(q_total_m3_s, n_parallel, area_m2, perimeter_m, width_m, length_m, n_series, oil, one_wall_heated=False):
    """Hydraulics and film coefficient of the slot channels.

    q_total_m3_s  total oil flow through the machine
    n_parallel    parallel slot channels per cartridge (the flow splits between them)
    area_m2       free flow area of one channel; perimeter_m its wetted perimeter
    width_m       representative long side of the rectangle, used for the aspect ratio
    length_m      channel length per cartridge (the stack length)
    n_series      cartridges the flow passes through in series
    oil           dict(rho, cp, mu (Pa·s), k)
    one_wall_heated  thin gap heated on one wall only (other adiabatic): laminar Nu scaled from the
                  Shah & London H1 value by 5.385 / 8.235, their parallel-plate ratio for one wall
                  heated at constant flux vs both walls

    Returns velocity, Re, Pr, Dh, h (W/m²K), friction factor and pressure drop in Pa. Each
    cartridge adds its friction loss plus entry and exit losses (K = 0.5 + 1.0 per channel pass).
    """
    dh = 4 * area_m2 / perimeter_m
    q_channel = q_total_m3_s / n_parallel
    v = q_channel / area_m2
    re = oil['rho'] * v * dh / oil['mu']
    pr = oil['mu'] * oil['cp'] / oil['k']
    alpha = duct_alpha(area_m2 / width_m, width_m)
    if re < 2300.0:
        f = f_re_laminar(alpha) / max(re, 1e-9)
        nu = nu_laminar_h1(alpha) * (5.385 / 8.235 if one_wall_heated else 1.0)
        regime = 'laminar'
    else:
        f = (0.790 * math.log(re) - 1.64) ** -2                                  # Petukhov
        nu = (f / 8) * (re - 1000) * pr / (1 + 12.7 * math.sqrt(f / 8) * (pr ** (2 / 3) - 1))   # Gnielinski
        regime = 'turbulent' if re > 4000 else 'transitional (Gnielinski)'
    h = nu * oil['k'] / dh
    dyn = 0.5 * oil['rho'] * v ** 2
    dp = n_series * (f * length_m / dh + 1.5) * dyn
    return dict(dh_m=dh, velocity_m_s=v, re=re, pr=pr, alpha=alpha, regime=regime, nu=nu, f=f, h_w_m2k=h, dp_pa=dp)


def oil_flow_for(heat_w, oil, rise_k):
    """Mass-balance oil flow (m³/s) that carries heat_w at the allowed temperature rise: Q = P / (ρ c_p ΔT)."""
    return heat_w / (oil['rho'] * oil['cp'] * rise_k)


def taylor_couette_h(omega_rad_s, rotor_r_m, gap_m):
    """Air-gap film coefficient (W/m²K) from the Taylor number, with Nu based on hydraulic diameter 2g.

    Ta = ω² r g³ / ν². Below Ta = 1708 the gap flow is laminar Couette flow and only conducts
    (Nu = 2). Above that, Taylor vortices form and the Becker & Kaye correlation applies.
    """
    ta = omega_rad_s ** 2 * rotor_r_m * gap_m ** 3 / AIR_NU_M2_S ** 2
    if ta < 1708.0:
        nu = 2.0
    elif ta < 1e4:
        nu = 0.128 * ta ** 0.367
    else:
        nu = 0.409 * ta ** 0.241
    nu = max(nu, 2.0)
    return dict(ta=ta, nu=nu, h_w_m2k=nu * AIR_K_W_MK / (2 * gap_m))


def rotor_temperature(rotor_loss_w, stator_bore_temp_c, h_gap, gap_area_m2, shaft_path_w_k=0.0):
    """Mean rotor temperature for a dry rotor.

    Heat leaves through two parallel paths: gap convection (h·A) and the shaft/bearing
    conduction conductance G. T_rotor = T_bore + P / (hA + G).
    """
    conductance = h_gap * gap_area_m2 + shaft_path_w_k
    return stator_bore_temp_c + rotor_loss_w / conductance if conductance > 0 else float('inf'), conductance


def shaft_conductance(shaft_r_m, length_m, n_paths=2):
    """Conductance (W/K) of a solid shaft from the rotor to the bearings along n parallel paths: G = n k A / L."""
    return n_paths * STEEL_K_W_MK * math.pi * shaft_r_m ** 2 / length_m if length_m > 0 else 0.0


def wedge_plate(dp_pa, span_m, width_m, thickness_m, modulus_pa, tensile_pa):
    """Slot wedge as a simply supported strip spanning the stack under uniform pressure.

    Beam of width b, thickness t, span L carrying w = p·b:
      bending stress σ = M/Z = (w L² / 8) / (b t² / 6)
      mid-span deflection δ = 5 w L⁴ / (384 E I), with I = b t³ / 12
    Returns stress (Pa), safety factor on tensile strength, and deflection (m).
    """
    w = dp_pa * width_m
    moment = w * span_m ** 2 / 8
    z = width_m * thickness_m ** 2 / 6
    inertia = width_m * thickness_m ** 3 / 12
    sigma = moment / z if z > 0 else float('inf')
    delta = 5 * w * span_m ** 4 / (384 * modulus_pa * inertia) if inertia > 0 else float('inf')
    return dict(stress_pa=sigma, sf=tensile_pa / sigma if sigma > 0 else float('inf'), deflection_m=delta)


def hoop_stress(pressure_pa, radius_m, wall_m):
    """Thin-wall hoop stress σ = p r / t (Pa) of a ring under internal gauge pressure."""
    return pressure_pa * radius_m / wall_m

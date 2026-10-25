#!/usr/bin/env python3

#              properties('air', T, p): density now from engineering_sim.core.env.Env (ISA + site + humidity), dynamic viscosity by
#              the SAME Sutherland law (NASA GRC: mu = 1.716e-5 (T/273.15)^1.5 (273.15+110.4)/(T+110.4)) evaluated
#              at the core temperature. The core has no viscosity yet -> requested core change (atmos.viscosity).
#              legacy_env(T_c, p_pa) builds the Env that reproduces a legacy (T, p) state at z = 0 for parity.
#              WATER (IAPWS-IF97) and the SU2 CFD pipeline are NOT moved: they are offline solvers (see migration doc).
import math

import numpy as np

T0_ISA = 288.15


def sutherland_mu(T_k):
    T_k = np.asarray(T_k, float)
    return 1.716e-5 * (T_k / 273.15) ** 1.5 * (273.15 + 110.4) / (T_k + 110.4)


def legacy_env(temperature_c=20.0, pressure_pa=101325.0):
    """Env config whose z=0 state is (T, p): ISA+dT with sea-level pressure p (dry air)."""
    return dict(atmos=dict(site_elev_m=0.0, dT_K=temperature_c + 273.15 - T0_ISA, p0_pa=pressure_pa, rh=0.0),
                wind=dict(u10=0.0, turbulence=False))


def properties_air(env, z=0.0):
    """manifold_fluids.properties('air')-compatible dict from the core environment at height z."""
    T, p = env.T_p([z])
    T, p = float(T[0]), float(p[0])
    rho = float(env.rho([z])[0])
    mu = float(sutherland_mu(T))
    c = float(env.sound_speed([z])[0])
    return dict(fluid="air", temperature_c=T - 273.15, temperature_k=T, pressure_absolute_pa=p,
                density_kg_m3=rho, dynamic_viscosity_pa_s=mu, kinematic_viscosity_m2_s=mu / rho,
                sound_speed_m_s=c, vapor_pressure_pa=None,
                model="unified core Env (ISA/site/humidity) + Sutherland viscosity",
                source="core/atmos.py; https://www.grc.nasa.gov/www/BGH/viscosity.html")


def legacy_properties_air(temperature_c=20.0, pressure_pa=101325.0):
    """Transcription of the legacy air branch (used only when the legacy module cannot be imported)."""
    kelvin = temperature_c + 273.15
    return dict(density_kg_m3=pressure_pa / (287.058 * kelvin), dynamic_viscosity_pa_s=float(sutherland_mu(kelvin)),
                sound_speed_m_s=math.sqrt(1.4 * 287.058 * kelvin))

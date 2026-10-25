# DESCRIPTION: Physical fluid states and coordinate-based spacing inputs for CFD cases.
"""Geometry in mm; solver properties in SI. Pressure is always absolute here."""
import math
from pathlib import Path
import sys

from iapws import IAPWS97

VERSION='1.0.0'
SOURCES=dict(water='https://iapws.org/technical-guidance/release/IF97',
    viscosity='https://iapws.org/technical-guidance/release/viscosity',
    air='https://www.grc.nasa.gov/www/BGH/viscosity.html')


def properties(fluid, temperature_c=20.0, pressure_pa=101325.0):
    """Return dry-air or fresh-water properties; water must be single-phase liquid.

    Air uses the ideal-gas law and Sutherland viscosity. Water uses IAPWS-IF97
    with IAPWS transport properties through pinned iapws 1.5.4. The water
    property package accepts MPa, hence the explicit Pa/1e6 conversion.
    """
    t, p = float(temperature_c), float(pressure_pa)
    if not math.isfinite(t) or not 5<=t<=60 or not math.isfinite(p) or not 20000<=p<=1000000:
        raise ValueError('Use 5–60 °C and 20–1000 kPa absolute for this comparison.')
    kelvin=t+273.15
    if fluid=='air':
        density=p/(287.058*kelvin)
        viscosity=1.716e-5*(kelvin/273.15)**1.5*(273.15+110.4)/(kelvin+110.4)
        sound=math.sqrt(1.4*287.058*kelvin)
        vapor=None
    elif fluid=='water':
        state=IAPWS97(T=kelvin,P=p/1e6)
        if state.phase not in ('Liquid','Compressible liquid'):
            raise ValueError('Requested water state is not liquid.')
        density,viscosity,sound=map(float,(state.rho,state.mu,state.w))
        vapor=float(IAPWS97(T=kelvin,x=0).P*1e6)
    else:
        raise ValueError('Fluid must be air or water.')
    return dict(fluid=fluid,temperature_c=t,temperature_k=kelvin,pressure_absolute_pa=p,
        density_kg_m3=density,dynamic_viscosity_pa_s=viscosity,
        kinematic_viscosity_m2_s=viscosity/density,sound_speed_m_s=sound,
        vapor_pressure_pa=vapor,model='IAPWS-IF97 / IAPWS viscosity' if fluid=='water' else 'Dry ideal gas / Sutherland',
        source=SOURCES['water' if fluid=='water' else 'air'])


def point_distance(a,b):
    """Measure Euclidean, axial and lateral separation of two assembled XYZ points."""
    if len(a)!=3 or len(b)!=3:
        raise ValueError('Each reference point needs three XYZ coordinates in mm.')
    values=[float(x) for x in list(a)+list(b)]
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Reference coordinates must be finite.')
    delta=[values[i]-values[i+3] for i in range(3)]
    return dict(point_a_mm=values[:3],point_b_mm=values[3:],distance_mm=math.hypot(*delta),
        axial_separation_mm=abs(delta[2]),lateral_separation_mm=math.hypot(*delta[:2]))


def gap_from_points(a,b,target_distance_mm,current_gap_mm):
    """Translate upstream FAN along +Z to reach a point-to-point distance.

    A belongs to FAN and B to stationary STATOR, both in assembled coordinates.
    Lateral offsets stay fixed. A must start upstream (higher Z). Distance is
    therefore not interchangeable with axial clearance when radii differ.
    """
    measured=point_distance(a,b)
    target=float(target_distance_mm)
    if not math.isfinite(target) or target<measured['lateral_separation_mm']:
        raise ValueError('Target is smaller than the fixed lateral offset.')
    if measured['point_a_mm'][2]<=measured['point_b_mm'][2]:
        raise ValueError('Fan reference A must be upstream of stator reference B.')
    axial=math.sqrt(target**2-measured['lateral_separation_mm']**2)
    gap=float(current_gap_mm)+axial-measured['axial_separation_mm']
    if not 1<=gap<=100:raise ValueError('Requested point distance produces a gap outside 1–100 mm.')
    # Ignore sub-nanometre arithmetic noise so a 15 mm datum reconstructs the
    # same geometry identity as an explicit 15 mm recipe or cached CAD build.
    return round(gap,10)


def similarity_screen(calculation,state,rpm):
    """Return an ideal load screen, explicitly without a gap-performance model.

    Constant flow and loading coefficients imply Q~n, T~rho*n² and P~rho*n³.
    Viscous effects, separation and cavitation can violate these assumptions,
    so this is a scale/load comparison, never a spacing ranking or CFD result.
    The old 0.65 'projected thrust' multiplier is deliberately not propagated.
    """
    fan=calculation['fan']
    rpm=float(rpm)
    if not math.isfinite(rpm) or rpm<=0:raise ValueError('RPM must be positive and finite.')
    ratio=rpm/fan['rpm']
    rho=state['density_kg_m3']
    baseline_rho=fan['assumptions_used']['rho']
    density_ratio=rho/baseline_rho
    velocity=fan['stations'][0]['v_axial_m_s']*ratio
    thrust=fan['thrust_n_ideal']*density_ratio*ratio**2
    power=thrust*velocity
    tip=fan['tip_speed_m_s']*ratio
    diameter=calculation['layout']['bore_mm']/1000
    # Sigma uses tip speed as a stated reference only; the actual minimum
    # pressure coefficient is unknown until a pressure field is computed.
    sigma=None if state['vapor_pressure_pa'] is None else (
        state['pressure_absolute_pa']-state['vapor_pressure_pa'])/(0.5*rho*tip**2)
    return dict(status='ideal_similarity_screen_only',rpm=rpm,
        ideal_thrust_n=thrust,ideal_fluid_power_w=power,
        ideal_torque_nm=power/(rpm*2*math.pi/60),
        volume_flow_m3_s=calculation['volume_flow_m3_s']*ratio,
        tip_speed_m_s=tip,tip_mach=tip/state['sound_speed_m_s'],
        reynolds_tip_speed_bore=rho*tip*diameter/state['dynamic_viscosity_pa_s'],
        cavitation_number_tip_reference=sigma,
        spacing_dependence=False,motor_capability_demonstrated=False,
        assumptions='Constant flow/loading coefficients; inviscid, noncavitating ideal fluid power. No spacing loss law.')

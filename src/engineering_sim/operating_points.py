import math
from . import correlations
from .execution import operation

@operation("engineering.operating_point",kind="static")
def solve(spec,s,volume_mm3=None,fluid=None):
    """Calculate one SI operating point; ratios with zero denominators remain null.

    fluid: optional CoolProp result for fluid_requests(spec,s)[0]; when given, real_* metrics are added
    beside (never instead of) the existing ideal/assumed-property metrics."""
    mode=spec['mode'];d=spec['derived'];ratio=lambda a,b:a/b if b else None
    if mode=='actuator':
        omega=s['rpm']*2*math.pi/60;out=s['torque']*omega;shaft=out/s['gear_efficiency'];electric=shaft/s['motor_efficiency']+s['idle_power']
        return dict(output_power_w=out,motor_shaft_power_w=shaft,electrical_power_w=electric,current_a=electric/s['voltage'],heat_w=electric-out,
                    efficiency_pct=ratio(out*100,electric),output_torque_nm=s['torque'],motor_torque_nm=s['torque']/s['ratio']/s['gear_efficiency'],motor_rpm=s['rpm']*s['ratio'],force_n=s['torque']/s['arm_m'])
    if mode=='propeller':
        diameter=s.get('diameter_m',d['diameter_m']);hub=s.get('hub_ratio',d['hub_ratio']);area=math.pi*diameter**2/4*(1-hub**2)
        tip=math.pi*diameter*s['rpm']/60;vi=s['phi']*tip;disk=s['forward_speed']+vi;flow=area*disk;thrust=2*s['rho']*flow*vi
        ideal=thrust*disk;shaft=ideal/s['fan_efficiency'];electric=shaft/s['motor_efficiency'];useful=thrust*s['forward_speed'];omega=s['rpm']*2*math.pi/60
        r=dict(thrust_n=thrust,shaft_power_w=shaft,electrical_power_w=electric,current_a=electric/s['voltage'],torque_nm=ratio(shaft,omega),
          propulsive_efficiency_pct=ratio(useful*100,shaft),overall_efficiency_pct=ratio(useful*100,electric),static_figure_of_merit_pct=s['fan_efficiency']*100 if s['forward_speed']==0 and shaft else None,
          thrust_per_watt_n_w=ratio(thrust,electric),heat_motor_w=electric-shaft,fluid_loss_w=shaft-ideal,ideal_fluid_power_w=ideal,
          volume_flow_m3_s=flow,mass_flow_kg_s=flow*s['rho'],disk_velocity_m_s=disk,tip_speed_m_s=tip,disk_area_m2=area)
        for end in ('inlet','outlet'):
            if end+'_area_m2' in d:r[end+'_velocity_m_s']=flow/d[end+'_area_m2']
        return r
    if mode=='compressor':
        dt=s['temperature']*(s['pressure_ratio']**(.4/1.4)-1)/s['efficiency'];shaft=s['mass_flow']*1005*dt;electric=shaft/s['drive_efficiency'];rho=s['pressure']/(correlations.R_AIR_USSA1976*s['temperature'])
        r=dict(shaft_power_w=shaft,electrical_power_w=electric,current_a=electric/s['voltage'],drive_heat_w=electric-shaft,
          outlet_temperature_k=s['temperature']+dt,outlet_pressure_pa=s['pressure']*s['pressure_ratio'],mass_flow_kg_s=s['mass_flow'],
          inlet_velocity_m_s=s['mass_flow']/rho/d['area_m2'],isentropic_efficiency_pct=s['efficiency']*100)
        if fluid:
            # Real-air work per kg from CoolProp enthalpies; power difference is relative to the ideal-gas headline.
            real=s['mass_flow']*fluid['work_j_kg']
            r.update(real_gas_shaft_power_w=real,real_gas_electrical_power_w=real/s['drive_efficiency'],real_gas_outlet_temperature_k=fluid['T2'],
                     real_gas_inlet_density_kg_m3=fluid['rho1'],real_vs_ideal_power_pct=ratio((real-shaft)*100,shaft))
        return r
    if mode=='channel':
        v=s['flow']/d['area_m2'];dp=(s['friction']*d['length_m']/d['diameter_m']+s['loss_k'])*s['rho']*v*v/2;power=dp*s['flow']
        r=dict(mean_velocity_m_s=v,pressure_loss_pa=dp,fluid_power_w=power,electrical_power_w=power/s['efficiency'],mass_flow_kg_s=s['rho']*s['flow'],efficiency_pct=s['efficiency']*100)
        if fluid and fluid['mu']>0:
            # Re per bore from the combined mean velocity (equal parallel bores share it); Haaland explicit Colebrook fit above Re 4000.
            re=fluid['rho']*v*d['diameter_m']/fluid['mu']
            if re<=0:f=None
            elif re<2300:f=correlations.darcy_laminar(re)
            else:f=correlations.darcy_haaland(re,s['roughness']/d['diameter_m'])
            dp_real=None if f is None else (f*d['length_m']/d['diameter_m']+s['loss_k'])*fluid['rho']*v*v/2
            r.update(real_fluid_density_kg_m3=fluid['rho'],real_fluid_viscosity_pa_s=fluid['mu'],reynolds_number=re,
                     flow_regime='laminar' if re<2300 else 'transitional' if re<4000 else 'turbulent',
                     real_fluid_friction_factor=f,real_fluid_pressure_loss_pa=dp_real,
                     real_fluid_electrical_power_w=None if dp_real is None else dp_real*s['flow']/s['efficiency'])
        return r
    if mode=='aero':
        q=.5*s['rho']*s['speed']**2;lift=q*s['area']*s['cl'];drag=q*s['area']*s['cd']
        return dict(lift_n=lift,drag_n=drag,lift_drag_ratio=s['cl']/s['cd'],dynamic_pressure_pa=q,drag_power_w=drag*s['speed'])
    if mode=='mass':
        if volume_mm3 is None:raise ValueError('Build current exact CAD before a material mass study.')
        mass=volume_mm3/1e9*s['density'];return dict(material_volume_mm3=volume_mm3,mass_kg=mass,gravity_load_n=mass*s['gravity'])
    raise ValueError('Flotation requires a queued exact-hull calculation.')

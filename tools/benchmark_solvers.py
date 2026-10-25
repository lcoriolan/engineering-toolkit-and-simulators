"""Run synthetic numerical benchmarks and save inputs, equations and actual errors."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import warp as wp
wp.config.quiet=True

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='benchmarks/solvers.json');parser.add_argument('--fea',action='store_true');args=parser.parse_args()
    rows=[]
    def add(name,equation,inputs,reference,actual,tolerance,unit,kind='analytical',assumptions='Synthetic inputs; idealized model only.'):
        error=abs(float(actual)-float(reference))
        rows.append(dict(name=name,equation=equation,inputs=inputs,reference=reference,actual=float(actual),absolute_error=error,
          relative_error=error/abs(reference) if reference else None,absolute_tolerance=tolerance,unit=unit,passed=error<=tolerance,
          classification=kind,assumptions=assumptions,command='python tools/benchmark_solvers.py --fea',backend_versions='backend-versions.json'))
    from engineering_sim.demo import run_world
    r=run_world(.2,.001,'cpu')
    add('World free-fall position','z=z0-g*t^2/2',dict(z0_m=10,g_m_s2=9.80665,t_s=.2,dt_s=.001),10-.5*9.80665*.2**2,r['position_m'][2],.0015,'m',assumptions='No drag/contact; semi-implicit Euler and float32 state, 200 CPU steps.')
    add('World free-fall velocity','vz=-g*t',dict(g_m_s2=9.80665,t_s=.2),-9.80665*.2,r['velocity_m_s'][2],.0002,'m/s')
    from engineering_sim.core.air_np import ref_state
    r=ref_state([0.,11000.],geometric=False)
    add('US76 sea-level density','rho=p/(R*T)',dict(geopotential_altitude_m=0,T_K=288.15,p_Pa=101325),1.225,r['rho'][0],.000245,'kg/m^3','published reference landmark','Dry US76 standard atmosphere; no weather deviation.')
    add('US76 tropopause pressure','p=p0*(T/T0)^(-g/(L*R))',dict(geopotential_altitude_m=11000,L_K_m=-.0065),22632.1,r['p_pa'][1],4.527,'Pa','published reference landmark')
    from engineering_sim.domains.water import props,waves,hull
    add('Freshwater density','temperature polynomial (UNESCO form in source)',dict(temperature_C=20,salinity_PSU=0),998.21,props.density(20,0),.02,'kg/m^3','reference landmark')
    omega=np.array([.5,1,2,4]);k=waves.dispersion_k(omega,5.)
    add('Finite-depth dispersion','omega^2=g*k*tanh(k*h)',dict(omega_rad_s=omega.tolist(),depth_m=5,g_m_s2=9.80665),0,np.max(abs(9.80665*k*np.tanh(k*5)-omega**2)),1e-9,'s^-2','equation residual')
    cols=hull.columns_from_mesh(*hull.box_mesh(2,1,.5),nx=32,ny=16)
    volume=hull.volume_curve(cols,[.3])[0]['volume_m3']
    add('Box displacement','V=L*B*draft',dict(length_m=2,width_m=1,draft_m=.3),.6,volume,6e-5,'m^3')
    from engineering_sim.domains.space import frames,forces,constants
    r,v=frames.kepler_to_rv(7000000.,.01,.5,.2,.1,.3)
    t,rr,vv=forces.rk4_host(r,v,2451545.,10.,600,terms=(),every=10)
    energy=.5*np.sum(vv[:,0]**2,axis=1)-constants.MU_EARTH/np.linalg.norm(rr[:,0],axis=1)
    add('Two-body orbital energy drift','epsilon=v^2/2-mu/r=constant',dict(a_m=7000000,e=.01,i_rad=.5,raan_rad=.2,argp_rad=.1,true_anomaly_rad=.3,dt_s=10,steps=600),0,np.max(abs(energy/energy[0]-1)),1e-9,'1','conservation invariant','Two-body point mass; no perturbations; host float64 RK4.')
    from engineering_sim.domains.room_acoustics.room import ShoeboxRoom
    room=ShoeboxRoom(4,3,2.5,.2,c=343)
    add('Sabine decay time','T60=24*ln(10)*V/(c*sum(alpha*S))',dict(volume_m3=30,area_m2=59,absorption=.2,c_m_s=343),24*math.log(10)*30/(343*59*.2),room.t60_sabine(),1e-12,'s',assumptions='Diffuse energy field, uniform absorption; not a measured room.')
    from engineering_sim.checks import tolerances,mechanisms
    st=tolerances.stack([dict(sensitivity=1,tolerance_mm=.03),dict(sensitivity=-1,tolerance_mm=.04)])
    add('Worst-case tolerance','sum(abs(sensitivity*tolerance))',dict(tolerances_mm=[.03,.04],sensitivities=[1,-1]),.07,st['worst_case_mm'],1e-14,'mm')
    add('RSS tolerance','sqrt(sum((sensitivity*tolerance)^2))',dict(tolerances_mm=[.03,.04],sensitivities=[1,-1]),.05,st['rss_mm'],1e-14,'mm',assumptions='Independent centered contributors on the same statistical basis; no process-capability claim.')
    mobility=mechanisms.mobility(['base','arm'],[dict(parent='base',child='arm',axis=[0,0,1],pivot=[0,0,0])],fixed_owner='base')
    add('Single revolute mobility','nullity=6*N-rank(C)',dict(bodies=2,fixed_bodies=1,revolute_joints=1),1,mobility['nullity'],0,'DOF','constraint-rank invariant')
    from engineering_sim.geometry import airfoil
    p=airfoil.properties(airfoil.coords('NACA0012',100))
    add('NACA0012 thickness','yt=5*t*(.2969*sqrt(x)-.126*x-.3516*x^2+.2843*x^3-.1036*x^4)',dict(t_over_c=.12,n_per_side=100,closed_TE=True),.12,p['t_c'],.0001,'chord fraction','geometric reference landmark','Closed trailing-edge coefficient differs from the original finite-gap NACA coefficient.')
    from engineering_sim.structures import rect_area_moment,tip_deflection_uniform
    add('Rectangular section inertia','I=b*h^3/12',dict(b_m=.02,h_m=.03),.02*.03**3/12,rect_area_moment(.02,.03),1e-20,'m^4')
    add('Uniform beam deflection','delta=w*L^4/(8*EI)',dict(w_N_m=10,L_m=1,EI_N_m2=1000),.00125,tip_deflection_uniform(10,1,1000),1e-15,'m')
    from engineering_sim.molding_process import slab_threshold_time
    t=slab_threshold_time(100,0,50,.01,1e-6,0)
    add('Slab cooling thickness scaling','t_crossing proportional to half_thickness^2/alpha',dict(T_initial_C=100,T_wall_C=0,T_threshold_C=50,half_thickness_m=[.01,.02],alpha_m2_s=1e-6,position_fraction=0),4,slab_threshold_time(100,0,50,.02,1e-6,0)/t,1e-12,'1','scaling invariant','Fixed wall, constant properties, no latent heat or flow.')
    from engineering_sim.flow.aero import LatticeSurface
    from engineering_sim.flow.vlm import solve
    surface=LatticeSurface('synthetic',lambda s,x:[x,2*s-1,0],8)
    a=solve([surface],[10,0,1],1.2,[0,0,0],chord_panels=3)
    b=solve([surface],[10,0,1],1.2,[0,0,0],chord_panels=3,scale=2)
    add('VLM length-scale invariance','F(lambda*geometry)=lambda^2*F(geometry)',dict(chord_m=1,span_m=2,panels=24,velocity_m_s=[10,0,1],rho_kg_m3=1.2,length_scale=2),4,np.linalg.norm(b['F_g'])/np.linalg.norm(a['F_g']),1e-9,'1','scaling invariant','Thin attached incompressible potential flow; core radius also scaled; no stall or viscous drag.')
    if args.fea:
        import ngsolve as ng
        from netgen.geom2d import unit_square
        from engineering_sim import fea
        mesh=ng.Mesh(unit_square.GenerateMesh(maxh=.12));exact=ng.sin(math.pi*ng.x)*ng.sin(math.pi*ng.y)
        result=fea.magnetostatic_2d(mesh,1.,2*math.pi**2*exact,'.*',order=3)
        error=math.sqrt(ng.Integrate((result['potential']-exact)**2,mesh))
        add('Manufactured Poisson L2 error','-laplacian(A)=2*pi^2*sin(pi*x)*sin(pi*y)',dict(domain='unit square',reluctivity=1,dirichlet_A=0,maxh=.12,order=3),0,error,1e-4,'normalized potential','manufactured solution','Dimensionless unit-domain verification of the weak operator; not a motor field benchmark.')
        add('Manufactured magnetic energy','U=integral(|grad A|^2)/2=pi^2/4',dict(domain='unit square',order=3),math.pi**2/4,result['energy_j_per_m'],math.pi**2/4*.001,'normalized energy per length','manufactured solution')
        mesh=ng.Mesh(unit_square.GenerateMesh(maxh=.15));r=fea.elasticity(mesh,1000.,0.,'left','right',(10.,0.),order=2)
        add('Uniform elastic extension energy','u_x=(traction/E)*x; U=traction^2/(2E)',dict(domain='unit square',E_Pa=1000,nu=0,traction_Pa=10,maxh=.15,order=2),.05,r['strain_energy_j'],5e-10,'J per unit thickness','analytical','Small strain; unit thickness; lateral contraction suppressed by nu=0.')
    out=dict(passed=sum(r['passed'] for r in rows),failed=sum(not r['passed'] for r in rows),benchmarks=rows)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'passed':out['passed'],'failed':out['failed']}))
    if out['failed']: raise SystemExit(1)
if __name__=='__main__':main()

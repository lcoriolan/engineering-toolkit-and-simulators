"""Synthetic inputs for analytical checks and numerical invariants."""
import math
import numpy as np
import pytest

def test_naca_and_rectangular_planform():
    from engineering_sim.geometry import airfoil,planform
    props=airfoil.properties(airfoil.coords('NACA0012',100))
    assert props['t_c']==pytest.approx(.12,abs=.0001)
    assert abs(props['camber_max'])<1e-12
    stations=planform.stations('rectangular',500.,100.,100.,0.,0.,0.,0.,0.,0.,30,"uniform")
    area,mac,x,y=planform.area_mac(stations)
    assert area==pytest.approx(100000.) and mac==pytest.approx(100.)

def test_mesh_and_mass_measurement():
    from engineering_sim.geometry import stl,measure
    vertices=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    faces=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
    assert abs(stl.mesh_volume(vertices,faces))==pytest.approx(1/6)
    mass,cg,I=measure.combine_mass([measure.point_item(1,[-1,0,0]),measure.point_item(1,[1,0,0])])
    assert mass==2 and np.max(abs(cg))==0
    assert I==pytest.approx(np.diag([0.,2.,2.]))
    a=np.linspace(0,2*math.pi,50);cx,cy,r,rms=measure.fit_circle(np.column_stack((2+3*np.cos(a),-1+3*np.sin(a))))
    assert [cx,cy,r]==pytest.approx([2,-1,3]) and rms<1e-12

def test_flexure_and_fourbar_closure():
    from engineering_sim.checks import flexures
    assert flexures.flexure_arc(10,0,0,10,0)==(10,0)
    angle=.2; x,z=flexures.flexure_arc(10,0,0,10,math.degrees(angle))
    assert math.hypot(x,z+50)==pytest.approx(50.)
    assert flexures.flexure_strain(1,100,math.degrees(.2))==pytest.approx(.001)
    r=flexures.linkage((0,0),(30,5),5,5,10,30)
    assert math.dist(r['pin'],r['target'])==pytest.approx(r['rod_length'])

def test_cooling_heat_balance():
    from engineering_sim.cooling import Duty,gap_channels,evaluate
    d=Duty(copper_loss_w=100,iron_loss_w=0,n_modules=1,winding=lambda gap:gap_channels(4,50,10,gap),iron=None,rotor=None,
           oil=dict(rho=1000,cp=1000,nu_cst=1,k=.6),inlet_c=20,coil_limit_c=100,iron_limit_c=100,pump_limit_bar=1,pump_margin=.5,
           flow_max_lpm=2,gap_min_mm=.5,gap_max_mm=2,gap_tol_mm=0,name='synthetic')
    r=evaluate(d,1,1,'series')
    assert r['mixed_outlet_c']==pytest.approx(26.)
    assert r['winding_oil_out_c']==pytest.approx(26.)
    assert r['coil_temp_c']>26. and r['dp_bar']>0

def test_slab_cooling_and_route_distance():
    from engineering_sim.molding_process import slab_threshold_time,segment_distance
    t=slab_threshold_time(100,0,50,.01,1e-6,0.)
    assert t>0
    assert slab_threshold_time(100,0,50,.02,1e-6,0.)==pytest.approx(4*t)
    assert segment_distance([0,0,0],[1,0,0],[0,2,0],[1,2,0])==pytest.approx(2.)

def test_grille_window_and_flow():
    from engineering_sim.grilles import ply_window,line_of_sight,ply_flow
    assert ply_window(3,1,2,0)==pytest.approx((-1,1))
    plies=[dict(dir_deg=0,pitch_mm=3,slat_mm=1,depth_mm=2,tilt_deg=0)]
    assert line_of_sight(plies)
    with pytest.raises(ValueError): line_of_sight([dict(plies[0],dir_deg=i) for i in (0,30,60)])
    r=ply_flow(.001,10,2,3,50,dict(density_kg_m3=1.2,mu_pa_s=1.8e-5,k_w_mk=.026,pr=.71),1.)
    assert r['velocity_m_s']==pytest.approx(.001/(1.2*60e-6),abs=.001)
    assert r['dp_pa']>0

def test_vlm_dimensional_scale_and_zero_lift():
    from engineering_sim.flow.vlm import solve
    from engineering_sim.flow.aero import LatticeSurface
    surface=LatticeSurface('synthetic',lambda s,x:[x,2*s-1,0],8)
    zero=solve([surface],[10,0,0],1.2,[0,0,0],chord_panels=3)
    assert np.linalg.norm(zero['F_g'])<1e-10
    a=solve([surface],[10,0,1],1.2,[0,0,0],chord_panels=3)
    b=solve([surface],[10,0,1],1.2,[0,0,0],chord_panels=3,scale=2)
    assert np.linalg.norm(a['F_g'])>0
    assert b['F_g']==pytest.approx(4*a['F_g'],rel=1e-10)
    assert a['relative_boundary_residual']<1e-10

def test_linear_modes_and_aeroelastic_units():
    from engineering_sim.flight_dynamics import eigenmode_metrics
    from engineering_sim.structures import effective_gj,divergence_pressure,reversal_pressure,flap_coefficients
    modes=eigenmode_metrics([-1+2j,-2+0j,.5+0j])
    assert modes[0]['zeta']==pytest.approx(1/math.sqrt(5))
    assert modes[1]['tau_s']==.5 and modes[2]['t2_s']==pytest.approx(2*math.log(2))
    assert effective_gj(np.linspace(0,1,10),np.ones(10)*100,np.ones(10))==pytest.approx(100)
    assert divergence_pressure(100,.05,1,2*math.pi,1)==pytest.approx(250*math.pi)
    assert reversal_pressure(100,1,-.2,1,2*math.pi,1)>0
    assert flap_coefficients(.7)['CL_delta']>0

def test_magnetic_bearing_zero_and_odd_force():
    from engineering_sim.electromagnetics.magnetic_bearing import nonlinear_axis_force,_allocate
    info=dict(pole_normal_cosine=1,gap_m=.001,turns=100,pole_face_area_mm2=100)
    assert nonlinear_axis_force(info,1,1,0)==0
    assert nonlinear_axis_force(info,2,1,0)==pytest.approx(-nonlinear_axis_force(info,1,2,0))
    assert _allocate([4,0,1,0,0],-1,1)==pytest.approx([2,0,2,0,1])

def test_gimbal_dynamic_settles():
    from engineering_sim.domains.motion_control.core_gimbal import GimbalPlant
    plant=GimbalPlant({'pan':(-90,90)},{'pan':0},payload={'d_cg_m':0})
    plant.command('pan',10)
    for _ in range(200):plant.advance(.01)
    assert plant.th['pan']==pytest.approx(10,abs=.01)

def test_dimension_parser():
    from engineering_sim.vision.dimensions import parse
    result=parse('DIA 10 ± 0.1 mm')
    assert result
    assert '10' in str(result)

def test_route_graph_shortest_paths():
    from engineering_sim.checks.mold_routing_study import paths
    graph={'nodes':[{}, {}, {}], 'edges':{0:[(1,1,1)],1:[(0,1,1),(2,1,1)],2:[(1,1,1)]}}
    r=paths(graph,[0])
    assert r['costs']==[0,1,2] and r['max_path_mm']==2
    with pytest.raises(ValueError): paths({'nodes':[{},{}],'edges':{0:[],1:[]}},[0])

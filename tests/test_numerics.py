"""Synthetic problems with analytic invariants and independent physical references."""
import math
import numpy as np
import pytest
import warp as wp
wp.config.quiet=True

def test_actual_world_particle_freefall():
    from engineering_sim.demo import run_world
    from engineering_sim.execution import capture
    with capture() as records:
        r=run_world(.2,.001,'cpu')
    assert r['steps']==200
    assert r['time_s']==pytest.approx(.2)
    assert r['velocity_m_s'][2]==pytest.approx(-9.80665*.2,rel=1e-4)
    assert r['position_m'][2]==pytest.approx(10-.5*9.80665*.2**2,abs=.0015)
    assert any(x['operation']=='demo.world' and x['state']=='complete' for x in records)

def test_standard_atmosphere():
    from engineering_sim.core.air_np import ref_state
    states=ref_state([0.,1000.,11000.],geometric=False)
    assert states['rho'][0]==pytest.approx(1.225,rel=2e-4)
    assert states['p_pa'][2]==pytest.approx(22632.1,rel=2e-4)
    assert np.all(np.diff(states['rho'])<0)

def test_water_density_and_dispersion():
    from engineering_sim.domains.water import props,waves
    assert props.density(20,0)==pytest.approx(998.21,abs=.02)
    assert props.density(15,35)==pytest.approx(1025.97,abs=.05)
    w=np.array([.5,1.,2.,4.]);k=waves.dispersion_k(w,5.)
    assert np.max(abs(9.80665*k*np.tanh(k*5)-w*w))<1e-9

def test_hull_displaced_volume():
    from engineering_sim.domains.water import hull
    length,width,depth=2.,1.,.5
    cols=hull.columns_from_mesh(*hull.box_mesh(length,width,depth),nx=32,ny=16)
    rows=hull.volume_curve(cols,[.1,.3])
    for r in rows:
        assert r['volume_m3']==pytest.approx(length*width*r['waterline_m'],rel=1e-4)

def test_wave_kernel_against_host_sum():
    from engineering_sim.domains.water.waves import SeaState
    sea=SeaState(Hs=.3,Tp=3.,n_freq=8,n_dir=4,seed=13)
    samples=sea.probe_time(2.,3.,0.,0.,.1,40)[:,0]
    reference=sea.eta_np(2.,3.,.1*np.arange(40))
    assert np.max(abs(samples-reference))<1e-4

def test_orbit_conserves_two_body_energy():
    from engineering_sim.domains.space import frames,forces,constants
    r,v=frames.kepler_to_rv(7000000.,.01,.5,.2,.1,.3)
    t,rr,vv=forces.rk4_host(r,v,2451545.,10.,600,terms=(),every=10)
    energy=.5*np.sum(vv[:,0]**2,axis=1)-constants.MU_EARTH/np.linalg.norm(rr[:,0],axis=1)
    assert np.max(abs(energy/energy[0]-1))<1e-9

def test_warp_orbit_matches_host_rk4():
    from engineering_sim.domains.space import frames,forces,propagate
    r,v=frames.kepler_to_rv(7000000.,.01,.5,.2,.1,.3)
    t,rr,vv=forces.rk4_host(r,v,2451545.,5.,60,terms=(),every=10)
    batch=propagate.OrbitBatch(r,v,2451545.,terms=(),device='cpu')
    _,rg,vg=batch.run(5.,60,every=10)
    assert np.linalg.norm(rg[-1,0]-rr[-1,0])<1e-4
    assert np.linalg.norm(vg[-1,0]-vv[-1,0])<1e-7

def test_room_sabine_and_distance_law():
    from engineering_sim.domains.room_acoustics.room import ShoeboxRoom
    room=ShoeboxRoom(4.,3.,2.5,.2,c=343.)
    expected=24*math.log(10)*30/(343*(2*(12+10+7.5)*.2))
    assert room.t60_sabine()==pytest.approx(expected)
    assert room.drr_diffuse(room.critical_distance())==pytest.approx(1.)
    assert room.drr_diffuse(1)/room.drr_diffuse(2)==pytest.approx(4.)

def test_magnetic_field_linear_in_remanence():
    from engineering_sim.electromagnetics.permanent_magnet import solve_field
    import json
    from pathlib import Path
    args=json.loads((Path(__file__).parents[1]/'examples/magnetic_field.json').read_text())
    a=solve_field(**args)
    doubled=dict(args,geometry=dict(args['geometry'],br_t=2.))
    b=solve_field(**doubled)
    assert a['b_r_t']>0
    assert b['b_r_t']==pytest.approx(2*a['b_r_t'],rel=1e-11)

def test_rejection_of_incomplete_field_map():
    from engineering_sim.electromagnetics.reluctance_drive import PhaseMap
    with pytest.raises(ValueError,match='Incomplete'):
        PhaseMap(dict(parameters={},topology=dict(phases=3,rotor_poles=4,stator_poles=6),rows=[dict(state='failed')]))

def test_reluctance_map_coenergy_and_drive():
    from engineering_sim.electromagnetics.reluctance_drive import PhaseMap,simulate
    rows=[]
    for angle in np.linspace(0,45,10):
        inductance=.02+.01*math.cos(math.radians(angle*4))
        for current in [1.,2.,4.,8.]:
            rows.append(dict(state='complete',angle=float(angle),result=dict(current_a=current,flux_linkage_wb_turn=inductance*current,coenergy_j=.5*inductance*current*current)))
    m=PhaseMap(dict(parameters={'synthetic':True},topology=dict(phases=3,rotor_poles=4,stator_poles=6),rows=rows,peak_current_a=2.,resistance_20c_ohm=1.))
    assert m.coenergy_integral_relative<1e-12
    result=simulate(m,voltage=24.,rpm=600.,cycles=8,step_deg=.5,temperature_c=20.)
    assert result['peak_current_a']<=2.2
    assert result['periodic_flux_error_wb_turn']<1e-5
    assert result['energy_balance_relative']<.04

def test_air_properties_and_flow_correlations():
    from engineering_sim.flow.manifold_fluids import properties
    water=properties('water',20.,101325.)
    air=properties('air',20.,101325.)
    assert water['density_kg_m3']==pytest.approx(998.21,abs=.02)
    assert water['dynamic_viscosity_pa_s']==pytest.approx(.001002,rel=.003)
    assert air['density_kg_m3']==pytest.approx(1.2041,rel=.001)

import math
import numpy as np
import pytest
from engineering_sim.checks import mechanisms,tolerances,rules,seals
from engineering_sim.checks.transforms import transform

def test_tolerance_bounds_and_rss():
    assert tolerances.fit_clearance((.01,.03),(-.02,0.))==pytest.approx((.01,.05))
    r=tolerances.stack([dict(sensitivity=1,tolerance_mm=.03),dict(sensitivity=-1,tolerance_mm=.04)])
    assert r==pytest.approx(dict(worst_case_mm=.07,rss_mm=.05))
    assert tolerances.thermal_shift(1e-5,100,100,2e-5,100,100)==pytest.approx(-.1)

def test_joint_closure_and_wrong_angle():
    angle=.4;pose=transform([0,1,0],[0,0,0],angle)
    r=mechanisms.joint_residual(np.eye(4),pose,np.eye(4),np.eye(4),angle)
    assert r['position_mm']<1e-12 and r['angle_radians']<1e-12
    bad=mechanisms.joint_residual(np.eye(4),pose,np.eye(4),np.eye(4),angle+.1)
    assert bad['angle_radians']==pytest.approx(.1)

def test_mobility_and_locked_hinge():
    owners=['base','arm'];joints=[dict(parent='base',child='arm',axis=[0,0,1],pivot=[0,0,0])]
    assert mechanisms.mobility(owners,joints,fixed_owner='base')['nullity']==1
    assert mechanisms.mobility(owners,joints,fixed_owner='base',locked=True)['nullity']==0

def test_nonrigid_pose_is_not_accepted_as_closure():
    affine=np.diag([2.,1.,1.,1.]);r=mechanisms.residual(affine,affine)
    assert r['rotation_orthogonality_error']>0 and r['rotation_determinant_error']>0

def test_dimensional_rules_reject_bad_wall_and_unknown_material():
    r=rules.evaluate([dict(type='min_wall',param='wall',min_mm=2.)],dict(wall=1.),'synthetic',{}, {})
    assert not r['ok'] and r['violations'][0]['suggested_legal']==2.
    rule=dict(type='cantilever_bending',load_param='F',length_param='L',width_param='b',thickness_param='t')
    r=rules.evaluate([rule],dict(F=1,L=10,b=10,t=2),'missing',{}, {})
    assert not r['ok']

def test_gland_formula_with_explicit_tolerance():
    r=seals.radial_gland_check(2.,20.,17.,3.,.01,0.,.02)
    assert r['squeeze']==pytest.approx(.25)
    assert r['fill']==pytest.approx(math.pi/4.5)
    assert r['squeeze_min']<r['squeeze']<r['squeeze_max']

def test_generic_thermal_and_section_relations():
    from engineering_sim import structures,thermal
    assert structures.rect_area_moment(.02,.03)==pytest.approx(.02*.03**3/12)
    assert structures.tip_deflection_uniform(10.,1.,1000.)==pytest.approx(.00125)
    assert thermal.hoop_stress(1e6,.1,.01)==pytest.approx(1e7)
    assert thermal.oil_flow_for(100.,dict(rho=1000.,cp=1000.),10.)==pytest.approx(1e-5)

def test_involute_pitch_and_backlash():
    from engineering_sim.cad.gears import profile,involute
    g=profile(24,1.,.1)
    assert g['pitch']==12.
    assert g['base']==pytest.approx(12*math.cos(math.radians(20)))
    assert 2*g['pitch']*g['angle'](g['pitch'])==pytest.approx(math.pi/2-.05)
    assert involute(g['base'],g['base'])==0.

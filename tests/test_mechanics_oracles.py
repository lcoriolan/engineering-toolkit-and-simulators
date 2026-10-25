"""Independent hand calculations and attributed reference cases, with declared tolerances."""
import pytest
from engineering_sim.science.mech import core,oracles
CASES=[case for data in oracles.load().values() for case in data['cases']]
@pytest.mark.parametrize('case',CASES,ids=[str(i)+'-'+c['calc'] for i,c in enumerate(CASES)])
def test_independent_reference(case):
    ok,detail=oracles.check_case(case)
    assert ok,detail

def test_registry_and_oracle_coverage():
    reg=core.registry()
    assert reg['errors']=={}
    assert len(reg['sections'])==50
    assert {c['calc'] for c in CASES}==set(reg['calcs'])

def test_dimensions_and_examples():
    for calc in core.registry()['calcs'].values():
        assert core.dimension_check(calc)==[],calc.id
        result=core.compute(calc.id,calc.example)
        assert set(calc.outputs)<=set(result['results']),calc.id

def test_wrong_reference_is_rejected():
    ok,_=oracles.check_case({'calc':'statics.moment','inputs':{'F':'10 N','d':'2 m'},'expect':{'M':'21 N*m'},'source':{'kind':'hand','derivation':'deliberately wrong'}})
    assert not ok

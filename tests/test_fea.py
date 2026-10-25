"""Small analytic FEM benchmarks on newly generated unit geometry."""
import math
import numpy as np
import pytest
import ngsolve as ng
from netgen.geom2d import unit_square
from engineering_sim import fea

def test_poisson_manufactured_magnetostatic_solution():
    mesh=ng.Mesh(unit_square.GenerateMesh(maxh=.12))
    exact=ng.sin(math.pi*ng.x)*ng.sin(math.pi*ng.y)
    result=fea.magnetostatic_2d(mesh,1.,2*math.pi**2*exact,'.*',order=3)
    error=math.sqrt(ng.Integrate((result['potential']-exact)**2,mesh))
    assert error<1e-4
    assert result['energy_j_per_m']==pytest.approx(math.pi**2/4,rel=1e-3)

def test_elastic_uniform_extension():
    mesh=ng.Mesh(unit_square.GenerateMesh(maxh=.15))
    # nu=0 removes lateral contraction; u=(traction/E*x,0) is exact.
    result=fea.elasticity(mesh,1000.,0.,'left','right',(10.,0.),order=2)
    exact=ng.CoefficientFunction((.01*ng.x,0.))
    error=math.sqrt(ng.Integrate(ng.InnerProduct(result['displacement']-exact,result['displacement']-exact),mesh))
    assert error<1e-8
    assert result['strain_energy_j']==pytest.approx(.05,rel=1e-8)

def test_vector_potential_zero_source():
    from netgen.csg import unit_cube
    mesh=ng.Mesh(unit_cube.GenerateMesh(maxh=.6))
    result=fea.magnetostatic_3d(mesh,1.,ng.CoefficientFunction((0.,0.,0.)),'.*')
    assert result['energy_j']==pytest.approx(0.,abs=1e-20)
    assert np.linalg.norm(result['potential'].vec.FV().NumPy())<1e-15

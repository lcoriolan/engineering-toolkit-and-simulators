"""Finite-element operators on caller-provided meshes; no design geometry is bundled.
All lengths and constitutive inputs use coherent SI units. NGSolve owns assembly/solving.
"""
from .execution import operation
@operation('fea.linear_elasticity',kind='static')
def elasticity(mesh,young_pa,poisson,fixed,traction_boundary,traction_pa,order=2):
    import ngsolve as ng
    if young_pa<=0 or not -1<poisson<.5:raise ValueError('Invalid isotropic elastic material.')
    fes=ng.VectorH1(mesh,order=order,dirichlet=fixed);u,v=fes.TnT()
    mu=young_pa/(2*(1+poisson));lam=young_pa*poisson/((1+poisson)*(1-2*poisson))
    def eps(x):return ng.Sym(ng.Grad(x))
    def stress(x):return 2*mu*eps(x)+lam*ng.Trace(eps(x))*ng.Id(mesh.dim)
    a=ng.BilinearForm(fes);a+=ng.InnerProduct(stress(u),eps(v))*ng.dx
    f=ng.LinearForm(fes);f+=ng.InnerProduct(ng.CoefficientFunction(tuple(traction_pa)),v)*ng.ds(definedon=mesh.Boundaries(traction_boundary))
    a.Assemble();f.Assemble();g=ng.GridFunction(fes);g.vec.data=a.mat.Inverse(fes.FreeDofs(),inverse='sparsecholesky')*f.vec
    return dict(displacement=g,stress=stress(g),strain_energy_j=float(.5*ng.InnerProduct(g.vec,a.mat*g.vec)),dofs=fes.ndof)
@operation('fea.magnetostatic_2d',kind='static')
def magnetostatic_2d(mesh,reluctivity,current_density,boundary,order=2):
    import ngsolve as ng
    fes=ng.H1(mesh,order=order,dirichlet=boundary);u,v=fes.TnT()
    a=ng.BilinearForm(fes);a+=reluctivity*ng.InnerProduct(ng.grad(u),ng.grad(v))*ng.dx
    f=ng.LinearForm(fes);f+=current_density*v*ng.dx
    a.Assemble();f.Assemble();g=ng.GridFunction(fes);g.vec.data=a.mat.Inverse(fes.FreeDofs(),inverse='sparsecholesky')*f.vec
    b=ng.CoefficientFunction((ng.grad(g)[1],-ng.grad(g)[0]))
    return dict(potential=g,flux_density=b,energy_j_per_m=float(ng.Integrate(.5*reluctivity*ng.InnerProduct(b,b),mesh)),dofs=fes.ndof)
@operation('fea.magnetostatic_3d',kind='static')
def magnetostatic_3d(mesh,reluctivity,current_density,boundary,order=1,gauge=1e-7):
    import ngsolve as ng
    fes=ng.HCurl(mesh,order=order,dirichlet=boundary,nograds=True);u,v=fes.TnT()
    a=ng.BilinearForm(fes);a+=(reluctivity*ng.InnerProduct(ng.curl(u),ng.curl(v))+gauge*ng.InnerProduct(u,v))*ng.dx
    f=ng.LinearForm(fes);f+=ng.InnerProduct(current_density,v)*ng.dx
    a.Assemble();f.Assemble();g=ng.GridFunction(fes);g.vec.data=a.mat.Inverse(fes.FreeDofs(),inverse='sparsecholesky')*f.vec
    b=ng.curl(g)
    return dict(potential=g,flux_density=b,energy_j=float(ng.Integrate(.5*reluctivity*ng.InnerProduct(b,b),mesh)),dofs=fes.ndof)

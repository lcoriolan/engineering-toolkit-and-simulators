# DESCRIPTION: Dimensionally consistent vortex-core kernel and bounded NumPy lattice solver.
"""SI units throughout: metres, seconds, kg, circulation m²/s, core radius m.

The former kernel applied one smoothing parameter to inverse quantities with
different dimensions (length and length squared). On small fin panels this
distorts the influence matrix as it is refined. Here each core correction has
the dimensions of its denominator, and uniform length scaling is testable.
The core is numerical desingularization, not a measured viscous wake model.
"""
import math
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.linalg.lapack import dgecon

VERSION = '1.0.0'


def horseshoe(field, left, right, core_m=1e-7, wake_direction=(1.,0.,0.)):
    """Unit-circulation velocity at M fields from N horseshoes; output M×N×3.

    Bound circulation runs left→right. The left wake points inward from
    infinity; the right wake points outward. A finite-segment Biot–Savart
    kernel uses cross² + core²*segment_length² (both m⁴). Wake denominators
    use perpendicular_distance² + core² (both m²). Collinear self terms
    vanish through their vector numerators without division by zero.
    """
    if not np.isfinite(core_m) or core_m<=0:raise ValueError('Positive finite core required')
    field=np.asarray(field,float);left=np.asarray(left,float);right=np.asarray(right,float)
    direction=np.asarray(wake_direction,float);direction/=np.linalg.norm(direction)
    a=field[:,None,:]-left[None,:,:];b=field[:,None,:]-right[None,:,:]
    segment=right-left;length2=np.sum(segment*segment,axis=1)
    if np.any(length2<=0):raise ValueError('Zero-length bound vortex')
    na=np.sqrt(np.sum(a*a,axis=2)+core_m**2)
    nb=np.sqrt(np.sum(b*b,axis=2)+core_m**2)
    cross=np.cross(a,b)
    angle=np.sum(segment[None,:,:]*(a/na[:,:,None]-b/nb[:,:,None]),axis=2)
    denominator=np.sum(cross*cross,axis=2)+core_m**2*length2[None,:]
    bound=cross*(angle/denominator)[:,:,None]
    # Cross products calculate perpendicular distances without subtracting
    # nearly equal squared lengths along long downstream wakes.
    ua=np.cross(direction,a);ub=np.cross(direction,b)
    da=np.sum(ua*ua,axis=2)+core_m**2;db=np.sum(ub*ub,axis=2)+core_m**2
    wa=(1+np.sum(a*direction,axis=2)/na)/da
    wb=(1+np.sum(b*direction,axis=2)/nb)/db
    return (bound-ua*wa[:,:,None]+ub*wb[:,:,None])/(4*math.pi)


def solve(surfaces, velocity, density, reference, chord_panels=4, core_m=1e-7,
          scale=1., diagnostics=False):
    """Solve attached, incompressible flow on explicit mean-surface panels.

    Surface adapters supply metre-space quads FL,BL,BR,FR. Velocity is a
    geometry-axis vector (m/s), density kg/m³ and reference a metre-space point.
    Scale changes every geometric length, including the core, for invariance
    tests. Results then have the correspondingly scaled force/moment units.
    Chunk field evaluations to limit temporary pair-array memory usage.
    """
    points=[]
    def cosine(a,b,n):
        """Cosine chord spacing matches the existing geometry adapter convention."""
        return a+(b-a)*(1-np.cos(np.linspace(0,math.pi,n)))/2
    for surface in surfaces:
        vertices,faces=surface.mesh_thin_surface('quad',chord_panels,cosine,True)
        points.append(np.asarray(vertices)[faces]*scale)
    quad=np.concatenate(points);fl,bl,br,fr=[quad[:,i,:] for i in range(4)]
    cross=np.cross(fr-bl,fl-br);area2=np.linalg.norm(cross,axis=1)
    if np.any(area2<=0):raise ValueError('Degenerate lattice panel')
    normals=cross/area2[:,None]
    left=.75*fl+.25*bl;right=.75*fr+.25*br;centre=(left+right)/2
    collocation=(.25*(fl+fr)+.75*(bl+br))/2
    velocity=np.asarray(velocity,float);reference=np.asarray(reference,float)*scale
    core=core_m*scale;n=len(quad);matrix=np.empty((n,n))
    for start in range(0,n,128):
        stop=min(start+128,n)
        induced=horseshoe(collocation[start:stop],left,right,core)
        matrix[start:stop]=np.einsum('ijk,ik->ij',induced,normals[start:stop])
    rhs=-normals@velocity
    factors,pivots=lu_factor(matrix);gamma=lu_solve((factors,pivots),rhs)
    residual=np.linalg.norm(matrix@gamma-rhs,np.inf)/max(np.linalg.norm(rhs,np.inf),1.)
    inverse_condition,info=dgecon(factors,np.linalg.norm(matrix,1))
    if info or not np.all(np.isfinite(gamma)):raise ValueError('Influence solve failed')
    velocities=np.empty((n,3))
    for start in range(0,n,128):
        stop=min(start+128,n)
        velocities[start:stop]=velocity+np.einsum('ijk,j->ik',horseshoe(centre[start:stop],left,right,core),gamma)
    forces=density*np.cross(velocities,right-left)*gamma[:,None]
    moments=np.cross(centre-reference,forces)
    fg=forces.sum(axis=0);mg=moments.sum(axis=0)
    result=dict(F_g=fg,M_g=mg,M_b=mg*np.array([-1,1,-1]),
                panel_count=n,relative_boundary_residual=residual,
                reciprocal_condition_estimate=inverse_condition,
                max_induced_speed_m_s=float(np.linalg.norm(velocities-velocity,axis=1).max()),
                core_m=core,solver_version=VERSION)
    if diagnostics:result.update(circulation=gamma,panel_forces=forces,panel_centres=centre)
    return result

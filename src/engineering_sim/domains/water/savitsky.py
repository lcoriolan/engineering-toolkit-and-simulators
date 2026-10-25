#!/usr/bin/env python3
# DESCRIPTION: Independent closed-form/bisection solution of Savitsky's equations for a prismatic hull, in the same
#              force arrangement the GPU vessel uses (normal force N at the centre of pressure along the keel normal,
#              friction Df along the keel acting b/4 tan(beta) above the keel, thrust horizontal through the CG):
#                C_L0  = tau^1.1 (0.012 lam^0.5 + 0.0055 lam^2.5 / Cv^2)        (tau in degrees)
#                C_Lb  = C_L0 - 0.0065 beta C_L0^0.6
#                lp/(lam b) = 0.75 - 1/(5.21 Cv^2/lam^2 + 2.39)
#                Df = 1/2 rho u^2 (lam b^2 / cos beta) (C_F,ITTC57(Re = u lam b / nu) + dCf)
#              Equilibrium: vertical  N cos(tau) - Df sin(tau) = m g;  pitch moment about CG = 0;
#                           horizontal T = N sin(tau) + Df cos(tau) (= total resistance R).
#              u = V cos(tau) is the flow speed along the keel (the GPU model's definition).
import math

from engineering_sim.core.constants import G0 as G

from .props import ittc57


def forces(V, tau_deg, lam, b, beta_deg, rho, nu, dCf=0.0004, g=G):
    tr, br = math.radians(tau_deg), math.radians(beta_deg)
    u = V * math.cos(tr)
    Cv = u / math.sqrt(g * b)
    CL0 = tau_deg ** 1.1 * (0.012 * lam ** 0.5 + 0.0055 * lam ** 2.5 / Cv ** 2)
    CLb = max(CL0 - 0.0065 * beta_deg * CL0 ** 0.6, 0.0)
    N = 0.5 * rho * u * u * b * b * CLb / math.cos(tr)
    lp = lam * b * (0.75 - 1.0 / (5.21 * Cv ** 2 / lam ** 2 + 2.39))
    Sw = lam * b * b / math.cos(br)
    Df = 0.5 * rho * u * u * Sw * (ittc57(u * lam * b / nu) + dCf)
    return dict(N=N, Df=Df, lp=lp, Cv=Cv, CL0=CL0, CLb=CLb, u=u, Sw=Sw)


def equilibrium(V, mass, b, beta_deg, lcg, vcg, rho=1025.0, nu=1.19e-6, dCf=0.0004, g=G):
    """lcg: CG forward of the transom [m]; vcg: CG above the keel [m]. Returns tau, lam, R, ..."""
    br = math.radians(beta_deg)

    def lam_for(tau):
        lo, hi = 1e-3, 30.0
        tr = math.radians(tau)
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            F = forces(V, tau, mid, b, beta_deg, rho, nu, dCf, g)
            if F["N"] * math.cos(tr) - F["Df"] * math.sin(tr) < mass * g:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def moment(tau):
        lam = lam_for(tau)
        F = forces(V, tau, lam, b, beta_deg, rho, nu, dCf, g)
        # body frame (x fwd, z up), about the CG; +My = bow DOWN. N up along body z at (lp - lcg, -vcg);
        # Df aft along body x at (lp - lcg, b/4 tan(beta) - vcg)
        rx = F["lp"] - lcg
        My_N = -rx * F["N"]                      # (r x F)_y = rz*Fx - rx*Fz
        rz_f = 0.25 * b * math.tan(br) - vcg
        My_f = rz_f * (-F["Df"])
        return My_N + My_f, lam, F

    lo, hi = 0.5, 15.0
    mlo = moment(lo)[0]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        mm = moment(mid)[0]
        if (mm > 0) == (mlo > 0):
            lo, mlo = mid, mm
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    M, lam, F = moment(tau)
    tr = math.radians(tau)
    R = F["N"] * math.sin(tr) + F["Df"] * math.cos(tr)
    return dict(V=V, tau_deg=tau, lam=lam, R_N=R, N=F["N"], Df=F["Df"], lp=F["lp"], Cv=F["Cv"], CLb=F["CLb"],
                residual_moment=M, keel_wetted_length=lam * b + b * math.tan(br) / (2 * math.pi * math.tan(tr)))

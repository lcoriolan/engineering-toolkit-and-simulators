#!/usr/bin/env python3
# DESCRIPTION: StackAero(rocket, n_top): aerodynamics of the attached stack (stages 0..n_top-1), vectorised in Mach and
#              Reynolds number (numpy), referenced to A_ref = pi d_ref^2 / 4, d_ref = largest body diameter.
#
#   NORMAL FORCE / CP (small angle; per component, summed; Barrowman 1967 NTRS 20010047838 + OpenRocket, Niskanen 2013)
#     nose        CNa = 2 (d_base/d_ref)^2 (Barrowman eq. 3-66, any pointed slender nose, Mach-independent in slender-
#                 body theory); x_cp = L - V_nose / A_base (Barrowman; cone -> 2/3 L exactly)
#     transition  CNa = 2 [(d2/d_ref)^2 - (d1/d_ref)^2]; x_cp = x0 + L/3 [1 + (1 - d1/d2) / (1 - (d1/d2)^2)]
#     fins        one fin, subsonic (Diederich / OpenRocket eq. 3.40, Prandtl-Glauert beta inside):
#                   CNa1 = 2 pi (s^2/A_ref) / (1 + sqrt(1 + (beta s^2 / (A_fin cos Gc))^2)),  M <= 0.8
#                 supersonic (Ackeret linear theory, finite-span tip-Mach-cone correction, AR = 2 s^2 / A_fin):
#                   CNa1 = (A_fin/A_ref) (4/beta) (1 - 1/(2 AR beta)),   M >= 1.5   (floor at 50 % of 2-D value)
#                 0.8 < M < 1.5: cubic Hermite in M between the two branches (OpenRocket also interpolates here;
#                 ASSUMED shape). Fin set = K_int * (N/2) * f_N * CNa1, K_int = 1 + R/(s + R) (Barrowman body-fin
#                 interference), f_N fin-fin interference (OpenRocket: 1.0 for N <= 4, 0.948 5, 0.913 6, 0.854 7, 0.81 8).
#                 Fin CP along the MAC: 0.25 MAC (M <= 0.5) -> (AR beta - 0.67)/(2 AR beta - 1) MAC (M >= 2),
#                 smooth in between (OpenRocket eq. 3.41-3.42 form).
#     body lift   nonlinear C_N = 1.1 (A_plan/A_ref) sin^2(alpha) at the planform centroid (OpenRocket eq. 3.26);
#                 zero slope at alpha = 0 (does not change the static margin), used in the 6-DOF at larger alpha.
#
#   DRAG (zero lift; Niskanen 2013 ch. 3 build-up, Barrowman 1967 base drag; all referenced to A_ref)
#     friction    Re on total length; Cf = max(turbulent 1/(1.50 ln Re - 5.6)^2, roughness-limited 0.032 (Rs/L)^0.2)
#                 above Re_crit = 51 (Rs/L)^-1.039, laminar 1.328/sqrt(Re) below 1e4 is replaced by 1.48e-2 (OpenRocket);
#                 compressibility: subsonic x (1 - 0.1 M^2); supersonic turbulent / (1 + 0.15 M^2)^0.58,
#                 roughness-limited / (1 + 0.18 M^2). Body wetted area x (1 + 1/(2 fB)), fins 2 sides x (1 + 2 t/MAC).
#     base        subsonic Hoerner (as used by Barrowman 1967): C_DB = 0.029 (d_b/d_ref)^3 / sqrt(C_Df,body);
#                 supersonic 0.25/M (OpenRocket eq. 3.94), capped at the vacuum-base limit 2/(gamma M^2); the rise
#                 from the subsonic value is a cubic over M 0.9 -> 1.0 (window CALIBRATED on NACA RM L50I08a). While a motor burns, the
#                 base area is reduced by the nozzle (motor) area (OpenRocket convention).
#     nose wave   cone (Hoerner, as OpenRocket eq. 3.86): M >= 1.3: 2.1 sin^2 phi + 0.5 sin phi / sqrt(M^2 - 1), phi =
#                 equivalent half angle atan(R/L); subsonic 0.8 sin^2 phi (cone), 0 for rounded profiles; 0.8 < M < 1.3
#                 cubic Hermite. Shape factor vs cone of equal fineness (ASSUMED from slender-body wave-drag ratios):
#                 cone 1.0, ogive 0.75, parabolic 0.65, vonkarman/haack 0.55, elliptical: blunt (stagnation, see LE).
#     transition  boattail: base area reduced to its aft diameter (pressure recovery neglected, ASSUMED);
#                 forward-facing (shoulder/interstage flare): cone formula on the step area.
#     fin LE      rounded (OpenRocket eq. 3.88-3.89): C_LE = (1 - M^2)^-0.417 - 1 (M < 0.9), 1 - 1.785 (M - 0.9)
#                 (0.9 <= M < 1), 1.214 - 0.502/M^2 + 0.1095/M^4 (M >= 1), x cos^2(Gamma_LE), on the fin frontal area;
#                 square: stagnation 0.85 q_stag/q; wedge/airfoil: sharp, Ackeret wave drag supersonic
#                 C_D = (2/beta) * 2 eps^2 * (l_LE + l_TE)/c on the planform (eps = edge half angle).
#     fin TE      square: base-drag formula on the TE area; tapered/wedge: 0.
#     extra       cd_extra (spec, default 0) for launch lugs, rail buttons, cameras (ASSUMED by the designer).
#   DECLARED VALIDITY: M <= 5 (zero-lift drag). Above it the build-up under-predicts the Aerobee 150A tunnel CD
#   (-30 % at M 7.2; the raceway shrouds are not modelled and the hypersonic terms are not in this method).
#   Barrowman states his method within 10 % of experiment for M 2-8; OpenRocket documents ~10-20 % on subsonic
#   CD for typical hobby rockets. Validation against NACA RM L50I08a (free flight, M 0.75-1.5) and the Aerobee 150A
#   tunnel data (M 2.2-7.2) is in tests/test_rocket_design.py and proof/ROCKET_DESIGN_2026-10-16.md.
import math

import numpy as np

SHAPE_WAVE = dict(cone=1.0, ogive=0.75, parabolic=0.65, vonkarman=0.55, haack=0.55, elliptical=1.0)
FIN_FN = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.948, 6: 0.913, 7: 0.854, 8: 0.81}
M_SUB, M_SUP = 0.8, 1.5
M_BASE0 = 0.9    # base-drag rise window 0.9 -> 1.0 (CALIBRATED on NACA RM L50I08a model B, the only transonic anchor)


def _hermite(M, m0, m1, f0, d0, f1, d1):
    t = np.clip((M - m0) / (m1 - m0), 0.0, 1.0)
    h = m1 - m0
    return ((2 * t ** 3 - 3 * t ** 2 + 1) * f0 + (t ** 3 - 2 * t ** 2 + t) * h * d0
            + (-2 * t ** 3 + 3 * t ** 2) * f1 + (t ** 3 - t ** 2) * h * d1)


def _beta(M):
    return np.sqrt(np.abs(1.0 - M * M))


class StackAero:
    def __init__(self, rocket, n_top, cd_extra=None):
        self.r = rocket
        self.n_top = n_top
        stages = rocket.stack(n_top)
        self.d_ref = max(st.d for st in stages)
        self.a_ref = math.pi * self.d_ref ** 2 / 4.0
        self.L = stages[-1].x_aft
        self.cd_extra = float(rocket.spec.get("cd_extra", 0.0) if cd_extra is None else cd_extra)
        self.noses, self.trans, self.fins, self.tubes = [], [], [], []
        for st in stages:
            for c in st.comps:
                if c.kind == "nose":
                    self.noses.append(c)
                elif c.kind == "transition":
                    self.trans.append(c)
                elif c.kind == "fins":
                    self.fins.append(c)
                elif c.kind == "tube" and c.name == "body":
                    self.tubes.append(c)
        last = stages[-1]
        self.d_base = last.d_aft
        self.motor_d = last.motor.dia_m if last.motor is not None else 0.0
        # wetted areas
        self.wet_body = sum(n.wet for n in self.noses) + sum(math.pi * t.d * t.length for t in self.tubes) \
            + sum(c.wet for c in self.trans)
        self.wet_fins = sum(2.0 * f.n * f.A for f in self.fins)
        self.fB = self.L / self.d_ref
        self.A_plan = sum(self._planform(c) for c in self.noses + self.tubes + self.trans)
        self.x_plan = (sum(self._planform(c) * self._plan_x(c) for c in self.noses + self.tubes + self.trans)
                       / max(self.A_plan, 1e-12))

    # ---------------------------------------------------------------- geometry helpers
    @staticmethod
    def _planform(c):
        if c.kind == "nose":
            return c.plan
        if c.kind == "transition":
            return 0.5 * (c.d1 + c.d2) * c.length
        return c.d * c.length

    @staticmethod
    def _plan_x(c):
        if c.kind == "nose":
            return c.plan_x
        return c.x0 + 0.5 * c.length

    @staticmethod
    def fin_geom(f):
        Cr, Ct, s, Xs = f.root, f.tip, f.span, f.sweep
        Lf = math.sqrt(s * s + (Xs + 0.5 * Ct - 0.5 * Cr) ** 2)
        mac = (2.0 / 3.0) * (Cr + Ct - Cr * Ct / (Cr + Ct))
        x_mac = Xs * (Cr + 2 * Ct) / (3.0 * (Cr + Ct))
        AR = 2.0 * s * s / f.A
        gam_le = math.atan2(Xs, s)
        cos_gc = s / Lf
        return dict(Lf=Lf, mac=mac, x_mac=x_mac, AR=AR, gam_le=gam_le, cos_gc=cos_gc)

    # ---------------------------------------------------------------- normal force
    def _cna1(self, f, M):
        g = self.fin_geom(f)
        s, A = f.span, f.A

        def sub(m):
            b = _beta(np.minimum(m, 0.999))
            return 2 * math.pi * s * s / self.a_ref / (1.0 + np.sqrt(1.0 + (b * s * s / (A * g["cos_gc"])) ** 2))

        def sup(m):
            b = _beta(np.maximum(m, 1.0001))
            k = np.maximum(1.0 - 1.0 / (2.0 * g["AR"] * b), 0.5)
            return A / self.a_ref * 4.0 / b * k

        M = np.asarray(M, float)
        out = np.where(M <= M_SUB, sub(M), sup(M))
        mid = (M > M_SUB) & (M < M_SUP)
        if np.any(mid):
            e = 1e-3
            f0, f1 = sub(np.array(M_SUB)), sup(np.array(M_SUP))
            d0 = (sub(np.array(M_SUB)) - sub(np.array(M_SUB - e))) / e
            d1 = (sup(np.array(M_SUP + e)) - sup(np.array(M_SUP))) / e
            out = np.where(mid, _hermite(M, M_SUB, M_SUP, f0, d0, f1, d1), out)
        return out

    def _fin_xcp_frac(self, f, M):
        g = self.fin_geom(f)
        M = np.asarray(M, float)
        b2 = _beta(np.array(2.0))
        hi = (g["AR"] * b2 - 0.67) / (2.0 * g["AR"] * b2 - 1.0)
        b = _beta(np.maximum(M, 2.0))
        sup = (g["AR"] * b - 0.67) / (2.0 * g["AR"] * b - 1.0)
        return np.where(M <= 0.5, 0.25, np.where(M >= 2.0, sup, _hermite(M, 0.5, 2.0, 0.25, 0.0, hi, 0.0)))

    def components(self, M):
        """[(CNa_i [1/rad], x_cp_i [m from nose])] per component at Mach M (scalar or array)."""
        M = np.asarray(M, float)
        out = []
        for n in self.noses:
            out.append((np.full_like(M, 2.0 * (n.d / self.d_ref) ** 2), np.full_like(M, n.x0 + n.L - n.vol /
                                                                                    (math.pi * n.R * n.R))))
        for c in self.trans:
            d1, d2 = c.d1, c.d2
            cna = 2.0 * ((d2 / self.d_ref) ** 2 - (d1 / self.d_ref) ** 2)
            r = d1 / d2
            x = c.x0 + c.length / 3.0 * (1.0 + (1.0 - r) / (1.0 - r * r)) if abs(1 - r) > 1e-9 else c.x0
            out.append((np.full_like(M, cna), np.full_like(M, x)))
        for f in self.fins:
            g = self.fin_geom(f)
            K = 1.0 + f.R / (f.span + f.R)
            fn = FIN_FN.get(f.n, 0.81)
            cna = K * (f.n / 2.0) * fn * self._cna1(f, M)
            x = f.x0 + g["x_mac"] + self._fin_xcp_frac(f, M) * g["mac"]
            out.append((cna, x))
        return out

    def cna_cp(self, M):
        comps = self.components(M)
        cna = sum(c for c, _ in comps)
        xcp = sum(c * x for c, x in comps) / cna
        return cna, xcp, comps

    # ---------------------------------------------------------------- drag
    def _cf(self, M, Re):
        Rs = self.r.roughness
        L = self.L
        Re = np.maximum(np.asarray(Re, float), 1.0)
        turb = np.where(Re < 1e4, 1.48e-2, 1.0 / (1.50 * np.log(np.maximum(Re, 1e4)) - 5.6) ** 2)
        rough = 0.032 * (Rs / L) ** 0.2
        Re_crit = 51.0 * (Rs / L) ** -1.039
        sub = M < 1.0
        turb_c = np.where(sub, turb * (1 - 0.1 * M * M), turb / (1 + 0.15 * M * M) ** 0.58)
        rough_c = np.where(sub, rough * (1 - 0.1 * M * M), rough / (1 + 0.18 * M * M))
        return np.where(Re > Re_crit, np.maximum(turb_c, rough_c), turb_c)

    def _cone_wave(self, M, phi):
        s = math.sin(phi)
        sup = lambda m: 2.1 * s * s + 0.5 * s / np.sqrt(np.maximum(m * m - 1.0, 1e-9))   # noqa: E731
        c0 = 0.8 * s * s
        e = 1e-3
        f1, d1 = sup(np.array(1.3)), (sup(np.array(1.3 + e)) - sup(np.array(1.3))) / e
        return np.where(M <= 0.8, c0, np.where(M >= 1.3, sup(M), _hermite(M, 0.8, 1.3, c0, 0.0, f1, d1)))

    def _stag(self, M):
        """Stagnation pressure ratio q_stag/q (OpenRocket eq. 3.81)."""
        return np.where(M < 1.0, 1 + M * M / 4 + M ** 4 / 40,
                        1.84 - 0.76 / np.maximum(M, 1) ** 2 + 0.166 / np.maximum(M, 1) ** 4
                        + 0.035 / np.maximum(M, 1) ** 6)

    def _base(self, M, cdf_body, area_ratio):
        sub = 0.029 * area_ratio ** 1.5 / np.sqrt(np.maximum(cdf_body, 1e-4))
        Ms = np.maximum(M, 1.0)
        sup = np.minimum(0.25 / Ms, 1.0 / (0.7 * Ms * Ms))          # never above the vacuum-base limit 2/(gamma M^2)
        return np.where(M <= M_BASE0, sub, np.where(M >= 1.0, sup, _hermite(M, M_BASE0, 1.0, sub, 0.0, sup, -0.25)))

    def cd_table(self, M, thrusting):
        """Spec override (stage key `cd_table` on the LOWEST attached stage): {coast: [[M, CD]], thrust: [[M, CD]],
        ref_diameter_m}; CD rescaled to this stack's A_ref. Used when a vehicle definition supplies measured or
        handbook drag (validation anchors); the build-up is then not used for drag."""
        tb = self.r.stages[self.n_top - 1].spec.get("cd_table")
        if not tb:
            return None
        rows = tb.get("thrust" if thrusting and tb.get("thrust") else "coast")
        Ms, Cs = np.array([p[0] for p in rows], float), np.array([p[1] for p in rows], float)
        k = (float(tb.get("ref_diameter_m", self.d_ref)) / self.d_ref) ** 2
        return np.interp(np.asarray(M, float), Ms, Cs) * k * float(tb.get("scale", 1.0))

    def cd(self, M, Re, thrusting=False, breakdown=False):
        """Zero-lift drag coefficient (ref A_ref) at Mach M, length Reynolds number Re (arrays broadcast)."""
        ov = self.cd_table(M, thrusting)
        if ov is not None:
            return (ov, dict(table=ov)) if breakdown else ov
        M = np.maximum(np.asarray(M, float), 1e-4)
        Re = np.asarray(Re, float)
        cf = self._cf(M, Re)
        mac_t = [(f, self.fin_geom(f)) for f in self.fins]
        fin_fac = (sum(2 * f.n * f.A * (1 + 2 * f.thickness / g["mac"]) for f, g in mac_t) / self.wet_fins
                   if self.fins else 1.0)
        cdf_body = cf * self.wet_body * (1 + 1 / (2 * self.fB)) / self.a_ref
        cdf_fins = cf * self.wet_fins * fin_fac / self.a_ref
        # nose pressure / wave drag
        cdn = np.zeros_like(M)
        for n in self.noses:
            phi = math.atan2(n.R, n.L)
            k = SHAPE_WAVE.get(n.shape, 1.0)
            if n.shape == "cone":
                w = self._cone_wave(M, phi)
            else:     # rounded profiles: ~0 subsonic pressure drag, ramp from M 0.8 to the scaled cone value at 1.3
                f13 = float(self._cone_wave(np.array(1.3), phi)) * k
                e = 1e-3
                d13 = float(self._cone_wave(np.array(1.3 + e), phi) - self._cone_wave(np.array(1.3), phi)) / e * k
                w = np.where(M <= 0.8, 0.0, np.where(M >= 1.3, self._cone_wave(M, phi) * k,
                                                     _hermite(M, 0.8, 1.3, 0.0, 0.0, f13, d13)))
            cdn = cdn + w * (n.d / self.d_ref) ** 2
        # transitions (forward-facing flares only)
        cdt = np.zeros_like(M)
        for c in self.trans:
            if c.d2 > c.d1:
                phi = math.atan2(0.5 * (c.d2 - c.d1), c.length)
                cdt = cdt + self._cone_wave(M, phi) * ((c.d2 ** 2 - c.d1 ** 2) / self.d_ref ** 2)
        # fins: leading / trailing edges and supersonic thickness wave drag
        cdle = np.zeros_like(M)
        cdte = np.zeros_like(M)
        cdwave = np.zeros_like(M)
        for f, g in mac_t:
            front = f.n * f.thickness * f.span / self.a_ref
            if f.le == "rounded":
                le = np.where(M < 0.9, (1 - np.minimum(M, 0.899) ** 2) ** -0.417 - 1,
                              np.where(M < 1.0, 1 - 1.785 * (M - 0.9), 1.214 - 0.502 / M ** 2 + 0.1095 / M ** 4))
                cdle = cdle + le * math.cos(g["gam_le"]) ** 2 * front
            elif f.le == "square":
                cdle = cdle + 0.85 * self._stag(M) * front
            else:     # wedge / airfoil: sharp edge; supersonic Ackeret thickness drag
                eps = math.radians(float(getattr(f, "edge_half_angle_deg", 4.0)))
                lb = 0.5 * f.thickness / math.tan(eps)
                n_edges = 2.0 if f.te != "square" else 1.0
                ack = (2.0 / _beta(np.maximum(M, 1.0001))) * 2 * eps * eps * min(n_edges * lb / g["mac"], 1.0)
                ramp = np.clip((M - 0.9) / 0.4, 0.0, 1.0) ** 2
                cdwave = cdwave + np.where(M >= 1.3, ack, ramp * ack) * f.n * f.A / self.a_ref
            if f.te == "square":
                cdte = cdte + front * 1.0       # x base coefficient below
        # base
        d_b = self.d_base
        if thrusting:
            a_b = max(d_b * d_b - self.motor_d * self.motor_d, 0.0) / self.d_ref ** 2
        else:
            a_b = d_b * d_b / self.d_ref ** 2
        cbase_unit = self._base(M, cdf_body, 1.0)
        cdb = cbase_unit * a_b
        cdte = cdte * cbase_unit
        total = cdf_body + cdf_fins + cdn + cdt + cdle + cdte + cdwave + cdb + self.cd_extra
        if breakdown:
            return total, dict(friction_body=cdf_body, friction_fins=cdf_fins, nose=cdn, transitions=cdt,
                               fin_le=cdle, fin_te=cdte, fin_wave=cdwave, base=cdb, extra=self.cd_extra + 0 * M)
        return total

    def margin_cal(self, M, x_cg):
        _, xcp, _ = self.cna_cp(M)
        return (xcp - x_cg) / self.d_ref

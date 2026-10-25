#!/usr/bin/env python3
# DESCRIPTION: A design spec (dict / YAML, schema rocket_design/1, see README) -> Rocket: an ordered list of stages
#              (index 0 = top/sustainer), each a list of components positioned from the overall nose tip (x [m], +aft).
#              Every dimension, wall thickness, material and mass is a parameter; derived quantities scale from them:
#                nose        shell of `thickness` on the profile (cone | ogive (tangent) | vonkarman | parabolic |
#                            elliptical) + shoulder tube; area/volume/CG/inertia by ring integration
#                tube        body tube OD `diameter`, wall `thickness`, `length`
#                transition  conical shell between two diameters (interstage / boattail)
#                fins        N trapezoidal flat-plate fins: root, tip, span, sweep (LE), thickness, LE profile
#                motor       a thrustcurve.org motor (motors.py): case at the motor centre, propellant as a solid
#                            cylinder burning in proportion to delivered impulse; aft end flush with the stage aft end
#                            unless `overhang` is given; mount tube + `rings` centering rings (plywood default)
#                recovery    canopy mass = ripstop areal density x canopy area x (1 + lines_factor) + shock cord
#                            (cord_len_factor x stage length); packed at `x` (default 0.15 m below the stage top/nose)
#                point       avionics / ballast / anything: mass at x
#              finish_factor (default 1.06, ASSUMED: paint, epoxy fillets, adhesives) multiplies structural mass.
#              mass_override: {stage: {dry_kg, cg_m}} replaces the build-up with MEASURED values (scaled CG/inertia
#              kept from the build-up shape). All values from tables are labelled in the provenance list.
import math

import numpy as np

from . import materials as MAT

NOSE_SHAPES = ("cone", "ogive", "vonkarman", "parabolic", "elliptical")


def nose_profile(shape, L, R, n=400):
    xi = np.linspace(0.0, 1.0, n)
    x = xi * L
    if shape == "cone":
        r = R * xi
    elif shape == "ogive":
        rho = (R * R + L * L) / (2.0 * R)
        r = np.sqrt(np.maximum(rho * rho - (L - x) ** 2, 0.0)) + R - rho
    elif shape in ("vonkarman", "haack"):
        th = np.arccos(np.clip(1.0 - 2.0 * xi, -1, 1))
        r = R / math.sqrt(math.pi) * np.sqrt(np.maximum(th - np.sin(2 * th) / 2.0, 0.0))
    elif shape == "parabolic":
        r = R * (2.0 * xi - xi * xi)
    elif shape == "elliptical":
        r = R * np.sqrt(np.maximum(1.0 - (1.0 - xi) ** 2, 0.0))
    else:
        raise ValueError("nose shape %r not in %s" % (shape, NOSE_SHAPES))
    return x, np.maximum(r, 0.0)


def _shell_rings(x, r, t, rho):
    """Thin shell of revolution: ring masses dm at (xm, rm)."""
    dx, dr = np.diff(x), np.diff(r)
    ds = np.hypot(dx, dr)
    xm, rm = 0.5 * (x[1:] + x[:-1]), 0.5 * (r[1:] + r[:-1])
    dm = rho * t * 2.0 * math.pi * rm * ds
    return xm, rm, dm


class Comp:
    """One mass/aero component. x0 = forward end from the overall nose tip [m]."""

    def __init__(self, kind, stage, x0, length, **kw):
        self.kind, self.stage, self.x0, self.length = kind, stage, float(x0), float(length)
        self.__dict__.update(kw)
        self.m = kw.get("m", 0.0)
        self.xc = kw.get("xc", self.x0 + 0.5 * self.length)
        self.I_ax = kw.get("I_ax", 0.0)          # about own axis (body X)
        self.I_tr_c = kw.get("I_tr_c", 0.0)      # transverse, about own CG


def _tube(stage, x0, L, d, t, rho, name, finish=1.0):
    R = 0.5 * d
    m = rho * math.pi * (d - t) * t * L * finish
    rm = R - 0.5 * t
    return Comp("tube", stage, x0, L, name=name, d=d, m=m, I_ax=m * rm * rm, I_tr_c=m * (rm * rm / 2 + L * L / 12))


def _shell_comp(kind, stage, x0, x, r, t, rho, name, finish=1.0, **kw):
    xm, rm, dm = _shell_rings(x, r, t, rho)
    dm = dm * finish
    m = float(dm.sum())
    xc = float((dm * xm).sum() / m) if m > 0 else x0 + 0.5 * (x[-1] - x[0])
    I_ax = float((dm * rm * rm).sum())
    I_tr = float((dm * (rm * rm / 2 + (xm - xc) ** 2)).sum())
    return Comp(kind, stage, x0, float(x[-1] - x[0]), name=name, m=m, xc=x0 + xc - x[0], I_ax=I_ax, I_tr_c=I_tr, **kw)


class Stage:
    def __init__(self, idx, spec):
        self.idx, self.spec = idx, spec
        self.name = spec.get("name", "stage%d" % idx)
        self.comps = []
        self.motor = None
        self.x_top = self.x_aft = 0.0
        self.d = float(spec["body"]["diameter"])
        self.d_aft = self.d


class Rocket:
    def __init__(self, spec, motor_db=None):
        self.spec = spec
        self.name = spec.get("name", "rocket")
        self.finish = float(spec.get("finish_factor", 1.06))
        self.roughness = MAT.FINISH_ROUGHNESS_M[spec.get("surface_finish", "smooth_paint")] \
            if isinstance(spec.get("surface_finish", "smooth_paint"), str) else float(spec["surface_finish"])
        self.mat_over = spec.get("materials") or {}
        self.provenance = []
        self.stages = []
        db = motor_db
        x = 0.0
        for i, ss in enumerate(spec["stages"]):
            st = Stage(i, ss)
            st.x_top = x
            if i == 0:
                x = self._nose(st, ss["nose"], x)
            if ss.get("transition"):
                x = self._transition(st, ss["transition"], x)
            b = ss["body"]
            mat = self._mat(b.get("material", "fiberglass"))
            st.comps.append(_tube(i, x, b["length"], st.d, b.get("thickness", 0.0015), mat["rho"], "body",
                                  self.finish))
            st.comps[-1].wetted = True
            body_x0 = x
            x += b["length"]
            st.x_aft = x
            if ss.get("boattail"):
                bt = ss["boattail"]
                c = self._cone_shell(i, x, bt["length"], st.d, bt["aft_diameter"], bt.get("thickness", 0.0015),
                                     self._mat(bt.get("material", "fiberglass"))["rho"], "boattail")
                st.comps.append(c)
                x += bt["length"]
                st.x_aft = x
                st.d_aft = bt["aft_diameter"]
            if ss.get("fins"):
                self._fins(st, ss["fins"], body_x0, b["length"])
            if ss.get("motor"):
                mot = ss["motor"]
                if isinstance(mot, dict):
                    from .motors import from_table, from_file
                    if mot.get("file"):
                        mot = from_file(mot["file"], mot.get("provenance"))
                    else:
                        mot = from_table(mot["name"], mot["thrust_n_vs_s"], mot["prop_kg"], mot["total_kg"],
                                         mot["dia_m"], mot["len_m"], mot.get("provenance", dict(label="ASSUMED")),
                                         mot.get("prop_kg_vs_s"))
                elif isinstance(mot, str):
                    if db is None:
                        from .motors import load
                        db = load()
                    mot = db.find(mot)
                self._motor(st, ss, mot)
            for rec_name in ("drogue", "main"):
                if (ss.get("recovery") or {}).get(rec_name):
                    self._chute(st, ss["recovery"][rec_name], rec_name)
            for p in ss.get("masses", []) or []:
                xp = st.x_top + float(p.get("x_from_stage_top", 0.5 * (st.x_aft - st.x_top)))
                st.comps.append(Comp("point", i, xp, 0.0, name=p.get("name", "mass"), m=float(p["kg"]), xc=xp,
                                     label=p.get("label", "ASSUMED")))
            if ss.get("ballast_kg"):
                xb = st.x_top + float(ss.get("ballast_x_from_stage_top",
                                             (ss["nose"]["length"] * 0.9) if i == 0 and ss.get("nose") else 0.05))
                st.comps.append(Comp("point", i, xb, 0.0, name="ballast", m=float(ss["ballast_kg"]), xc=xb,
                                     label="DESIGN"))
            if ss.get("coupler"):
                cp = ss["coupler"]
                mt = self._mat(cp.get("material", "fiberglass"))
                st.comps.append(_tube(i, st.x_aft - 0.5 * cp["length"], cp["length"], st.d - 2 * b.get("thickness", 0.0015),
                                      cp.get("thickness", 0.0015), mt["rho"], "coupler", self.finish))
            mo = (spec.get("mass_override") or {}).get(i) or (spec.get("mass_override") or {}).get(st.name)
            if mo:
                self._override(st, mo)
            if ss.get("mass_direct"):
                self._direct(st, ss["mass_direct"])
            self.stages.append(st)
        self.length = x
        self.motor_db = db

    # ---------------------------------------------------------------- builders
    def _mat(self, name):
        m = MAT.material(name, self.mat_over)
        self.provenance.append(("material", name, m["src"]))
        return m

    def _nose(self, st, ns, x):
        L, R = ns["length"], 0.5 * st.d
        shape = ns.get("shape", "ogive")
        xs, rs = nose_profile(shape, L, R)
        mat = self._mat(ns.get("material", "fiberglass"))
        c = _shell_comp("nose", st.idx, x, xs + x, rs, ns.get("thickness", 0.002), mat["rho"], "nose", self.finish,
                        shape=shape, d=st.d)
        dx = np.diff(xs)
        vol = float(np.sum(math.pi * (0.5 * (rs[1:] + rs[:-1])) ** 2 * dx))
        c.vol, c.wet = vol, float(np.sum(2 * math.pi * 0.5 * (rs[1:] + rs[:-1]) * np.hypot(dx, np.diff(rs))))
        c.L, c.R = L, R
        xm = 0.5 * (xs[1:] + xs[:-1])
        strip = 2.0 * 0.5 * (rs[1:] + rs[:-1]) * dx
        c.plan, c.plan_x = float(strip.sum()), x + float((strip * xm).sum() / strip.sum())
        st.comps.append(c)
        if ns.get("shoulder_length"):
            sl = ns["shoulder_length"]
            st.comps.append(_tube(st.idx, x + L, sl, st.d - 2 * ns.get("thickness", 0.002), ns.get("thickness", 0.002),
                                  mat["rho"], "nose_shoulder", self.finish))
        if ns.get("tip_mass_kg"):
            st.comps.append(Comp("point", st.idx, x + 0.2 * L, 0.0, name="nose_weight", m=float(ns["tip_mass_kg"]),
                                 xc=x + 0.2 * L, label="ASSUMED"))
        return x + L

    def _cone_shell(self, stage, x0, L, d1, d2, t, rho, name):
        xs = np.linspace(0, L, 60)
        rs = 0.5 * d1 + (0.5 * d2 - 0.5 * d1) * xs / L
        c = _shell_comp("transition", stage, x0, xs + x0, rs, t, rho, name, self.finish, d1=d1, d2=d2)
        c.wet = float(math.pi * (0.5 * d1 + 0.5 * d2) * math.hypot(L, 0.5 * (d2 - d1)))
        return c

    def _transition(self, st, ts, x):
        d_up = self.stages[st.idx - 1].d_aft
        c = self._cone_shell(st.idx, x, ts["length"], d_up, st.d, ts.get("thickness", 0.0015),
                             self._mat(ts.get("material", "fiberglass"))["rho"], "transition")
        st.comps.append(c)
        return x + ts["length"]

    def _fins(self, st, fs, body_x0, body_len):
        Cr, Ct, s, Xs, t = fs["root"], fs["tip"], fs["span"], fs.get("sweep", fs["root"] - fs["tip"]), fs["thickness"]
        N = int(fs.get("n", 3))
        x_le = body_x0 + body_len - Cr - float(fs.get("from_aft", 0.0))
        mat = self._mat(fs.get("material", "g10"))
        A = 0.5 * (Cr + Ct) * s
        m = N * A * t * mat["rho"] * self.finish
        xcf = (Xs * (Cr + 2 * Ct) + (Cr * Cr + Cr * Ct + Ct * Ct)) / (3.0 * (Cr + Ct))
        R = 0.5 * st.d
        yc = R + s * (Cr + 2 * Ct) / (3.0 * (Cr + Ct))
        c = Comp("fins", st.idx, x_le, Cr, name="fins", m=m, xc=x_le + xcf, I_ax=m * (yc * yc + s * s / 18.0),
                 I_tr_c=m * (Cr * Cr / 12.0 + 0.5 * yc * yc), n=N, root=Cr, tip=Ct, span=s, sweep=Xs, thickness=t,
                 R=R, G=mat["G"], le=fs.get("le_profile", "rounded"), te=fs.get("te_profile", "square"),
                 cant_deg=float(fs.get("cant_deg", 0.0)), A=A, material=fs.get("material", "g10"))
        st.comps.append(c)

    def _motor(self, st, ss, motor):
        st.motor = motor
        L, dm = motor.len_m, motor.dia_m
        x0 = st.x_aft - L + float(ss.get("motor_overhang", 0.0))
        st.motor_x0 = x0
        r = 0.5 * dm
        case = Comp("motor_case", st.idx, x0, L, name="motor_case", m=motor.case_kg,
                    I_ax=motor.case_kg * r * r, I_tr_c=motor.case_kg * (r * r / 2 + L * L / 12))
        st.comps.append(case)
        st.prop = Comp("propellant", st.idx, x0, L, name="propellant", m=motor.prop_kg, r=0.8 * r)
        mt = ss.get("mount") or {}
        mat = self._mat(mt.get("material", "cardboard"))
        tt = mt.get("thickness", 0.0012)
        st.comps.append(_tube(st.idx, x0, L, dm + 2 * tt, tt, mat["rho"], "motor_mount", self.finish))
        ring_mat = self._mat(mt.get("ring_material", "plywood"))
        nr, rt = int(mt.get("rings", 3)), mt.get("ring_thickness", 0.006)
        Ro, Ri = 0.5 * st.d - 0.0015, 0.5 * dm + tt
        if Ro > Ri:
            mr = ring_mat["rho"] * math.pi * (Ro * Ro - Ri * Ri) * rt * self.finish
            for k in range(nr):
                xr = x0 + (L - rt) * k / max(nr - 1, 1)
                st.comps.append(Comp("point", st.idx, xr, rt, name="ring%d" % k, m=mr, xc=xr + rt / 2,
                                     I_ax=mr * (Ro * Ro + Ri * Ri) / 2))
        self.provenance.append(("motor", motor.name, motor.provenance))

    def _chute(self, st, rs, name):
        D = float(rs["diameter"])
        fab = MAT.SURFACE["ripstop_nylon"]["areal"]
        m = fab * math.pi * D * D / 4.0 * (1.0 + float(rs.get("lines_factor", 0.5)))
        m += MAT.SURFACE["shock_cord"]["linear"] * float(rs.get("cord_len_factor", 3.0)) * (st.x_aft - st.x_top)
        if rs.get("mass_kg") is not None:
            m = float(rs["mass_kg"])
        xp = st.x_top + float(rs.get("x_from_stage_top", 0.15 + (self.spec["stages"][0]["nose"]["length"]
                                                                  if st.idx == 0 else 0.0)))
        st.comps.append(Comp("point", st.idx, xp, 0.0, name=name + "_chute", m=m, xc=xp,
                             label="DERIVED (fabric areal density x area)"))
        st.chutes = getattr(st, "chutes", {})
        st.chutes[name] = dict(D=D, cd=float(rs.get("cd", 0.8)), deploy=rs.get("deploy"),
                               agl_m=rs.get("agl_m"), inflate_s=float(rs.get("inflate_s", 1.0 if name == "main" else 0.5)))

    def _override(self, st, mo):
        """Replace the dry build-up with MEASURED dry mass / CG: scale structural masses, shift by a point mass."""
        dry = [c for c in st.comps if c.kind not in ("motor_case",)]
        m0 = sum(c.m for c in dry)
        k = float(mo["dry_kg"]) / m0 if mo.get("dry_kg") else 1.0
        for c in dry:
            c.m *= k
            c.I_ax *= k
            c.I_tr_c *= k
        if mo.get("cg_m") is not None:
            m = sum(c.m for c in dry)
            xc = sum(c.m * c.xc for c in dry) / m
            shift = float(mo["cg_m"]) - xc       # move every dry component by `shift` (rigid CG shift)
            for c in dry:
                c.xc += shift
        self.provenance.append(("mass_override", st.name, mo.get("src", "MEASURED")))

    def _direct(self, st, md):
        """A published vehicle definition: dry mass (incl. motor case unless case_included is false), dry CG and dry
        inertia about that CG replace the build-up; the propellant sits at prop_cg_from_nose_m."""
        keep_case = not md.get("case_included", True)
        geom = [c for c in st.comps if c.kind == "fins"]
        for c in st.comps:
            c.m, c.I_ax, c.I_tr_c = 0.0, 0.0, 0.0
        if not keep_case:
            pass
        else:
            for c in st.comps:
                if c.kind == "motor_case":
                    c.m = 0.0
        xc = float(md["cg_from_nose_m"])
        st.comps.append(Comp("point", st.idx, xc, 0.0, name="dry (published)", m=float(md["dry_kg"]), xc=xc,
                             I_ax=float(md.get("I_roll", 0.0)), I_tr_c=float(md.get("I_pitch", 0.0)),
                             label=md.get("src", "MEASURED (published vehicle definition)")))
        if st.motor is not None and md.get("prop_cg_from_nose_m") is not None:
            st.prop.x0 = float(md["prop_cg_from_nose_m"]) - 0.5 * st.prop.length
        self.provenance.append(("mass_direct", st.name, md.get("src", "published")))
        del geom

    # ---------------------------------------------------------------- queries
    def stack(self, n_top):
        """Stages 0..n_top-1 still attached (n_top = number of stages in this configuration)."""
        return self.stages[:n_top]

    def mass_props(self, n_top, motor_t):
        """Mass, CG [m from nose], principal inertia (roll, pitch, pitch) of the attached stack.
        motor_t[i] = time since ignition of stage i's motor (None/negative = not ignited)."""
        m = mx = 0.0
        parts = []
        for st in self.stack(n_top):
            for c in st.comps:
                parts.append((c.m, c.xc, c.I_ax, c.I_tr_c))
            if st.motor is not None:
                tm = motor_t[st.idx] if motor_t is not None else None
                mp = st.motor.prop_kg if tm is None or tm < 0 else st.motor.prop_mass(tm)
                p = st.prop
                L = p.length
                parts.append((mp, p.x0 + 0.5 * L, 0.5 * mp * p.r * p.r, mp * (3 * p.r * p.r + L * L) / 12.0))
        for mi, xi, _, _ in parts:
            m += mi
            mx += mi * xi
        xcg = mx / m
        I_ax = sum(p[2] for p in parts)
        I_tr = sum(p[3] + p[0] * (p[1] - xcg) ** 2 for p in parts)
        return dict(m=m, x_cg=xcg, I=np.array([I_ax, I_tr, I_tr]))

    def stage_dry_mass(self, i):
        st = self.stages[i]
        return sum(c.m for c in st.comps)

    def length_of(self, n_top):
        return self.stages[n_top - 1].x_aft

    def d_ref(self, n_top):
        return max(st.d for st in self.stack(n_top))

    def summary(self):
        rows = []
        for st in self.stages:
            rows.append(dict(stage=st.name, x_top=round(st.x_top, 4), x_aft=round(st.x_aft, 4), d=st.d,
                             dry_kg=round(sum(c.m for c in st.comps if c.kind != "motor_case"), 4),
                             motor=st.motor.name if st.motor else None,
                             motor_total_kg=round(st.motor.total_kg, 4) if st.motor else 0.0,
                             components=[(c.name, round(c.m, 4), round(c.xc, 4)) for c in st.comps]))
        mp = self.mass_props(len(self.stages), None)
        return dict(name=self.name, length_m=round(self.length, 4), liftoff_kg=round(mp["m"], 4),
                    cg_liftoff_m=round(mp["x_cg"], 4), stages=rows)

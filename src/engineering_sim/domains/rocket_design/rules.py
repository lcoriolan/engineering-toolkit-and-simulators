#!/usr/bin/env python3
# DESCRIPTION: Derived dimensions so a design is defined by a few parameters and everything else scales:
#                airframe OD from motor diameter (standard hobby fiberglass sizes, ASSUMED catalogue values)
#                body length = motor length + avionics bay + recovery bay (+ margin)
#                parachute diameter from mass and target descent rate: D = sqrt(8 m g / (pi rho Cd v^2))
#                  (ground air density of the site; main 6 m/s, drogue 25 m/s defaults, ASSUMED common practice)
#                fin thickness from span (thickness_ratio x span, floored at the stock sheet sizes)
#              derive(spec, site_rho) fills every `auto` value; all rules are parameters in spec["rules"].
import copy
import math

AIRFRAME_OD_M = {   # motor dia mm -> candidate airframe ODs [m] (fiberglass tube catalogue sizes, ASSUMED)
    18: [0.0254], 24: [0.0336, 0.0419], 29: [0.0419, 0.0563], 38: [0.0563, 0.0660], 54: [0.0792, 0.1016],
    75: [0.1016, 0.1570], 98: [0.1570, 0.2030], 152: [0.2030, 0.2540], 161: [0.2030, 0.2540],
}
FIN_SHEETS_M = [0.0016, 0.0024, 0.0032, 0.0047, 0.0064, 0.0095, 0.0127]   # common G10/CF/Al sheet thicknesses


def airframe_od(motor_dia_mm, choice=0):
    keys = sorted(AIRFRAME_OD_M)
    k = min(keys, key=lambda d: abs(d - motor_dia_mm))
    opts = AIRFRAME_OD_M[k]
    return opts[min(int(choice), len(opts) - 1)]


def chute_diameter(mass_kg, v_mps, rho=1.0, cd=0.8, g=9.80665):
    return math.sqrt(8.0 * mass_kg * g / (math.pi * rho * cd * v_mps * v_mps))


def sheet(t):
    return min(FIN_SHEETS_M, key=lambda s: abs(s - t))


def derive(spec, motor_db, site_rho=1.0):
    """Resolve 'auto' entries. Two passes: geometry from the motor, then chutes from the built mass."""
    from .airframe import Rocket
    sp = copy.deepcopy(spec)
    rules = sp.get("rules", {})
    for st in sp["stages"]:
        mot = motor_db.find(st["motor"]) if st.get("motor") else None
        if mot is not None:
            if st["body"].get("diameter") in (None, "auto"):
                st["body"]["diameter"] = airframe_od(mot.dia_m * 1000, st["body"].get("od_choice", 0))
            if st["body"].get("length") in (None, "auto"):
                st["body"]["length"] = mot.len_m + float(st["body"].get("bay_len", rules.get("bay_len", 0.35)))
        if st.get("nose") and st["nose"].get("length") in (None, "auto"):
            st["nose"]["length"] = float(st["nose"].get("fineness", 4.0)) * st["body"]["diameter"]
        fs = st.get("fins")
        if fs:
            d = st["body"]["diameter"]
            for key, rel in (("root", "root_cal"), ("tip", "tip_ratio"), ("span", "span_cal"), ("sweep", "sweep_ratio")):
                if fs.get(key) in (None, "auto"):
                    if key == "root":
                        fs["root"] = float(fs.get("root_cal", 2.0)) * d
                    elif key == "tip":
                        fs["tip"] = float(fs.get("tip_ratio", 0.4)) * fs["root"]
                    elif key == "span":
                        fs["span"] = float(fs.get("span_cal", 1.0)) * d
                    else:
                        fs["sweep"] = float(fs.get("sweep_ratio", 1.0)) * (fs["root"] - fs["tip"])
            if fs.get("thickness") in (None, "auto"):
                fs["thickness"] = sheet(float(fs.get("thickness_ratio", 0.06)) * fs["span"])
        for k, v in (st.get("recovery") or {}).items():
            if v.get("diameter") in (None, "auto"):
                v["diameter"] = 0.5          # placeholder for the mass pass
    r = Rocket(sp, motor_db)
    for i, st in enumerate(sp["stages"]):
        rec = st.get("recovery") or {}
        if not rec:
            continue
        m_desc = sum(c.m for s in r.stages[: i + 1] for c in s.comps) if i == 0 else r.stage_dry_mass(i)
        if i == 0:
            m_desc = sum(c.m for c in r.stages[0].comps)
        for name, v in rec.items():
            if spec["stages"][i].get("recovery", {}).get(name, {}).get("diameter") in (None, "auto"):
                vt = float(v.get("descent_mps", 6.0 if name == "main" else 25.0))
                v["diameter"] = round(chute_diameter(m_desc, vt, site_rho, float(v.get("cd", 0.8))), 3)
    return sp

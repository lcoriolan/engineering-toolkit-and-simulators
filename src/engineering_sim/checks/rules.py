def _nearest(value, allowed):
    return min(allowed, key=lambda a: (abs(a - value), -a))

def _material_property(materials, material, key, rtype, param):
    """Return (value, None) or (None, blocking violation). No default is ever substituted: a typo'd material
    or a missing property must stop the gate, not pass it against steel's modulus or an infinite yield."""
    rec = materials.get(material)
    if rec is None:
        why = "unknown material %r (not in caller material table)" % (material,)
    elif rec.get(key) is None:
        why = "material %r has no %s in caller material table" % (material, key)
    else:
        return float(rec[key]), None
    return None, {"rule": rtype, "param": param, "severity": "error", "current": None, "suggested_legal": None,
                  "msg": "%s: %s; no default is assumed." % (rtype, why)}

def _eval_rule(r, lookup, material, shop, materials):
    """Evaluate ONE rule via `lookup(name)->value`. Returns a violation dict or None.

    `lookup` resolves a parameter name to its current value (feature-local first, then published
    globals / loads, depending on the caller). This keeps rule definitions identical whether a
    rule is feature-local or model-level.
    """
    rtype = r["type"]

    if rtype == "min_wall":
        p = r["param"]; v = float(lookup(p) or 0); lo = float(r["min_mm"])
        if v < lo - 1e-9:
            return {"rule": rtype, "param": p, "severity": "error", "current": v,
                    "suggested_legal": lo,
                    "msg": "%s = %.2f mm is below the %.2f mm minimum. %s" % (p, v, lo, r.get("why", ""))}

    elif rtype == "hole_vs_endmill":
        p = r["dia_param"]; v = float(lookup(p) or 0); lo = min(shop["end_mill_dia_mm"])
        if v < lo - 1e-9:
            return {"rule": rtype, "param": p, "severity": "error", "current": v,
                    "suggested_legal": lo,
                    "msg": "%s = %.2f mm is smaller than the smallest cutter (%.2f mm). %s"
                           % (p, v, lo, r.get("why", ""))}

    elif rtype == "edge_distance":
        dp, mp, k = r["hole_dia_param"], r["margin_param"], float(r["k"])
        dia = float(lookup(dp) or 0); margin = float(lookup(mp) or 0); need = k * dia
        if margin < need - 1e-9:
            return {"rule": rtype, "param": mp, "severity": "error", "current": margin,
                    "suggested_legal": round(need, 3),
                    "msg": "edge margin %.2f mm < %.1f x dia (%.2f mm). %s"
                           % (margin, k, need, r.get("why", ""))}

    elif rtype == "standard_size":
        p = r["param"]; v = float(lookup(p) or 0); allowed = shop["standard_metric_mm"]
        if not any(abs(v - a) < 1e-6 for a in allowed):
            near = _nearest(v, allowed)
            return {"rule": rtype, "param": p, "severity": "warn", "current": v,
                    "suggested_legal": near,
                    "msg": "%s = %.2f mm is non-standard; nearest standard is %.2f mm. %s"
                           % (p, v, near, r.get("why", ""))}

    elif rtype == "pocket_floor":
        depth = float(lookup(r["depth_param"]) or 0)
        base_t = float(lookup(r["base_thickness_pub"]) or 0)
        floor = base_t - depth; lo = float(r["min_floor_mm"])
        if floor < lo - 1e-9:
            legal = round(base_t - lo, 3)
            return {"rule": rtype, "param": r["depth_param"], "severity": "error", "current": depth,
                    "suggested_legal": max(0.5, legal),
                    "msg": "pocket leaves %.2f mm floor (< %.2f mm min). %s" % (floor, lo, r.get("why", ""))}

    elif rtype == "plate_deflection":
        # Deterministic beam/plate deflection check (the AUTHORITY - never the LLM's arithmetic).
        # delta = k * P * L^3 / (E * b * t^3), with support factor k (cantilever tip load k=4;
        # simply-supported central load k=0.25). Solve the minimum thickness that meets the limit.
        load = lookup(r["load_param"])
        if load is None:
            return None                                # inert unless a load is supplied
        load = float(load)
        L = float(lookup(r["length_param"]) or 0)
        b = float(lookup(r["width_param"]) or 0)
        t = float(lookup(r["thickness_param"]) or 0)
        E_gpa, bad = _material_property(materials, material, "modulus_gpa", rtype, r["thickness_param"])
        if bad:
            return bad
        E = E_gpa * 1000.0                             # GPa -> MPa (N/mm^2)
        dmax = float(r["max_deflection_mm"])
        k = 4.0 if r.get("support", "cantilever") == "cantilever" else 0.25
        if b <= 0 or t <= 0 or E <= 0 or dmax <= 0:
            return None
        delta = k * load * L ** 3 / (E * b * t ** 3)
        if delta > dmax + 1e-9:
            t_min = (k * load * L ** 3 / (E * b * dmax)) ** (1.0 / 3.0)
            return {"rule": rtype, "param": r["thickness_param"], "severity": "error",
                    "current": round(t, 3), "suggested_legal": round(t_min, 3),
                    "msg": "%s deflection %.3f mm > %.3f mm limit (%s, load %.0f N over %.0f mm). %s"
                           % (r.get("support", "cantilever"), delta, dmax, material, load, L, r.get("why", ""))}

    elif rtype == "cantilever_bending":
        load = lookup(r["load_param"])
        if load is None:
            return None                                # inert unless a load is supplied
        load = float(load)
        Lc = float(lookup(r["length_param"]) or 0)
        b = float(lookup(r["width_param"]) or 0)
        t = float(lookup(r["thickness_param"]) or 0)
        yld, bad = _material_property(materials, material, "yield_mpa", rtype, r["thickness_param"])
        if bad:
            return bad
        sf = float(shop.get("default_safety_factor", 1.5))
        if b <= 0 or t <= 0:
            return None
        sigma = (load * Lc) * (t / 2.0) / (b * t ** 3 / 12.0)     # sigma = 6*F*Lc/(b*t^2)
        allow = yld / sf
        if sigma > allow + 1e-6:
            t_min = (6.0 * load * Lc / (b * allow)) ** 0.5
            return {"rule": rtype, "param": r["thickness_param"], "severity": "error",
                    "current": round(t, 3), "suggested_legal": round(t_min, 3),
                    "msg": "root bending %.0f MPa > allowable %.0f MPa (%s yield %.0f / SF %.1f). %s"
                           % (sigma, allow, material, yld, sf, r.get("why", ""))}

    elif rtype == "servo_fit":
        # The servo body occupies the radial annulus; keep a real clearance to the duct.
        # Dimensions are published by body_tube and inner_duct, so this remains stack-aware.
        radial = float(lookup(r["radial_param"]) or 0)
        gap = float(lookup(r["gap_param"]) or 0)
        body_dia = float(lookup(r.get("body_dia_param", "body_dia")) or 0)
        body_wall = float(lookup(r.get("body_wall_param", "body_wall")) or 0)
        duct_od = float(lookup(r.get("duct_od_param", "duct_od")) or 0)
        capacity = (body_dia / 2.0 - body_wall) - duct_od / 2.0
        if capacity <= 0:
            return None
        if radial + gap > capacity + 1e-9:
            legal = max(0.1, capacity - gap)
            return {"rule": rtype, "param": r["radial_param"], "severity": "error",
                    "current": radial, "suggested_legal": round(legal, 3),
                    "msg": "servo radial envelope %.2f mm + clearance %.2f mm exceeds %.2f mm annulus. %s"
                           % (radial, gap, capacity, r.get("why", ""))}

    elif rtype == "servo_torque_margin":
        required = lookup(r["required_param"])
        available = lookup(r["available_param"])
        if required is None or available is None:
            return None
        required = float(required); available = float(available)
        sf = float(lookup(r.get("safety_factor_param", "")) or r.get("safety_factor", 1.0))
        need = required * sf
        if available < need - 1e-9:
            return {"rule": rtype, "param": r["available_param"], "severity": "error",
                    "current": available, "suggested_legal": round(need, 3),
                    "msg": "servo torque %.2f kg-cm is below %.2f kg-cm required (%.2f x %.1f safety factor). %s"
                           % (available, need, required, sf, r.get("why", ""))}

    elif rtype == "mass_limit":
        value = lookup(r["mass_param"])
        if value is None:
            return None
        value = float(value); limit = float(r["max_g"])
        if value > limit + 1e-9:
            return {"rule": rtype, "param": r["mass_param"], "severity": "error",
                    "current": round(value, 3), "suggested_legal": limit,
                    "msg": "measured mass %.1f g exceeds %.1f g limit. %s"
                           % (value, limit, r.get("why", ""))}

    elif rtype == "cg_shift":
        value = lookup(r["shift_param"])
        datum = lookup(r.get("datum_param", ""))
        if value is not None and datum is not None:
            value = float(value) - float(datum)
        if value is None:
            return None
        value = float(value); limit = float(r["max_mm"])
        if abs(value) > limit + 1e-9:
            return {"rule": rtype, "param": r["shift_param"], "severity": "error",
                    "current": round(value, 3), "suggested_legal": 0.0,
                    "msg": "CG shift %.2f mm exceeds +/- %.2f mm limit. %s"
                           % (value, limit, r.get("why", ""))}
    return None

def evaluate(rules,parameters,material,materials,shop):
    """Evaluate caller-supplied dimensional, material and manufacturing rule records."""
    violations=[]
    for rule in rules:
        issue=_eval_rule(rule,parameters.get,material,shop,materials)
        if issue:violations.append(issue)
    return {'ok':not any(r['severity']=='error' for r in violations),'violations':violations}

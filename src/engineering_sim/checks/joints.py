import math
import itertools

def mount_loads(layout, mass_kg, torque_nm, a):
    """Four-foot equilibrium under bounded shaft force, torque and inertia cases.

Foot coordinates are (x,z); normal load is +Y uplift. Positive moments follow
the right-hand rule. The shaft is H above the support plane. Each case includes
simultaneous axial load, radial direction, gravity and a signed inertia axis.
The elastic bolt-group helper supplies in-plane direct + eccentric shear. Normal
forces use a rigid four-point plate distribution, before a declared prying factor.
"""
    from engineering_sim.science.mech.bolted import _bolt_group
    fr = layout['iec']
    spacing = layout['length'] + (layout['front_cap_t'] + layout['rear_cap_t']) / 2
    pts = [(sx * fr['A'] / 2, sz * spacing / 2) for sz in (-1, 1) for sx in (-1, 1)]
    sx2, sz2 = (sum((x * x for x, z in pts)), sum((z * z for x, z in pts)))
    nose = -spacing / 2 - layout['front_cap_t'] / 2 - a.shaft_overhang_mm
    rows = []
    inertias = [(axis, sign) for axis in range(3) for sign in (-1, 1)]
    for radial, az, tq, (axis, sign) in itertools.product(range(8), (-1, 1), (-1, 1), inertias):
        phi = radial * math.pi / 4
        shaft = [a.shaft_radial_n * math.cos(phi), a.shaft_radial_n * math.sin(phi), az * a.shaft_axial_n]
        force = list(shaft)
        force[1] -= mass_kg * 9.80665
        force[axis] += sign * a.shock_g * mass_kg * 9.80665
        mx = fr['H'] * force[2] - nose * shaft[1]
        mz = -fr['H'] * force[0] + tq * a.torque_multiplier * torque_nm * 1000
        my = nose * shaft[0]
        direct = _bolt_group(dict(bolts=[[x / 1000, z / 1000] for x, z in pts], F_x=force[0], F_y=force[2], x_p=0.0, y_p=0.0))
        polar = sx2 + sz2
        for i, (x, z) in enumerate(pts):
            fx = direct['bolt_forces'][i][0] + my * z / polar
            fz = direct['bolt_forces'][i][1] - my * x / polar
            normal = force[1] / 4 + mz * x / sx2 - mx * z / sz2
            rows.append(dict(front=z < 0, fx=fx, fz=fz, normal=normal, applied_force=force, moments_nmm=[mx, my, mz]))
    return dict(rows=rows, foot_spacing_z_mm=spacing, equilibrium_model='rigid four-foot load split; no flange-foot load sharing credited')

def joint_screen(g, load_rows, bolt_key, arrangement, a, fasteners):
    """Screen whole local joint with pessimistic independent load envelopes.

The sleeve variant transfers shear by positive bore bearing, not by pretending
that sleeve-only preload clamps the plastic. A clearance movement allowance is
therefore retained. The insert variant has only a declared pull-out sensitivity;
its measured pull-out, rotation and creep are still mandatory open gates.
"""
    from engineering_sim.science.mech.bolted import _load_sharing, _bolt_fatigue
    b = fasteners[bolt_key]
    P = max((max(0.0, r['normal']) for r in load_rows)) * a.prying_factor
    Fx, Fz = (max((abs(r[k]) for r in load_rows)) for k in ('fx', 'fz'))
    V = math.hypot(Fx, Fz)
    area, t, w, z, e = (b['area'], g['pad'], g['leg'], g['depth'], g['eccentricity'])
    rejection = []
    if min(t, w, g['height'] - t) <= 0:
        rejection.append('nonpositive drafted foot section')
    if b['d'] + 0.3 > g['sleeve_id']:
        rejection.append('bolt does not clear fixed-stock sleeve bore')
    if arrangement == 'sleeved_through_bolt' and (b['head_od'] - g['sleeve_id']) / 2 < a.min_head_sleeve_overlap_mm:
        rejection.append('bolt head does not provide required radial overlap over sleeve end')
    if b['washer_od'] > z or b['washer_od'] / 2 > g['edge']:
        rejection.append('washer lacks complete pad support/tool land')
    if b['washer_od'] / 2 + w / 2 > e:
        rejection.append('washer overlaps foot leg; revise pattern or seating geometry')
    if rejection:
        return dict(state='rejected', reasons=rejection, worst_margin=0.0)
    As = math.pi / 4 * ((g['sleeve_od'] - g['clearance']) ** 2 - g['sleeve_id'] ** 2)
    Aw = math.pi / 4 * (b['washer_od'] ** 2 - g['sleeve_od'] ** 2)
    if min(As, Aw) <= 0:
        return dict(state='rejected', reasons=['no sleeve or washer bearing section'], worst_margin=0.0)
    Fi = a.preload_fraction * area * a.bolt_proof_mpa
    grip = g['nominal_pad'] + 2 * b['washer_t'] + b['d']
    kb = a.steel_modulus_mpa * area / grip
    km = (a.steel_modulus_mpa * As if arrangement == 'sleeved_through_bolt' else a.polymer_modulus_mpa * Aw) / g['nominal_pad']
    retention = a.metal_preload_retention if arrangement == 'sleeved_through_bolt' else a.polymer_preload_retention
    low, high = (Fi * (1 - a.preload_scatter) * retention, Fi * (1 + a.preload_scatter))
    sharing = _load_sharing(dict(k_b=kb * 1000, k_m=km * 1000, F_i=low, P=P))
    C = sharing['C']
    Fb = max(high + C * P, P)
    sigma_vm = math.hypot(Fb / area, math.sqrt(3) * V / area)
    h = g['height'] - t
    sf, kt = (a.strength_factor, a.notch_factor)
    zl = g['leg_depth']
    leg_sigma = P / (w * zl) + 6 * (Fx * h + P * e) / (zl * w * w) + 6 * Fz * h / (w * zl * zl)
    pad_sigma = 6 * P * e / (z * t * t)
    comp = high * (a.sleeve_polymer_preload_fraction if arrangement == 'sleeved_through_bolt' else 1.0) / Aw
    sleeve_sigma = high / As
    head_area = math.pi / 4 * (min(b['head_od'], g['sleeve_od'] - g['clearance']) ** 2 - g['sleeve_id'] ** 2)
    bearing = V / ((g['sleeve_od'] - g['clearance']) * t)
    edge = g['edge'] - g['sleeve_od'] / 2
    tear = V / (2 * edge * t)
    punch = P / (math.pi * b['washer_od'] * t)
    washer_sigma = 3 * P * (b['washer_od'] - g['sleeve_od']) / (2 * math.pi * g['sleeve_od'] * b['washer_t'] ** 2)

    def ratio(capacity, demand):
        """Finite JSON sentinel for an unloaded mode; never divide by zero."""
        return min(1000000000000.0, capacity / max(demand, 1e-30))
    margins = dict(bolt_combined_proof=ratio(a.bolt_proof_mpa, a.proof_factor * sigma_vm), separation=ratio(sharing['P_sep'], P), leg_nominal=ratio(a.polymer_strength_mpa, sf * kt * leg_sigma), pad_nominal=ratio(a.polymer_strength_mpa, sf * kt * pad_sigma), host_preload_compression=ratio(a.polymer_compression_mpa, sf * comp), sleeve_compression=ratio(a.sleeve_yield_mpa, sf * sleeve_sigma), host_bearing=ratio(a.polymer_compression_mpa, sf * bearing), edge_tearout=ratio(a.polymer_shear_mpa, sf * tear), washer_pullthrough=ratio(a.polymer_shear_mpa, sf * punch), washer_bending=ratio(a.washer_yield_mpa, sf * washer_sigma))
    if arrangement == 'sleeved_through_bolt':
        margins['sleeve_end_bearing'] = a.sleeve_yield_mpa * head_area / (sf * high)
    if arrangement == 'metal_insert':
        pullout = math.pi * g['sleeve_od'] * g['nominal_pad'] * a.polymer_shear_mpa * a.insert_transfer_factor
        margins['insert_host_pullout'] = pullout / (sf * max(high + C * P, P))
    if P > 0:
        fatigue = _bolt_fatigue(dict(F_i=high, C=C, Pmax=P, Pmin=0.0, A_t=area * 1e-06, Se=a.corrected_bolt_endurance_mpa * 1000000.0, Sut=a.bolt_ultimate_mpa * 1000000.0))
        margins['fatigue_sensitivity'] = fatigue['n_fatigue'] / sf if sharing['P_sep'] >= P else 0.0
    else:
        margins['fatigue_sensitivity'] = 1000000000000.0
    governing = min(margins, key=margins.get)
    volume = (e + g['edge'] + w / 2) * t * z + w * (g['height'] - t) * zl - math.pi * g['sleeve_od'] ** 2 * t / 4
    mass = volume * a.polymer_density_kg_mm3 + (area * grip + As * g['nominal_pad'] + 2 * Aw * b['washer_t']) * a.steel_density_kg_mm3
    return dict(state='screened', worst_margin=margins[governing], governing=governing, margins=margins, force_envelope_n=dict(uplift=P, fx=Fx, fz=Fz), preload_n=dict(nominal=Fi, minimum_retained=low, maximum=high), joint_constant=C, mass_per_foot_kg=mass, geometry=g, bolt=bolt_key, arrangement=arrangement, conditional_only=True, open_gates=['hot/oil-conditioned layer and creep coupons', 'sleeve/head/nut/washer contact and tolerances', 'full cap/end-plate load transfer', 'vibration spectrum, locking and life', 'catalogue nut/thread proof and engagement', 'physical joint proof test'])

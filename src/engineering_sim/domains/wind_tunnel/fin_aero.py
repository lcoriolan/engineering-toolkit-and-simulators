#!/usr/bin/env python3
# Fin aeroelastic screening and two-degree-of-freedom time integration.
# Air density and sound speed come from engineering_sim.core.env.Env.
# LEGACY_ENV selects ISA sea level at 15 C; sound speed uses
# 331.3 sqrt(1 + Tc/273.15), with density and temperature supplied by Env.
# State: x = [h_tip, theta, h_tip_dot, theta_dot], xdot = A(U,rho,c)x + B forcing.
# Lateral gust v produces incidence v/U. Servo commands act through a spring
# in series with blade torsion; their hinge-row contribution is k_t*delta.
# The model neglects walls, body/fin interference, wakes, drag, servo control
# loops, backlash, joints, anisotropy and fixture compliance. Results are
# screening estimates, not validated aeroelastic limits.
import json
import math
import pathlib

import numpy as np

KGCM = 0.0980665
LEGACY_ENV = dict(atmos=dict(site_elev_m=0.0, dT_K=0.0, rh=0.0), wind=dict(u10=0.0, turbulence=False))


def load_inputs(section_path, material_path):
    """Read section and material properties from caller-supplied JSON."""
    return json.loads(pathlib.Path(section_path).read_text()), json.loads(pathlib.Path(material_path).read_text())


def strips(spec, count=400):
    root, tip = spec["exposed_root_radius_mm"] / 1000, spec["tip_radius_mm"] / 1000
    length = tip - root
    eta = (np.arange(count) + 0.5) / count
    leading = (spec["exposed_root_leading_z_mm"] * (1 - eta) + spec["tip_leading_z_mm"] * eta) / 1000
    trailing = (spec["exposed_root_trailing_z_mm"] * (1 - eta) + spec["tip_trailing_z_mm"] * eta) / 1000
    chord = leading - trailing
    return {"r": root + eta * length, "eta": eta, "dr": length / count, "length": length,
            "leading": leading, "trailing": trailing, "chord": chord, "area": np.sum(chord) * length / count}


def lift_slope(area, span, speed, sound, oswald):
    aspect = 2 * span ** 2 / area
    incompressible = 2 * math.pi / (1 + 2 / (oswald * aspect))
    mach = speed / sound
    if mach >= 0.85:
        raise ValueError("Reduced subsonic screen must not extrapolate toward Mach 1")
    return incompressible / math.sqrt(1 - mach ** 2)


def section_integrals(chord, thickness):
    edge = 0.0005
    area = chord * (thickness + edge) / 2
    inertia = chord * (thickness ** 3 + thickness ** 2 * edge + thickness * edge ** 2 + edge ** 3) / 48
    return area, inertia


def beam_fe(spec, pressure, PH, count=40, modulus_mpa=None):
    """Euler-Bernoulli tapered-blade cantilever FE; structure-only, with no air dependence."""
    E = (modulus_mpa or PH["petg_E_assumed_MPa"]) * 1e6
    rho = PH["petg_density_assumed_kg_m3"]
    t = spec["thickness_mm"] / 1000
    s = strips(spec, count)
    le = s["dr"]
    ndof = 2 * (count + 1)
    K, M, F = np.zeros((ndof, ndof)), np.zeros((ndof, ndof)), np.zeros(ndof)
    for i, chord in enumerate(s["chord"]):
        area, inertia = section_integrals(chord, t)
        k = E * inertia / le ** 3 * np.array([[12, 6 * le, -12, 6 * le], [6 * le, 4 * le ** 2, -6 * le, 2 * le ** 2],
                                             [-12, -6 * le, 12, -6 * le], [6 * le, 2 * le ** 2, -6 * le, 4 * le ** 2]])
        m = rho * area * le / 420 * np.array([[156, 22 * le, 54, -13 * le], [22 * le, 4 * le ** 2, 13 * le, -3 * le ** 2],
                                             [54, 13 * le, 156, -22 * le], [-13 * le, -3 * le ** 2, -22 * le, 4 * le ** 2]])
        f = pressure * chord * le / 2 * np.array([1, le / 6, 1, -le / 6])
        dofs = np.arange(2 * i, 2 * i + 4)
        K[np.ix_(dofs, dofs)] += k
        M[np.ix_(dofs, dofs)] += m
        F[dofs] += f
    free = np.arange(2, ndof)
    Kf, Mf = K[np.ix_(free, free)], M[np.ix_(free, free)]
    solution = np.zeros(ndof)
    solution[free] = np.linalg.solve(Kf, F[free])
    reaction = K @ solution - F
    L = np.linalg.cholesky(Mf)
    Linv = np.linalg.inv(L)
    eigenvalues = np.linalg.eigvalsh(Linv @ Kf @ Linv.T)
    unit_load = np.zeros(len(free)); unit_load[-2] = 1
    compliance = np.linalg.solve(Kf, unit_load)[-2]
    root_chord = (spec["exposed_root_leading_z_mm"] - spec["exposed_root_trailing_z_mm"]) / 1000
    _, Iroot = section_integrals(root_chord, t)
    stress = abs(reaction[1]) * t / 2 / Iroot
    return {"elements": count, "E_MPa_assumed": E / 1e6, "tip_deflection_mm": float(solution[-2] * 1000),
            "root_nominal_stress_MPa": float(stress / 1e6), "root_bending_moment_Nm": float(abs(reaction[1])),
            "reaction_force_N": float(abs(reaction[0])), "applied_force_N": float(F[::2].sum()),
            "first_bending_mode_Hz": float(math.sqrt(max(0, eigenvalues[0])) / (2 * math.pi)),
            "tip_stiffness_N_m": float(1 / compliance),
            "nominal_strength_ratio_assumed": PH["petg_strength_assumed_MPa"] / (stress / 1e6)}


def aeroelastic_matrices(spec, beam, servo_stiffness, cp_offset_fraction, speed, rho, sound, PH):
    """Return the aeroelastic state matrix A and gust/servo-command forcing columns."""
    s = strips(spec)
    c, dr = s["chord"], s["dr"]
    t = spec["thickness_mm"] / 1000
    phi = s["eta"] ** 2 * (3 - s["eta"]) / 2
    mac = float(np.sum(c * c) * dr / s["area"])
    axis = spec["hinge_z_mm"] / 1000
    u = (np.arange(1000) + 0.5) / 1000
    tx = 0.0005 + (t - 0.0005) * np.where(u < .35, u / .35, (1 - u) / .65)
    u_cg = float(np.sum(u * tx) / np.sum(tx))
    u_var = float(np.sum((u - u_cg) ** 2 * tx) / np.sum(tx))
    y_var = float(np.sum(tx ** 3) / (12 * np.sum(tx)))
    area, inertia = section_integrals(c, t)
    dm = PH["petg_density_assumed_kg_m3"] * area * dr
    d_cg = axis - (s["leading"] - u_cg * c)
    mass = np.array([[np.sum(dm * phi ** 2), -np.sum(dm * phi * d_cg)],
                     [-np.sum(dm * phi * d_cg), np.sum(dm * (d_cg ** 2 + u_var * c * c + y_var))]])
    if np.min(np.linalg.eigvalsh(mass)) <= 0:
        raise RuntimeError("Non-positive reduced mass matrix")
    shear = PH["petg_E_assumed_MPa"] * 1e6 / (2 * (1 + PH["petg_nu_assumed"]))
    blade_torsion_stiffness = 1 / np.sum(dr / (shear * 4 * inertia))
    kt = 1 / (1 / servo_stiffness + 1 / blade_torsion_stiffness)
    kb = beam["tip_stiffness_N_m"]
    stiffness = np.diag([kb, kt])
    zeta = PH["modal_damping_ratio_assumed"]
    damping = np.diag([2 * zeta * math.sqrt(kb * mass[0, 0]), 2 * zeta * math.sqrt(kt * mass[1, 1])])
    d_ac = axis - (s["leading"] - .25 * c) + cp_offset_fraction * mac
    d_75 = axis - (s["leading"] - .75 * c)
    factor = .5 * rho * speed ** 2 * lift_slope(s["area"], s["length"], speed, sound, PH["oswald_efficiency_assumed"])
    ka = factor * np.array([[0, np.sum(phi * c) * dr], [0, -np.sum(d_ac * c) * dr]])
    ca = factor / speed * np.array([[-np.sum(phi * phi * c) * dr, np.sum(phi * d_75 * c) * dr],
                                   [np.sum(d_ac * phi * c) * dr, -np.sum(d_ac * d_75 * c) * dr]])
    Minv = np.linalg.inv(mass)
    A = np.block([[np.zeros((2, 2)), np.eye(2)],
                  [-np.linalg.solve(mass, stiffness - ka), -np.linalg.solve(mass, damping - ca)]])
    # forcing: gust incidence alpha_g acts like theta in the aero stiffness column; servo command via k_t
    f_gust = np.r_[0, 0, Minv @ ka[:, 1]]
    f_cmd = np.r_[0, 0, Minv @ np.array([0.0, kt])]
    return dict(A=A, f_gust=f_gust, f_cmd=f_cmd, mass=mass, kt=kt, kb=kb, phi=phi, strips=s)


def aeroelastic_eigs(spec, beam, servo_k, cp_fraction, speed, rho, sound, PH):
    return np.linalg.eigvals(aeroelastic_matrices(spec, beam, servo_k, cp_fraction, speed, rho, sound, PH)["A"])


def onset_speed(spec, beam, servo_k, cp_fraction, rho, sound, PH, speeds=np.arange(1., 251., 1.)):
    """Scan in 1 m/s increments for the first speed with a positive real eigenvalue."""
    for v in speeds:
        try:
            ev = aeroelastic_eigs(spec, beam, servo_k, cp_fraction, v, rho, sound, PH)
        except ValueError:
            return None, "Mach limit"
        most = ev[np.argmax(ev.real)]
        if most.real > 1e-5:
            return float(v), ("oscillatory instability" if abs(most.imag) > 1. else "static divergence")
    return None, "none within scan"


def screen(spec, P, rho, sound, speed=None):
    """Static force, torque and beam screening for one fin at the supplied air properties."""
    PH = P["physics"]
    U = speed if speed is not None else PH["max_speed_mph"] * 0.44704
    q = .5 * rho * U ** 2
    s = strips(spec)
    c, dr = s["chord"], s["dr"]
    a = lift_slope(s["area"], s["length"], U, sound, PH["oswald_efficiency_assumed"])
    cp = float(np.sum(c * (s["leading"] - .25 * c)) * dr / s["area"])
    mac = float(np.sum(c * c) * dr / s["area"])
    lever = spec["hinge_z_mm"] / 1000 - cp
    force_per_rad = q * s["area"] * a
    angle = math.radians(P["fin"]["deflection_assumed_deg"])
    beam = beam_fe(spec, q * a * angle, PH, 80)
    nominal = force_per_rad * angle * abs(lever)
    worst = force_per_rad * angle * (abs(lever) + P["fin"]["cp_uncertainty_fraction_mac"] * mac)
    return dict(name=spec["name"], speed_m_s=U, rho=rho, sound=sound, mach=U / sound, q_Pa=q,
                lift_slope_per_rad=a, normal_force_at_assumed_deflection_N=force_per_rad * angle,
                nominal_torque_kgf_cm=nominal / KGCM, CP10pct_worst_torque_kgf_cm=worst / KGCM,
                tip_deflection_mm=beam["tip_deflection_mm"], root_stress_MPa=beam["root_nominal_stress_MPa"],
                first_bending_mode_Hz=beam["first_bending_mode_Hz"], beam=beam)


def simulate(spec, P, rho, sound, speed, servo_k, cp_fraction, gust_alpha, t, delta_cmd, x0=None):
    """Time-domain RK4 of the 2-DOF model. gust_alpha(t_k), delta_cmd(t_k) arrays on the uniform grid t."""
    PH = P["physics"]
    beam = beam_fe(spec, 1.0, PH, 80)   # load-independent stiffness using 80 elements
    mats = aeroelastic_matrices(spec, beam, servo_k, cp_fraction, speed, rho, sound, PH)
    A, fg, fc = mats["A"], mats["f_gust"], mats["f_cmd"]
    dt = t[1] - t[0]
    x = np.zeros(4) if x0 is None else np.asarray(x0, float)
    X = np.zeros((len(t), 4))
    for k in range(len(t)):
        X[k] = x
        if k == len(t) - 1:
            break
        g0, g1 = gust_alpha[k], gust_alpha[k + 1]
        d0, d1 = delta_cmd[k], delta_cmd[k + 1]
        f = lambda xx, g, d: A @ xx + fg * g + fc * d
        gm, dm_ = (g0 + g1) / 2, (d0 + d1) / 2
        k1 = f(x, g0, d0); k2 = f(x + dt / 2 * k1, gm, dm_); k3 = f(x + dt / 2 * k2, gm, dm_); k4 = f(x + dt * k3, g1, d1)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X, mats, beam


# ---------------------------------------------------------------------------------------------------------------

# Sweep declared operating conditions and report screening metrics.


DEFAULT_TUNNEL_ATMOS = dict(site_elev_m=0.0, dT_K=0.0, rh=0.0)   # core ISA sea level; set the real tunnel site


def core_air(atmos=None, z=0.0):
    """Test-section air from the unified core atmosphere (host reference; needs the warp package, not a GPU).
    atmos: core atmos keys (site_elev_m, dT_K, p0_pa, rh). Returns dict(rho, sound, T_K, p_pa, atmos)."""
    from engineering_sim.core import atmos as CA
    cfg = dict(DEFAULT_TUNNEL_ATMOS, **(atmos or {}))
    s = CA.ref_state([z], **cfg)
    return dict(rho=float(s["rho"][0]), sound=float(s["c"][0]), T_K=float(s["T_K"][0]), p_pa=float(s["p_pa"][0]),
                atmos=cfg, source="unified_sim core.atmos.ref_state (ISA + site + humidity)")


def screen_sweep(P, G, rho, sound):
    """Retired screen_physics.main() sweeps, air (rho, sound) supplied by the caller (core_air()).
    Returns dict(speed, q, summaries, torque_rows, stability_rows, control) — the evidence rows, unchanged schema."""
    PH = P["physics"]
    U = PH["max_speed_mph"] * 0.44704
    osw = PH["oswald_efficiency_assumed"]
    summaries, torque_rows, stability_rows, control_columns = [], [], [], []
    q = .5 * rho * U ** 2
    allowed = P["servo"]["rated_torque_kg_cm"] * KGCM / 2
    for spec in G["fin_specs"]:
        s = strips(spec)
        c, dr = s["chord"], s["dr"]
        a = lift_slope(s["area"], s["length"], U, sound, osw)
        cp = float(np.sum(c * (s["leading"] - .25 * c)) * dr / s["area"])
        mac = float(np.sum(c * c) * dr / s["area"])
        lever = spec["hinge_z_mm"] / 1000 - cp
        force_per_rad = q * s["area"] * a
        angle = math.radians(P["fin"]["deflection_assumed_deg"])
        pressure = q * a * angle
        mesh_runs = [beam_fe(spec, pressure, PH, n) for n in [10, 20, 40, 80]]
        beam = mesh_runs[-1]
        convergence = abs(mesh_runs[-1]["tip_deflection_mm"] / mesh_runs[-2]["tip_deflection_mm"] - 1)
        modulus_runs = [beam_fe(spec, pressure, PH, 40, E) for E in PH["petg_E_sweep_MPa"]]
        for degrees in [1., 2., 5., 10.]:
            for cp_fraction in [0., .02, .05, .10]:
                force = force_per_rad * math.radians(degrees)
                torque = force * (abs(lever) + cp_fraction * mac)
                torque_rows.append({"fin": spec["name"], "speed_mph": PH["max_speed_mph"],
                    "deflection_deg": degrees, "CP_uncertainty_MAC_fraction": cp_fraction,
                    "force_N": force, "worst_torque_kgf_cm": torque / KGCM,
                    "torque_with_SF2_kgf_cm": 2 * torque / KGCM,
                    "within_user_2p2_rating_with_SF2": bool(torque <= allowed)})
        nominal = force_per_rad * angle * abs(lever)
        worst = force_per_rad * angle * (abs(lever) + P["fin"]["cp_uncertainty_fraction_mac"] * mac)
        delta_limit = math.degrees(allowed / (force_per_rad * (abs(lever) + .10 * mac)))
        summaries.append({"name": spec["name"], "hinge_frac": spec["hinge_frac"],
            "hinge_z_mm": spec["hinge_z_mm"], "exposed_area_mm2": s["area"] * 1e6,
            "exposed_span_mm": s["length"] * 1000, "MAC_mm": mac * 1000,
            "nominal_CP_z_mm": cp * 1000, "hinge_ahead_of_nominal_CP_mm": lever * 1000,
            "lift_slope_per_rad_assumed": a, "normal_force_per_degree_N": force_per_rad * math.pi / 180,
            "normal_force_at_assumed_deflection_N": force_per_rad * angle,
            "nominal_torque_kgf_cm": nominal / KGCM, "nominal_torque_SF2_kgf_cm": 2 * nominal / KGCM,
            "CP10pct_worst_torque_kgf_cm": worst / KGCM, "CP10pct_worst_torque_SF2_kgf_cm": 2 * worst / KGCM,
            "CP10pct_SF2_deflection_limit_deg_screen_only": delta_limit,
            "beam_FE": beam, "beam_mesh_convergence": mesh_runs,
            "tip_displacement_change_40_to_80_fraction": convergence, "modulus_sensitivity": modulus_runs})
        r_cp = float(np.sum(c * s["r"]) * dr / s["area"])
        for azimuth in spec["station_angles_deg"]:
            phi = math.radians(azimuth)
            force = force_per_rad * np.array([-math.sin(phi), math.cos(phi), 0.])
            position = np.array([r_cp * math.cos(phi), r_cp * math.sin(phi), cp - .610])
            control_columns.append(np.concatenate([force, np.cross(position, force)]))
        for servo_k in PH["hinge_stiffness_sweep_Nm_rad"]:
            for cp_fraction in [-.10, -.05, -.02, 0., .02, .05, .10]:
                onset, kind = None, "none within scan"
                for speed in np.arange(1., 251., 1.):
                    ev = aeroelastic_eigs(spec, beam, servo_k, cp_fraction, speed, rho, sound, PH)
                    most = ev[np.argmax(ev.real)]
                    if most.real > 1e-5:
                        onset = float(speed)
                        kind = "oscillatory instability" if abs(most.imag) > 1. else "static divergence"
                        break
                target_ev = aeroelastic_eigs(spec, beam, servo_k, cp_fraction, U, rho, sound, PH)
                stability_rows.append({"fin": spec["name"], "servo_stiffness_Nm_rad_ASSUMED": servo_k,
                    "CP_offset_MAC_fraction_ASSUMED": cp_fraction,
                    "screen_onset_speed_m_s": onset, "instability_type": kind,
                    "max_real_eigenvalue_at_target_speed_per_s": float(max(target_ev.real)),
                    "onset_to_target_speed_ratio": None if onset is None else onset / U})
    return dict(speed=U, q=q, summaries=summaries, torque_rows=torque_rows, stability_rows=stability_rows,
                control=np.column_stack(control_columns))

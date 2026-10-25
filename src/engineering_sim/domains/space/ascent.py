# DESCRIPTION: 3-DOF point-mass launch vehicle integrated in the inertial frame from a launch site (LLA) to orbit.
#              Air: density.AirColumn (core.env at the site below core_top -> US76 continuation -> Vallado >= 86 km),
#              the SAME column the orbit propagator uses, so there is no seam between "atmospheric" and "space"
#              flight. Wind (h_agl < wind_top, default 86 km = core top) comes from engineering_sim.core.env.wind (core v0.2:
#              MOS surface layer -> boundary layer -> geostrophic/free air, gusts decaying with height) rotated
#              ENU -> ECEF -> ECI and added to the co-rotating atmosphere. (v0.1 stopped wind at 3 km.)
#              Gravity: forces.accel_gravity (J2..J4). Thrust: F = mdot g0 Isp_vac - p_amb A_exit (back-pressure).
#              Guidance (simple, parametric): vertical rise -> pitch kick -> gravity turn on stage 1 (thrust along
#              the air-relative velocity) -> stage 2 linear pitch program in the local vertical plane of the launch
#              azimuth -> cutoff when the osculating perigee reaches the target, then coast.
#              Vehicle numbers in DEFAULT_VEHICLE are ASSUMED illustrative (small-launcher class), not any real
#              vehicle's data.
import numpy as np

from .constants import G0
from .frames import lla_to_ecef, ecef_to_eci, eci_to_ecef, enu_matrix, geodetic_alt_fast, rv_to_kepler, rot_z, gmst
from .forces import accel_gravity, v_rel_air

DEFAULT_VEHICLE = dict(
    payload=150.0, fairing=50.0, fairing_sep_h=110e3, cd=0.35, area=1.13,
    stages=[dict(dry=950.0, prop=9250.0, thrust_vac=250e3, isp_vac=311.0, a_exit=0.70),
            dict(dry=250.0, prop=2050.0, thrust_vac=25.8e3, isp_vac=343.0, a_exit=0.30)],
    coast_after_s1=3.0)
DEFAULT_GUIDE = dict(launch_az_deg=90.0, h_kick=400.0, kick_deg=11.0, s2_pitch0_deg=15.0, s2_pitch1_deg=-10.0,
                     target_perigee_m=200e3)


def fly(air, lat_deg, lon_deg, jd0, vehicle=None, guide=None, h=0.1, t_max=1200.0, wind_top=86000.0, log_every=10,
        coast_s=0.0):
    veh = dict(DEFAULT_VEHICLE, **(vehicle or {}))
    gd = dict(DEFAULT_GUIDE, **(guide or {}))
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    site_h = air.site
    r_e = lla_to_ecef(lat, lon, site_h)
    r, v = ecef_to_eci(r_e, jd0, np.zeros(3))
    E = enu_matrix(lat, lon)                                 # rows E,N,U in ECEF
    az = np.deg2rad(gd["launch_az_deg"])
    stages = [dict(s) for s in veh["stages"]]
    mass = veh["payload"] + veh["fairing"] + sum(s["dry"] + s["prop"] for s in stages)
    si, prop_left, fairing_on, phase = 0, stages[0]["prop"], True, "vertical"
    t, coast_left, s2_t0, s2_pitch0 = 0.0, 0.0, None, None
    log = dict(t=[], r=[], v=[], h=[], rho=[], q=[], mach=[], mass=[], phase=[], stage=[])
    events = []

    def air_vel(r, v, tt):
        vr = v_rel_air(r[None], v[None])[0]
        hh = geodetic_alt_fast(r[None])[0]
        if hh - site_h < wind_top and hasattr(air.env, "wind"):
            jd = jd0 + tt / 86400.0
            re = eci_to_ecef(r, jd)
            enu = E @ (re - r_e)
            wind_enu = air.env.wind(np.array([[enu[0], enu[1], max(hh - site_h, 0.5)]]), tt)[0]
            wind_ecef = E.T @ np.asarray(wind_enu, float)
            vr = vr - rot_z(gmst(jd)) @ wind_ecef
        return vr

    def thrust_dir(r, v, tt, vr):
        jd = jd0 + tt / 86400.0
        up = r / np.linalg.norm(r)
        # downrange direction: launch azimuth in the ENU of the CURRENT ground point (ECI)
        Ri = rot_z(gmst(jd))
        latc = np.arcsin(r[2] / np.linalg.norm(r))
        lonc = np.arctan2(r[1], r[0]) - gmst(jd)
        Ec = enu_matrix(latc, lonc)
        east, north = Ri @ Ec[0], Ri @ Ec[1]
        down = np.cos(az) * north + np.sin(az) * east
        if phase == "vertical":
            return up
        if phase == "kick":
            k = np.deg2rad(gd["kick_deg"])
            return np.cos(k) * up + np.sin(k) * down
        if phase == "gravity_turn":
            return vr / np.linalg.norm(vr)
        if phase == "s2":
            frac = min((tt - s2_t0) / max(stages[1]["prop"] * G0 * stages[1]["isp_vac"] / stages[1]["thrust_vac"], 1.0), 1.0)
            p = np.deg2rad(s2_pitch0 + (gd["s2_pitch1_deg"] - s2_pitch0) * frac)
            return np.sin(p) * up + np.cos(p) * down
        return None

    def deriv(r, v, m, tt, thrusting):
        vr = air_vel(r, v, tt)
        hh = geodetic_alt_fast(r[None])[0]
        rho = float(air.rho(hh))
        area = veh["area"]
        a = accel_gravity(r[None])[0] - 0.5 * rho * veh["cd"] * area * np.linalg.norm(vr) * vr / m
        mdot = 0.0
        if thrusting:
            st = stages[si]
            F = st["thrust_vac"] - float(air.pressure(hh)) * st["a_exit"]
            d = thrust_dir(r, v, tt, vr)
            a = a + F / m * d
            mdot = -st["thrust_vac"] / (G0 * st["isp_vac"])
        return v, a, mdot, rho, vr

    k = 0
    while t < t_max:
        thrusting = phase in ("vertical", "kick", "gravity_turn", "s2") and coast_left <= 0
        hh = geodetic_alt_fast(r[None])[0]
        if k % log_every == 0:
            _, _, _, rho, vr = deriv(r, v, mass, t, False)
            vrn = np.linalg.norm(vr)
            T_amb = float(air.pressure(hh)) / max(rho * 287.05287, 1e-30)
            for key, val in (("t", t), ("r", r.copy()), ("v", v.copy()), ("h", hh), ("rho", rho),
                             ("q", 0.5 * rho * vrn ** 2), ("mach", vrn / np.sqrt(1.4 * 287.05287 * max(T_amb, 150.0))),
                             ("mass", mass), ("phase", phase), ("stage", si)):
                log[key].append(val)
        # phase logic
        if phase == "vertical" and hh - site_h > gd["h_kick"]:
            phase = "kick"
            events.append(("kick", t, hh))
        elif phase == "kick":
            vr = air_vel(r, v, t)
            up = r / np.linalg.norm(r)
            if np.degrees(np.arccos(np.clip(np.dot(vr, up) / np.linalg.norm(vr), -1, 1))) >= gd["kick_deg"]:
                phase = "gravity_turn"
                events.append(("gravity_turn", t, hh))
        if fairing_on and hh > veh["fairing_sep_h"]:
            fairing_on, mass = False, mass - veh["fairing"]
            events.append(("fairing_sep", t, hh))
        if thrusting and prop_left <= 0:
            if si == 0:
                mass -= stages[0]["dry"]
                si, prop_left, phase = 1, stages[1]["prop"], "s2"
                coast_left = veh["coast_after_s1"]
                s2_t0 = t + coast_left
                vr = air_vel(r, v, t)
                up = r / np.linalg.norm(r)
                s2_pitch0 = gd["s2_pitch0_deg"] if gd["s2_pitch0_deg"] is not None else \
                    90.0 - np.degrees(np.arccos(np.clip(np.dot(v, up) / np.linalg.norm(v), -1, 1)))
                events.append(("staging", t, hh))
            else:
                phase = "coast"
                events.append(("s2_depletion", t, hh))
        if phase == "s2" and coast_left <= 0:
            kep = rv_to_kepler(r, v)
            if kep["a"] * (1 - kep["e"]) - 6378137.0 >= gd["target_perigee_m"]:
                phase = "coast"
                events.append(("seco", t, hh))
        if phase == "coast" and coast_s <= 0:
            break
        if phase == "coast":
            coast_s -= h
        if hh < site_h - 1.0 and t > 5:
            events.append(("impact", t, hh))
            break
        thrusting = phase in ("vertical", "kick", "gravity_turn", "s2") and coast_left <= 0
        coast_left -= h if coast_left > 0 else 0.0
        # RK4 (mass integrated alongside)
        s = (r, v, mass)
        f = lambda rr, vv, mm, tt: deriv(rr, vv, mm, tt, thrusting)[:3]  # noqa: E731
        k1 = f(*s, t)
        k2 = f(*[x + h / 2 * d for x, d in zip(s, k1)], t + h / 2)
        k3 = f(*[x + h / 2 * d for x, d in zip(s, k2)], t + h / 2)
        k4 = f(*[x + h * d for x, d in zip(s, k3)], t + h)
        r, v, mass = [x + h / 6 * (a + 2 * b + 2 * c + d) for x, a, b, c, d in zip(s, k1, k2, k3, k4)]
        if thrusting:
            prop_left -= stages[si]["thrust_vac"] / (G0 * stages[si]["isp_vac"]) * h
        t += h
        k += 1
    kep = rv_to_kepler(r, v)
    res = dict(t_end=t, r=r, v=v, mass=mass, events=events, perigee_m=kep["a"] * (1 - kep["e"]) - 6378137.0,
               apogee_m=kep["a"] * (1 + kep["e"]) - 6378137.0, incl_deg=np.degrees(kep["i"]), phase=phase,
               max_q=max(log["q"]) if log["q"] else 0.0)
    return res, {key: (np.array(val) if key != "phase" else val) for key, val in log.items()}


# =====================================================================================================================
# 6-DOF launcher with thrust-vector control and closed-loop guidance (2026-10-15 ATTITUDE)
# =====================================================================================================================
# fly6(): same vehicle, same air column, same wind, same events as fly(), plus
#   attitude    body +X = thrust axis; quaternion body -> ECI and body rates, advanced by core.aero.step_rigid_np
#               (float64 mirror of core/rigid6.step_rigid); mass, CG and inertia per stage (uniform-rod segments)
#   aero        axial drag (cd, area) on the air-relative velocity in body axes + normal force cn_alpha at x_cp
#               (unfinned launcher: CP ahead of the CG = statically UNSTABLE, held by TVC) + aero/jet damping
#   TVC         PD attitude loop (bandwidth tvc_wn, damping tvc_zeta) -> gimbal side force, limited to
#               +/- gimbal_max_deg and gimbal_rate_dps; roll held by a rate damper (RCS, ASSUMED)
#   guidance    stage 1: vertical -> kick -> gravity turn along the GROUND-relative velocity (Earth-fixed frame; the
#               open-loop fly() steers along the AIR-relative velocity, which a westerly jet aloft swings backwards at
#               low speed - the main source of its wind sensitivity) with q*alpha load relief: when q*alpha (air) would
#               exceed qalpha_max the command is rotated toward the air-relative velocity just enough to hold the
#               limit. Commands are slew-rate limited (slew_dps). Stage 2: closed-loop linear-tangent steering tan(chi) = A + B t in the
#               plane at stage-2 ignition, re-solved every guid_dt s by shooting a vacuum point-mass model to the
#               target insertion (radius r_T, radial velocity 0, energy of the target orbit); cutoff exactly when
#               the osculating energy reaches the target (step shortened to hit it). This removes the open-loop
#               sensitivity of fly() to winds aloft.
# Vehicle geometry and controller numbers are ASSUMED illustrative.
DEFAULT_VEHICLE6 = dict(diam=1.2, len_s1=11.5, len_upper=6.5, len_s2_stack=5.0, cn_alpha=2.0, x_cp_frac=0.17,
                        gimbal_max_deg=5.0, gimbal_rate_dps=20.0, tvc_wn=4.0, tvc_zeta=0.7, roll_damp=2.0)
DEFAULT_GUIDE6 = dict(target_perigee_m=200e3, target_apogee_m=500e3, guid_dt=2.0, t_go_freeze=8.0,
                      qalpha_max_kpa_deg=80.0, slew_dps=3.0)


def _mass_props6(veh, v6, si, prop_left, fairing_on):
    """mass, CG (m from nose), principal inertia (roll, pitch, yaw) of the stack. Uniform-rod segments (ASSUMED)."""
    st = veh["stages"]
    pay = veh["payload"] + (veh["fairing"] if fairing_on else 0.0)
    if si == 0:
        segs = [(pay + st[1]["dry"] + st[1]["prop"], 0.5 * v6["len_upper"], v6["len_upper"]),
                (st[0]["dry"] + prop_left, v6["len_upper"] + 0.5 * v6["len_s1"], v6["len_s1"])]
        L = v6["len_upper"] + v6["len_s1"]
    else:
        segs = [(pay + st[1]["dry"] + prop_left, 0.5 * v6["len_s2_stack"], v6["len_s2_stack"])]
        L = v6["len_s2_stack"]
    m = sum(s[0] for s in segs)
    x_cg = sum(s[0] * s[1] for s in segs) / m
    r = 0.5 * v6["diam"]
    I_p = sum(s[0] * (s[2] ** 2 / 12.0 + r * r / 4.0) + s[0] * (s[1] - x_cg) ** 2 for s in segs)
    I_r = 0.5 * m * r * r
    return m, x_cg, np.array([I_r, I_p, I_p]), L


def _lt_shoot(r0, rd0, vh0, m0, T, mdot, A, B, r_T, E_T, t_max, mu, dt=1.0):
    """Vacuum point-mass, polar in-plane. Steering tan(chi) = A + B t. Integrate until the specific energy reaches E_T
    (linear interpolation of the crossing) or propellant/time runs out. Returns (r, rdot, vh, t_cut, reached)."""
    def f(y, t):
        r, rd, vh, m = y
        chi = np.arctan(A + B * t)
        aT = T / m
        return np.array([rd, vh * vh / r - mu / (r * r) + aT * np.sin(chi), (r * aT * np.cos(chi) - rd * vh) / r,
                         -mdot])
    y = np.array([r0, rd0, vh0, m0], float)
    t = 0.0
    E = 0.5 * (rd0 ** 2 + vh0 ** 2) - mu / r0
    while t < t_max:
        h = min(dt, t_max - t)
        k1 = f(y, t); k2 = f(y + h / 2 * k1, t + h / 2); k3 = f(y + h / 2 * k2, t + h / 2); k4 = f(y + h * k3, t + h)
        yn = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        En = 0.5 * (yn[1] ** 2 + yn[2] ** 2) - mu / yn[0]
        if En >= E_T:
            fr = (E_T - E) / max(En - E, 1e-12)
            yc = y + fr * (yn - y)
            return yc[0], yc[1], yc[2], t + fr * h, True
        y, E, t = yn, En, t + h
    return y[0], y[1], y[2], t, False


def _lt_solve(state, A, B, r_T, E_T, t_max, mu, iters=6):
    """Newton on (A, B) so that r(t_cut) = r_T and rdot(t_cut) = 0 at the energy cutoff."""
    def res(a, b):
        r, rd, vh, tc, ok = _lt_shoot(*state, a, b, r_T, E_T, t_max, mu)
        return np.array([(r - r_T) / 1e3, rd / 10.0]), ok, tc
    x = np.array([A, B], float)
    best = None
    for _ in range(iters):
        F, ok, tc = res(*x)
        if best is None or np.linalg.norm(F) < best[1]:
            best = (x.copy(), np.linalg.norm(F), tc, ok)
        if np.linalg.norm(F) < 1e-3:
            break
        J = np.zeros((2, 2))
        for j, dx in enumerate((1e-3, 1e-5)):
            xp = x.copy(); xp[j] += dx
            J[:, j] = (res(*xp)[0] - F) / dx
        try:
            step = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            break
        lim = np.array([0.5, 0.01])
        step = np.clip(step, -lim, lim)
        x = x + step
    return best


def fly6(air, lat_deg, lon_deg, jd0, vehicle=None, guide=None, vehicle6=None, guide6=None, h=0.02, t_max=1200.0,
         wind_top=86000.0, log_every=25, wind_every_s=0.1, ideal_attitude=False):
    """6-DOF ascent (see block comment). ideal_attitude=True makes the body follow the command exactly (TVC off,
    no rotational dynamics) - a check that isolates the guidance. Returns (res, log) like fly()."""
    from engineering_sim.core import aero as AE
    from .constants import MU_EARTH, WGS84_A
    veh = dict(DEFAULT_VEHICLE, **(vehicle or {}))
    gd = dict(DEFAULT_GUIDE, **(guide or {}))
    v6 = dict(DEFAULT_VEHICLE6, **(vehicle6 or {}))
    g6 = dict(DEFAULT_GUIDE6, **(guide6 or {}))
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    site_h = air.site
    r_e = lla_to_ecef(lat, lon, site_h)
    r, v = ecef_to_eci(r_e, jd0, np.zeros(3))
    E = enu_matrix(lat, lon)
    az = np.deg2rad(gd["launch_az_deg"])
    stages = [dict(s) for s in veh["stages"]]
    area = veh["area"]
    si, prop_left, fairing_on, phase = 0, stages[0]["prop"], True, "vertical"
    t, coast_left = 0.0, 0.0
    r_T = WGS84_A + g6["target_perigee_m"]
    a_T = WGS84_A + 0.5 * (g6["target_perigee_m"] + g6["target_apogee_m"])
    E_T = -MU_EARTH / (2.0 * a_T)
    q = AE.q_from_x_to(r / np.linalg.norm(r))
    w = np.zeros(3)
    F_lat = np.zeros(2)                                  # gimbal side force (body y, z) [N], rate limited
    wind_hold, next_wind = np.zeros(3), 0.0
    plane_n, lt, t_solve, next_guid, chi_hold = None, None, None, 0.0, None
    events = []
    log = dict(t=[], r=[], v=[], h=[], rho=[], q=[], mach=[], mass=[], phase=[], stage=[], alpha=[], qalpha=[],
               gimbal=[], att_err=[])
    k = 0

    def wind_eci(r, tt):
        nonlocal wind_hold, next_wind
        hh = geodetic_alt_fast(r[None])[0]
        if not (hh - site_h < wind_top and hasattr(air.env, "wind")):
            return np.zeros(3)
        if tt >= next_wind:
            jd = jd0 + tt / 86400.0
            re = eci_to_ecef(r, jd)
            enu = E @ (re - r_e)
            wind_enu = air.env.wind(np.array([[enu[0], enu[1], max(hh - site_h, 0.5)]]), tt)[0]
            wind_hold = rot_z(gmst(jd)) @ (E.T @ np.asarray(wind_enu, float))
            next_wind = tt + wind_every_s
        return wind_hold

    def basis(r, tt):
        jd = jd0 + tt / 86400.0
        up = r / np.linalg.norm(r)
        Ri = rot_z(gmst(jd))
        latc = np.arcsin(r[2] / np.linalg.norm(r))
        lonc = np.arctan2(r[1], r[0]) - gmst(jd)
        Ec = enu_matrix(latc, lonc)
        east, north = Ri @ Ec[0], Ri @ Ec[1]
        return up, np.cos(az) * north + np.sin(az) * east

    def command(r, v, vr, tt, m, qd):
        nonlocal plane_n, lt, t_solve, next_guid, chi_hold
        up, down = basis(r, tt)
        if phase == "vertical":
            return up
        if phase == "kick":
            kk = np.deg2rad(gd["kick_deg"])
            return np.cos(kk) * up + np.sin(kk) * down
        if phase == "gravity_turn":
            vg = v_rel_air(r[None], v[None])[0]
            g_dir = vg / np.linalg.norm(vg)
            a_dir = vr / np.linalg.norm(vr)
            al = float(np.arccos(np.clip(g_dir @ a_dir, -1.0, 1.0)))
            lim = g6["qalpha_max_kpa_deg"] * 1e3 / max(qd, 1e-9)              # allowed alpha [deg]
            if np.degrees(al) > lim and al > 1e-9:                             # load relief toward the air vector
                f = 1.0 - np.deg2rad(lim) / al
                ax = np.cross(g_dir, a_dir); ax /= np.linalg.norm(ax)
                ang = f * al
                g_dir = g_dir * np.cos(ang) + np.cross(ax, g_dir) * np.sin(ang) + ax * (ax @ g_dir) * (1 - np.cos(ang))
            return g_dir
        if phase == "s2":
            if plane_n is None:
                plane_n = np.cross(r, v); plane_n /= np.linalg.norm(plane_n)
            hor = np.cross(plane_n, up); hor /= np.linalg.norm(hor)
            st = stages[1]
            T = st["thrust_vac"]
            mdot = T / (G0 * st["isp_vac"])
            rn = np.linalg.norm(r)
            rd = float(np.dot(v, up))
            vh = float(np.dot(v, hor))
            if lt is None or tt >= next_guid:
                t_left = prop_left / mdot
                x0 = (lt[0], lt[1]) if lt is not None else (np.tan(np.deg2rad(20.0)), -2e-3)
                if lt is not None:            # re-reference the held law to the new solve time
                    x0 = (lt[0] + lt[1] * (tt - t_solve), lt[1])
                sol = _lt_solve((rn, rd, vh, m, T, mdot), x0[0], x0[1], r_T, E_T, t_left, MU_EARTH)
                if sol is not None and (lt is None or sol[2] > g6["t_go_freeze"]):
                    lt, t_solve = sol[0], tt
                    events_g.append((tt, float(sol[1]), float(sol[2]), bool(sol[3])))
                next_guid = tt + g6["guid_dt"]
            chi = np.arctan(lt[0] + lt[1] * (tt - t_solve))
            return np.sin(chi) * up + np.cos(chi) * hor
        return None

    events_g = []
    d_prev = None
    while t < t_max:
        thrusting = phase in ("vertical", "kick", "gravity_turn", "s2") and coast_left <= 0
        hh = geodetic_alt_fast(r[None])[0]
        m, x_cg, Iner, L = _mass_props6(veh, v6, si, prop_left, fairing_on)
        rho = float(air.rho(hh))
        vr = v_rel_air(r[None], v[None])[0] - wind_eci(r, t)
        vrn = float(np.linalg.norm(vr))
        T_amb = float(air.pressure(hh)) / max(rho * 287.05287, 1e-30)
        mach = vrn / np.sqrt(1.4 * 287.05287 * max(T_amb, 150.0))
        qd = 0.5 * rho * vrn ** 2
        vb = AE.qrot_inv(q, vr)
        alpha = float(np.arctan2(np.hypot(vb[1], vb[2]), vb[0])) if vrn > 1e-3 else 0.0
        # ---- phase logic (same events as fly())
        if phase == "vertical" and hh - site_h > gd["h_kick"]:
            phase = "kick"; events.append(("kick", t, hh))
        elif phase == "kick":
            up, dn = basis(r, t)
            vg = v_rel_air(r[None], v[None])[0]
            tilt = np.degrees(np.arctan2(float(vg @ dn), float(vg @ up)))
            if tilt >= gd["kick_deg"]:
                phase = "gravity_turn"; events.append(("gravity_turn", t, hh))
        if fairing_on and hh > veh["fairing_sep_h"]:
            fairing_on = False; events.append(("fairing_sep", t, hh))
        if thrusting and prop_left <= 0:
            if si == 0:
                si, prop_left, phase = 1, stages[1]["prop"], "s2"
                coast_left = veh["coast_after_s1"]
                next_guid = t + coast_left
                events.append(("staging", t, hh))
            else:
                phase = "coast"; events.append(("s2_depletion", t, hh))
        if phase == "coast":
            break
        if hh < site_h - 1.0 and t > 5:
            events.append(("impact", t, hh)); break
        thrusting = phase in ("vertical", "kick", "gravity_turn", "s2") and coast_left <= 0
        m, x_cg, Iner, L = _mass_props6(veh, v6, si, prop_left, fairing_on)
        st = stages[si]
        Tm = st["thrust_vac"] - float(air.pressure(hh)) * st["a_exit"] if thrusting else 0.0
        d_raw = command(r, v, vr, t, m, qd) if thrusting else None
        if d_raw is not None:                                    # slew-rate limit on the commanded direction
            if d_prev is None:
                d_cmd = d_raw
            else:
                ang = float(np.arccos(np.clip(d_prev @ d_raw, -1.0, 1.0)))
                mx = np.deg2rad(g6["slew_dps"]) * h
                if ang > mx:
                    ax = np.cross(d_prev, d_raw); ax /= np.linalg.norm(ax)
                    d_cmd = d_prev * np.cos(mx) + np.cross(ax, d_prev) * np.sin(mx)
                else:
                    d_cmd = d_raw
            d_prev = d_cmd / np.linalg.norm(d_cmd)
        else:
            d_cmd = None
        if ideal_attitude and d_cmd is not None:
            q = AE.q_from_x_to(d_cmd); w = np.zeros(3)
        # ---- TVC: PD on attitude error -> gimbal side force at the nozzle (body x = x_cg - L)
        lever = L - x_cg
        F_cmd = np.zeros(2)
        err = 0.0
        if d_cmd is not None and Tm > 0 and not ideal_attitude:
            db = AE.qrot_inv(q, d_cmd)
            e = np.cross([1.0, 0, 0], db)                            # body-frame error axis * sin(angle)
            err = float(np.degrees(np.arctan2(np.linalg.norm(e), db[0])))
            wn, zt = v6["tvc_wn"], v6["tvc_zeta"]
            tau_des = Iner * (wn * wn * e - 2.0 * zt * wn * w)
            F_cmd = np.array([-tau_des[2] / lever, tau_des[1] / lever])
            fmax = Tm * np.sin(np.deg2rad(v6["gimbal_max_deg"]))
            n = np.linalg.norm(F_cmd)
            if n > fmax:
                F_cmd *= fmax / n
        dF = Tm * np.sin(np.deg2rad(v6["gimbal_rate_dps"] * h)) if Tm > 0 else np.inf
        F_lat = F_lat + np.clip(F_cmd - F_lat, -dF, dF) if Tm > 0 else np.zeros(2)
        gimbal = float(np.degrees(np.arcsin(min(np.linalg.norm(F_lat) / Tm, 1.0)))) if Tm > 0 else 0.0
        # ---- log
        if k % log_every == 0:
            for key, val in (("t", t), ("r", r.copy()), ("v", v.copy()), ("h", hh), ("rho", rho), ("q", qd),
                             ("mach", mach), ("mass", m), ("phase", phase), ("stage", si),
                             ("alpha", np.degrees(alpha)), ("qalpha", qd * np.degrees(alpha)), ("gimbal", gimbal),
                             ("att_err", err)):
                log[key].append(val)
        # ---- body forces/torques (frozen over the step)
        Fx = np.sqrt(max(Tm * Tm - float(F_lat @ F_lat), 0.0))
        Fb_thr = np.array([Fx, F_lat[0], F_lat[1]])
        Fb_aer = np.zeros(3)
        tau = np.zeros(3)
        Vb = float(np.linalg.norm(vb))
        Fb_aer[0] = -0.5 * rho * veh["cd"] * area * Vb * vb[0]
        x_cp = v6["x_cp_frac"] * L
        Fn = np.array([0.0, -0.5 * rho * area * v6["cn_alpha"] * Vb * vb[1],
                       -0.5 * rho * area * v6["cn_alpha"] * Vb * vb[2]])
        Fb_aer += Fn
        tau += np.cross([x_cg - x_cp, 0, 0], Fn)
        tau += np.cross([x_cg - L, 0, 0], Fb_thr)
        mdot = st["thrust_vac"] / (G0 * st["isp_vac"]) if thrusting else 0.0
        c2 = AE.damping_c2a(rho, Vb, area, [(v6["cn_alpha"], x_cp)], x_cg) + mdot * (L - x_cg) ** 2
        tau[1] -= c2 * w[1]; tau[2] -= c2 * w[2]
        tau[0] -= v6["roll_damp"] * Iner[0] * w[0]
        Fw_nongrav = AE.qrot(q, Fb_thr + Fb_aer)

        # ---- translation: RK4 on (r, v, m) with the body forces held over the step (attitude frozen)
        def deriv(rr, vv, mm):
            return vv, accel_gravity(rr[None])[0] + Fw_nongrav / mm, -mdot
        E0 = 0.5 * float(v @ v) - MU_EARTH / np.linalg.norm(r)

        def rk4(hs):
            s = (r, v, m)
            k1 = deriv(*s)
            k2 = deriv(*[x + hs / 2 * d for x, d in zip(s, k1)])
            k3 = deriv(*[x + hs / 2 * d for x, d in zip(s, k2)])
            k4 = deriv(*[x + hs * d for x, d in zip(s, k3)])
            return [x + hs / 6 * (a + 2 * b + 2 * c + d) for x, a, b, c, d in zip(s, k1, k2, k3, k4)]
        hs = h
        rn_, vn_, mn_ = rk4(hs)
        cut = False
        if phase == "s2" and thrusting:
            E1 = 0.5 * float(vn_ @ vn_) - MU_EARTH / np.linalg.norm(rn_)
            if E1 >= E_T:                                    # shorten the step to hit the target energy
                hs = h * (E_T - E0) / max(E1 - E0, 1e-12)
                rn_, vn_, mn_ = rk4(hs)
                cut = True
        # ---- rotation: THE rigid-body step (core.aero.step_rigid_np = core/rigid6.step_rigid), translation discarded
        if not ideal_attitude:
            _, q, _, w = AE.step_rigid_np(np.zeros(3), q, np.zeros(3), w, np.zeros(3), tau, (m, m, m), Iner, Iner, hs)
        r, v = rn_, vn_
        if thrusting:
            prop_left -= mdot * hs
        coast_left -= hs if coast_left > 0 else 0.0
        t += hs
        k += 1
        if cut:
            phase = "coast"
            events.append(("seco", t, geodetic_alt_fast(r[None])[0]))
            break
    kep = rv_to_kepler(r, v)
    lg = {key: (np.array(val) if key != "phase" else val) for key, val in log.items()}
    res = dict(t_end=t, r=r, v=v, mass=m, events=events, perigee_m=kep["a"] * (1 - kep["e"]) - WGS84_A,
               apogee_m=kep["a"] * (1 + kep["e"]) - WGS84_A, incl_deg=np.degrees(kep["i"]), phase=phase,
               max_q=float(lg["q"].max()) if len(lg["q"]) else 0.0,
               max_qalpha_kpa_deg=float(lg["qalpha"].max()) / 1e3 if len(lg["qalpha"]) else 0.0,
               max_alpha_deg_in_air=float(lg["alpha"][lg["q"] > 1e3].max()) if (lg["q"] > 1e3).any() else 0.0,
               max_gimbal_deg=float(lg["gimbal"].max()) if len(lg["gimbal"]) else 0.0,
               prop_left_kg=prop_left, guidance=events_g, target=dict(r_T=r_T, a_T=a_T))
    return res, lg

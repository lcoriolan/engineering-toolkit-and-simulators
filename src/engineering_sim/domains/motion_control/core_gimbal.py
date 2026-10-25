#!/usr/bin/env python3
# Motion-control backend: set_position, get_position, set_velocity, enable and
# limits. Starts disabled and reports feedback_kind "simulated". Air density
# and gusting wind use engineering_sim.core.env.Env; state uses core.statelog.
# Plant modes (parameter `plant`):
#   "ideal": instant position, hard clamping and ignored commands when disabled.
#            units_per_deg enables integer native quantisation and reversed scales.
#            NumPy is optional for this mode.
#                "dynamic" — (default) per-axis rigid servo, MODELED:
#                     J th'' = sat(Kp (th_goal - th) + Kd (w_hint - th'), tau_max) - b th' + tau_wind + tau_grav
#                     Kp = J wn^2, Kd = 2 zeta J wn (servo inner loop as a 2nd-order system), |th'| <= w_max,
#                     mechanical hard stops at the travel limits (velocity zeroed into a stop).
#                  Wind load (core air + gusts at the mount point, sampled once per advance() call — frozen over
#                  one control tick, ASSUMED << gust time scale):
#                     F = 0.5 rho Cd A |u| u   (payload treated as an isotropic bluff body, ASSUMED)
#                     tau = (d_cp b) x F, projected on the pan axis (z) and the tilt axis a_t = (sin p, -cos p, 0),
#                     b = boresight (cos t cos p, cos t sin p, sin t) in the core ENU frame (pan 0 = +E, CCW +).
#                  Gravity on tilt: tau_g = ((d_cg b) x (0,0,-m g)) . a_t.  Pan/tilt inertia coupling ignored.
#              Time: `clock` = "external" (harness calls advance(dt); deterministic) or "wall" (service use:
#              each get_position advances by the monotonic delta, capped at max_wall_step_s).
#              Plant ODE is 2 scalar axes integrated on the CPU (semi-implicit Euler, sub-step dt_int); a GPU kernel
#              buys nothing at this size. The environment it sees is the core's (GPU wind field).
#              ALL plant constants below are ASSUMED generic small-gimbal values, parametric, not a measured unit.
import math
import threading
import time

try:
    import numpy as np
except ImportError:          # "ideal" plant on a GPU-/numpy-less host: no numpy needed
    np = None

AXES_ORDER = ("pan", "tilt")

DEFAULT_AXIS = dict(J=0.02, wn_hz=4.0, zeta=0.7, tau_max=1.5, w_max_deg_s=120.0, b=0.005)
DEFAULT_PAYLOAD = dict(mount_xyz=[0.0, 0.0, 3.0], area_m2=0.05, cd=1.2, d_cp_m=0.04, d_cg_m=0.01, mass_kg=1.0)

DEFAULT_ENV = dict(atmos={}, wind=dict(u10=0.0, turbulence=False))
LEGACY_ENV = dict(atmos=dict(), wind=dict(u10=0.0, turbulence=False))   # compatibility profile: still air without turbulence
try:
    from engineering_sim.core.constants import G0            # the one core gravity constant (core v0.2)
except ImportError:
    G0 = 9.80665


def boresight(pan_deg, tilt_deg):
    p, t = math.radians(pan_deg), math.radians(tilt_deg)
    return np.array([math.cos(t) * math.cos(p), math.cos(t) * math.sin(p), math.sin(t)])


def tilt_axis(pan_deg):
    p = math.radians(pan_deg)
    return np.array([math.sin(p), -math.cos(p), 0.0])


def quat_pan_tilt(pan_deg, tilt_deg):
    """Body->world quaternion (w,x,y,z): yaw = pan about +Z, then pitch UP by tilt (body +X = boresight)."""
    from engineering_sim.core.statelog import quat_from_euler
    return quat_from_euler(0.0, -math.radians(tilt_deg), math.radians(pan_deg))


class GimbalPlant:
    def __init__(self, limits, home, plant="dynamic", axes=None, payload=None, env=None, dt_int=1e-3):
        self.limits = {a: tuple(map(float, limits[a])) for a in limits}
        self.plant = plant
        self.axes = {a: dict(DEFAULT_AXIS, **((axes or {}).get(a, {}))) for a in limits}
        self.payload = dict(DEFAULT_PAYLOAD, **(payload or {}))
        self.env = env
        self.dt_int = float(dt_int)
        self.th = {a: float(home[a]) for a in limits}
        self.w = {a: 0.0 for a in limits}
        self.goal = dict(self.th)
        self.hint = {a: 0.0 for a in limits}
        self.t = 0.0
        self.last_wind = np.zeros(3) if np is not None else (0.0, 0.0, 0.0)
        self.last_tau_wind = {a: 0.0 for a in limits}

    def command(self, axis, goal_deg, hint_deg_s=None):
        self.goal[axis] = goal_deg
        if hint_deg_s is not None:
            self.hint[axis] = hint_deg_s
        if self.plant == "ideal":
            self.th[axis] = goal_deg
            self.w[axis] = 0.0

    def _loads(self):
        """Wind + gravity torques [N m] per axis at the current pose (wind sampled from the core)."""
        pan = self.th.get("pan", 0.0)
        tilt = self.th.get("tilt", 0.0)
        P = self.payload
        if self.env is not None:
            u = np.asarray(self.env.wind([P["mount_xyz"]], self.t)[0], float)
            rho = float(self.env.rho([P["mount_xyz"][2]])[0])
        else:
            u, rho = np.zeros(3), 1.225
        F = 0.5 * rho * P["cd"] * P["area_m2"] * np.linalg.norm(u) * u
        b = boresight(pan, tilt)
        tau = np.cross(P["d_cp_m"] * b, F)
        tg = np.cross(P["d_cg_m"] * b, np.array([0.0, 0.0, -P["mass_kg"] * G0]))
        at = tilt_axis(pan)
        out = {"pan": float(tau[2]), "tilt": float(tau @ at + tg @ at)}
        self.last_wind = u
        self.last_tau_wind = {"pan": float(tau[2]), "tilt": float(tau @ at)}
        return {a: out.get(a, 0.0) for a in self.th}

    def advance(self, dt, enabled=True):
        if dt <= 0:
            return
        if self.plant == "ideal":
            self.t += dt
            return
        loads = self._loads()
        n = max(1, int(math.ceil(dt / self.dt_int)))
        h = dt / n
        for _ in range(n):
            for a in self.th:
                c = self.axes[a]
                J = c["J"]
                wn = 2 * math.pi * c["wn_hz"]
                wmax = math.radians(c["w_max_deg_s"])
                th, w = math.radians(self.th[a]), math.radians(self.w[a])
                tau_m = 0.0
                if enabled:
                    tau_m = J * wn * wn * (math.radians(self.goal[a]) - th) + \
                        2 * c["zeta"] * J * wn * (math.radians(self.hint[a]) - w)
                    tau_m = max(-c["tau_max"], min(c["tau_max"], tau_m))
                acc = (tau_m - c["b"] * w + loads[a]) / J
                w = max(-wmax, min(wmax, w + acc * h))
                th = th + w * h
                lo, hi = (math.radians(v) for v in self.limits[a])
                if th < lo:
                    th, w = lo, max(w, 0.0)
                if th > hi:
                    th, w = hi, min(w, 0.0)
                self.th[a], self.w[a] = math.degrees(th), math.degrees(w)
        self.t += dt


class CoreGimbalDriver:
    """MotorDriver backed by GimbalPlant with axis, command and limit checks."""
    feedback_kind = "simulated"

    def __init__(self, limits, home, plant="dynamic", env=None, axes=None, payload=None, clock="external",
                 units_per_deg=None, zero_native=0.0, max_wall_step_s=0.1, dt_int=1e-3, log=None, log_hz=30.0):
        self.feedback_source = "unified-sim-core-%s-plant" % plant
        self._lock = threading.RLock()
        self._enabled = False
        self._limits = {a: tuple(map(float, limits[a])) for a in limits}
        self._cal = None
        self._native_limits = {}
        if units_per_deg is not None:
            from .calibration import LinearCalibration
            self._cal = LinearCalibration(units_per_deg=units_per_deg, zero_native=zero_native)
            for a, (lo, hi) in self._limits.items():
                ends = sorted(self._cal.to_native(v) for v in (lo, hi))
                self._native_limits[a] = (math.ceil(ends[0]), math.floor(ends[1]))
            home = {a: self._cal.to_degrees(self._quantize(a, home[a])) for a in home}
        self.plant = GimbalPlant(self._limits, home, plant, axes, payload, env, dt_int)
        self._velocity = {a: 0.0 for a in limits}
        self.clock, self.max_wall_step_s = clock, max_wall_step_s
        self._wall = time.monotonic()
        self.commands = []
        self.log, self.log_dt, self._next_log, self.auto_log = log, 1.0 / log_hz, 0.0, True

    def _number(self, value):
        if not math.isfinite(value):
            raise ValueError("non-finite actuator command")

    def _quantize(self, axis, angle_deg):
        lo, hi = self._native_limits[axis]
        return min(hi, max(lo, round(self._cal.to_native(angle_deg))))

    def set_position(self, axis, angle_deg):
        self._number(angle_deg)
        with self._lock:
            if not self._enabled:
                return
            if self._cal is not None:
                angle_deg = self._cal.to_degrees(self._quantize(axis, angle_deg))
            lo, hi = self.limits(axis)
            value = min(hi, max(lo, angle_deg))
            self.plant.command(axis, value, self._velocity[axis])
            self.commands.append((axis, value))
            if len(self.commands) > 1000:
                del self.commands[:500]

    def get_position(self, axis):
        with self._lock:
            if self.clock == "wall":
                now = time.monotonic()
                self.advance(min(now - self._wall, self.max_wall_step_s))
                self._wall = now
            return self.plant.th[axis]

    def set_velocity(self, axis, deg_per_s):
        self._number(deg_per_s)
        with self._lock:
            self._velocity[axis] = deg_per_s if self._enabled else 0.0

    def enable(self, on):
        with self._lock:
            self._enabled = bool(on)
            if not on:
                self._velocity = {a: 0.0 for a in self._limits}
                # cancel queued motion: hold where the plant is now
                for a in self._limits:
                    self.plant.goal[a] = self.plant.th[a]
                    self.plant.hint[a] = 0.0

    def limits(self, axis):
        return tuple(self._limits[axis])

    def advance(self, dt):
        with self._lock:
            self.plant.advance(dt, self._enabled)
            if self.log is not None and self.auto_log and self.plant.t + 1e-9 >= self._next_log:
                self._next_log += self.log_dt
                self.log_frame()

    def log_frame(self, extra_bodies=None):
        p = self.plant
        P = p.payload
        mx = np.array(P["mount_xyz"], float)
        pan, tilt = p.th.get("pan", 0.0), p.th.get("tilt", 0.0)
        bx = [mx * [1, 1, 0.5], mx, mx + 0.6 * boresight(pan, tilt)]
        bq = [[1, 0, 0, 0], quat_pan_tilt(pan, tilt), quat_pan_tilt(pan, tilt)]
        if extra_bodies:
            for x, q in extra_bodies:
                bx.append(x); bq.append(q)
        series = [pan, tilt, p.goal.get("pan", 0.0), p.goal.get("tilt", 0.0), float(np.linalg.norm(p.last_wind)),
                  p.last_tau_wind.get("pan", 0.0)]
        self.log.frame(p.t, body_x=bx, body_q=bq, series=series, probe_wind=[p.last_wind])

    def setup_log(self, log, auto=True):
        """Register this gimbal's glyphs/channels on a core StateLog. auto=False: caller calls log_frame()."""
        self.log, self.auto_log = log, auto
        P = self.plant.payload
        log.add_body("mast", "kinematic", "box", [0.08, 0.08, P["mount_xyz"][2]])
        log.add_body("gimbal_payload", "vehicle", "box", [0.30, 0.22, 0.18])
        log.add_body("boresight", "vehicle", "cylinder", [0.02, 1.0])
        for name, unit in (("pan", "deg"), ("tilt", "deg"), ("pan_goal", "deg"), ("tilt_goal", "deg"),
                           ("wind", "m/s"), ("tau_wind_pan", "N m")):
            log.add_series(name, unit)
        log.m["probes"] = [list(map(float, P["mount_xyz"]))]


def create_driver(*, params, axes):
    """motion-control plugin factory. params (all optional):
       plant: "dynamic" | "ideal";  env: "core-default" | "legacy" | {atmos:{..}, wind:{..}};
       axes: {pan:{J,wn_hz,zeta,tau_max,w_max_deg_s,b}, tilt:{..}};  payload: {...};  clock: "wall" | "external";
       units_per_deg / zero_native (native quantisation, as native_simulator)."""
    allowed = {"plant", "env", "axes", "payload", "clock", "units_per_deg", "zero_native", "dt_int"}
    if set(params) - allowed:
        raise ValueError("core gimbal params allow only %s" % sorted(allowed))
    env_cfg = params.get("env", "core-default")
    env = None
    if params.get("plant", "dynamic") != "ideal":
        from engineering_sim.core.env import Env
        env = Env.from_dict(DEFAULT_ENV if env_cfg == "core-default" else LEGACY_ENV if env_cfg == "legacy" else env_cfg)
    return CoreGimbalDriver({a: (s["min_deg"], s["max_deg"]) for a, s in axes.items()},
                            {a: s["home_deg"] for a, s in axes.items()},
                            plant=params.get("plant", "dynamic"), env=env, axes=params.get("axes"),
                            payload=params.get("payload"), clock=params.get("clock", "wall"),
                            units_per_deg=params.get("units_per_deg"), zero_native=params.get("zero_native", 0.0),
                            dt_int=params.get("dt_int", 1e-3))

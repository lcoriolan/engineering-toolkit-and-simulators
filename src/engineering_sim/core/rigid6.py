#!/usr/bin/env python3
# DESCRIPTION: One batch of rigid bodies on the physics GPU. Domains add forces/torques into the shared
#              accumulators (any number of kernels, atomically), then ONE integrate launch advances every body.
#
#              State (world frame ENU, z up; body frame +X forward, +Y port/left, +Z up; origin = CG):
#                x  world CG position [m]          q  attitude quaternion body->world (Warp order x,y,z,w)
#                v  world CG velocity [m/s]        w  body angular velocity [rad/s]
#              Mass model: mass m, principal body inertia I (diag, body axes ASSUMED principal),
#                added mass ma (surge, sway, heave) / added inertia Ia (roll, pitch, yaw), both scaled each step by
#                s in [0,1] (a domain sets s, e.g. submerged fraction, so added mass vanishes in air).
#              Accumulators (cleared by clear()): f = world force [N] (gravity is added by integrate()),
#                tau = body torque about CG [N m].
#              Integrator: semi-implicit Euler; quaternion advanced by the exact body-rate rotation over dt.
#              step_rigid() is the ONE rigid-body step: Rigid6.integrate uses it, and domains with their own state
#              layout (domains/land vehicles + contact bodies) call it from inside their kernels (2026-10-15 UNIFY).
#              ASSUMED simplifications (documented in proof/WATER_DOMAIN_2026-10-15.md): diagonal added mass with no
#              Munk moment / added-mass Coriolis coupling; gyroscopic term uses rigid inertia only.
#
#   USAGE:     rb = Rigid6(); i = rb.add("pwc", mass=822.4, inertia=(Ixx, Iyy, Izz), pos=(0,0,0))
#              rb.upload(device); per step: rb.clear(); <domain kernels call add_wrench>; rb.integrate(dt)
import math

import numpy as np
import warp as wp

from . import physics_device

from .constants import G0          # the one gravity constant (core v0.2)


@wp.func
def add_wrench(f: wp.array(dtype=wp.vec3), tau: wp.array(dtype=wp.vec3), i: int, x: wp.vec3, q: wp.quat,
               F_world: wp.vec3, P_world: wp.vec3):
    """Accumulate a world force applied at a world point (atomic; safe from many threads)."""
    wp.atomic_add(f, i, F_world)
    wp.atomic_add(tau, i, wp.quat_rotate_inv(q, wp.cross(P_world - x, F_world)))


@wp.func
def add_wrench_body(f: wp.array(dtype=wp.vec3), tau: wp.array(dtype=wp.vec3), i: int, q: wp.quat,
                    F_body: wp.vec3, r_body: wp.vec3):
    """Accumulate a body-frame force applied at a body-frame point r (relative to CG)."""
    wp.atomic_add(f, i, wp.quat_rotate(q, F_body))
    wp.atomic_add(tau, i, wp.cross(r_body, F_body))


@wp.func
def step_rigid(p: wp.vec3, rot: wp.quat, vel: wp.vec3, om_b: wp.vec3, F_world: wp.vec3, tau_body: wp.vec3,
               m_eff: wp.vec3, I_eff: wp.vec3, I_rigid: wp.vec3, dt: float, i: int, x: wp.array(dtype=wp.vec3),
               q: wp.array(dtype=wp.quat), v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3)):
    """THE rigid-body step of the unified sim (every 6-DOF body in every domain goes through here).
    p, rot, vel, om_b = current state of body i (values, = x[i], q[i], v[i], w[i]); F_world = total world force
    INCLUDING gravity [N]; tau_body = body-frame torque about the CG [N m]; m_eff / I_eff = per-axis effective mass /
    principal inertia (rigid + added; = (m, m, m) / I_rigid for a body in air); I_rigid = rigid inertia (gyroscopic
    term). Semi-implicit Euler: v, x first; then body rate; attitude by the exact rotation over dt."""
    Fb = wp.quat_rotate_inv(rot, F_world)
    ab = wp.vec3(Fb[0] / m_eff[0], Fb[1] / m_eff[1], Fb[2] / m_eff[2])
    vn = vel + wp.quat_rotate(rot, ab) * dt
    v[i] = vn
    x[i] = p + vn * dt
    T = tau_body - wp.cross(om_b, wp.cw_mul(I_rigid, om_b))
    wd = wp.vec3(T[0] / I_eff[0], T[1] / I_eff[1], T[2] / I_eff[2])
    wn = om_b + wd * dt
    w[i] = wn
    qi = rot
    om = wp.length(wn)
    if om > 1.0e-9:
        qi = qi * wp.quat_from_axis_angle(wn / om, om * dt)
    q[i] = wp.normalize(qi)


@wp.kernel
def _integrate(dt: float, g: float, x: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.quat),
               v: wp.array(dtype=wp.vec3), w: wp.array(dtype=wp.vec3), m: wp.array(dtype=float),
               I: wp.array(dtype=wp.vec3), ma: wp.array(dtype=wp.vec3), Ia: wp.array(dtype=wp.vec3),
               s: wp.array(dtype=float), f: wp.array(dtype=wp.vec3), tau: wp.array(dtype=wp.vec3),
               free: wp.array(dtype=int)):
    i = wp.tid()
    if free[i] == 0:
        return
    mi = m[i]
    si = s[i]
    mai = ma[i]
    Ii = I[i]
    Iai = Ia[i]
    step_rigid(x[i], q[i], v[i], w[i], f[i] + wp.vec3(0.0, 0.0, -g * mi), tau[i],
               wp.vec3(mi + mai[0] * si, mi + mai[1] * si, mi + mai[2] * si),
               wp.vec3(Ii[0] + Iai[0] * si, Ii[1] + Iai[1] * si, Ii[2] + Iai[2] * si), Ii, dt, i, x, q, v, w)


def quat_wxyz_to_xyzw(qw):
    w, x, y, z = qw
    return (x, y, z, w)


def euler_to_quat_wxyz(roll, pitch, yaw):
    """ZYX Euler [rad] -> (w,x,y,z). pitch > 0 = bow DOWN about +Y(port) in this frame (right-hand rule)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def quat_to_euler(q_wxyz):
    """(w,x,y,z) arrays [...,4] -> roll, pitch, yaw [rad] (ZYX)."""
    q = np.asarray(q_wxyz, float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class Rigid6:
    """Host container for a batch of 6-DOF rigid bodies. All arguments are parameters."""

    def __init__(self, g=G0):
        self.rows, self.g, self.n = [], float(g), 0

    def add(self, name, mass, inertia, pos=(0, 0, 0), quat_wxyz=(1, 0, 0, 0), vel=(0, 0, 0), omega_body=(0, 0, 0),
            added_mass=(0, 0, 0), added_inertia=(0, 0, 0), free=True):
        self.rows.append(dict(name=name, m=float(mass), I=tuple(map(float, inertia)), x=tuple(map(float, pos)),
                              q=quat_wxyz_to_xyzw(quat_wxyz), v=tuple(map(float, vel)),
                              w=tuple(map(float, omega_body)), ma=tuple(map(float, added_mass)),
                              Ia=tuple(map(float, added_inertia)), free=int(bool(free))))
        return len(self.rows) - 1

    def upload(self, device=None):
        dev = self.device = physics_device(device)
        R = self.rows
        self.n = len(R)
        vec = lambda k: wp.array(np.array([r[k] for r in R], np.float32).reshape(-1, 3), dtype=wp.vec3, device=dev)
        self.x, self.v, self.w, self.I, self.ma, self.Ia = (vec(k) for k in ("x", "v", "w", "I", "ma", "Ia"))
        self.q = wp.array(np.array([r["q"] for r in R], np.float32).reshape(-1, 4), dtype=wp.quat, device=dev)
        self.m = wp.array(np.array([r["m"] for r in R], np.float32), dtype=float, device=dev)
        self.s = wp.ones(self.n, dtype=float, device=dev)
        self.free = wp.array(np.array([r["free"] for r in R], np.int32), dtype=int, device=dev)
        self.f = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self.tau = wp.zeros(self.n, dtype=wp.vec3, device=dev)
        self.names = [r["name"] for r in R]

    def clear(self):
        self.f.zero_()
        self.tau.zero_()

    def integrate(self, dt):
        wp.launch(_integrate, self.n, [float(dt), self.g, self.x, self.q, self.v, self.w, self.m, self.I, self.ma,
                                       self.Ia, self.s, self.f, self.tau, self.free], device=self.device)

    def state(self):
        """numpy copies: x[N,3], v[N,3], q_wxyz[N,4], w[N,3]."""
        qx = self.q.numpy()
        return self.x.numpy().copy(), self.v.numpy().copy(), qx[:, [3, 0, 1, 2]].copy(), self.w.numpy().copy()

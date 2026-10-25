# DESCRIPTION: Robotics section: serial-chain degrees of freedom and workspace, Denavit–Hartenberg
#              forward kinematics, homogeneous transforms, closed-form and numeric inverse kinematics,
#              geometric Jacobian and joint velocities, closed-form planar 2R dynamics torques,
#              payload capacity and cubic/quintic trajectory polynomials.
"""Robotics calculators.

Conventions: joints are revolute about +z for the DH examples and the planar arms live in the x–y plane
with angles measured from +x (CCW positive) and gravity acting along −y. DH tables give angles in
degrees and lengths in metres inside the json rows. The standard DH transform is
T = Rot_z(θ)·Trans_z(d)·Trans_x(a)·Rot_x(α); the modified (Craig) convention is
T = Rot_x(α)·Trans_x(a)·Rot_z(θ)·Trans_z(d).

Relations are the standard ones in robotics texts (Craig, *Introduction to Robotics*; Spong,
*Robot Modeling and Control*; Siciliano & Khatib, *Springer Handbook of Robotics*). Cited by relation
name, recalled rather than re-read, so no page numbers are given.
"""
import math

import numpy as np

from engineering_sim.science.mech.core import Calc, MechError, V, section
from engineering_sim.science.mech import charts

SECTION = section('robotics', 'Robotics', order=400,
                  summary='Degrees of freedom, workspace, DH forward kinematics, inverse kinematics, '
                          'Jacobian, dynamics torques, payload capacity and trajectory polynomials.')

PI = math.pi


# ---- 4x4 helpers -----------------------------------------------------------------------------------
def _rot_z(th):
    c, s = math.cos(th), math.sin(th)
    return [[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _rot_x(al):
    c, s = math.cos(al), math.sin(al)
    return [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _trans_z(d):
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, d], [0.0, 0.0, 0.0, 1.0]]


def _trans_x(a):
    return [[1.0, 0.0, 0.0, a], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _mm(A, B):
    """4x4 matrix product."""
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _ident():
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


# ---- forward kinematics (planar n-link) ------------------------------------------------------------
def _fk_planar(links, angles):
    """Tip position [x, y] of a planar n-link arm (links and angles in SI/rad)."""
    x = y = 0.0
    acc = 0.0
    for L, th in zip(links, angles):
        acc += th
        x += L * math.cos(acc)
        y += L * math.sin(acc)
    return x, y


def _jacobian_planar(links, angles):
    """2xn geometric (basic) Jacobian of a planar n-link arm: J = [dx/dth; dy/dth]."""
    n = len(links)
    J = np.zeros((2, n))
    for i in range(n):
        acc = 0.0
        for k in range(n):
            acc += angles[k]
            if k >= i:
                J[0, i] -= links[k] * math.sin(acc)
                J[1, i] += links[k] * math.cos(acc)
    return J


# ---- polynomial trajectories -----------------------------------------------------------------------
def _cubic_coeffs(q0, v0, q1, v1, T):
    A = np.array([[T ** 2, T ** 3], [2.0 * T, 3.0 * T ** 2]], dtype=float)
    b = np.array([q1 - q0 - v0 * T, v1 - v0], dtype=float)
    a2, a3 = np.linalg.solve(A, b)
    return [q0, v0, float(a2), float(a3)]


def _quintic_coeffs(q0, v0, a0, q1, v1, a1, T):
    A = np.array([[T ** 3, T ** 4, T ** 5],
                  [3.0 * T ** 2, 4.0 * T ** 3, 5.0 * T ** 4],
                  [6.0 * T, 12.0 * T ** 2, 20.0 * T ** 3]], dtype=float)
    b = np.array([q1 - q0 - v0 * T - (a0 / 2.0) * T ** 2,
                  v1 - v0 - a0 * T,
                  a1 - a0], dtype=float)
    a3, a4, a5 = np.linalg.solve(A, b)
    return [q0, v0, a0 / 2.0, float(a3), float(a4), float(a5)]


def _poly(coeffs, t):
    """Evaluate a polynomial and its first two derivatives at t -> (q, qd, qdd)."""
    q = qd = qdd = 0.0
    for k, a in enumerate(coeffs):
        q += a * t ** k
        if k >= 1:
            qd += k * a * t ** (k - 1)
        if k >= 2:
            qdd += k * (k - 1) * a * t ** (k - 2)
    return q, qd, qdd


def _traj_chart(kind):
    def fn(values):
        T = values['T']
        if kind == 'cubic':
            c = _cubic_coeffs(values['q0'], values['v0'], values['q1'], values['v1'], T)
        else:
            c = _quintic_coeffs(values['q0'], values['v0'], values['a0'],
                                values['q1'], values['v1'], values['a1'], T)
        ts = [T * i / 200.0 for i in range(201)]
        q = [_poly(c, tt)[0] for tt in ts]
        qd = [_poly(c, tt)[1] for tt in ts]
        qdd = [_poly(c, tt)[2] for tt in ts]
        return charts.chart('Trajectory position, velocity and acceleration', 't [s]', 'q, qd, qdd',
                            [charts.series('q', ts, q), charts.series('qd', ts, qd),
                             charts.series('qdd', ts, qdd)])
    return fn


# ---- procedures ------------------------------------------------------------------------------------
def _serial_dof(v):
    """DoF of a serial open chain from its joint types: R/P=1, U/C=2, S=3 degrees of freedom each."""
    joints = v['joints']
    if not isinstance(joints, list) or not joints:
        raise MechError("joints: give a list of types, e.g. ['R', 'R', 'P']")
    freedoms = {'R': 1, 'P': 1, 'U': 2, 'C': 2, 'S': 3}
    dof = 0
    for j in joints:
        if j not in freedoms:
            raise MechError(f"joints: unknown type {j!r}; use one of {sorted(freedoms)}")
        dof += freedoms[j]
    return {'dof': dof, 'n_joints': len(joints)}


def _workspace_2r(v):
    L1, L2 = v['L1'], v['L2']
    if L1 <= 0 or L2 <= 0:
        raise MechError('link lengths must be positive')
    return {'R_max': L1 + L2, 'R_min': abs(L1 - L2), 'A': 4.0 * PI * L1 * L2}


def _dh_forward(v):
    """Denavit–Hartenberg forward kinematics. Rows [a (m), alpha (deg), d (m), theta (deg)]."""
    rows = v['dh']
    conv = v['convention']
    if not isinstance(rows, list) or not rows:
        raise MechError('dh: give a non-empty list of [a, alpha, d, theta] rows')
    T = _ident()
    for r in rows:
        if not isinstance(r, (list, tuple)) or len(r) != 4:
            raise MechError('dh: each row must be [a, alpha, d, theta]')
        a, al, d, th = (float(x) for x in r)
        al, th = math.radians(al), math.radians(th)
        if conv == 'standard':
            Ti = _mm(_rot_z(th), _mm(_trans_z(d), _mm(_trans_x(a), _rot_x(al))))
        elif conv == 'modified':
            Ti = _mm(_rot_x(al), _mm(_trans_x(a), _mm(_rot_z(th), _trans_z(d))))
        else:
            raise MechError("convention: must be 'standard' or 'modified'")
        T = _mm(T, Ti)
    return {'T': T, 'pos': [T[0][3], T[1][3], T[2][3]],
            'rot': [[T[i][j] for j in range(3)] for i in range(3)], 'dof': len(rows)}


def _homogeneous_transform(v):
    """Build a 4x4 homogeneous transform from a rotation matrix and translation, and its inverse."""
    p = v['translation']
    R = v['rotation']
    if not isinstance(p, (list, tuple)) or len(p) != 3:
        raise MechError('translation: give [x, y, z] in m')
    if not isinstance(R, (list, tuple)) or len(R) != 3 or any(len(row) != 3 for row in R):
        raise MechError('rotation: give a 3x3 rotation matrix')
    tx, ty, tz = (float(x) for x in p)
    R = [[float(x) for x in row] for row in R]
    Rn = np.array(R)
    if not np.allclose(Rn.T @ Rn, np.eye(3), atol=1e-6):
        raise MechError('rotation: R must be orthonormal (RᵀR = I); check that the columns are mutually '
                        'orthogonal unit vectors')
    if abs(float(np.linalg.det(Rn)) - 1.0) > 1e-6:
        raise MechError('rotation: det R must be +1 for a proper rotation (a reflection has det = −1)')
    T = [[R[0][0], R[0][1], R[0][2], tx],
         [R[1][0], R[1][1], R[1][2], ty],
         [R[2][0], R[2][1], R[2][2], tz],
         [0.0, 0.0, 0.0, 1.0]]
    RT = [[R[j][i] for j in range(3)] for i in range(3)]  # transpose
    q = [-sum(RT[i][j] * p[j] for j in range(3)) for i in range(3)]
    Ti = [[RT[0][0], RT[0][1], RT[0][2], q[0]],
          [RT[1][0], RT[1][1], RT[1][2], q[1]],
          [RT[2][0], RT[2][1], RT[2][2], q[2]],
          [0.0, 0.0, 0.0, 1.0]]
    return {'T': T, 'T_inv': Ti}


def _planar_2r_ik(v):
    """Closed-form inverse kinematics of a planar 2R arm: two elbow branches.

    θ2 is the joint-2 angle relative to link 1. The branch with θ2 = +acos(c2) is elbow-DOWN in the
    standard (Spong) sense (elbow below the base–tip line); θ2 = −acos(c2) is elbow-UP. `theta1`/
    `theta2` carry the elbow-down branch, `theta1_b`/`theta2_b` the elbow-up branch.
    """
    x, y, L1, L2 = v['x'], v['y'], v['L1'], v['L2']
    if L1 <= 0 or L2 <= 0:
        raise MechError('link lengths must be positive')
    c2 = (x * x + y * y - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    if c2 < -1.0 or c2 > 1.0:
        raise MechError(f'target ({x}, {y}) is outside the workspace '
                        f'|L1-L2| = {abs(L1-L2)} .. L1+L2 = {L1+L2}')

    def t1(t2):
        return math.atan2(y, x) - math.atan2(L2 * math.sin(t2), L1 + L2 * math.cos(t2))

    t2a = math.acos(c2)
    th1a, th2a = t1(t2a), t2a          # θ2 > 0: elbow down
    th1b, th2b = t1(-t2a), -t2a        # θ2 < 0: elbow up
    return {'theta1': th1a, 'theta2': th2a, 'theta1_b': th1b, 'theta2_b': th2b,
            'elbow_a': [L1 * math.cos(th1a), L1 * math.sin(th1a)],
            'elbow_b': [L1 * math.cos(th1b), L1 * math.sin(th1b)]}


def _planar_3r_ik(v):
    """Closed-form IK of a planar 3R arm for a given tip position AND orientation (phi = theta1+theta2+theta3)."""
    x, y, phi, L1, L2, L3 = (v[k] for k in ('x', 'y', 'phi', 'L1', 'L2', 'L3'))
    if L1 <= 0 or L2 <= 0 or L3 <= 0:
        raise MechError('link lengths must be positive')
    xw = x - L3 * math.cos(phi)
    yw = y - L3 * math.sin(phi)
    c2 = (xw * xw + yw * yw - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    if c2 < -1.0 or c2 > 1.0:
        raise MechError('wrist point (with the given orientation) is outside the 2-link reach')

    def t1(t2):
        return math.atan2(yw, xw) - math.atan2(L2 * math.sin(t2), L1 + L2 * math.cos(t2))

    t2a = math.acos(c2)
    th1a, th2a = t1(t2a), t2a
    th1b, th2b = t1(-t2a), -t2a
    return {'theta1': th1a, 'theta2': th2a, 'theta3': phi - th1a - th2a,
            'theta1_b': th1b, 'theta2_b': th2b, 'theta3_b': phi - th1b - th2b}


def _numeric_ik(v):
    """Numeric IK of a general planar n-link arm by damped least squares (deterministic)."""
    links = v['links']
    guess = v['guess']
    target = v['target']
    if not isinstance(links, list) or not links or any(L <= 0 for L in links):
        raise MechError('links: give positive link lengths [L1, ...] in m')
    if not isinstance(guess, list) or len(guess) != len(links):
        raise MechError('guess: give one initial joint angle (rad) per link')
    if not isinstance(target, list) or len(target) != 2:
        raise MechError('target: give [x, y] in m')
    theta = np.array([float(t) for t in guess])
    x_t, y_t = float(target[0]), float(target[1])
    # Reachability pre-check: a planar n-link arm reaches exactly the annulus [R_min, R_max].
    r_t = math.hypot(x_t, y_t)
    r_max = sum(links)
    r_min = max(0.0, 2.0 * max(links) - r_max)
    if r_t > r_max + 1e-9 or r_t < r_min - 1e-9:
        raise MechError(f'target ({x_t}, {y_t}) is outside the workspace: reachable radius is '
                        f'{r_min:g} .. {r_max:g} m (target is at r = {r_t:g} m)')
    lam = 1e-3
    err = 0.0
    for _ in range(400):
        x, y = _fk_planar(links, theta)
        e = np.array([x_t - x, y_t - y])
        err = float(np.hypot(*e))
        if err < 1e-9:
            break
        J = _jacobian_planar(links, theta)
        JTJ = J @ J.T
        step = J.T @ np.linalg.solve(JTJ + lam * lam * np.eye(2), e)
        theta = theta + step
    theta = np.arctan2(np.sin(theta), np.cos(theta))  # wrap each angle to (−π, π]
    x, y = _fk_planar(links, theta)
    err = float(math.hypot(x_t - x, y_t - y))
    if err > 1e-6:
        raise MechError(f'numeric IK did not converge to the target ({x_t}, {y_t}) '
                        f'(final error {err:.3g} m after 400 iterations); the damped least-squares solver '
                        f'fell into a local minimum — try a different initial guess')
    return {'angles': [float(t) for t in theta], 'position': [x, y], 'error': err}


def _planar_jacobian(v):
    """Geometric Jacobian of a planar n-link arm and joint velocities for a given tip velocity.

    joint_vel is the minimum-norm (pseudo-inverse) solution of J θ̇ = v; `residual` reports how much of
    the requested tip velocity is actually achieved (‖J θ̇ − v‖), which is non-zero only at a singularity.
    """
    links = v['links']
    angles = v['angles']
    vel = v['vel']
    if not isinstance(links, list) or not links or any(L <= 0 for L in links):
        raise MechError('links: give positive link lengths [L1, ...] in m')
    if not isinstance(angles, list) or len(angles) != len(links):
        raise MechError('angles: give one joint angle (rad) per link')
    if not isinstance(vel, list) or len(vel) != 2:
        raise MechError('vel: give [vx, vy] in m/s')
    angles = [float(a) for a in angles]
    x, y = _fk_planar(links, angles)
    J = _jacobian_planar(links, angles)
    vv = np.array([float(vel[0]), float(vel[1])])
    theta_dot = np.linalg.pinv(J) @ vv  # min-norm solution of J theta_dot = v
    residual = float(np.hypot(*(J @ theta_dot - vv)))
    det = float(np.linalg.det(J)) if J.shape == (2, 2) else float(np.linalg.det(J @ J.T))
    sv = np.linalg.svd(J, compute_uv=False)
    warnings = []
    if sv[-1] < 1e-6 * sv[0]:
        warnings.append(f'Jacobian is (near-)singular (smallest singular value {sv[-1]:.3g} vs largest '
                        f'{sv[0]:.3g}); the min-norm joint velocity may not reproduce the requested tip '
                        f'velocity — see the residual')
    return {'position': [x, y], 'jacobian': J.tolist(), 'joint_vel': theta_dot.tolist(), 'det': det,
            'residual': residual, '_warnings': warnings}


def _planar_2r_dynamics(v):
    """Closed-form Lagrange/Newton–Euler torques of a planar 2R arm (relative joint angles, gravity −y).

    θ1 is the joint-1 angle from +x; θ2 is the joint-2 angle RELATIVE to link 1 (not absolute), so the
    inertia-matrix coupling term carries cos θ2. Gravity acts along −y.
    """
    L1, m1, m2 = v['L1'], v['m1'], v['m2']
    lc1, lc2, I1, I2 = v['lc1'], v['lc2'], v['I1'], v['I2']
    th1, th2 = v['theta1'], v['theta2']
    qd1, qd2, qdd1, qdd2 = v['qd1'], v['qd2'], v['qdd1'], v['qdd2']
    g = v['g']
    if min(L1, m1, m2, lc1, lc2) <= 0 or I1 < 0 or I2 < 0:
        raise MechError('lengths, masses and CoM distances must be positive; inertias non-negative')
    c2, s2 = math.cos(th2), math.sin(th2)
    c1, c12 = math.cos(th1), math.cos(th1 + th2)
    M11 = m1 * lc1 ** 2 + I1 + m2 * (L1 ** 2 + lc2 ** 2 + 2.0 * L1 * lc2 * c2) + I2
    M12 = m2 * (lc2 ** 2 + L1 * lc2 * c2) + I2
    M22 = m2 * lc2 ** 2 + I2
    # Christoffel-symbol Coriolis/centrifugal terms (h = -m2 L1 lc2 sin th2).
    tau1 = M11 * qdd1 + M12 * qdd2 - m2 * L1 * lc2 * s2 * (2.0 * qd1 * qd2 + qd2 ** 2) \
        + (m1 * lc1 + m2 * L1) * g * c1 + m2 * lc2 * g * c12
    tau2 = M12 * qdd1 + M22 * qdd2 + m2 * L1 * lc2 * s2 * qd1 ** 2 + m2 * lc2 * g * c12
    return {'tau1': tau1, 'tau2': tau2}


def _payload_capacity(v):
    """Static payload capacity of a planar 2R arm (links as point masses at mid-length, gravity −y).

    The payload sits at the tip. Each joint limits the payload through |τ_link + m·g·arm| ≤ τ_max, solved exactly
    for either sign of the link torque and moment arm (feasible interval clipped at m ≥ 0; capacity = upper end). A joint whose payload arm is ≈ 0
    (payload directly above/below it) imposes no static limit and is skipped with a warning.
    """
    L1, L2, th1, th2 = v['L1'], v['L2'], v['theta1'], v['theta2']
    m1, m2, g = v['m1'], v['m2'], v['g']
    tau1_max, tau2_max = v['tau1_max'], v['tau2_max']
    if min(L1, L2, m1, m2) <= 0 or g <= 0:
        raise MechError('lengths, masses and gravity must be positive')
    if tau1_max < 0 or tau2_max < 0:
        raise MechError('torque limits must be non-negative')
    c1, c12 = math.cos(th1), math.cos(th1 + th2)
    arm1 = L1 * c1 + L2 * c12          # signed payload moment arm about joint 1
    arm2 = L2 * c12                    # signed payload moment arm about joint 2
    tau1_link = m1 * g * (L1 / 2.0) * c1 + m2 * g * (L1 * c1 + (L2 / 2.0) * c12)  # signed holding torque
    tau2_link = m2 * g * (L2 / 2.0) * c12
    eps = 1e-9 * (L1 + L2)
    warnings = []

    def joint_range(j, tau_max, tau_link, arm):
        """Feasible payload interval for one joint from |τ_link + m·g·arm| ≤ τ_max (exact for either sign).

        The two bounds (±τ_max − τ_link)/(g·arm) bracket the feasible m; clip below at 0. Returns the upper
        end (capacity) or None when the arm is ≈ 0.
        """
        if abs(arm) < eps:
            warnings.append(f'joint {j}: payload moment arm ≈ 0 — the payload is not statically limited by joint {j}')
            return None
        b = sorted([(tau_max - tau_link) / (g * arm), (-tau_max - tau_link) / (g * arm)])
        lo, hi = max(0.0, b[0]), b[1]
        if hi < lo:
            raise MechError(f'no payload satisfies the joint-{j} torque limit in this pose')
        if lo > 0.0:
            warnings.append(f'joint {j}: the links alone exceed the torque limit; a payload of at least '
                            f'{lo:.6g} kg is needed to balance them')
        return hi

    m_cap_1 = joint_range(1, tau1_max, tau1_link, arm1)
    m_cap_2 = joint_range(2, tau2_max, tau2_link, arm2)
    caps = [c for c in (m_cap_1, m_cap_2) if c is not None]
    if not caps:
        raise MechError('the payload sits directly above the base (both moment arms ≈ 0): no static torque '
                        'limits the payload here')
    return {'m_payload_max': min(caps), 'm_cap_1': m_cap_1, 'm_cap_2': m_cap_2,
            'tau1_link': tau1_link, 'tau2_link': tau2_link, '_warnings': warnings}


def _traj_cubic(v):
    T = v['T']
    if T <= 0:
        raise MechError('T: travel time must be positive')
    coeffs = _cubic_coeffs(v['q0'], v['v0'], v['q1'], v['v1'], T)
    q, qd, qdd = _poly(coeffs, v['t'])
    warnings = []
    if v['t'] > T:
        warnings.append(f't = {v["t"]:g} exceeds the travel time T = {T:g}; the polynomial is being '
                        f'evaluated outside its designed interval [0, T]')
    return {'coeffs': coeffs, 'q_t': q, 'qd_t': qd, 'qdd_t': qdd, '_warnings': warnings}


def _traj_quintic(v):
    T = v['T']
    if T <= 0:
        raise MechError('T: travel time must be positive')
    coeffs = _quintic_coeffs(v['q0'], v['v0'], v['a0'], v['q1'], v['v1'], v['a1'], T)
    q, qd, qdd = _poly(coeffs, v['t'])
    warnings = []
    if v['t'] > T:
        warnings.append(f't = {v["t"]:g} exceeds the travel time T = {T:g}; the polynomial is being '
                        f'evaluated outside its designed interval [0, T]')
    return {'coeffs': coeffs, 'q_t': q, 'qd_t': qd, 'qdd_t': qdd, '_warnings': warnings}


CALCS = [
    # ---- degrees of freedom and workspace ----------------------------------------------------------
    Calc('serial_dof', 'Degrees of freedom of a serial open chain',
         vars=[V('joints', '', "joint types in order, e.g. ['R','R','P']; R/P=1, U/C=2, S=3 freedoms",
                 kind='json'),
               V('dof', '', 'total degrees of freedom', integer=True),
               V('n_joints', '', 'number of joints', integer=True)],
         procedure=_serial_dof, outputs=['dof', 'n_joints'],
         example={'joints': ['R', 'R', 'P']},
         keywords=['mobility', 'joint space', 'manipulator'],
         assumptions=['Serial (open) chain: the DoF is the sum of the joint freedoms. Closed-loop mobility '
                      'is mechanisms.dof_planar / dof_spatial (Grübler/Kutzbach).'],
         refs=['Mobility of a serial manipulator = Σ joint freedoms']),

    Calc('planar_2r_workspace', 'Workspace of a planar 2-link arm',
         vars=[V('L1', 'm', 'link 1 length'), V('L2', 'm', 'link 2 length'),
               V('R_max', 'm', 'maximum reach L1 + L2'), V('R_min', 'm', 'inner radius |L1 - L2|'),
               V('A', 'm^2', 'reachable area (annulus)')],
         procedure=_workspace_2r, outputs=['R_max', 'R_min', 'A'],
         example={'L1': 0.5, 'L2': 0.3},
         keywords=['reach', 'annulus', 'dexterous workspace'],
         assumptions=['Revolute joints with no joint limits; the reachable set is the annulus '
                      'R_min ≤ r ≤ R_max. Area = π(R_max² − R_min²) = 4π L1 L2.'],
         refs=['Annular workspace of a planar 2R arm']),

    # ---- forward kinematics and homogeneous transforms ---------------------------------------------
    Calc('dh_forward', 'Denavit–Hartenberg forward kinematics',
         vars=[V('dh', '', 'joint rows [a (m), alpha (deg), d (m), theta (deg)]', kind='json'),
               V('convention', '', 'DH convention', kind='text', choices=['standard', 'modified']),
               V('T', '', '4x4 homogeneous transform of the tool frame', kind='json'),
               V('pos', '', 'tool position [x, y, z] in m', kind='json'),
               V('rot', '', '3x3 rotation matrix of the tool frame', kind='json'),
               V('dof', '', 'number of joints (rows)', integer=True)],
         procedure=_dh_forward, outputs=['T', 'pos', 'rot', 'dof'],
         example={'dh': [[0.5, 0, 0, 30], [0.3, 0, 0, 60]], 'convention': 'standard'},
         keywords=['dh parameters', 'homogeneous transform', 'forward kinematics', 'frame'],
         assumptions=['Standard DH: T = Rot_z(θ)Trans_z(d)Trans_x(a)Rot_x(α); modified (Craig): '
                      'T = Rot_x(α)Trans_x(a)Rot_z(θ)Trans_z(d). Angles are given in degrees in the table.',
                      'The joint rows are fixed parameters or joint variables: substitute joint values for '
                      'revolute θ or prismatic d.'],
         refs=['Denavit–Hartenberg convention; product of link transforms']),

    Calc('homogeneous_transform', 'Homogeneous transform from rotation and translation, with inverse',
         vars=[V('translation', '', 'translation [x, y, z] in m', kind='json'),
               V('rotation', '', '3x3 rotation matrix', kind='json'),
               V('T', '', '4x4 homogeneous transform', kind='json'),
               V('T_inv', '', 'inverse transform', kind='json')],
         procedure=_homogeneous_transform, outputs=['T', 'T_inv'],
         example={'translation': [1, 2, 3], 'rotation': [[0, -1, 0], [1, 0, 0], [0, 0, 1]]},
         keywords=['rigid body', 'pose', 'frame', 'transformation matrix'],
         assumptions=['R must be a proper rotation matrix; the inverse of T = [R p; 0 1] is '
                      '[Rᵀ −Rᵀp; 0 1].'],
         refs=['Homogeneous representation of a rigid-body transform']),

    # ---- inverse kinematics -------------------------------------------------------------------------
    Calc('planar_2r_ik', 'Inverse kinematics of a planar 2R arm (two elbow branches)',
         vars=[V('x', 'm', 'target x'), V('y', 'm', 'target y'), V('L1', 'm', 'link 1 length'),
               V('L2', 'm', 'link 2 length'),
               V('theta1', 'rad', 'θ1, elbow-down branch (θ2 = +acos)', signed=True, label='θ1'),
               V('theta2', 'rad', 'θ2, elbow-down branch (positive)', signed=True, label='θ2'),
               V('theta1_b', 'rad', 'θ1, elbow-up branch (θ2 = −acos)', signed=True),
               V('theta2_b', 'rad', 'θ2, elbow-up branch (negative)', signed=True),
               V('elbow_a', '', '[x, y] of the elbow, elbow-down branch', kind='json'),
               V('elbow_b', '', '[x, y] of the elbow, elbow-up branch', kind='json')],
         procedure=_planar_2r_ik, outputs=['theta1', 'theta2', 'theta1_b', 'theta2_b', 'elbow_a', 'elbow_b'],
         example={'x': 0.6, 'y': 0.3, 'L1': 0.5, 'L2': 0.4},
         keywords=['closed form', 'two link', 'atan2'],
         assumptions=['Angles measured from +x, with θ2 relative to link 1. In the standard (Spong) sense the '
                      'branch with θ2 = +acos(c2) places the elbow below the base–tip line (elbow down) and '
                      'θ2 = −acos(c2) places it above (elbow up). Two solutions exist for every reachable '
                      'point; at the workspace boundary they coincide. Unreachable targets raise an error.'],
         refs=['Law of cosines and atan2 for the planar 2R inverse kinematics']),

    Calc('planar_3r_ik', 'Inverse kinematics of a planar 3R arm (position and orientation)',
         vars=[V('x', 'm', 'target x'), V('y', 'm', 'target y'),
               V('phi', 'rad', 'end-effector orientation (θ1+θ2+θ3)', signed=True, label='φ'),
               V('L1', 'm', 'link 1 length'), V('L2', 'm', 'link 2 length'), V('L3', 'm', 'link 3 length'),
               V('theta1', 'rad', 'θ1, branch a', signed=True), V('theta2', 'rad', 'θ2, branch a', signed=True),
               V('theta3', 'rad', 'θ3, branch a', signed=True),
               V('theta1_b', 'rad', 'θ1, branch b', signed=True), V('theta2_b', 'rad', 'θ2, branch b', signed=True),
               V('theta3_b', 'rad', 'θ3, branch b', signed=True)],
         procedure=_planar_3r_ik, outputs=['theta1', 'theta2', 'theta3', 'theta1_b', 'theta2_b', 'theta3_b'],
         example={'x': 0.7, 'y': 0.5, 'phi': math.radians(30), 'L1': 0.5, 'L2': 0.4, 'L3': 0.3},
         keywords=['redundant', 'orientation', 'closed form'],
         assumptions=['With a specified tip orientation φ the redundancy is removed: the wrist point is '
                      'solved with the 2R formulas and θ3 = φ − θ1 − θ2.'],
         refs=['Wrist-centring reduction of the planar 3R IK to a 2R problem']),

    Calc('numeric_ik', 'Numeric inverse kinematics of a planar n-link arm (damped least squares)',
         vars=[V('links', '', 'link lengths [L1, ...] in m', kind='json'),
               V('guess', '', 'initial joint angles [θ1, ...] in rad', kind='json'),
               V('target', '', 'target tip position [x, y] in m', kind='json'),
               V('angles', '', 'solution joint angles [θ1, ...] in rad', kind='json'),
               V('position', '', 'reached tip position [x, y] in m', kind='json'),
               V('error', 'm', 'final tip-to-target distance')],
         procedure=_numeric_ik, outputs=['angles', 'position', 'error'],
         example={'links': [0.5, 0.4], 'guess': [0.3, 0.3], 'target': [0.5, 0.4]},
         keywords=['iterative', 'jacobian transpose', 'least squares'],
         assumptions=['Damped least squares (Levenberg) with 400 fixed iterations; converges to the local '
                      'solution nearest the guess. Redundant arms yield the minimum-norm step direction. '
                      'Deterministic (no random restarts).'],
         refs=['Jacobian pseudo-inverse IK (damped least squares)']),

    # ---- Jacobian and joint velocities --------------------------------------------------------------
    Calc('planar_jacobian', 'Geometric Jacobian and joint velocities of a planar n-link arm',
         vars=[V('links', '', 'link lengths [L1, ...] in m', kind='json'),
               V('angles', '', 'joint angles [θ1, ...] in rad', kind='json'),
               V('vel', '', 'tip velocity [vx, vy] in m/s', kind='json'),
               V('position', '', 'tip position [x, y] in m', kind='json'),
               V('jacobian', '', '2 x n geometric Jacobian', kind='json'),
               V('joint_vel', '', 'joint velocities [θ̇1, ...] in rad/s (min-norm)', kind='json'),
               V('det', '', 'det J (2R) or det(J·Jᵀ) for n>2', signed=True),
               V('residual', 'm/s', '‖J θ̇ − v‖: unachieved part of the requested tip velocity')],
         procedure=_planar_jacobian, outputs=['position', 'jacobian', 'joint_vel', 'det', 'residual'],
         example={'links': [0.5, 0.4], 'angles': [math.radians(30), math.radians(60)], 'vel': [0.0, 1.0]},
         keywords=['singularity', 'velocity kinematics', 'pseudo-inverse'],
         assumptions=['Geometric (basic) Jacobian J = [∂x/∂θ; ∂y/∂θ]. joint_vel is the minimum-norm '
                      'least-squares solution of J θ̇ = v. At a singularity the pseudo-inverse still returns a '
                      'finite (min-norm) answer, but it cannot reproduce an arbitrary tip velocity — the '
                      'residual ‖J θ̇ − v‖ reports the shortfall and a warning is raised.'],
         refs=['Geometric Jacobian of a planar arm; J θ̇ = v']),

    # ---- dynamics -----------------------------------------------------------------------------------
    Calc('planar_2r_dynamics', 'Planar 2R arm joint torques (closed-form Lagrange dynamics)',
         vars=[V('L1', 'm', 'link 1 length'),
               V('m1', 'kg', 'link 1 mass'), V('m2', 'kg', 'link 2 mass'),
               V('lc1', 'm', 'distance from joint 1 to link-1 CoM'), V('lc2', 'm', 'distance from joint 2 to link-2 CoM'),
               V('I1', 'kg*m^2', 'link 1 inertia about its CoM'), V('I2', 'kg*m^2', 'link 2 inertia about its CoM'),
               V('theta1', 'rad', 'joint 1 angle (from +x)', signed=True),
               V('theta2', 'rad', 'joint 2 angle relative to link 1', signed=True),
               V('qd1', 'rad/s', 'joint 1 velocity', signed=True), V('qd2', 'rad/s', 'joint 2 velocity', signed=True),
               V('qdd1', 'rad/s^2', 'joint 1 acceleration', signed=True),
               V('qdd2', 'rad/s^2', 'joint 2 acceleration', signed=True),
               V('g', 'm/s^2', 'gravitational acceleration'),
               V('tau1', 'N*m', 'joint 1 torque', signed=True), V('tau2', 'N*m', 'joint 2 torque', signed=True)],
         procedure=_planar_2r_dynamics, outputs=['tau1', 'tau2'],
         example={'L1': 0.5, 'm1': 2.0, 'm2': 1.5, 'lc1': 0.25, 'lc2': 0.2,
                  'I1': 0.02, 'I2': 0.01, 'theta1': math.radians(30), 'theta2': math.radians(60),
                  'qd1': 1.0, 'qd2': 0.5, 'qdd1': 0.2, 'qdd2': 0.1, 'g': 9.80665},
         keywords=['lagrange', 'newton euler', 'inertia matrix', 'coriolis', 'gravity torque'],
         assumptions=['Rigid links, revolute joints, gravity along −y. θ1 is measured from +x and θ2 is the '
                      'joint-2 angle RELATIVE to link 1 (the relative-angle Lagrangian derivation), so the '
                      'inertia coupling carries cos θ2; the link-2 CoM enters only through lc2 (no tip mass). '
                      'Inertia matrix M(q), Coriolis/centrifugal and gravity terms from the standard '
                      'Lagrangian derivation.'],
         refs=['Lagrangian dynamics of a planar 2R manipulator (inertia, Coriolis and gravity terms)']),

    # ---- payload and trajectories -------------------------------------------------------------------
    Calc('payload_capacity', 'Static payload capacity of a planar 2R arm',
         vars=[V('L1', 'm', 'link 1 length'), V('L2', 'm', 'link 2 length'),
               V('theta1', 'rad', 'joint 1 angle', signed=True), V('theta2', 'rad', 'joint 2 angle', signed=True),
               V('m1', 'kg', 'link 1 mass'), V('m2', 'kg', 'link 2 mass'),
               V('tau1_max', 'N*m', 'joint 1 torque limit'), V('tau2_max', 'N*m', 'joint 2 torque limit'),
               V('g', 'm/s^2', 'gravitational acceleration'),
               V('m_payload_max', 'kg', 'maximum payload mass at the tip'),
               V('m_cap_1', 'kg', 'payload limited by joint 1'), V('m_cap_2', 'kg', 'payload limited by joint 2'),
               V('tau1_link', 'N*m', 'gravity torque of the links alone about joint 1', signed=True),
               V('tau2_link', 'N*m', 'gravity torque of the links alone about joint 2', signed=True)],
         procedure=_payload_capacity, outputs=['m_payload_max', 'm_cap_1', 'm_cap_2', 'tau1_link', 'tau2_link'],
         example={'L1': 1.0, 'L2': 1.0, 'theta1': 0.0, 'theta2': 0.0, 'm1': 1.0, 'm2': 1.0,
                  'tau1_max': 50.0, 'tau2_max': 12.0, 'g': 9.80665},
         no_invert={'tau1_max': 'in the example the elbow (joint 2) limit binds, so m_payload_max does not '
                                'depend on τ1_max; recover τ1_max through m_cap_1 instead'},
         keywords=['gripper', 'end effector', 'static load', 'torque limit'],
         assumptions=['Static (no acceleration); links modelled as point masses at mid-length; gravity −y. '
                      'Each joint limits the payload through |τ_link + m·g·arm| ≤ τ_max, using magnitudes so '
                      'poses reaching either side of a joint are handled. A joint with payload moment arm ≈ 0 '
                      'does not limit the payload. Payload sits at the tip.'],
         refs=['Static joint-torque limit on the payload (min over joints)']),

    Calc('trajectory_cubic', 'Cubic (rest-to-rest or via) joint trajectory polynomial',
         vars=[V('q0', '', 'start position (any coordinate unit)', signed=True),
               V('v0', '1/s', 'start velocity (coordinate per second)', signed=True),
               V('q1', '', 'end position (any coordinate unit)', signed=True),
               V('v1', '1/s', 'end velocity (coordinate per second)', signed=True),
               V('T', 's', 'travel time'), V('t', 's', 'evaluation time'),
               V('coeffs', '', 'polynomial coefficients [a0..a3]', kind='json'),
               V('q_t', '', 'position at t', signed=True),
               V('qd_t', '1/s', 'velocity at t', signed=True),
               V('qdd_t', '1/s^2', 'acceleration at t', signed=True)],
         procedure=_traj_cubic, outputs=['coeffs', 'q_t', 'qd_t', 'qdd_t'],
         example={'q0': 0.0, 'v0': 0.0, 'q1': 1.0, 'v1': 0.0, 'T': 1.0, 't': 0.5},
         plots=[{'id': 'profile', 'title': 'Cubic trajectory q, qd, qdd', 'fn': _traj_chart('cubic')}],
         keywords=['path planning', 'rest to rest', 'smooth'],
         assumptions=['Four boundary conditions (position and velocity at t = 0 and t = T). q is a generic '
                      'coordinate (joint angle, displacement …), so position is unit-agnostic and velocity/'
                      'acceleration are per-second. Acceleration is piecewise linear, so the rest-to-rest '
                      'profile peaks at the ends. T must be positive; evaluating at t > T warns.'],
         refs=['Cubic polynomial joint-space interpolation']),

    Calc('trajectory_quintic', 'Quintic joint trajectory polynomial (position, velocity and acceleration)',
         vars=[V('q0', '', 'start position (any coordinate unit)', signed=True),
               V('v0', '1/s', 'start velocity (coordinate per second)', signed=True),
               V('a0', '1/s^2', 'start acceleration (coordinate per second squared)', signed=True),
               V('q1', '', 'end position (any coordinate unit)', signed=True),
               V('v1', '1/s', 'end velocity (coordinate per second)', signed=True),
               V('a1', '1/s^2', 'end acceleration (coordinate per second squared)', signed=True),
               V('T', 's', 'travel time'), V('t', 's', 'evaluation time'),
               V('coeffs', '', 'polynomial coefficients [a0..a5]', kind='json'),
               V('q_t', '', 'position at t', signed=True),
               V('qd_t', '1/s', 'velocity at t', signed=True),
               V('qdd_t', '1/s^2', 'acceleration at t', signed=True)],
         procedure=_traj_quintic, outputs=['coeffs', 'q_t', 'qd_t', 'qdd_t'],
         example={'q0': 0.0, 'v0': 0.0, 'a0': 0.0, 'q1': 1.0, 'v1': 0.0, 'a1': 0.0, 'T': 1.0, 't': 0.5},
         plots=[{'id': 'profile', 'title': 'Quintic trajectory q, qd, qdd', 'fn': _traj_chart('quintic')}],
         keywords=['path planning', 'jerk', 'smooth'],
         assumptions=['Six boundary conditions (position, velocity and acceleration at both ends), so '
                      'acceleration is continuous — no jerk spike at the endpoints. q is a generic coordinate, '
                      'so position is unit-agnostic and velocity/acceleration are per-second. T must be '
                      'positive; evaluating at t > T warns.'],
         refs=['Quintic polynomial joint-space interpolation']),
]

"""Linear flight dynamics from caller-supplied derivatives, mass and conditions; SI units."""
from __future__ import annotations

import json

import numpy as np

# ---------------------------------------------------------------------------
# Declared handling-quality screen limits (MIL-F-8785C Class I, Level 1, cited)
# ---------------------------------------------------------------------------
HQ_LIMITS = {
    # Short period damping ratio band (linear theory; MIL-F-8785C Class I Cat B/C range)
    "short_period_zeta_min": 0.3,
    "short_period_zeta_max": 2.0,
    # Phugoid damping ratio floor
    "phugoid_zeta_min": 0.04,
    # Dutch roll: damping ratio and damped natural frequency product
    "dutch_roll_zeta_min": 0.08,
    "dutch_roll_zeta_omega_min": 0.15,
    # Roll mode time constant ceiling (s)
    "roll_tau_max_s": 1.4,
    # Spiral minimum time to double (s) if unstable; stable spiral passes
    "spiral_t2_min_s": 12.0,
}
HQ_LIMITS_BASIS = (
    "Declared handling-quality screen limits adapted from MIL-F-8785C Class I "
    "Category B/C Level 1 ranges (short period, phugoid, Dutch roll, roll mode, "
    "spiral) as quoted in Nelson, 'Flight Stability and Automatic Control', 2nd ed. "
    "These are linear-theory screens for a conventionally controlled airplane; they "
    "are not a flight-test compliance statement."
)

# Canonical derivative keys (per rad / per dimensionless rate). Optional keys default 0.
_DERIV_KEYS = {
    "CL_a": 0.0, "CD0": 0.0, "CL": None, "CD_a": 0.0, "Cm_a": 0.0,
    "Cm_q": 0.0, "CL_q": 0.0, "Cm_a_dot": 0.0, "CL_a_dot": 0.0,
    "CY_b": 0.0, "Cl_b": 0.0, "Cn_b": 0.0, "Cl_p": 0.0, "Cn_p": 0.0,
    "Cl_r": 0.0, "Cn_r": 0.0, "CY_p": 0.0, "CY_r": 0.0,
    "CL_de": 0.0, "Cm_de": 0.0, "Cl_da": 0.0, "Cn_da": 0.0,
}

# Greek-subscript aliases -> canonical ASCII keys.
_ALIASES = {
    "CL_alpha": "CL_a", "Cm_alpha": "Cm_a", "Cm_q": "Cm_q", "CL_q": "CL_q",
    "Cm_alpha_dot": "Cm_a_dot", "CL_alpha_dot": "CL_a_dot",
    "CY_beta": "CY_b", "Cl_beta": "Cl_b", "Cn_beta": "Cn_b",
    "Cl_p": "Cl_p", "Cn_p": "Cn_p", "Cl_r": "Cl_r", "Cn_r": "Cn_r",
    "CY_p": "CY_p", "CY_r": "CY_r",
    "CL_de": "CL_de", "Cm_de": "Cm_de", "Cl_da": "Cl_da", "Cn_da": "Cn_da",
    "Cl_delta_a": "Cl_da", "Cn_delta_a": "Cn_da",
    "Cl_b": "Cl_b", "Cn_b": "Cn_b", "CL_a": "CL_a", "Cm_a": "Cm_a",
}


from engineering_sim.execution import operation as _simulation_operation

def normalise_derivatives(derivs):
    """Return ``derivs`` with alias keys mapped to canonical ASCII keys (copy).

    Also applies symbol aliases (CL_alpha -> CL_a, ...). Unknown keys are
    passed through unchanged so extra optional fields do not raise.
    """
    out = {}
    for key, value in (derivs or {}).items():
        out[_ALIASES.get(key, key)] = value
    for key, default in _DERIV_KEYS.items():
        if key not in out:
            out[key] = default
    return out


def _as_json(value):
    """Recursively convert numpy scalars/arrays to plain Python for JSON output."""
    if isinstance(value, np.ndarray):
        return [_as_json(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _as_json(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"re": float(value.real), "im": float(value.imag)}
    return value


# ---------------------------------------------------------------------------
# Dimensional derivative conversion (Nelson Ch. 4, SI)
# ---------------------------------------------------------------------------
def _qbar(flight):
    return 0.5 * flight["rho"] * flight["V_ms"] ** 2


def dimensional_longitudinal(derivs, mass, flight):
    """Convert non-dimensional longitudinal derivatives to dimensional form (SI).

    With qbar = 1/2 rho V^2 and U0 = V (level flight), and the incompressible Mach
    terms C_Du = C_Lu = C_mu = 0:

        X_u = -(qbar S/(m V)) (C_Du + 2 CD0)        [1/s]
        X_w =  (qbar S/(m V)) (C_L   - C_Dalpha)    [1/s]
        Z_u = -(qbar S/(m V)) (C_Lu  + 2 C_L)       [1/s]
        Z_w = -(qbar S/(m V)) (C_Lalpha + CD0)      [1/s]
        Z_wdot = (qbar S c/(2 m V^2)) C_Lalphadot   [m/m, dimensionless]
        Z_q = -(qbar S c/(2 m V)) C_Lq              [m/s]
        M_w =  (qbar S c/(Iyy V)) C_malpha          [1/(m s)]
        M_wdot = (qbar S c^2/(2 Iyy V^2)) C_malphadot  [1/m]
        M_q =  (qbar S c^2/(2 Iyy V)) C_mq          [1/s]

    Returns a dict of dimensional derivatives plus the dimensionless pitch metric.
    The derivatives are already normalised by mass/inertia (Nelson's convention),
    i.e. Z_w = dw/dt per w, M_w = dq/dt per w, so no further division is applied
    when building the state matrix.
    """
    d = normalise_derivatives(derivs)
    V = flight["V_ms"]; rho = flight["rho"]; S = flight["S_m2"]; c = flight["c_m"]
    m = mass["mass_kg"]; Iyy = mass["Iyy"]
    qbar = _qbar(flight)
    CL = d["CL"] if d["CL"] is not None else (2 * mass["mass_kg"] * 9.80665 / (rho * S * V ** 2))
    Xu = -(qbar * S / (m * V)) * (2 * d["CD0"])
    Xw = (qbar * S / (m * V)) * (CL - d["CD_a"])
    Zu = -(qbar * S / (m * V)) * (2 * CL)
    Zw = -(qbar * S / (m * V)) * (d["CL_a"] + d["CD0"])
    Zwdot = (qbar * S * c / (2 * m * V ** 2)) * d["CL_a_dot"]
    Zq = -(qbar * S * c / (2 * m * V)) * d["CL_q"]
    Mw = (qbar * S * c / (Iyy * V)) * d["Cm_a"]
    Mwdot = (qbar * S * c ** 2 / (2 * Iyy * V ** 2)) * d["Cm_a_dot"]
    Mq = (qbar * S * c ** 2 / (2 * Iyy * V)) * d["Cm_q"]
    return {"Xu": Xu, "Xw": Xw, "Zu": Zu, "Zw": Zw, "Zwdot": Zwdot, "Zq": Zq,
            "Mw": Mw, "Mwdot": Mwdot, "Mq": Mq, "Mu": 0.0, "CL": CL}


def longitudinal_matrix(derivs, mass, flight):
    """Build the longitudinal state matrix A (4x4) for x = [u, w, q, theta] (SI).

    Level flight (theta0 = 0), stability axes, small perturbations:

        u_dot = X_u u + X_w w + 0 q         - g theta
        w_dot = Z_u u + Z_w w + (Z_q + U0) q
        q_dot = (M_u + M_wdot Z_u) u + (M_w + M_wdot Z_w) w
                + (M_q + M_wdot (U0 + Z_q)) q
        theta_dot = q

    The q_dot row folds in the downwash-lag (C_malphadot -> M_wdot) correction by
    substituting w_dot into the pitching-moment equation. With M_wdot = 0 this
    reduces to the plain [M_u, M_w, M_q] row.
    """
    dd = dimensional_longitudinal(derivs, mass, flight)
    g = 9.80665
    U0 = flight["V_ms"]
    A = np.array([
        [dd["Xu"], dd["Xw"], 0.0, -g],
        [dd["Zu"], dd["Zw"], U0 + dd["Zq"], 0.0],
        [dd["Mu"] + dd["Mwdot"] * dd["Zu"],
         dd["Mw"] + dd["Mwdot"] * dd["Zw"],
         dd["Mq"] + dd["Mwdot"] * (U0 + dd["Zq"]), 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])
    return A, dd


def dimensional_lateral(derivs, mass, flight):
    """Convert non-dimensional lateral-directional derivatives to dimensional (SI).

    With reference b (span), V = U0:

        Y_b = (qbar S/m) C_Ybeta                 [m/s^2]
        L_b = (qbar S b/Ixx) C_lbeta             [1/s^2]
        N_b = (qbar S b/Izz) C_nbeta             [1/s^2]
        Y_p = (qbar S b/(2 m V)) C_Yp            [m/s]
        Y_r = (qbar S b/(2 m V)) C_Yr            [m/s]
        L_p = (qbar S b^2/(2 V Ixx)) C_lp        [1/s]
        L_r = (qbar S b^2/(2 V Ixx)) C_lr        [1/s]
        N_p = (qbar S b^2/(2 V Izz)) C_np        [1/s]
        N_r = (qbar S b^2/(2 V Izz)) C_nr        [1/s]

    The roll/yaw derivatives are normalised by inertia (Nelson's convention):
    L_b = dp/dt per beta, N_b = dr/dt per beta, etc.
    """
    d = normalise_derivatives(derivs)
    V = flight["V_ms"]; rho = flight["rho"]; S = flight["S_m2"]; b = flight["b_m"]
    m = mass["mass_kg"]; Ix = mass["Ixx"]; Iz = mass["Izz"]
    qbar = _qbar(flight)
    Yb = (qbar * S / m) * d["CY_b"]
    Lb = (qbar * S * b / Ix) * d["Cl_b"]
    Nb = (qbar * S * b / Iz) * d["Cn_b"]
    Yp = (qbar * S * b / (2 * m * V)) * d["CY_p"]
    Yr = (qbar * S * b / (2 * m * V)) * d["CY_r"]
    Lp = (qbar * S * b ** 2 / (2 * V * Ix)) * d["Cl_p"]
    Lr = (qbar * S * b ** 2 / (2 * V * Ix)) * d["Cl_r"]
    Np = (qbar * S * b ** 2 / (2 * V * Iz)) * d["Cn_p"]
    Nr = (qbar * S * b ** 2 / (2 * V * Iz)) * d["Cn_r"]
    return {"Yb": Yb, "Lb": Lb, "Nb": Nb, "Yp": Yp, "Yr": Yr,
            "Lp": Lp, "Lr": Lr, "Np": Np, "Nr": Nr}


def lateral_matrix(derivs, mass, flight):
    """Build the lateral state matrix A (4x4) for x = [v, p, r, phi] (SI).

    Level flight (theta0 = 0), stability axes, small perturbations, and using
    beta = v/U0 so that a per-beta derivative acts on v with the factor 1/U0:

        v_dot = (Y_b/U0) v + Y_p p + (Y_r - U0) r + g phi
        p_dot = (L_b/U0) v + L_p p + L_r r
        r_dot = (N_b/U0) v + N_p p + N_r r
        phi_dot = p

    Product-of-inertia (Ixz) coupling is applied when mass['Ixz'] is non-zero,
    using the standard roll/yaw coupling form (Nelson Eq. 5.62): with
    Gamma = 1 - Ixz^2/(Ixx Izz), the coupled inertia rolls/yaw accelerations are
        p_dot = (L + (Ixz/Ixx) N)/Gamma
        r_dot = (N + (Ixz/Izz) L)/Gamma
    where L, N are the dimensional rolling/yawing moment accelerations.
    """
    dd = dimensional_lateral(derivs, mass, flight)
    g = 9.80665
    U0 = flight["V_ms"]
    Ix = mass["Ixx"]; Iz = mass["Izz"]; Ixz = mass.get("Ixz", 0.0)
    Gamma = 1.0 - Ixz ** 2 / (Ix * Iz) if (Ix and Iz) else 1.0
    # Aero moment accelerations (per v / p / r) before inertia coupling.
    Lv = dd["Lb"] / U0; Lp = dd["Lp"]; Lr = dd["Lr"]
    Nv = dd["Nb"] / U0; Np = dd["Np"]; Nr = dd["Nr"]
    kx = Ixz / Ix if Ix else 0.0
    kz = Ixz / Iz if Iz else 0.0
    A = np.array([
        [dd["Yb"] / U0, dd["Yp"], dd["Yr"] - U0, g],
        [(Lv + kx * Nv) / Gamma, (Lp + kx * Np) / Gamma, (Lr + kx * Nr) / Gamma, 0.0],
        [(Nv + kz * Lv) / Gamma, (Np + kz * Lp) / Gamma, (Nr + kz * Lr) / Gamma, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ])
    return A, dd


# ---------------------------------------------------------------------------
# Eigenmode metrics
# ---------------------------------------------------------------------------
def eigenmode_metrics(eigvals):
    """Return mode metrics for a list of eigenvalues (SI, 1/s).

    Complex conjugate pair (assume im > 0 branch present): omega_n = |lambda|,
    zeta = -Re/|lambda|. Real stable root lambda<0: time constant tau = -1/lambda.
    Real unstable root lambda>0: time to double T2 = ln 2 / lambda.
    """
    out = []
    for lam in eigvals:
        if abs(lam.imag) > 1e-9:
            wn = abs(lam)
            out.append({"lambda": complex(lam.real, lam.imag), "omega_n": wn,
                        "zeta": -lam.real / wn, "type": "oscillatory"})
        else:
            r = lam.real
            if r < 0:
                out.append({"lambda": float(r), "omega_n": -r, "zeta": 1.0,
                            "tau_s": -1.0 / r, "type": "stable_real"})
            else:
                out.append({"lambda": float(r), "omega_n": r, "zeta": -1.0,
                            "t2_s": np.log(2.0) / r, "type": "unstable_real"})
    return out


def _content(eigvec):
    """Return the normalised magnitude of each component of an eigenvector."""
    mags = np.abs(eigvec)
    total = mags.sum()
    return (mags / total) if total > 0 else mags


# ---------------------------------------------------------------------------
# Mode identification by eigenvector content (not ordering)
# ---------------------------------------------------------------------------
def identify_longitudinal_modes(A, evals=None, evecs=None):
    """Identify the short-period and phugoid modes of a 4x4 longitudinal A matrix.

    Identification uses eigenvector CONTENT, not eigenvalue ordering: the phugoid is
    dominated by u and theta (speed/height exchange), the short period by w (=alpha)
    and q (pitch). A phugoid score (|u|+|theta|)/(|w|+|q|) is computed for each
    oscillatory pair; the higher score is the phugoid. If the short period has
    collapsed to real roots, it is reported as such. Returns {mode: {metrics, content}}.

    ``evals``/``evecs`` may be passed pre-computed (columns of ``evecs`` match
    ``evals``) so callers can test order-independence directly.
    """
    if evals is None or evecs is None:
        evals, evecs = np.linalg.eig(A)
    evals = np.asarray(evals); evecs = np.asarray(evecs)
    n = len(evals)
    used = [False] * n
    result = {"short_period": None, "phugoid": None, "real_roots": []}

    def score(vec):
        # Phugoid: u/theta dominant. Short period: w(=alpha)/q dominant.
        c = _content(vec)
        u, w, q, th = c[0], c[1], c[2], c[3]
        return (u + th) / (w + q + 1e-12)

    pairs = []
    for i in range(n):
        if used[i]:
            continue
        if abs(evals[i].imag) > 1e-9:
            # pair this root with its conjugate
            for j in range(i + 1, n):
                if abs(evals[i] - np.conj(evals[j])) < 1e-6:
                    used[i] = used[j] = True
                    lam = evals[i]
                    pairs.append({
                        "lambda": complex(lam.real, lam.imag),
                        "omega_n": abs(lam), "zeta": -lam.real / abs(lam),
                        "content_u_w_q_theta": _content(evecs[:, i]).tolist(),
                        "_score": score(evecs[:, i]),
                    })
                    break

    # Highest phugoid score -> phugoid; the other(s) -> short period.
    ordered = sorted(pairs, key=lambda p: -p["_score"])
    if len(ordered) >= 1:
        result["phugoid"] = ordered[0]
    if len(ordered) >= 2:
        result["short_period"] = ordered[1]

    for i in range(n):
        if not used[i]:
            result["real_roots"].append({
                "lambda": float(evals[i].real),
                "content_u_w_q_theta": _content(evecs[:, i]).tolist(),
            })
    for key in ("short_period", "phugoid"):
        if result[key]:
            result[key].pop("_score", None)
    return result


def identify_lateral_modes(A, evals=None, evecs=None):
    """Identify the Dutch-roll, roll and spiral modes of a 4x4 lateral A matrix.

    The Dutch roll is the oscillatory pair (v/beta and r content). The two real roots
    are separated by eigenvector CONTENT: the roll mode is p-dominant, the spiral mode
    is phi-dominant; ties fall back to |lambda| magnitude (roll is the faster root).

    ``evals``/``evecs`` may be passed pre-computed (columns match) for order testing.
    """
    if evals is None or evecs is None:
        evals, evecs = np.linalg.eig(A)
    evals = np.asarray(evals); evecs = np.asarray(evecs)
    n = len(evals)
    used = [False] * n
    result = {"dutch_roll": None, "roll": None, "spiral": None, "real_roots": []}

    for i in range(n):
        if used[i]:
            continue
        if abs(evals[i].imag) > 1e-9:
            for j in range(i + 1, n):
                if abs(evals[i] - np.conj(evals[j])) < 1e-6:
                    used[i] = used[j] = True
                    lam = evals[i]
                    result["dutch_roll"] = {
                        "lambda": complex(lam.real, lam.imag),
                        "omega_n": abs(lam), "zeta": -lam.real / abs(lam),
                        "content_v_p_r_phi": _content(evecs[:, i]).tolist(),
                    }
                    break

    real = []
    for i in range(n):
        if not used[i]:
            c = _content(evecs[:, i])
            real.append({"lambda": float(evals[i].real),
                         "content_v_p_r_phi": c.tolist()})
            result["real_roots"].append({"lambda": float(evals[i].real),
                                         "content_v_p_r_phi": c.tolist()})
    if len(real) == 2:
        # Roll: p-dominant and faster. Spiral: phi-dominant and slower.
        # Sort most-negative (fastest) first: ascending by the real part.
        ordered = sorted(real, key=lambda r: r["lambda"].real)
        r1, r2 = ordered
        # content check: p-dominant -> roll
        if r1["content_v_p_r_phi"][1] >= r1["content_v_p_r_phi"][3] and \
           r2["content_v_p_r_phi"][3] >= r2["content_v_p_r_phi"][1]:
            result["roll"], result["spiral"] = r1, r2
        else:
            # fall back to speed ordering if content is ambiguous
            result["roll"], result["spiral"] = ordered
    elif len(real) == 3:
        # no oscillatory Dutch roll: keep all real roots, tag fastest as roll
        ordered = sorted(real, key=lambda r: r["lambda"].real)
        result["roll"] = ordered[0]
    elif len(real) >= 4:
        # No oscillatory pair and >=4 real roots (e.g. statically unstable yaw with a
        # divergent real mode): label the fastest (most negative) root as roll and the
        # slowest (smallest |lambda|, closest to zero) root as spiral; remaining roots
        # stay in real_roots.
        result["roll"] = min(real, key=lambda r: r["lambda"].real)
        result["spiral"] = min(real, key=lambda r: abs(r["lambda"].real))
    return result


# ---------------------------------------------------------------------------
# Screening evaluations
# ---------------------------------------------------------------------------
@_simulation_operation('airframe.dynamics.longitudinal', kind='static')
def evaluate_longitudinal(derivs, mass, flight, limits=None):
    """Run the longitudinal model and its handling-quality checks.

    Returns the schema result shape {inputs, values, checks, findings, method, declared}.
    Checks (declared, linear-theory screens):
      * short period damping ratio within [zeta_min, zeta_max]
      * phugoid damping ratio above zeta_min
    """
    lim = dict(HQ_LIMITS); lim.update(limits or {})
    A, dd = longitudinal_matrix(derivs, mass, flight)
    modes = identify_longitudinal_modes(A)
    checks = {}
    findings = []
    sp = modes["short_period"]; ph = modes["phugoid"]
    if sp:
        z = sp["zeta"]
        checks["short_period_zeta_in_band"] = lim["short_period_zeta_min"] <= z <= lim["short_period_zeta_max"]
        if not checks["short_period_zeta_in_band"]:
            findings.append("Short-period damping ratio %.3f outside [%.2f, %.2f]." % (
                z, lim["short_period_zeta_min"], lim["short_period_zeta_max"]))
    else:
        checks["short_period_zeta_in_band"] = False
        findings.append("Short period did not resolve to an oscillatory mode (real roots); see real_roots.")
    if ph:
        z = ph["zeta"]
        checks["phugoid_zeta_ok"] = z > lim["phugoid_zeta_min"]
        if not checks["phugoid_zeta_ok"]:
            findings.append("Phugoid damping ratio %.4f below floor %.3f." % (z, lim["phugoid_zeta_min"]))
    else:
        checks["phugoid_zeta_ok"] = False
        findings.append("Phugoid did not resolve to an oscillatory mode.")
    values = {"A_matrix": _as_json(A), "dimensional_derivatives": _as_json(dd), "modes": _as_json(modes)}
    return {"inputs": {"derivatives": _as_json(normalise_derivatives(derivs)),
                       "mass": _as_json(mass), "flight": _as_json(flight)},
            "values": values, "checks": _as_json(checks), "findings": findings,
            "method": "Linearised 4-DOF longitudinal small-perturbation eigenanalysis (Nelson Ch. 4); rigid-body, no flexibility, no propulsion dynamics.",
            "declared": {"limits": lim, "limits_basis": HQ_LIMITS_BASIS}}


@_simulation_operation('airframe.dynamics.lateral', kind='static')
def evaluate_lateral(derivs, mass, flight, limits=None):
    """Run the lateral-directional model and its handling-quality checks.

    Checks (declared, linear-theory screens):
      * Dutch roll damping ratio >= zeta_min AND zeta*omega_n >= zeta_omega_min
      * roll mode time constant <= roll_tau_max_s
      * spiral: stable, or time-to-double >= spiral_t2_min_s
    """
    lim = dict(HQ_LIMITS); lim.update(limits or {})
    A, dd = lateral_matrix(derivs, mass, flight)
    modes = identify_lateral_modes(A)
    checks = {}; findings = []
    dr = modes["dutch_roll"]
    if dr:
        z = dr["zeta"]; wn = dr["omega_n"]
        checks["dutch_roll_zeta_ok"] = z >= lim["dutch_roll_zeta_min"]
        checks["dutch_roll_zeta_omega_ok"] = z * wn >= lim["dutch_roll_zeta_omega_min"]
        if not checks["dutch_roll_zeta_ok"]:
            findings.append("Dutch-roll damping %.3f below floor %.2f." % (z, lim["dutch_roll_zeta_min"]))
        if not checks["dutch_roll_zeta_omega_ok"]:
            findings.append("Dutch-roll zeta*omega_n %.3f below floor %.2f." % (z * wn, lim["dutch_roll_zeta_omega_min"]))
    else:
        checks["dutch_roll_zeta_ok"] = False
        checks["dutch_roll_zeta_omega_ok"] = False
        findings.append("No oscillatory Dutch-roll mode resolved.")
    roll = modes["roll"]
    if roll:
        tau = -1.0 / roll["lambda"] if roll["lambda"] < 0 else float("inf")
        checks["roll_tau_ok"] = tau <= lim["roll_tau_max_s"]
        if not checks["roll_tau_ok"]:
            findings.append("Roll-mode time constant %.3f s exceeds %.2f s." % (tau, lim["roll_tau_max_s"]))
    else:
        checks["roll_tau_ok"] = False
    spiral = modes["spiral"]
    # Integrator fix (2026-09-15 01:55 MDT | 2026-09-15 03:55 EDT | 2026-09-15 07:55 Zulu): one check matching the declared screen (MIL-F-8785C Class I: spiral stable OR
    # minimum time to double ≥ 12 s). Round 2 also failed a separate "spiral_stable" check, which demanded more
    # than the declared criterion. Stability and time to double are kept as reported values.
    spiral_info = {}
    if spiral:
        lam = spiral["lambda"]
        stable = lam < 0
        t2 = None if stable else float(np.log(2.0) / lam)
        spiral_info = {"stable": bool(stable), "time_to_double_s": t2}
        checks["spiral_ok"] = bool(stable or t2 >= lim["spiral_t2_min_s"])
        if not checks["spiral_ok"]:
            findings.append("Spiral mode unstable with time-to-double %.1f s below %.0f s." % (t2, lim["spiral_t2_min_s"]))
        elif not stable:
            findings.append("Spiral mode slightly unstable (time to double %.0f s ≥ %.0f s): acceptable by the declared screen." % (t2, lim["spiral_t2_min_s"]))
    else:
        checks["spiral_ok"] = False
        findings.append("Spiral mode not resolved.")
    values = {"A_matrix": _as_json(A), "dimensional_derivatives": _as_json(dd), "modes": _as_json(modes), "spiral": spiral_info}
    return {"inputs": {"derivatives": _as_json(normalise_derivatives(derivs)),
                       "mass": _as_json(mass), "flight": _as_json(flight)},
            "values": values, "checks": _as_json(checks), "findings": findings,
            "method": "Linearised 4-DOF lateral-directional small-perturbation eigenanalysis (Nelson Ch. 5); rigid-body, no flexibility.",
            "declared": {"limits": lim, "limits_basis": HQ_LIMITS_BASIS}}


# ---------------------------------------------------------------------------
# Round-2 adapter: configuration record + measured baseline -> SI dynamics inputs
# ---------------------------------------------------------------------------
_DYNAMICS_DECLARED = {
    "CD0": {"value": 0.03, "basis": "declared zero-lift drag coefficient (typical for this class; verify)."},
    "body_folded": {"value": "Munk body pitch/yaw correction added to Cm_alpha / Cn_beta when the stability result reports it separately.",
                    "basis": "stability.body_pitch_correction is reported, not folded into the raw VLM coefficients."},
}

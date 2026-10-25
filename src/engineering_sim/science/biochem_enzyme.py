# DESCRIPTION: Biochemistry area — enzyme kinetics: Michaelis-Menten model fitting with parameter
#              uncertainty and AIC, progress-curve integration, kcat/Km efficiency vs the diffusion
#              limit, and Arrhenius/Eyring analysis with CODATA constants (numpy/scipy only).
"""Enzyme-kinetics tools for the science toolkit Biochemistry area.

Imports numpy/scipy lazily inside handlers (see ops_biochem). Reported quantities carry cite
records: fitted parameters are ``estimated`` (method: nonlinear least squares, scipy curve_fit),
standard errors/CIs and R^2/AIC are ``derived`` (exact formulas over the fitted results), and the
diffusion-limit / CODATA constants are ``cited`` (CODATA 2018 read from physics.nist.gov during
this build; the diffusion-limit range is unverified because its source was not fetched).

Honest limits: Michaelis-Menten is an initial-rate model — the fitted Km/Vmax are only meaningful
for the substrate/inhibitor range covered by the data; error bars are the linearised (covariance)
approximation, not a Monte Carlo; AIC ranks models but does not prove a mechanism.
"""
from __future__ import annotations

import math

from . import cite

# CODATA 2018 fundamental constants, read from
# https://physics.nist.gov/cuu/Constants/Table/allascii.txt on 2026-09-14 (verified during build).
_CODATA_URL = "https://physics.nist.gov/cuu/Constants/Table/allascii.txt"
_R = cite.value(8.314462618, "J/(mol K)", "cited", verified=True, source=_CODATA_URL,
                retrieved="2026-09-14", note="CODATA 2018 molar gas constant (exact)")
_KB = cite.value(1.380649e-23, "J/K", "cited", verified=True, source=_CODATA_URL,
                 retrieved="2026-09-14", note="CODATA 2018 Boltzmann constant (exact)")
_H = cite.value(6.62607015e-34, "J s", "cited", verified=True, source=_CODATA_URL,
                retrieved="2026-09-14", note="CODATA 2018 Planck constant (exact)")

# Models: (python params order) and the rate law f(S, I).
_MODELS = {
    "mm": (["Vmax", "Km"],
           lambda S, I, Vmax, Km: Vmax * S / (Km + S)),
    "competitive": (["Vmax", "Km", "Ki"],
                    lambda S, I, Vmax, Km, Ki: Vmax * S / (Km * (1.0 + I / Ki) + S)),
    "uncompetitive": (["Vmax", "Km", "Ki"],
                      lambda S, I, Vmax, Km, Ki: Vmax * S / (Km + S * (1.0 + I / Ki))),
    "noncompetitive": (["Vmax", "Km", "Ki"],
                       lambda S, I, Vmax, Km, Ki: Vmax * S / ((Km + S) * (1.0 + I / Ki))),
    "hill": (["Vmax", "Kh", "n"],
             lambda S, I, Vmax, Kh, n: Vmax * S ** n / (Kh ** n + S ** n)),
    "substrate_inhibition": (["Vmax", "Km", "Ksi"],
                             lambda S, I, Vmax, Km, Ksi: Vmax * S / (Km + S + S * S / Ksi)),
}


from engineering_sim.execution import operation as _simulation_operation

def _as_arrays(S, v):
    import numpy as np
    S = np.asarray(S, dtype=float)
    v = np.asarray(v, dtype=float)
    if S.ndim != 1 or v.ndim != 1 or S.shape != v.shape or S.size < 3:
        raise ValueError("S and v must be equal-length 1-D lists with at least 3 points.")
    if np.any(S <= 0):
        raise ValueError("Substrate concentrations must be positive.")
    if np.any(v < 0):
        raise ValueError("Rates must be non-negative.")
    return S, v


def fit_michaelis_menten(S, v, model="mm", I=None):
    """Fit a kinetic model by nonlinear least squares; report parameters + SEs/95 % CIs, R^2, AIC,
    residuals, and the three linearised plots (Lineweaver-Burk, Eadie-Hofstee, Hanes-Woolf)."""
    import numpy as np
    from scipy.optimize import curve_fit

    if model not in _MODELS:
        raise ValueError("Unknown model %r; choose from %s" % (model, sorted(_MODELS)))
    S, v = _as_arrays(S, v)
    names, rate = _MODELS[model]
    needs_I = model in ("competitive", "uncompetitive", "noncompetitive")
    if needs_I:
        if I is None:
            raise ValueError("Model %r requires an inhibitor concentration list I." % model)
        Iarr = np.asarray(I, dtype=float)
        if Iarr.shape != S.shape or np.any(Iarr < 0):
            raise ValueError("I must match S in length and be non-negative.")
    else:
        Iarr = np.zeros_like(S)

    # Initial guesses and bounds: parameters are positive; hill n is bounded to a sane range.
    vmax0 = float(np.max(v) * 1.2)
    km0 = float(np.median(S))
    p0 = {"Vmax": vmax0, "Km": km0, "Ki": 10.0, "Kh": km0, "n": 1.0, "Ksi": 1000.0}
    lower = {"Vmax": 0.0, "Km": 1e-12, "Ki": 1e-12, "Kh": 1e-12, "n": 0.1, "Ksi": 1e-12}
    upper = {"Vmax": np.inf, "Km": np.inf, "Ki": np.inf, "Kh": np.inf, "n": 10.0, "Ksi": np.inf}
    p0_vec = [p0[n] for n in names]
    bounds = ([lower[n] for n in names], [upper[n] for n in names])

    def f(xdata, *params):
        return rate(xdata, Iarr, *params)

    popt, pcov = curve_fit(f, S, v, p0=p0_vec, bounds=bounds, maxfev=200000)

    # Parameter uncertainty from the covariance matrix (linearised approximation).
    perr = np.sqrt(np.diag(pcov))
    tcrit = 1.96  # z for a 95 % interval (large-sample approximation; note it)
    params = {}
    details = []
    for name, est, err in zip(names, popt, perr):
        rec = cite.value(float(est), _units_for(name), "estimated",
                         method="nonlinear least squares (scipy curve_fit), model=%s" % model)
        params[name] = rec
        details.append({
            "name": name,
            "estimate": rec,
            "se": cite.value(float(err), _units_for(name), "derived",
                             method="sqrt of the covariance-matrix diagonal"),
            "ci95_low": cite.value(float(est - tcrit * err), _units_for(name), "derived",
                                   method="estimate - 1.96 * SE (z interval, approximate)"),
            "ci95_high": cite.value(float(est + tcrit * err), _units_for(name), "derived",
                                    method="estimate + 1.96 * SE (z interval, approximate)"),
        })

    fitted = rate(S, Iarr, *popt)
    resid = v - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((v - np.mean(v)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    n = int(S.size)
    k = len(names)
    rss = ss_res
    # AIC is only defined when RSS > 0; guard against non-finite values (the worker rejects NaN).
    aic = n * math.log(rss / n) + 2.0 * k if rss > 0 else None
    aicc = aic + 2.0 * k * (k + 1) / (n - k - 1) if (aic is not None and (n - k - 1) > 0) else None

    linearized = {
        "lineweaver_burk": {"x": (1.0 / S).tolist(), "y": (1.0 / v).tolist()},
        "eadie_hofstee": {"x": (v / S).tolist(), "y": v.tolist()},
        "hanes_woolf": {"x": S.tolist(), "y": (S / v).tolist()},
    }

    return {
        "model": model,
        "params": params,
        "param_details": details,
        "r2": cite.value(r2, "", "derived", method="1 - SS_res / SS_tot"),
        "rss": cite.value(rss, "", "derived", method="sum of squared residuals"),
        "aic": cite.value(aic, "", "derived", method="n * ln(RSS/n) + 2k (k = fitted parameters)"),
        "aicc": cite.value(aicc, "", "derived", method="AIC with small-sample correction"),
        "n": n, "k": k,
        "fitted": fitted.tolist(),
        "residuals": resid.tolist(),
        "linearized": linearized,
        "notes": [
            "The nonlinear least-squares fit is the estimate of record; the Lineweaver-Burk, "
            "Eadie-Hofstee and Hanes-Woolf transforms are provided for display only (they distort "
            "the error structure).",
            "Parameter errors and 95 % intervals are the linearised covariance approximation "
            "(z = 1.96), valid for well-conditioned data; they are not a bootstrap/Monte Carlo.",
            "AIC ranks models on the same data (lower is better); it does not prove a mechanism.",
        ],
    }


def _units_for(name):
    """Return the display unit for a fitted parameter."""
    if name == "Vmax":
        return "rate units"
    if name == "n":
        return ""
    return "conc units"


def compare_models(S, v, I=None):
    """Fit every applicable model and return them sorted by AIC (a convenience for model choice)."""
    def _aic_key(f):
        val = f["aic"]["value"]
        return float(val) if val is not None else float("inf")
    fits = []
    for model in _MODELS:
        needs_I = model in ("competitive", "uncompetitive", "noncompetitive")
        if needs_I and I is None:
            continue
        try:
            fits.append(fit_michaelis_menten(S, v, model=model, I=I))
        except Exception:
            continue
    fits.sort(key=_aic_key)
    return fits


@_simulation_operation('science_bundle.biochem.kinetics', kind='transient')
def simulate(model, params, S0, E0, t_end):
    """Integrate a progress curve (scipy solve_ivp). model 'mm' or 'product_inhibition'."""
    import numpy as np
    from scipy.integrate import solve_ivp

    if model not in ("mm", "product_inhibition"):
        raise ValueError("simulate model must be 'mm' or 'product_inhibition'.")
    S0 = float(S0)
    E0 = float(E0)
    t_end = float(t_end)
    if S0 <= 0 or E0 <= 0 or t_end <= 0:
        raise ValueError("S0, E0 and t_end must be positive.")
    kcat = float(params.get("kcat", 0.0))
    Km = float(params.get("Km", 0.0))
    if kcat <= 0 or Km <= 0:
        raise ValueError("params must include positive kcat and Km.")
    Kp = float(params.get("Kp", 0.0))
    if model == "product_inhibition" and Kp <= 0:
        raise ValueError("product_inhibition requires a positive Kp in params.")

    def rhs(t, P):
        S = max(S0 - P[0], 0.0)
        if model == "product_inhibition":
            return [kcat * E0 * S / (Km * (1.0 + P[0] / Kp) + S)]
        return [kcat * E0 * S / (Km + S)]

    sol = solve_ivp(rhs, [0.0, t_end], [0.0], method="LSODA", dense_output=True,
                    rtol=1e-8, atol=1e-10)
    times = np.linspace(0.0, t_end, 200)
    P = sol.sol(times)[0]
    v0 = kcat * E0 * S0 / (Km + S0)
    return {
        "model": model,
        "times": times.tolist(),
        "product": P.tolist(),
        "substrate": (S0 - P).tolist(),
        "v0": cite.value(v0, "conc/s", "derived",
                         method="v0 = kcat * E0 * S0 / (Km + S0)"),
        "notes": [
            "Units are as supplied: S0, Km, Kp and E0 in concentration units, kcat in s^-1, "
            "time in s; the progress curve is dP/dt = kcat*E0*S/(Km + S) with S = S0 - P "
            "(product inhibition uses Km*(1 + P/Kp) in the denominator).",
            "The quasi-steady-state assumption holds only when E0 << S0 + Km; the model does not "
            "simulate the transient pre-steady-state burst.",
        ],
    }


def kcat_efficiency(kcat, Km):
    """Catalytic efficiency kcat/Km and its comparison to the diffusion-controlled limit."""
    kcat = float(kcat)
    Km = float(Km)
    if kcat <= 0 or Km <= 0:
        raise ValueError("kcat and Km must be positive.")
    eff = kcat / Km
    lo, hi = 1e8, 1e9  # M^-1 s^-1
    ratio = eff / hi
    return {
        "efficiency": cite.value(eff, "M^-1 s^-1", "derived",
                                 method="kcat / Km (Km supplied in M)"),
        "diffusion_limit": cite.value(hi, "M^-1 s^-1", "unverified",
                                      source="Alberty & Hammes, J. Phys. Chem. 62, 154 (1958); Eigen & Hammes, Adv. Enzymol. 25, 1 (1963)",
                                      note="diffusion-controlled association ~1e8-1e9 M^-1 s^-1; source not fetched during build"),
        "fraction_of_limit": cite.value(ratio, "", "derived", method="kcat/Km / 1e9 M^-1 s^-1"),
        "notes": [
            "Km must be given in M for the efficiency to be in M^-1 s^-1.",
            "Enzymes with kcat/Km near 1e8-1e9 M^-1 s^-1 are 'catalytically perfect': every "
            "encounter is productive and the rate is diffusion-limited.",
        ],
    }


def arrhenius_eyring(k_values, T_values):
    """Arrhenius (Ea, ln A) and Eyring (dH#, dS#, dG# at 298.15 K) analysis with CODATA constants."""
    import numpy as np

    k = np.asarray(k_values, dtype=float)
    T = np.asarray(T_values, dtype=float)
    if k.ndim != 1 or T.ndim != 1 or k.shape != T.shape or k.size < 3:
        raise ValueError("k_values and T_values must be equal-length 1-D lists with at least 3 points.")
    if np.any(k <= 0) or np.any(T <= 0):
        raise ValueError("Rates and temperatures must be positive.")

    R = cite.num(_R)
    KB = cite.num(_KB)
    H = cite.num(_H)
    invT = 1.0 / T
    lnk = np.log(k)
    slope, intercept = np.polyfit(invT, lnk, 1)
    Ea_J = -slope * R
    lnA = float(intercept)
    A = math.exp(lnA)
    # R^2 of the Arrhenius line.
    pred = slope * invT + intercept
    ss_res = float(np.sum((lnk - pred) ** 2))
    ss_tot = float(np.sum((lnk - np.mean(lnk)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)

    T_mean = float(np.mean(T))
    # Eyring: k = (kB T / h) exp(-dG#/RT)  =>  dG# = -RT ln(k h / (kB T)).
    def dG_at(k_i, T_i):
        return -R * T_i * math.log(k_i * H / (KB * T_i))

    # dG# at the mean temperature from the fitted line k(T_mean) = A exp(-Ea/(R T_mean)).
    k_mean = A * math.exp(-Ea_J / (R * T_mean))
    dG_mean = dG_at(k_mean, T_mean)
    dH_mean = Ea_J - R * T_mean  # dH# = Ea - RT (Eyring, solution pre-factor)
    dS_mean = (dH_mean - dG_mean) / T_mean
    # dG#(298.15 K) assuming dH# and dS# are constant over the range (the Eyring-plot convention).
    dG_298 = dH_mean - 298.15 * dS_mean

    return {
        "ea_kjmol": cite.value(Ea_J / 1000.0, "kJ/mol", "estimated",
                               method="Arrhenius: Ea = -R * slope of ln k vs 1/T (linear least squares)"),
        "ln_a": cite.value(lnA, "", "derived", method="intercept of ln k vs 1/T"),
        "a": cite.value(A, "same units as k", "derived", method="exp(ln A)"),
        "arrhenius_r2": cite.value(r2, "", "derived", method="1 - SS_res/SS_tot of ln k vs 1/T"),
        "dh_dagger_kjmol": cite.value(dH_mean / 1000.0, "kJ/mol", "derived",
                                      method="dH# = Ea - R*T_mean (Eyring)"),
        "ds_dagger_jmolK": cite.value(dS_mean, "J/(mol K)", "derived",
                                      method="dS# = (dH# - dG#) / T_mean"),
        "dg_dagger_kjmol": cite.value(dG_mean / 1000.0, "kJ/mol", "derived",
                                      method="dG# = -RT ln(k h / (kB T)), averaged over the data"),
        "dg_dagger_298_kjmol": cite.value(dG_298 / 1000.0, "kJ/mol", "derived",
                                          method="dG# at 298.15 K from the Arrhenius pre-factor"),
        "t_mean": cite.value(T_mean, "K", "derived", method="mean of T_values"),
        "constants": {"R": _R, "kB": _KB, "h": _H},
        "notes": [
            "Arrhenius fits ln k vs 1/T with a straight line; a curved plot indicates a "
            "temperature-dependent activation energy or a multi-step mechanism.",
            "dH# = Ea - RT assumes the simple Eyring pre-factor (kB T / h) with no transmission "
            "coefficient; in solution it is an approximation.",
            "Rates may be in any self-consistent units (Ea is unit-independent); A shares the "
            "rate units.",
        ],
    }

import math
from engineering_sim import correlations

def ply_window(p, t, d, tilt_deg):
    """Projected-slope window (lo, hi) transmitted by one ply (see module docstring). Raises when the slats close the gap."""
    w = t / math.cos(math.radians(tilt_deg))
    if w >= p:
        raise ValueError(f'Slat width {w:.2f} mm (t/cosβ) closes the {p} mm pitch; no air path.')
    k = (p - w) / d
    s = math.tan(math.radians(tilt_deg))
    return (s - k, s + k)

def line_of_sight(plies):
    """True when some straight ray passes every ply. Plies are dicts (dir_deg, pitch_mm, slat_mm, depth_mm, tilt_deg)."""
    by_dir = {}
    for ply in plies:
        key = round(ply['dir_deg'] % 180.0, 6)
        lo, hi = ply_window(ply['pitch_mm'], ply['slat_mm'], ply['depth_mm'], ply['tilt_deg'])
        a, b = by_dir.get(key, (-math.inf, math.inf))
        by_dir[key] = (max(a, lo), min(b, hi))
    if len(by_dir)>2: raise ValueError('This interval method supports at most two independent slat directions.')
    return all((lo <= hi for lo, hi in by_dir.values()))

def _duct(re, a, pr):
    """(Nu, Darcy f) for a rectangular duct of aspect a ≤ 1 at Reynolds re: Shah–London laminar, Gnielinski turbulent, blended."""
    nu_l = correlations.nusselt_rect_duct_laminar_h1(a)
    f_l = correlations.f_re_rect_duct_laminar(a) / re

    def turb(r):
        f = correlations.darcy_petukhov(r)
        return (correlations.nusselt_gnielinski(r, pr, f), f)
    if re <= 2300.0:
        return (nu_l, f_l)
    if re >= 3000.0:
        return turb(re)
    w = (re - 2300.0) / 700.0
    nu_t, f_t = turb(3000.0)
    nu_l2, f_l2 = (nu_l, 96.0 * (f_l * re / 96.0) / 2300.0)
    return ((1 - w) * nu_l2 + w * nu_t, (1 - w) * f_l2 + w * f_t)

def ply_flow(mdot, n, s_mm, h_mm, path_mm, fluid, minor_loss_coefficient):
    """Flow state in one ply of n helical channels (width s, height h) carrying mdot: velocity, Re, Nu, h, Δp."""
    rho, mu, k_air, pr = (fluid[x] for x in ('density_kg_m3', 'mu_pa_s', 'k_w_mk', 'pr'))
    A = n * s_mm * h_mm * 1e-06
    dh = 4 * s_mm * h_mm / (2 * (s_mm + h_mm)) * 0.001
    v = mdot / (rho * A)
    re = max(rho * v * dh / mu, 1e-06)
    a = min(s_mm, h_mm) / max(s_mm, h_mm)
    nu, f = _duct(re, a, pr)
    dp = (f * (path_mm * 0.001) / dh + minor_loss_coefficient) * rho * v * v / 2
    return dict(area_mm2=round(A * 1000000.0, 1), dh_mm=round(dh * 1000.0, 3), velocity_m_s=round(v, 3), reynolds=round(re, 1), aspect=round(a, 4), regime='laminar' if re <= 2300 else 'turbulent' if re >= 3000 else 'transition', nusselt=round(nu, 3), darcy_f=round(f, 5), h_w_m2k=nu * k_air / dh, dp_pa=dp)

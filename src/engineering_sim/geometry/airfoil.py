import math
from pathlib import Path

def _spacing(n_per_side, cosine):
    """Cosine (or uniform) x/c grid, n_per_side + 1 points from 0 (LE) to 1 (TE)."""
    if n_per_side < 2:
        raise ValueError('n_per_side must be >= 2')
    if cosine:
        return [0.5 * (1.0 - math.cos(math.pi * i / n_per_side)) for i in range(n_per_side + 1)]
    return [i / n_per_side for i in range(n_per_side + 1)]

def _parse_naca(code):
    """Four NACA digits -> (max camber m, camber position p, max thickness t), all fractions."""
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f'NACA profile needs exactly four digits, got {code!r}')
    m, p, t = (int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:4]) / 100.0)
    if m > 0.0 and (not 1 <= int(code[1]) <= 9):
        raise ValueError(f'NACA camber position must be 1..9 for a cambered section, got {code!r}')
    if t <= 0.0:
        raise ValueError(f'NACA thickness must be > 0, got {code!r}')
    return (m, p, t)

def _naca4_mean_line(m, p, x):
    """PDAS four-digit mean line: (yc, dyc/dx) at x/c. Symmetric (m=0) -> (0, 0)."""
    if m == 0.0:
        return (0.0, 0.0)
    if x < p:
        yc = m / (p * p) * (2.0 * p * x - x * x)
        slope = 2.0 * m / (p * p) * (p - x)
    else:
        yc = m / (1.0 - p) ** 2 * (1.0 - 2.0 * p + 2.0 * p * x - x * x)
        slope = 2.0 * m / (1.0 - p) ** 2 * (p - x)
    return (yc, slope)

def _naca4_half_thickness(t, te_thickness_frac, x):
    """PDAS half-thickness (fraction) plus the smooth printable-TE term te_thickness_frac/2 * x^4."""
    c = CONSTANTS['naca4_thickness']
    yt = 5.0 * t * (c['a0'] * math.sqrt(x) + c['a1'] * x + c['a2'] * x * x + c['a3'] * x ** 3 + c['a4'] * x ** 4)
    return yt + 0.5 * te_thickness_frac * x ** 4

def _naca_surfaces(m, p, t, te_thickness_frac, xs):
    """Upper/lower (x/c, z/c) pairs on grid ``xs`` for a NACA four-digit section."""
    upper, lower = ([], [])
    for x in xs:
        yc, slope = _naca4_mean_line(m, p, x)
        th = math.atan(slope)
        s, c = (math.sin(th), math.cos(th))
        yt = _naca4_half_thickness(t, te_thickness_frac, x)
        upper.append((x - yt * s, yc + yt * c))
        lower.append((x + yt * s, yc - yt * c))
    return (upper, lower)

def _lininterp(rows, x):
    """Piecewise-linear interpolation on x-sorted (x, z) rows; clamped at the ends (no overshoot)."""
    if x <= rows[0][0]:
        return rows[0][1]
    if x >= rows[-1][0]:
        return rows[-1][1]
    for (xa, za), (xb, zb) in zip(rows, rows[1:]):
        if xa <= x <= xb:
            return za + (zb - za) * (x - xa) / (xb - xa)
    return rows[-1][1]

def _read_numeric(path):
    """Parse float (x, z) rows from a text .dat, skipping comments, blank and non-numeric lines."""
    rows = []
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            a, b = (float(parts[0]), float(parts[1]))
        except ValueError:
            continue
        rows.append((a, b))
    if not rows:
        raise ValueError(f'no numeric coordinate rows in {path}')
    return rows

def _split_surfaces(rows):
    """Split raw .dat rows into x-sorted (upper, lower) surfaces, Selig or Lednicer.

    Lednicer: first row holds the two point counts, then upper (LE->TE) and lower (LE->TE)
    blocks. Selig: one closed loop TE -> upper -> LE -> lower -> TE, split at the minimum-x
    (leading-edge) point. Both returns are ascending in x.
    """
    if len(rows) > 2 and rows[0][0] > 1.0 and (abs(rows[0][0] - round(rows[0][0])) < 1e-09):
        n_up = int(round(rows[0][0]))
        n_lo = int(round(rows[0][1]))
        upper = sorted(rows[1:1 + n_up])
        lower = sorted(rows[1 + n_up:1 + n_up + n_lo])
        return (upper, lower)
    i_le = min(range(len(rows)), key=lambda i: rows[i][0])
    upper = sorted(rows[:i_le + 1])
    lower = sorted(rows[i_le:])
    return (upper, lower)

def _dat_surfaces(upper, lower, gap, te_thickness_frac, xs):
    """Resample tabulated surfaces on ``xs`` and apply the TE-gap taper (fraction form).

    parametric airfoil keeps the measured mean line and tapers the original trailing-edge gap to the
    printable thickness: upper += (te_thickness_frac/2 - gap/2) x^4, lower -= (same).
    """
    add = 0.5 * te_thickness_frac - 0.5 * gap
    up = [(x, _lininterp(upper, x) + add * x ** 4) for x in xs]
    lo = [(x, _lininterp(lower, x) - add * x ** 4) for x in xs]
    return (up, lo)

def properties(coords):
    """Section properties from a ``coords`` loop: max thickness, max camber and their x/c, and area.

    The loop is split at the leading edge (minimum x/c) into x-ascending upper and lower
    surfaces; thickness(x) = z_upper(x) - z_lower(x) and camber(x) = (z_upper + z_lower)/2 are
    evaluated at the union of both surfaces' x/c breakpoints (exact for the piecewise-linear
    surfaces), and area is the shoelace of the closed loop. Returns
    ``{t_c, t_x, camber_max, camber_x, area}`` — ``t_c``/``camber_max`` in chord fractions,
    ``area`` in chord^2. ``camber_max`` is the signed maximum mean-line height.
    """
    le = min(range(len(coords)), key=lambda i: coords[i][0])
    upper = sorted(coords[:le + 1], key=lambda p: p[0])
    lower = sorted(coords[le:], key=lambda p: p[0])
    xs = sorted({round(p[0], 12) for p in upper} | {round(p[0], 12) for p in lower})
    thick, camber = ([], [])
    for x in xs:
        zu = _lininterp(upper, x)
        zl = _lininterp(lower, x)
        thick.append(zu - zl)
        camber.append(0.5 * (zu + zl))
    i_t = max(range(len(thick)), key=lambda i: thick[i])
    i_c = max(range(len(camber)), key=lambda i: camber[i])
    area = 0.5 * abs(sum((a[0] * b[1] - b[0] * a[1] for a, b in zip(coords, coords[1:] + coords[:1]))))
    return dict(t_c=thick[i_t], t_x=xs[i_t], camber_max=camber[i_c], camber_x=xs[i_c], area=area)

CONSTANTS = {'naca4_thickness': dict(a0=0.2969, a1=-0.126, a2=-0.3516, a3=0.2843, a4=-0.1036, open_a4=-0.1015, source='PDAS NACA 4-digit thickness equations (https://www.pdas.com/naca456thick4.html, page updated 2022-05-23, accessed 2026-09-11). a4 = -0.1036 is the closed-trailing-edge coefficient used by parametric airfoil 3.9.0 (model.section) so that the added te_thickness term alone sets the TE thickness; the standard open-TE coefficient is -0.1015.')}

def coords(profile, n_per_side=32, te_thickness_frac=0.0, cosine=True):
    """NACA four-digit or DAT:<caller-path>; unit chord, closed trailing-edge convention."""
    name=str(profile);key=name.upper();xs=_spacing(n_per_side,cosine)
    if key.startswith('DAT:'):
        upper,lower=_split_surfaces(_read_numeric(Path(name[4:])))
        upper,lower=_dat_surfaces(upper,lower,upper[-1][1]-lower[-1][1],te_thickness_frac,xs)
    else:
        code=key.removeprefix('NACA');m,p,t=_parse_naca(code)
        upper,lower=_naca_surfaces(m,p,t,te_thickness_frac,xs)
    return [upper[n_per_side]]+[upper[i] for i in range(n_per_side-1,-1,-1)]+[lower[i] for i in range(1,n_per_side+1)]

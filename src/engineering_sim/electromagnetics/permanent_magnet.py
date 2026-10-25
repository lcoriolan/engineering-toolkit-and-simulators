import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from ..execution import operation
MU0=4e-7*math.pi

@operation("magnetics.permanent_magnet_2d",kind="static")
def solve_field(geometry, segments_per_pair, with_iron=False, grid=120, exterior=False):
    """Solve Az on the square grid and return B at the winding radius.

    Domain: square of half-width R = 1.35 * rotor outer radius. Cells are
    labelled air (nu = 1/mu0), copper (same, mu_r 1), magnet (nu = 1/(mu0*mu_r_m),
    plus its M source) and iron (nu = 1/(mu0*mu_r_fe)). The magnetization curl
    source is computed per magnet cell from the segment's Halbach angle.

    `exterior` flips the magnetization rotation sign — the exterior-field
    Halbach pattern — and samples the field OUTSIDE the ring at the winding
    radius (the rim-drive geometry: magnets on the prop tips, winding in the
    duct wall).
    """
    g = dict(geometry)
    required={'r_gap_mm','gap_mm','magnet_t_mm','iron_t_mm','winding_wall_mm','poles','br_t'}
    if set(g)!=required or any(not math.isfinite(float(g[k])) or float(g[k])<=0 for k in required):
        raise ValueError('Supply all positive geometry/material fields; there is no design preset.')
    if type(grid) is not int or not 16<=grid<=1000:raise ValueError('Grid must be 16..1000.')
    if type(segments_per_pair) is not int or segments_per_pair<1:raise ValueError('Segment count must be positive.')
    r_i = (g['r_gap_mm'] + g['gap_mm']) / 1000.0        # magnet inner radius, m
    r_o = r_i + g['magnet_t_mm'] / 1000.0               # magnet outer radius, m
    r_fe_o = r_o + g['iron_t_mm'] / 1000.0 if with_iron else r_o
    R = 1.35 * (r_fe_o)
    h = 2.0 * R / grid
    n = grid
    xs = -R + h * (np.arange(n) + 0.5)                  # cell centres
    ys = xs
    # Per-cell material: 0 air/copper, 1 magnet, 2 iron.
    material = np.zeros((n, n), dtype=int)
    mx = np.zeros((n, n))
    my = np.zeros((n, n))
    segments_total = g['poles'] * segments_per_pair     # segments around the ring
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            r = math.hypot(x, y)
            if r < r_i:
                continue                                # air + winding annulus
            theta = math.atan2(y, x)
            if r <= r_o:
                material[j, i] = 1
                # Halbach direction: the ideal pattern rotates the
                # magnetization at (p+1) times the azimuth, where p = pole
                # PAIRS (dipole p=1 -> double rate; 14 poles p=7 -> 8x).
                # Discrete segments sample that rotation at their azimuths.
                seg = int((theta / (2.0 * math.pi) * segments_total) + 0.5) % segments_total
                ang = (g['poles'] / 2 + 1) * (2.0 * math.pi * seg / segments_total)
                if exterior:
                    ang = -ang       # exterior pattern: field projects outward
                mx[j, i] = g['br_t'] / MU0 * math.cos(ang)
                my[j, i] = g['br_t'] / MU0 * math.sin(ang)
            elif with_iron and r <= r_fe_o:
                material[j, i] = 2
    # nu per cell (m/H): air 1/mu0, magnet 1/(mu0*1.05), iron 1/(mu0*2000).
    nu = np.where(material == 2, 1.0 / (MU0 * 2000.0), 1.0 / (MU0 * 1.0))
    nu[material == 1] = 1.0 / (MU0 * 1.05)
    # Magnetization source via cell CIRCULATION, not central differences:
    # a central-difference curl turns every magnetization jump into an
    # adjacent +/- dipole pair whose field is suppressed by the grid spacing
    # (the classic FDM trap for bound currents). Circulation around each cell
    # places the current in the single layer of cells that straddle the jump:
    #   src = (Mx_bottom - Mx_top + My_right - My_left) / h
    # with edge values averaged from neighbouring cells and M = 0 outside the
    # domain (the halo matches the Dirichlet far field).
    mx = np.pad(mx, 1)          # halo of zero magnetization around the domain
    my = np.pad(my, 1)
    mx_bottom = 0.5 * (mx[1:-1, 1:-1] + mx[:-2, 1:-1])
    mx_top = 0.5 * (mx[1:-1, 1:-1] + mx[2:, 1:-1])
    my_left = 0.5 * (my[1:-1, 1:-1] + my[1:-1, :-2])
    my_right = 0.5 * (my[1:-1, 1:-1] + my[1:-1, 2:])
    source = (mx_bottom - mx_top + my_right - my_left) / h
    # Five-point stencil with harmonic face conductivities (declared screen).
    rows, cols, vals = [], [], []
    idx = np.arange(n * n).reshape(n, n)
    for j in range(n):
        for i in range(n):
            k = idx[j, i]
            diag = 0.0
            for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                jj, ii = j + dj, i + di
                if not (0 <= jj < n and 0 <= ii < n):
                    continue
                face = 2.0 * nu[j, i] * nu[jj, ii] / (nu[j, i] + nu[jj, ii]) if nu[j, i] + nu[jj, ii] else 0.0
                rows.append(k); cols.append(idx[jj, ii]); vals.append(-face / h ** 2)
                diag += face / h ** 2
            rows.append(k); cols.append(k); vals.append(diag)
    A = coo_matrix((vals, (rows, cols)), shape=(n * n, n * n)).tocsr()
    # Dirichlet Az = 0 on the domain boundary (the far-field shunt). Without
    # this the system is singular: pure Neumann plus a divergence-free source
    # leaves the gauge free and the solve returns the zero field.
    boundary = np.zeros((n, n), dtype=bool)
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = True
    keep = ~boundary.ravel()
    A = A[keep][:, keep]
    # RHS is the POINTWISE curl(M), paired with the h^-2-divided operator
    # -div(nu grad A) = curl(M). Integrating the RHS by h^2 here would
    # multiply curl(M) ~ M/h by h, making the whole solution collapse like
    # h^2 as the grid refines (verified: disk benchmark field went
    # 1.8e-5 -> 1.3e-6 -> 2.3e-8 T on 10/40/300 grids instead of 0.628 T).
    b = source.ravel()
    b = b[keep]
    az_full = np.zeros(n * n)
    az_full[keep] = spsolve(A, b)
    az = az_full.reshape(n, n)
    # B = (dAz/dy, -dAz/dx); evaluate radial component on the winding ring.
    dazdy = np.zeros_like(az)
    dazdx = np.zeros_like(az)
    dazdy[1:-1, :] = (az[2:, :] - az[:-2, :]) / (2.0 * h)
    dazdx[:, 1:-1] = (az[:, 2:] - az[:, :-2]) / (2.0 * h)
    bx, by = dazdy, -dazdx
    samples = []
    # Exterior (rim) sampling: the winding sits OUTSIDE the magnet ring, at
    # r_o + gap + half the winding build. Interior sampling stays at the
    # declared winding radius inside the bore.
    sample_r = (r_o + (g['gap_mm'] + g['winding_wall_mm']) / 1000.0) if exterior else \
               g['r_gap_mm'] / 1000.0
    for t in np.linspace(0.0, 2.0 * math.pi, 361):
        x, y = sample_r * math.cos(t), sample_r * math.sin(t)
        ix = min(n - 1, int((x + R) / h))
        iy = min(n - 1, int((y + R) / h))
        br = bx[iy, ix] * math.cos(t) + by[iy, ix] * math.sin(t)
        samples.append(br)
    # The signed radial field averages to zero over 360 degrees for ANY pole
    # count (as many + as -); the winding sees the MAGNITUDE, so the screen
    # averages |Br| around the ring.
    br_mean = float(np.mean(np.abs(samples)))
    return dict(b_r_t=br_mean, r_i_m=r_i, r_o_m=r_o, iron=with_iron,
                segments_per_pair=segments_per_pair, grid=grid)

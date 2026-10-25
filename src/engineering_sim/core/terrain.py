#!/usr/bin/env python3
# DESCRIPTION: Regular heightfield h[iy, ix] in the core ENU world frame (x = E, y = N, z = up, metres), with a
#              surface material per cell. GPU side: TerrainParams struct + wp.funcs (bilinear height, analytic
#              normal of the bilinear patch, material lookup, AGL). Host side: numpy mirrors for tests/renderers.
#
#              Sources (all parametric):
#                Terrain.flat / Terrain.plane(grade_x, grade_y) / Terrain.lanes(grades)   analytic test ground
#                Terrain.fbm(size, dx, amp, hurst, octaves, seed)                         procedural (spectral fBm)
#                Terrain.from_geotiff(path, center_lonlat, size_m, dx)                    DEM import (Copernicus
#                    GLO-30/90 COG etc.: tiled/stripped, DEFLATE/none, predictor 1/2/3 — decoded here with numpy +
#                    zlib, no GDAL/imagecodecs needed), resampled to a local ENU grid (equirectangular about the
#                    centre: error < 0.1 % over a few km, ASSUMED adequate for vehicle-scale terrain)
#                Terrain.from_xyz(path) / Terrain.load(npz) / .save(npz)
#              .add_fbm(...) overlays procedural roughness on any source (e.g. 90 m DEM + 1 m bumps).
#
#              Materials: SURFACES below. Tyre coefficients for dry/wet tarmac, snow and ice are the Magic Formula
#              constants of the MathWorks Simscape "Tire-Road Interaction (Magic Formula)" block (reference set,
#              not a measured tyre). Gravel/grass/soil/sand/mud tyre coefficients and every rolling-resistance
#              value are ASSUMED representative values in the ranges given by Wong, "Theory of Ground Vehicles"
#              (4th ed., ch. 1-2) and Gillespie, "Fundamentals of Vehicle Dynamics" (ch. 2). Soft-soil sinkage
#              (Bekker) is NOT modelled — soft surfaces only raise rolling resistance and lower friction.
#
#              Wind over terrain (hook for CORE): core wind profiles are functions of height above ground. A domain
#              moving over terrain calls wind_at(agl_point(p, tp), ...) — `agl_point` swaps z for z - h(x, y),
#              clamped to >= z_min (the roughness length). core/wind.py itself is unchanged.
import math
import struct
import zlib

import numpy as np
import warp as wp

# name: (B, C, D (=peak mu), E, crr, mu_scale_for_rollers/casters)
SURFACES = {
    "dry_tarmac": (10.0, 1.9, 1.00, 0.97, 0.012),
    "wet_tarmac": (12.0, 2.3, 0.82, 1.00, 0.015),
    "snow":       (5.0, 2.0, 0.30, 1.00, 0.030),
    "ice":        (4.0, 2.0, 0.10, 1.00, 0.010),
    "gravel":     (8.0, 1.6, 0.65, 0.90, 0.030),   # ASSUMED
    "grass":      (7.0, 1.6, 0.55, 0.90, 0.050),   # ASSUMED (dry, short)
    "dry_soil":   (7.0, 1.6, 0.60, 0.90, 0.060),   # ASSUMED (firm)
    "sand":       (5.0, 1.5, 0.45, 0.80, 0.150),   # ASSUMED (loose; sinkage not modelled)
    "mud":        (5.0, 1.5, 0.30, 0.80, 0.120),   # ASSUMED
}
SURFACE_NAMES = list(SURFACES)
SURFACE_SOURCE = {k: ("MathWorks Simscape Magic Formula reference constants" if k in
                      ("dry_tarmac", "wet_tarmac", "snow", "ice") else "ASSUMED (Wong/Gillespie ranges)")
                  for k in SURFACES}


@wp.struct
class TerrainParams:
    x0: float          # world x of column 0
    y0: float          # world y of row 0
    dx: float          # cell size [m] (square cells)
    nx: int
    ny: int
    h: wp.array2d(dtype=float)      # [ny, nx] height [m], world z
    mat: wp.array2d(dtype=int)      # [ny, nx] surface index into the tables below
    mf: wp.array(dtype=wp.vec4)     # per surface (B, C, D, E)
    crr: wp.array(dtype=float)      # per surface rolling-resistance coefficient


@wp.func
def _cell(x: float, y: float, tp: TerrainParams):
    fx = wp.clamp((x - tp.x0) / tp.dx, 0.0, float(tp.nx - 1) - 1.0e-4)
    fy = wp.clamp((y - tp.y0) / tp.dx, 0.0, float(tp.ny - 1) - 1.0e-4)
    ix = int(wp.floor(fx))
    iy = int(wp.floor(fy))
    return wp.vec4(float(ix), float(iy), fx - float(ix), fy - float(iy))


@wp.func
def height(x: float, y: float, tp: TerrainParams):
    c = _cell(x, y, tp)
    ix = int(c[0])
    iy = int(c[1])
    u = c[2]
    v = c[3]
    h00 = tp.h[iy, ix]
    h10 = tp.h[iy, ix + 1]
    h01 = tp.h[iy + 1, ix]
    h11 = tp.h[iy + 1, ix + 1]
    return (h00 * (1.0 - u) + h10 * u) * (1.0 - v) + (h01 * (1.0 - u) + h11 * u) * v


@wp.func
def normal(x: float, y: float, tp: TerrainParams):
    """Unit normal of the bilinear patch (exact gradient of `height`)."""
    c = _cell(x, y, tp)
    ix = int(c[0])
    iy = int(c[1])
    u = c[2]
    v = c[3]
    h00 = tp.h[iy, ix]
    h10 = tp.h[iy, ix + 1]
    h01 = tp.h[iy + 1, ix]
    h11 = tp.h[iy + 1, ix + 1]
    dhdx = ((h10 - h00) * (1.0 - v) + (h11 - h01) * v) / tp.dx
    dhdy = ((h01 - h00) * (1.0 - u) + (h11 - h10) * u) / tp.dx
    return wp.normalize(wp.vec3(-dhdx, -dhdy, 1.0))


@wp.func
def height_normal(x: float, y: float, tp: TerrainParams):
    """(nx, ny, nz, h) from ONE set of 4 corner reads (height + exact bilinear-patch normal)."""
    c = _cell(x, y, tp)
    ix = int(c[0])
    iy = int(c[1])
    u = c[2]
    v = c[3]
    h00 = tp.h[iy, ix]
    h10 = tp.h[iy, ix + 1]
    h01 = tp.h[iy + 1, ix]
    h11 = tp.h[iy + 1, ix + 1]
    dhdx = ((h10 - h00) * (1.0 - v) + (h11 - h01) * v) / tp.dx
    dhdy = ((h01 - h00) * (1.0 - u) + (h11 - h10) * u) / tp.dx
    n = wp.normalize(wp.vec3(-dhdx, -dhdy, 1.0))
    return wp.vec4(n[0], n[1], n[2], (h00 * (1.0 - u) + h10 * u) * (1.0 - v) + (h01 * (1.0 - u) + h11 * u) * v)


@wp.func
def material(x: float, y: float, tp: TerrainParams):
    c = _cell(x, y, tp)
    ix = int(c[0])
    iy = int(c[1])
    if c[2] > 0.5:
        ix = ix + 1
    if c[3] > 0.5:
        iy = iy + 1
    return tp.mat[iy, ix]


@wp.func
def agl_point(p: wp.vec3, tp: TerrainParams, z_min: float):
    """Same point with z replaced by height above the local ground (for AGL wind/air lookups)."""
    return wp.vec3(p[0], p[1], wp.max(p[2] - height(p[0], p[1], tp), z_min))


class Terrain:
    """Host-side heightfield + material map. .params(device) -> TerrainParams for kernels."""

    def __init__(self, h, dx, x0=None, y0=None, mat=None, meta=None):
        self.h = np.ascontiguousarray(h, np.float32)
        self.ny, self.nx = self.h.shape
        self.dx = float(dx)
        self.x0 = -0.5 * (self.nx - 1) * self.dx if x0 is None else float(x0)
        self.y0 = -0.5 * (self.ny - 1) * self.dx if y0 is None else float(y0)
        self.mat = np.zeros_like(self.h, np.int32) if mat is None else np.ascontiguousarray(mat, np.int32)
        self.meta = dict(meta or {})
        self._p = {}

    # ---------- constructors ----------
    @classmethod
    def flat(cls, size=400.0, dx=2.0, z=0.0, surface="dry_tarmac"):
        n = int(round(size / dx)) + 1
        t = cls(np.full((n, n), z, np.float32), dx, meta=dict(source="flat", z=z))
        return t.set_surface(surface)

    @classmethod
    def plane(cls, grade_x=0.0, grade_y=0.0, size=400.0, dx=2.0, surface="dry_tarmac"):
        """Planar ground z = grade_x x + grade_y y (grade = rise/run = tan(angle))."""
        t = cls.flat(size, dx, 0.0, surface)
        X, Y = t.grid()
        t.h[:] = (grade_x * X + grade_y * Y).astype(np.float32)
        t.meta = dict(source="plane", grade_x=grade_x, grade_y=grade_y)
        return t

    @classmethod
    def lanes(cls, angles_deg, lane_w=12.0, length=120.0, dx=1.0, surface="dry_tarmac"):
        """Parallel test lanes along +x, lane k at y = k*lane_w, each a plane rising at angles_deg[k] along x
        (z = x tan a, exact inside a lane: grade is constant across whole cells). Returns (terrain, lane_y)."""
        n_l = len(angles_deg)
        ny = int(round(n_l * lane_w / dx)) + 1
        nx = int(round(length / dx)) + 1
        x0, y0 = -0.5 * length, -0.5 * lane_w
        x = x0 + dx * np.arange(nx)
        y = y0 + dx * np.arange(ny)
        lane = np.clip(np.floor((y - y0) / lane_w).astype(int), 0, n_l - 1)
        g = np.tan(np.radians(np.asarray(angles_deg, float)))[lane]
        h = (g[:, None] * x[None, :]).astype(np.float32)
        t = cls(h, dx, x0, y0, meta=dict(source="lanes", angles_deg=list(map(float, angles_deg)), lane_w=lane_w))
        return t.set_surface(surface), np.arange(n_l) * lane_w

    @classmethod
    def fbm(cls, size=600.0, dx=2.0, amp=6.0, hurst=0.8, octaves=None, seed=1, surface="grass", lam_max=None):
        """Spectral fBm: power spectrum ~ k^-(2H+2), band [2 dx, lam_max]; scaled to RMS height `amp`."""
        n = int(round(size / dx)) + 1
        rng = np.random.default_rng(seed)
        kx = np.fft.fftfreq(n, dx)
        K = np.hypot(kx[None, :], kx[:, None])
        kmin = 1.0 / (lam_max or size)
        S = np.where((K >= kmin) & (K <= 0.5 / dx), np.maximum(K, kmin) ** (-(2 * hurst + 2) / 2), 0.0)
        z = np.real(np.fft.ifft2(S * np.exp(2j * np.pi * rng.random((n, n)))))
        z = (z - z.mean()) / max(z.std(), 1e-12) * amp
        t = cls(z.astype(np.float32), dx, meta=dict(source="fbm", amp=amp, hurst=hurst, seed=seed, lam_max=lam_max))
        return t.set_surface(surface)

    @classmethod
    def from_geotiff(cls, path, center_lonlat=None, size_m=2000.0, dx=10.0, surface="dry_soil", zero_at_center=True):
        """DEM GeoTIFF (EPSG:4326 lon/lat grid) -> local ENU heightfield about center_lonlat (default: tile centre)."""
        dem, (lon0, lat0, dlon, dlat), nodata = read_geotiff(path)
        if nodata is not None:
            dem = np.where(dem == nodata, np.nan, dem)
        rows, cols = dem.shape
        if center_lonlat is None:
            center_lonlat = (lon0 + dlon * cols / 2, lat0 - dlat * rows / 2)
        lonc, latc = center_lonlat
        R_E = 6371008.8
        n = int(round(size_m / dx)) + 1
        e = (np.arange(n) - (n - 1) / 2) * dx
        E, N = np.meshgrid(e, e)
        lat = latc + np.degrees(N / R_E)
        lon = lonc + np.degrees(E / (R_E * math.cos(math.radians(latc))))
        fc = (lon - lon0) / dlon - 0.5          # pixel-is-area centres
        fr = (lat0 - lat) / dlat - 0.5
        h = _bilinear(dem, fr, fc)
        if np.isnan(h).any():
            h = np.where(np.isnan(h), np.nanmean(h), h)
        z_ref = float(_bilinear(dem, np.array([(lat0 - latc) / dlat - 0.5]), np.array([(lonc - lon0) / dlon - 0.5]))[0])
        if zero_at_center:
            h = h - z_ref
        t = cls(h.astype(np.float32), dx, meta=dict(source="geotiff", path=str(path), center_lonlat=[lonc, latc],
                                                    size_m=size_m, dem_res_deg=[dlon, dlat],
                                                    z_ref_msl_m=z_ref if zero_at_center else 0.0))
        return t.set_surface(surface)

    @classmethod
    def from_xyz(cls, path, dx=None, surface="dry_soil"):
        """Regular-grid ASCII 'x y z' (any order of rows) -> heightfield."""
        a = np.loadtxt(path)
        xs, ys = np.unique(a[:, 0]), np.unique(a[:, 1])
        dx = dx or float(np.median(np.diff(xs)))
        h = np.full((len(ys), len(xs)), np.nan)
        h[np.searchsorted(ys, a[:, 1]), np.searchsorted(xs, a[:, 0])] = a[:, 2]
        h = np.where(np.isnan(h), np.nanmean(h), h)
        return cls(h, dx, xs[0], ys[0], meta=dict(source="xyz", path=str(path))).set_surface(surface)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        import json
        return cls(z["h"], float(z["dx"]), float(z["x0"]), float(z["y0"]), z["mat"], json.loads(str(z["meta"])))

    def save(self, path):
        import json
        np.savez_compressed(path, h=self.h, mat=self.mat, dx=self.dx, x0=self.x0, y0=self.y0,
                            meta=json.dumps(self.meta), surfaces=json.dumps(SURFACE_NAMES))
        return path

    # ---------- edits ----------
    def grid(self):
        return np.meshgrid(self.x0 + self.dx * np.arange(self.nx), self.y0 + self.dx * np.arange(self.ny))

    def add_fbm(self, amp=0.1, hurst=0.8, seed=2, lam_max=20.0):
        f = Terrain.fbm(size=(max(self.nx, self.ny) - 1) * self.dx, dx=self.dx, amp=amp, hurst=hurst, seed=seed,
                        lam_max=lam_max)
        self.h += f.h[:self.ny, :self.nx]
        self.meta.setdefault("overlays", []).append(dict(fbm_amp=amp, hurst=hurst, seed=seed, lam_max=lam_max))
        self._p = {}
        return self

    def set_surface(self, name, mask=None):
        k = SURFACE_NAMES.index(name)
        if mask is None:
            self.mat[:] = k
        else:
            self.mat[mask] = k
        self._p = {}
        return self

    def surface_by_slope(self, rules=((0.0, "grass"), (15.0, "dry_soil"), (30.0, "gravel"))):
        """Material from local slope [deg]: last rule whose threshold <= slope wins (illustrative, ASSUMED)."""
        s = self.slope_deg()
        for thr, name in rules:
            self.set_surface(name, s >= thr)
        return self

    # ---------- host mirrors ----------
    def height_np(self, x, y):
        fx = np.clip((np.asarray(x, float) - self.x0) / self.dx, 0, self.nx - 1 - 1e-4)
        fy = np.clip((np.asarray(y, float) - self.y0) / self.dx, 0, self.ny - 1 - 1e-4)
        return _bilinear(self.h.astype(float), fy, fx)

    def normal_np(self, x, y, e=None):
        e = e or 0.25 * self.dx
        gx = (self.height_np(x + e, y) - self.height_np(x - e, y)) / (2 * e)
        gy = (self.height_np(x, y + e) - self.height_np(x, y - e)) / (2 * e)
        n = np.array([-float(gx), -float(gy), 1.0])
        return n / np.linalg.norm(n)

    def slope_deg(self):
        gy, gx = np.gradient(self.h.astype(float), self.dx)
        return np.degrees(np.arctan(np.hypot(gx, gy)))

    def describe(self):
        return dict(nx=self.nx, ny=self.ny, dx=self.dx, x0=self.x0, y0=self.y0,
                    z_min=float(self.h.min()), z_max=float(self.h.max()),
                    surfaces=sorted({SURFACE_NAMES[i] for i in np.unique(self.mat)}), **self.meta)

    def params(self, device):
        if device in self._p:
            return self._p[device]
        tp = TerrainParams()
        tp.x0, tp.y0, tp.dx, tp.nx, tp.ny = self.x0, self.y0, self.dx, self.nx, self.ny
        tp.h = wp.array(self.h, dtype=float, device=device)
        tp.mat = wp.array(self.mat, dtype=int, device=device)
        tab = np.array([SURFACES[k] for k in SURFACE_NAMES], np.float32)
        tp.mf = wp.array(tab[:, :4], dtype=wp.vec4, device=device)
        tp.crr = wp.array(tab[:, 4], dtype=float, device=device)
        self._p[device] = tp
        return tp


def _bilinear(a, fr, fc):
    r0 = np.clip(np.floor(fr).astype(int), 0, a.shape[0] - 2)
    c0 = np.clip(np.floor(fc).astype(int), 0, a.shape[1] - 2)
    u, v = np.clip(fc - c0, 0, 1), np.clip(fr - r0, 0, 1)
    return ((a[r0, c0] * (1 - u) + a[r0, c0 + 1] * u) * (1 - v) + (a[r0 + 1, c0] * (1 - u) + a[r0 + 1, c0 + 1] * u) * v)


# ---------------- minimal GeoTIFF reader (numpy + zlib) ----------------
_TYPES = {1: "B", 2: "s", 3: "H", 4: "I", 5: "II", 6: "b", 7: "B", 8: "h", 9: "i", 11: "f", 12: "d", 16: "Q"}


def read_geotiff(path):
    """-> (array[rows, cols] float64, (lon0, lat0, dlon, dlat) of the top-left corner, nodata or None).
    Supports single-band int/float, stripped or tiled, compression 1 (none) / 8 / 32946 (deflate),
    predictor 1, 2 (horizontal) and 3 (floating point). First IFD only (full resolution)."""
    buf = open(path, "rb").read()
    bo = {b"II": "<", b"MM": ">"}[buf[:2]]
    magic = struct.unpack(bo + "H", buf[2:4])[0]
    if magic == 43:
        off = struct.unpack(bo + "Q", buf[8:16])[0]
        n = struct.unpack(bo + "Q", buf[off:off + 8])[0]; p, esz, cnt_fmt, val_sz = off + 8, 20, "Q", 8
    else:
        off = struct.unpack(bo + "I", buf[4:8])[0]
        n = struct.unpack(bo + "H", buf[off:off + 2])[0]; p, esz, cnt_fmt, val_sz = off + 2, 12, "I", 4
    tags = {}
    for k in range(n):
        e = buf[p + k * esz:p + (k + 1) * esz]
        tag, typ = struct.unpack(bo + "HH", e[:4])
        cnt = struct.unpack(bo + cnt_fmt, e[4:4 + val_sz])[0]
        fmt = _TYPES.get(typ, "B")
        item = struct.calcsize(fmt)
        size = item * cnt
        raw = e[4 + val_sz:] if size <= val_sz else buf[struct.unpack(bo + cnt_fmt, e[4 + val_sz:])[0]:][:size]
        if typ == 2:
            tags[tag] = raw[:size].split(b"\0")[0].decode(errors="replace")
        elif typ in (5,):
            v = struct.unpack(bo + "I" * (2 * cnt), raw[:size])
            tags[tag] = [v[2 * i] / v[2 * i + 1] for i in range(cnt)]
        else:
            tags[tag] = list(struct.unpack(bo + fmt * cnt, raw[:size]))
    W, H = tags[256][0], tags[257][0]
    bps = tags.get(258, [8])[0]
    comp = tags.get(259, [1])[0]
    pred = tags.get(317, [1])[0]
    sfmt = tags.get(339, [1])[0]
    if tags.get(277, [1])[0] != 1:
        raise ValueError("single-band DEMs only")
    dt = np.dtype({1: "u", 2: "i", 3: "f"}[sfmt] + str(bps // 8)).newbyteorder(bo)
    if 322 in tags:
        tw, th, offs, cnts = tags[322][0], tags[323][0], tags[324], tags[325]
    else:
        tw, th, offs, cnts = W, tags.get(278, [H])[0], tags[273], tags[279]
    out = np.zeros((H, W), np.float64)
    nxt = -(-W // tw)
    for k, (o, c) in enumerate(zip(offs, cnts)):
        raw = buf[o:o + c]
        if comp in (8, 32946):
            raw = zlib.decompress(raw)
        elif comp != 1:
            raise ValueError("compression %d not supported" % comp)
        rows = len(raw) // (tw * dt.itemsize)
        if pred == 3:
            b = np.frombuffer(raw, np.uint8).reshape(rows, tw * dt.itemsize)
            b = np.cumsum(b, axis=1, dtype=np.uint8)
            b = b.reshape(rows, dt.itemsize, tw).transpose(0, 2, 1)       # byte planes, MSB first
            tile = np.ascontiguousarray(b).view(np.dtype(dt.kind + str(dt.itemsize)).newbyteorder(">"))
            tile = tile.reshape(rows, tw).astype(np.float64)
        else:
            tile = np.frombuffer(raw, dt).reshape(rows, tw).astype(np.float64)
            if pred == 2:
                tile = np.cumsum(tile, axis=1)
        r0, c0 = (k // nxt) * th, (k % nxt) * tw
        rr, cc = min(rows, H - r0), min(tw, W - c0)
        if rr > 0 and cc > 0:
            out[r0:r0 + rr, c0:c0 + cc] = tile[:rr, :cc]
    scale, tie = tags.get(33550), tags.get(33922)
    if scale is None or tie is None:
        raise ValueError("no GeoTIFF georeference tags")
    lon0 = tie[3] - tie[0] * scale[0]
    lat0 = tie[4] + tie[1] * scale[1]
    nod = tags.get(42113)
    nodata = float(nod) if isinstance(nod, str) and nod.strip() else None
    return out, (lon0, lat0, scale[0], scale[1]), nodata

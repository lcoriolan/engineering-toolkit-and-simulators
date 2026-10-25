# DESCRIPTION: Density above the core column that responds to the Sun and the geomagnetic field, not just height:
#                NRLMSIS 2.1 (default), NRLMSIS 2.0 or NRLMSISE-00 through `pymsis` 0.11.0 (pinned wheel in

#                Lucas, G. et al., https://github.com/SWxTREC/pymsis). Inputs: F10.7 (previous day), F10.7 81-day
#                centred mean, Ap (daily + 3-hourly history), day of year, UT -> local solar time, geodetic lat/lon,
#                geodetic altitude. So the diurnal bulge (~14-16 LT), season, latitude and storms are all in.
#              Below 84 km the ONE core column (density.AirColumn: core.env -> US76) is used; 84-88 km is the same
#              log-linear blend density.py uses (column -> thermosphere), so launch/re-entry stay continuous.
#              `model: exp` keeps the old static Vallado table (density.AirColumn) unchanged — the default.
#
#              GPU path: a 3-D table of log(rho) over (geodetic altitude, local solar time, geodetic latitude),
#              rebuilt every `refresh_s` of sim time from the space weather at that time (MSIS 2.1: ~0.4 s for the
#              default 1421 x 24 x 19 grid). Local time is the axis (not longitude) so the bulge stays fixed
#              relative to the Sun between refreshes; only the slow terms (season, F10.7, Ap, UT-dependent terms)
#              are held for one refresh interval. Interpolation: linear in log(rho) along altitude, local time
#              (periodic) and latitude — the kernel in propagate.py and `rho_table()` here share the formula.
#
#              Space weather (`SpaceWeather`): constant indices, a CelesTrak SW-All.csv (cached copy in data/, never
#              downloaded at run time), or a simple CSV time series (columns: time ISO or jd, f107, f107a, ap).
#              SW-All convention follows pymsis/MSIS: f107 = observed F10.7 of the PREVIOUS day, f107a = observed
#              81-day centred mean of the day, ap[0] = daily Ap, ap[1..6] = 3-hourly history (storm-time mode).
#              Monthly predicted rows (PRM) have no Ap: F10.7 is interpolated, Ap is ASSUMED = mean daily Ap of
#              the last 81 observed days. MODELED throughout.
import csv
import math

import numpy as np

from .density import AirColumn, BLEND_HI, BLEND_LO

MODELS = {"msis2.1": 2.1, "msis2.0": 2.0, "msis00": 0, "exp": None}
JD_UNIX0 = 2440587.5


def jd_to_datetime64(jd):
    """Julian date (UT) -> numpy datetime64[us]."""
    us = np.round((np.asarray(jd, float) - JD_UNIX0) * 86400e6).astype("int64")
    return us.astype("datetime64[us]")


def datetime64_to_jd(d):
    return np.asarray(d, "datetime64[us]").astype("int64") / 86400e6 + JD_UNIX0


# ------------------------------------------------------------------------------------------------ space weather
class SpaceWeather:
    """F10.7 / F10.7a / Ap source. `indices(jd)` -> (f107, f107a, ap[7]) for scalar jd."""

    def __init__(self, f107=150.0, f107a=None, ap=4.0, file=None, fmt=None):
        self.file = file
        self.kind = "constant"
        if file is None:
            self.f107, self.f107a = float(f107), float(f107 if f107a is None else f107a)
            ap = np.atleast_1d(np.asarray(ap, float))
            self.ap = np.full(7, ap[0]) if ap.size == 1 else ap.astype(float)[:7]
            return
        with open(file, encoding="utf-8") as fh:
            head = fh.readline()
        if fmt == "celestrak" or (fmt is None and head.startswith("DATE,BSRN")):
            self._load_celestrak(file)
        else:
            self._load_series(file)

    @classmethod
    def from_cfg(cls, cfg):
        if cfg is None:
            return cls()
        if isinstance(cfg, SpaceWeather):
            return cfg
        if isinstance(cfg, str):
            return cls(file=cfg)
        cfg = dict(cfg)
        return cls(f107=cfg.get("f107", 150.0), f107a=cfg.get("f107a"), ap=cfg.get("ap", 4.0),
                   file=cfg.get("file"), fmt=cfg.get("format"))

    # CelesTrak SW-All.csv ------------------------------------------------------------------------------
    def _load_celestrak(self, path):
        self.kind = "celestrak"
        days, f107o, f107c, apd, ap3, typ = [], [], [], [], [], []
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                days.append(np.datetime64(row["DATE"], "D"))
                f = float(row["F10.7_OBS"]) if row["F10.7_OBS"] else np.nan
                fc = float(row["F10.7_OBS_CENTER81"]) if row["F10.7_OBS_CENTER81"] else np.nan
                if not (0.0 < f <= 400.0):          # solar radio burst / missing -> centred mean (pymsis rule)
                    f = fc
                f107o.append(f)
                f107c.append(fc)
                apd.append(float(row["AP_AVG"]) if row["AP_AVG"] else np.nan)
                ap3.append([float(row["AP%d" % k]) if row["AP%d" % k] else np.nan for k in range(1, 9)])
                typ.append(row["F10.7_DATA_TYPE"])
        self.days = np.array(days)
        self.day_jd = datetime64_to_jd(self.days.astype("datetime64[us]"))
        self.f107o, self.f107c = np.array(f107o), np.array(f107c)
        self.apd, self.ap3 = np.array(apd), np.array(ap3)
        self.dtype = np.array(typ)
        obs = np.where(np.isfinite(self.apd))[0]
        self.last_obs_day = int(obs[-1])
        self.ap_fill = float(np.mean(self.apd[obs[-81:]]))          # ASSUMED Ap where none is given (PRM rows)
        self.apd = np.where(np.isfinite(self.apd), self.apd, self.ap_fill)
        self.ap3 = np.where(np.isfinite(self.ap3), self.ap3, self.apd[:, None])
        gap = np.where(np.diff(self.day_jd) > 1.0 + 1e-9)[0]
        self.n_daily = int(gap[0]) + 1 if len(gap) else len(self.day_jd)    # rows before the monthly (PRM) section
        self.ap3_flat = self.ap3[:self.n_daily].reshape(-1)            # 3-hourly series from day 0, 00 UT

    def _day_value(self, arr, jd):
        """Daily value: exact day where daily rows exist, linear in time between monthly PRM rows."""
        x = float(jd)
        i = int(np.searchsorted(self.day_jd, x, side="right") - 1)
        if i < 0:
            raise ValueError("space weather: date before %s" % self.days[0])
        if i >= len(self.day_jd) - 1:
            return float(arr[-1])
        if self.day_jd[i + 1] - self.day_jd[i] <= 1.0 + 1e-9:
            return float(arr[i])
        w = (x - self.day_jd[i]) / (self.day_jd[i + 1] - self.day_jd[i])
        return float((1 - w) * arr[i] + w * arr[i + 1])

    def _ap_hist(self, jd):
        """MSIS ap[7] at jd (pymsis get_f107_ap convention); flat daily Ap outside the daily-row section."""
        k = int(math.floor((float(jd) - self.day_jd[0]) * 8.0 + 1e-9))       # 3-hour slot from day 0, 00 UT
        a = self.ap3_flat
        if not 0 <= k < len(a):
            return np.full(7, self._day_value(self.apd, jd))

        def at(j):
            return a[j] if 0 <= j < len(a) else np.nan

        def mean8(j_last):
            seg = np.array([at(j) for j in range(j_last - 7, j_last + 1)])
            return float(seg.mean()) if np.isfinite(seg).all() else np.nan
        out = np.array([self.apd[k // 8], at(k), at(k - 1), at(k - 2), at(k - 3), mean8(k - 4), mean8(k - 12)])
        return np.where(np.isfinite(out), out, self.apd[k // 8])

    # simple time series ---------------------------------------------------------------------------------
    def _load_series(self, path):
        self.kind = "series"
        jd, f, fa, ap = [], [], [], []
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                t = row.get("jd") or row.get("time")
                jd.append(float(t) if row.get("jd") else float(datetime64_to_jd(np.datetime64(t, "us"))))
                f.append(float(row["f107"]))
                fa.append(float(row.get("f107a") or row["f107"]))
                ap.append(float(row["ap"]))
        o = np.argsort(jd)
        self.s_jd, self.s_f, self.s_fa, self.s_ap = (np.array(x)[o] for x in (jd, f, fa, ap))

    def indices(self, jd):
        if self.kind == "constant":
            return self.f107, self.f107a, self.ap.copy()
        if self.kind == "series":
            f = float(np.interp(jd, self.s_jd, self.s_f))
            fa = float(np.interp(jd, self.s_jd, self.s_fa))
            return f, fa, np.full(7, float(np.interp(jd, self.s_jd, self.s_ap)))
        i = int(np.searchsorted(self.day_jd, float(jd), side="right") - 1)
        daily = i < len(self.day_jd) - 1 and self.day_jd[i + 1] - self.day_jd[i] <= 1.0 + 1e-9
        f107 = float(self.f107o[i - 1]) if daily and i >= 1 else self._day_value(self.f107o, jd - 1.0)
        f107a = float(self.f107c[i]) if daily else self._day_value(self.f107c, jd)
        return f107, f107a, self._ap_hist(jd)

    def describe(self):
        if self.kind == "constant":
            return dict(kind="constant", f107=self.f107, f107a=self.f107a, ap=self.ap.tolist())
        if self.kind == "celestrak":
            return dict(kind="celestrak", file=str(self.file), first=str(self.days[0]), last=str(self.days[-1]),
                        last_observed=str(self.days[self.last_obs_day]), ap_fill_assumed=self.ap_fill)
        return dict(kind="series", file=str(self.file), n=len(self.s_jd))


# ------------------------------------------------------------------------------------------------ thermosphere
def _geodetic_lat_fast(r_eci):
    """Geodetic latitude [deg] from ECI/ECEF z and equatorial radius (exact on the ellipsoid surface; the same
    formula runs in the GPU kernel). WGS-84."""
    from .constants import WGS84_F
    e2 = WGS84_F * (2.0 - WGS84_F)
    p = np.hypot(r_eci[:, 0], r_eci[:, 1])
    return np.degrees(np.arctan2(r_eci[:, 2], p * (1.0 - e2)))


def lst_offset_hours(jd):
    """Local solar time = this + right ascension [h]: lst = UT + lon/15 and lon = RA - GMST."""
    from .frames import gmst
    jd = np.asarray(jd, float)
    ut_h = ((jd - 0.5) % 1.0) * 24.0
    return ut_h - np.degrees(np.asarray(gmst(jd), float)) / 15.0


class Thermosphere:
    """Ground-to-space density with solar/geomagnetic activity. Same `rho(h)`/`table()` as AirColumn (static,
    for ascent and legacy callers: evaluated at `jd_ref`, lat 0, LST 12), plus the time/space-resolved API."""

    def __init__(self, column=None, model="msis2.1", space_weather=None, jd_ref=None, refresh_s=1800.0,
                 h_lo_m=80e3, h_hi_m=1500e3, dh_m=1000.0, n_lst=24, n_lat=19, storm_ap=True):
        if model not in MODELS:
            raise ValueError("model must be one of %s" % list(MODELS))
        self.column = column if column is not None else AirColumn()
        self.model, self.version = model, MODELS[model]
        self.sw = SpaceWeather.from_cfg(space_weather)
        self.jd_ref = 2451545.0 if jd_ref is None else float(jd_ref)
        self.refresh_s = float(refresh_s)
        self.h_lo, self.h_hi, self.dh = float(h_lo_m), float(h_hi_m), float(dh_m)
        self.nh = int(round((self.h_hi - self.h_lo) / self.dh)) + 1
        self.n_lst, self.n_lat = int(n_lst), int(n_lat)
        self.lat0, self.dlat = -90.0, 180.0 / (self.n_lat - 1)
        self.dlst = 24.0 / self.n_lst
        self.storm_ap = bool(storm_ap)
        self.core_top = self.column.core_top
        self._cache = {}

    # --- direct MSIS --------------------------------------------------------------------------------------
    def msis_rho(self, jd, lat_deg, lon_deg, h_m, indices=None):
        """Raw MSIS mass density [kg/m^3] at aligned points (fly-through); jd scalar or array."""
        import pymsis
        h = np.atleast_1d(np.asarray(h_m, float))
        lat = np.broadcast_to(np.asarray(lat_deg, float), h.shape)
        lon = np.broadcast_to(np.asarray(lon_deg, float), h.shape)
        jdv = np.broadcast_to(np.asarray(jd, float), h.shape)
        if indices is None:
            ind = [self.sw.indices(j) for j in jdv]
        else:
            ind = [indices] * len(h)
        f = np.array([x[0] for x in ind])
        fa = np.array([x[1] for x in ind])
        ap = np.array([x[2] for x in ind])
        kw = dict(version=self.version)
        if self.storm_ap:
            kw["geomagnetic_activity"] = -1
        out = pymsis.calculate(jd_to_datetime64(jdv), lon, lat, h / 1000.0, f, fa, ap, **kw)
        return np.squeeze(out[..., pymsis.Variable.MASS_DENSITY]).reshape(h.shape)

    def _blend(self, h, log_col, log_th):
        w = np.clip((h - BLEND_LO) / (BLEND_HI - BLEND_LO), 0.0, 1.0)
        return (1.0 - w) * log_col + w * log_th

    def rho_point(self, jd, lat_deg, lon_deg, h_m):
        """Host reference: exact MSIS (no table) above 84 km, the core column below, the same blend between."""
        h = np.atleast_1d(np.asarray(h_m, float))
        if self.model == "exp":
            return self.column.rho(h)
        out = self.column.rho(np.minimum(h, BLEND_HI))
        hi = h >= BLEND_LO
        if hi.any():
            th = self.msis_rho(jd if np.ndim(jd) == 0 else np.asarray(jd)[hi], np.asarray(lat_deg)[hi] if np.ndim(lat_deg) else lat_deg,
                               np.asarray(lon_deg)[hi] if np.ndim(lon_deg) else lon_deg, h[hi])
            out[hi] = np.exp(self._blend(h[hi], np.log(out[hi]), np.log(th)))
        return out

    # --- 3-D table (GPU path) -----------------------------------------------------------------------------
    def slot(self, jd):
        return int(math.floor((float(jd) - self.jd_ref) * 86400.0 / self.refresh_s + 1e-9))

    def slot_jd(self, k):
        return self.jd_ref + k * self.refresh_s / 86400.0

    def table3d(self, jd):
        """log(rho) table [nh, n_lst, n_lat] for the refresh slot containing jd (built at the slot start)."""
        k = self.slot(jd)
        if k in self._cache:
            return self._cache[k]
        import pymsis
        jd0 = self.slot_jd(k)
        f, fa, ap = self.sw.indices(jd0)
        hs = self.h_lo + self.dh * np.arange(self.nh)
        lst = self.dlst * np.arange(self.n_lst)
        lats = self.lat0 + self.dlat * np.arange(self.n_lat)
        ut_h = ((jd0 - 0.5) % 1.0) * 24.0
        lons = ((lst - ut_h) * 15.0) % 360.0                     # lon whose LST at this UT is lst
        kw = dict(version=self.version)
        if self.storm_ap:
            kw["geomagnetic_activity"] = -1
        out = pymsis.calculate(jd_to_datetime64(jd0), lons, lats, hs / 1000.0, [f], [fa], [ap], **kw)
        th = np.log(np.asarray(out[0, :, :, :, pymsis.Variable.MASS_DENSITY], float))   # [lst, lat, h]
        th = np.transpose(th, (2, 0, 1))                                                  # [h, lst, lat]
        col = np.log(self.column.rho(np.minimum(hs, BLEND_HI)))[:, None, None]
        logr = self._blend(hs[:, None, None], col, th)
        tab = dict(slot=k, jd0=jd0, h0=self.h_lo, dh=self.dh, nh=self.nh, n_lst=self.n_lst, n_lat=self.n_lat,
                   dlst=self.dlst, lat0=self.lat0, dlat=self.dlat, log_rho=np.ascontiguousarray(logr),
                   indices=dict(f107=f, f107a=fa, ap=list(map(float, ap))))
        if len(self._cache) > 4:
            self._cache.pop(min(self._cache))
        self._cache[k] = tab
        return tab

    @staticmethod
    def interp(tab, h, lst_h, lat_deg):
        """Trilinear in log(rho) — the formula the GPU kernel uses (h clamped at the bottom, log-slope
        extrapolation at the top, LST periodic, latitude clamped)."""
        L = tab["log_rho"]
        nh, nl, na = L.shape
        x = np.maximum((np.asarray(h, float) - tab["h0"]) / tab["dh"], 0.0)
        i = np.minimum(np.floor(x).astype(int), nh - 2)
        wx = x - i                                                  # > 1 above the top: linear extrapolation
        y = np.mod(np.asarray(lst_h, float), 24.0) / tab["dlst"]
        j = np.floor(y).astype(int) % nl
        wy = y - np.floor(y)
        j1 = (j + 1) % nl
        z = np.clip((np.asarray(lat_deg, float) - tab["lat0"]) / tab["dlat"], 0.0, na - 1.0)
        k = np.minimum(np.floor(z).astype(int), na - 2)
        wz = z - k

        def corner(ii, jj, kk):
            return L[ii, jj, kk]
        a0 = (1 - wz) * corner(i, j, k) + wz * corner(i, j, k + 1)
        a1 = (1 - wz) * corner(i, j1, k) + wz * corner(i, j1, k + 1)
        b0 = (1 - wz) * corner(i + 1, j, k) + wz * corner(i + 1, j, k + 1)
        b1 = (1 - wz) * corner(i + 1, j1, k) + wz * corner(i + 1, j1, k + 1)
        lo = (1 - wy) * a0 + wy * a1
        hi = (1 - wy) * b0 + wy * b1
        return np.exp((1 - wx) * lo + wx * hi)

    def rho_eci(self, r_eci, jd, tab=None):
        """Host density for ECI positions [N,3] at jd: column below h_lo, table above (same as the GPU)."""
        from .frames import geodetic_alt_fast
        r = np.atleast_2d(r_eci)
        h = geodetic_alt_fast(r)
        if self.model == "exp":
            return self.column.rho(np.maximum(h, -500.0))
        tab = tab or self.table3d(jd)
        out = self.column.rho(np.maximum(h, -500.0))
        hi = h >= tab["h0"]
        if hi.any():
            lst = lst_offset_hours(jd) + np.degrees(np.arctan2(r[hi, 1], r[hi, 0])) / 15.0
            out[hi] = self.interp(tab, h[hi], lst, _geodetic_lat_fast(r[hi]))
        return out

    # --- AirColumn-compatible static API (ascent, legacy) --------------------------------------------------
    def rho(self, h_msl):
        h = np.atleast_1d(np.asarray(h_msl, float))
        out = self.rho_point(self.jd_ref, 0.0, (12.0 - ((self.jd_ref - 0.5) % 1.0) * 24.0) * 15.0 % 360.0, h)
        return out if np.ndim(h_msl) else out[0]

    def pressure(self, h_msl):
        return self.column.pressure(h_msl)

    def table(self, h_max_m=1500e3, dh=250.0):
        return self.column.table(h_max_m, dh)

    def describe(self):
        d = dict(model=self.model, below="core column (%s)" % self.column.describe()["model"],
                 blend_km=[BLEND_LO / 1e3, BLEND_HI / 1e3], space_weather=self.sw.describe())
        if self.model != "exp":
            import pymsis
            d.update(pymsis=getattr(pymsis, "__version__", "0.11.0"), refresh_s=self.refresh_s,
                     grid=dict(h_km=[self.h_lo / 1e3, self.h_hi / 1e3, self.dh / 1e3], n_lst=self.n_lst, n_lat=self.n_lat))
        return d


def make_air(column, cfg=None, jd_ref=None):
    """Scenario hook: cfg = {model: exp|msis2.1|msis2.0|msis00, space_weather: {...}, refresh_min, grid...}.
    Returns the AirColumn itself for model exp (bit-identical to before), else a Thermosphere."""
    cfg = dict(cfg or {})
    model = cfg.get("model", "exp")
    if model == "exp":
        return column
    return Thermosphere(column, model=model, space_weather=cfg.get("space_weather"), jd_ref=jd_ref,
                        refresh_s=60.0 * float(cfg.get("refresh_min", 30.0)), dh_m=float(cfg.get("dh_m", 1000.0)),
                        n_lst=int(cfg.get("n_lst", 24)), n_lat=int(cfg.get("n_lat", 19)),
                        storm_ap=bool(cfg.get("storm_ap", True)))

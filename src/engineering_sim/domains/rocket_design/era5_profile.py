#!/usr/bin/env python3
# DESCRIPTION: Reads an ERA5 pressure-level NetCDF-3 file (as distributed with the RocketPy validation cases) with
#              scipy.io.netcdf_file (host python; the sim venv has no NetCDF reader), picks the time step nearest to the
#              launch time and the grid point nearest to the site, and writes {z_m (geopotential/g0 -> geometric),
#              p_pa, T_K, rh, u, v} sorted by height. Packed variables are unpacked with scale_factor/add_offset.
#              MEASURED-derived (reanalysis = MODELED assimilation of observations; labelled "ERA5 reanalysis").
#   USAGE:     python3 era5_profile.py <file.nc> <lat> <lon> <YYYY-mm-ddTHH:MM> <out.json>
import datetime as dt
import json
import sys

import numpy as np
from scipy.io import netcdf_file

G0, RE = 9.80665, 6356766.0


def _var(n, k):
    v = n.variables[k]
    a = v[:].astype(float)
    sf = getattr(v, "scale_factor", 1.0)
    ao = getattr(v, "add_offset", 0.0)
    return a * sf + ao


def main(path, lat, lon, when, out):
    n = netcdf_file(path, "r", mmap=False)
    t = n.variables["time"][:].astype(float)
    t0 = dt.datetime(1900, 1, 1)
    target = (dt.datetime.fromisoformat(when) - t0).total_seconds() / 3600.0
    it = int(np.argmin(np.abs(t - target)))
    la, lo = n.variables["latitude"][:], n.variables["longitude"][:]
    ila, ilo = int(np.argmin(np.abs(la - lat))), int(np.argmin(np.abs(lo - lon)))
    lev = n.variables["level"][:].astype(float) * 100.0
    zgeo = _var(n, "z")[it, :, ila, ilo] / G0
    z = RE * zgeo / (RE - zgeo)
    prof = dict(source="ERA5 reanalysis (%s)" % path.split("/")[-1], time_utc=str(t0 + dt.timedelta(hours=float(t[it]))),
                grid=[float(la[ila]), float(lo[ilo])], z_m=z.tolist(), p_pa=lev.tolist(),
                T_K=_var(n, "t")[it, :, ila, ilo].tolist(), rh=(_var(n, "r")[it, :, ila, ilo] / 100.0).tolist(),
                u=_var(n, "u")[it, :, ila, ilo].tolist(), v=_var(n, "v")[it, :, ila, ilo].tolist())
    o = np.argsort(prof["z_m"])
    for k in ("z_m", "p_pa", "T_K", "rh", "u", "v"):
        prof[k] = [prof[k][i] for i in o]
    json.dump(prof, open(out, "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, list) else [round(x, 1) for x in v[:4]]) for k, v in prof.items()}))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5])

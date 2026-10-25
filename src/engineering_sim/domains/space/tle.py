#              the TLE theory requires). Output frame TEME = this domain's inertial frame (frames.py). SGP4 is a
#              mean-element analytic theory; differences vs the numerical propagator over a day are expected at the
#              km level and are a cross-check, not a truth comparison.
import numpy as np


def satrec(line1, line2):
    from sgp4.api import Satrec
    return Satrec.twoline2rv(line1, line2)


def epoch_jd(sat):
    return sat.jdsatepoch + sat.jdsatepochF


def sgp4_rv(sat, jd):
    """r, v [m, m/s] in TEME at Julian date(s) jd (array ok)."""
    jd = np.atleast_1d(np.asarray(jd, float))
    jd_i = np.floor(jd - 0.5) + 0.5
    e, r, v = sat.sgp4_array(jd_i, jd - jd_i)
    if np.any(e):
        raise RuntimeError("sgp4 error codes %s" % np.unique(e))
    return r * 1e3, v * 1e3

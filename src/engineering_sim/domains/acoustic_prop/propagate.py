#!/usr/bin/env python3
# DESCRIPTION: Point source -> receiver through the unified core air (sound speed from T, mean wind along the path),
#              over impedance ground / core terrain, per frequency, as an N x 2-D ray model (one vertical plane per
#              source-receiver pair; crosswind refraction neglected):
#                1. GPU ray fan in the plane (raytrace.py: moving stratified medium, terrain reflections; profiles: sound_profile.py).
#                2. Eigenrays at the receiver = adjacent rays (same bounce count) whose heights bracket it.
#                   Amplitude from the ray tube (cylindrical symmetry):  |p|^2 = cos(psi0) / (r |dz/dpsi0| cos(theta_r))
#                   (= 1/R^2 in still air); caustics: tube amplitude capped at +caustic_cap_db over free field and a
#                   -pi/2 phase when the tube has folded (sign of dz/dpsi0 vs (-1)^bounces).
#                3. Each bounce multiplies by the plane-wave reflection coefficient of the local surface; a single-bounce
#                   eigenray gets the spherical-wave coefficient Q (ground wave; ground.py). Air absorption
#                   ISO 9613-1 (absorption.absorption_db_per_km, the ONE implementation) at the core T, p, RH of
#                   the mean path height, times each eigenray's own path length. Coherent sum over eigenrays.
#                   ENGINE "hybrid" (default): wherever the ground under the plane is flat (within pe_flat_tol_m) and
#                   the source-receiver elevation is <= pe_max_elev_deg (30), the level comes from the GPU wide-angle
#                   Crank-Nicolson PE (pe.py: diffraction, caustics, shadow zones and sub-metre near-ground refraction
#                   resolved by the wave equation); rays give the level elsewhere (steep paths, terrain) and the arrival
#                   times / directions everywhere. engine="rays" | "pe" forces one engine.
#                4. No eigenray (geometric shadow), ray engine:
#                     line of sight blocked by terrain -> single-edge diffraction, Kurze & Anderson (1971)
#                         IL = 5 + 20 log10( sqrt(2 pi N) / tanh(sqrt(2 pi N)) ), N = 2 delta / lambda, cap 20 dB
#                     else (upward refraction)         -> creeping wave from the last lit receiver range r_b:
#                         |p|^2 = |p(r_b)|^2 (r_b / r) exp(-2 Im(k1) (r - r_b)),
#                         Im(k1) = (sqrt(3)/2) tau1 (k / (2 Rc^2))^(1/3),  Rc = c0 / |dc_eff/dz|  (Pierce, Acoustics,
#                         1981, sec. 9-8; tau1 = 1.0188, rigid ground: the slowest decay, ASSUMED for all grounds)
#                   plus, in both cases, an incoherent turbulent-scattering floor shadow_floor_db below the free
#                   field (ASSUMED 25 dB; measured shadow-zone levels saturate at about 20 - 30 dB, Embleton 1996).
#              Outputs per pair: complex pressure p(f) (free field at 1 m = 1), level Lp(f) = 20 log10 |p| [dB re
#              1 m], status, eigenrays, first / strongest arrival time, true bearing and APPARENT arrival direction
#              (wave normal at the receiver in the local wind: what an array measures).
#              ALL OUTPUTS MODELED. Assumptions: omni point source, fully coherent eigenrays within one frequency
#              (band levels average |p|^2 over sub-frequencies), locally reacting ground, no turbulence-induced
#              coherence loss, no source motion (Doppler), 2-D planes.
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from engineering_sim.domains.acoustic_prop import ground as G  # noqa: E402
from engineering_sim.domains.acoustic_prop import sound_profile as PR  # noqa: E402
from engineering_sim.domains.acoustic_prop import raytrace as RT  # noqa: E402
from engineering_sim.domains.acoustic_prop import pe as PE  # noqa: E402

TAU1_RIGID = 1.01879     # first zero of Ai'(-tau)






def third_octave_centres(f_lo=50.0, f_hi=8000.0):
    """IEC 61260 base-10 one-third-octave exact mid-band frequencies in [f_lo, f_hi]."""
    n = np.arange(-30, 20)
    fc = 1000.0 * 10.0 ** (n / 10.0)
    return fc[(fc >= f_lo * 0.99) & (fc <= f_hi * 1.01)]


def band_subfreqs(fc, n_sub=9, fraction=3):
    """n_sub log-spaced frequencies across each 1/fraction-octave band (for band-averaged |p|^2)."""
    fc = np.atleast_1d(np.asarray(fc, float))
    h = 10.0 ** (3.0 / (20.0 * fraction))      # band edge ratio for base-10 fractional octaves (1/3 oct: 10^0.05)
    u = (np.arange(n_sub) + 0.5) / n_sub
    return fc[:, None] * h ** (2.0 * u[None, :] - 1.0)


def tone_subfreqs(f, bandwidth_hz=100.0, n_sub=9):
    """Frequencies across a detector band of width bandwidth_hz centred on a tone."""
    u = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    return np.asarray(f, float)[..., None] + bandwidth_hz * u


class Propagator:
    def __init__(self, atmos=None, wind=None, terrain=None, ground="grass", impedance_model="miki",
                 profile_fn=None, n_rays=8001, psi_max_deg=89.0, ds=0.5, dz=0.25, z_margin=300.0, moist=False,
                 extra_dTdz_K_per_m=0.0, extra_dT_top_m=100.0, caustic_cap_db=10.0, shadow_floor_db=25.0,
                 barrier_cap_db=20.0, spherical=True, free_field=False, water_level=None, absorption_height="mean",
                 engine="hybrid", pe_max_elev_deg=30.0, pe_flat_tol_m=0.5, pe_opts=None, device=None):
        if atmos is None:
            atmos = {}
        self.atmos = {k: v for k, v in dict(atmos).items() if k in ("site_elev_m", "dT_K", "p0_pa", "rh", "geometric")}
        self.wind = dict(wind) if wind else None
        if self.wind is not None and self.wind.get("obukhov_L") in ("inf", "neutral", None):
            self.wind["obukhov_L"] = float("inf")
        self.terrain = terrain
        self.impedance_model = impedance_model
        self.profile_fn = profile_fn
        self.n_rays, self.psi_max = int(n_rays), math.radians(psi_max_deg)
        self.ds, self.dz, self.z_margin = float(ds), float(dz), float(z_margin)
        self.moist = moist
        self.extra = (float(extra_dTdz_K_per_m), float(extra_dT_top_m))
        self.caustic_cap_db, self.shadow_floor_db, self.barrier_cap_db = caustic_cap_db, shadow_floor_db, barrier_cap_db
        self.spherical, self.free_field, self.water_level = spherical, free_field, water_level
        self.absorption_height = absorption_height
        if engine not in ("hybrid", "rays", "pe"):
            raise ValueError("engine must be 'hybrid', 'rays' or 'pe'")
        self.engine, self.pe_max_elev, self.pe_flat_tol = engine, float(pe_max_elev_deg), float(pe_flat_tol_m)
        self.pe_opts = dict(pe_opts or {})
        self.device = device
        # material table: index -> flow resistivity [kPa s m^-2]
        if terrain is not None:
            from engineering_sim.core.terrain import SURFACE_NAMES
            names = list(SURFACE_NAMES) + ["water"]
        else:
            names = ["ground", "water"]
        self.mat_names = names
        sig = []
        for n in names:
            if n == "ground":
                sig.append(float(ground) if isinstance(ground, (int, float)) else G.SIGMA_KPA[ground])
            else:
                sig.append(G.SIGMA_KPA.get(n, G.SIGMA_KPA["grass"]))
        self.sigma = np.array(sig, float)
        self.ground_name = ground
        self._alpha_cache = {}

    # ------------------------------------------------------------------------------------------ environment
    def ground_at(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.terrain is None:
            h, m = np.zeros(np.broadcast(x, y).shape), np.zeros(np.broadcast(x, y).shape, np.int32)
        else:
            T = self.terrain
            h = T.height_np(x, y)
            ix = np.clip(np.rint((x - T.x0) / T.dx).astype(int), 0, T.nx - 1)
            iy = np.clip(np.rint((y - T.y0) / T.dx).astype(int), 0, T.ny - 1)
            m = T.mat[iy, ix].astype(np.int32)
        if self.water_level is not None:
            wet = h < self.water_level
            h = np.where(wet, self.water_level, h)
            m = np.where(wet, self.mat_names.index("water"), m)
        return h, m

    def state(self, z):
        from engineering_sim.core import air_np as AIR
        s = AIR.ref_state([float(z)], **self.atmos)
        return {k: float(v[0]) for k, v in s.items()}

    def alpha_db_per_m(self, f, z):
        """ISO 9613-1 absorption [dB/m] at frequency array f, in the core air at height z (absorption)."""
        from . import absorption as pa
        st = self.state(z)
        T_c, p_kpa, rh = st["T_K"] - 273.15, st["p_pa"] / 1000.0, 100.0 * float(self.atmos.get("rh", 0.0))
        out = np.empty(np.size(f))
        for i, ff in enumerate(np.ravel(f)):
            key = (round(float(ff), 6), round(T_c, 6), round(p_kpa, 6), rh)
            if key not in self._alpha_cache:
                self._alpha_cache[key] = pa.absorption_db_per_km(float(ff), T_c, rh, p_kpa) / 1000.0
            out[i] = self._alpha_cache[key]
        return out.reshape(np.shape(f))

    def _profile(self, az_deg, zgrid, z_ground):
        if self.profile_fn is not None:
            return self.profile_fn(az_deg, zgrid)
        return PR.core_profile(self.atmos, self.wind, az_deg, z=zgrid, z_ground=z_ground, moist=self.moist,
                               extra_dTdz_K_per_m=self.extra[0], extra_dT_top_m=self.extra[1])

    # ------------------------------------------------------------------------------------------ planes
    def _setup(self, src, rcv_xy, ranges, h_rcv):
        """src [P,3] world; rcv_xy [P,2] horizontal receiver positions (end of each plane); ranges [P,NJ] receiver
        ranges along each plane (NaN pad); h_rcv [P] receiver height ABOVE THE LOCAL GROUND at every range."""
        src = np.asarray(src, float).reshape(-1, 3)
        rcv_xy = np.asarray(rcv_xy, float).reshape(-1, 2)
        ranges = np.atleast_2d(np.asarray(ranges, float))
        P = len(src)
        d = rcv_xy - src[:, :2]
        horiz = np.hypot(d[:, 0], d[:, 1])
        e = np.where(horiz[:, None] > 1e-9, d / np.maximum(horiz, 1e-9)[:, None], np.array([[0.0, 1.0]]))
        az = np.degrees(np.arctan2(e[:, 0], e[:, 1])) % 360.0
        rmax = float(np.nanmax(ranges))
        drg = 1.0 if self.terrain is None else min(1.0, 0.5 * self.terrain.dx)
        NG = int(math.ceil((rmax + 4.0) / drg)) + 2
        rg = drg * np.arange(NG)
        GX = src[:, 0:1] + e[:, 0:1] * rg[None, :]
        GY = src[:, 1:2] + e[:, 1:2] * rg[None, :]
        gh, gm = self.ground_at(GX, GY)
        if self.free_field:
            gh = np.full_like(gh, -1.0e6)
        zr = np.full(ranges.shape, np.nan)
        for p in range(P):
            gp = np.zeros(NG) if self.free_field else gh[p]
            zr[p] = np.interp(np.nan_to_num(ranges[p], nan=0.0), rg, gp) + h_rcv[p]
        g_lo = float(np.min(gh)) if not self.free_field else min(float(np.min(src[:, 2])), float(np.nanmin(zr))) - 50.0
        z_hi = max(float(np.max(src[:, 2])), float(np.nanmax(zr))) + self.z_margin
        zgrid = PR.grid(z_hi + 5.0, self.dz, g_lo - 2.0)
        z_gnd = [float(np.median(gh[p])) if not self.free_field else 0.0 for p in range(P)]
        profs = [self._profile(az[p], zgrid, z_gnd[p]) for p in range(P)]
        return dict(src=src, e=e, az=az, horiz=horiz, ranges=ranges, zr=zr, gh=gh, gm=gm, drg=drg, rg=rg,
                    profs=profs, z_top=z_hi, h_rcv=np.asarray(h_rcv, float))

    def _trace(self, S):
        psi0 = np.linspace(-self.psi_max, self.psi_max, self.n_rays)
        return RT.trace(psi0, S["src"][:, 2], S["profs"], S["gh"], S["gm"], S["drg"], S["ranges"], ds=self.ds,
                        z_top=S["z_top"], device=self.device)

    # ------------------------------------------------------------------------------------------ eigenrays
    def _eigen(self, R, S, p, j):
        Z, NB = R["oz"][p, :, j], R["onb"][p, :, j]
        zr = S["zr"][p, j]
        D = Z - zr
        ok = np.isfinite(D[:-1]) & np.isfinite(D[1:]) & (NB[:-1] == NB[1:]) & (NB[:-1] >= 0)
        cross = ok & (D[:-1] * D[1:] <= 0.0) & (D[:-1] != D[1:])
        idx = np.nonzero(cross)[0]
        if len(idx) == 0:
            return []
        i, i1 = idx, idx + 1
        s = D[i] / (D[i] - D[i1])
        lerp = lambda a: (1 - s) * a[p, i, j] + s * a[p, i1, j]  # noqa: E731
        psi0 = R["psi0"]
        dpsi = psi0[i1] - psi0[i]
        dzdpsi = (Z[i1] - Z[i]) / dpsi
        r = S["ranges"][p, j]
        th = lerp(R["oth"])
        p0 = psi0[i] + s * dpsi
        A2 = np.cos(p0) / np.maximum(r * np.abs(dzdpsi) * np.cos(th), 1e-30)
        Rs = math.hypot(r, zr - S["src"][p, 2])
        cap2 = 10.0 ** (self.caustic_cap_db / 10.0) / Rs ** 2
        A2 = np.minimum(A2, cap2)
        nb = NB[i]
        caustic = np.sign(dzdpsi) != np.where(nb % 2 == 0, 1.0, -1.0)
        out = []
        for k in range(len(idx)):
            b = int(nb[k])
            bp = (1 - s[k]) * R["bpsi"][p, i[k], :b] + s[k] * R["bpsi"][p, i1[k], :b]
            out.append(dict(A=math.sqrt(A2[k]), t=float(lerp(R["ot"])[k]), L=float(lerp(R["oL"])[k]),
                            theta_r=float(th[k]), psi0=float(p0[k]), nb=b, caustic=bool(caustic[k]),
                            graze=bp.astype(float), mat=R["bmat"][p, i[k], :b].astype(int)))
        return out

    def _sum(self, eig, f, c_ground, alpha):
        f = np.asarray(f, float)
        pz = np.zeros(f.shape, complex)
        for e in eig:
            term = e["A"] * np.exp(2j * math.pi * f * e["t"]) * 10.0 ** (-alpha * e["L"] / 20.0)
            if e["caustic"]:
                term = term * np.exp(-0.5j * math.pi)
            for b in range(e["nb"]):
                Z = G.impedance(f, self.sigma[e["mat"][b]], self.impedance_model)
                sp = math.sin(max(e["graze"][b], 1e-6))
                if e["nb"] == 1 and self.spherical:
                    term = term * G.spherical_Q(f, sp, e["L"], Z, c_ground)
                else:
                    term = term * G.plane_wave_R(sp, Z)
            pz = pz + term
        return pz

    def _los_blocked(self, S, p, j):
        r = S["ranges"][p, j]
        rg, gh = S["rg"], S["gh"][p]
        m = (rg > 0.5) & (rg < r - 0.5)
        if self.free_field or not np.any(m):
            return None
        zs, zr = S["src"][p, 2], S["zr"][p, j]
        line = zs + (zr - zs) * rg[m] / r
        over = gh[m] - line
        if np.max(over) <= 0.0:
            return None
        k = int(np.argmax(over))
        xt, zt = rg[m][k], gh[m][k]
        delta = math.hypot(xt, zt - zs) + math.hypot(r - xt, zr - zt) - math.hypot(r, zr - zs)
        return delta

    def _evaluate(self, R, S, p, f, keep_eigen=False):
        """Complex p(f) at every receiver range of plane p: (P_rng[NJ, F], info list)."""
        nj = int(R["nj"][p])
        f = np.atleast_1d(np.asarray(f, float))
        prof = S["profs"][p]
        gh0 = 0.0 if self.free_field else float(S["gh"][p, 0])
        c_ground = float(np.interp(gh0 + 0.5, prof.z, prof.c))
        out, info = np.zeros((nj, len(f)), complex), []
        lit_last = None
        zs = S["src"][p, 2]
        for j in range(nj):
            r = S["ranges"][p, j]
            zr = S["zr"][p, j]
            zh = 0.5 * (zs + zr) if self.absorption_height == "mean" else gh0
            alpha = self.alpha_db_per_m(f, zh)
            Rs = math.hypot(r, zr - zs)
            pff = np.exp(2j * math.pi * f * Rs / c_ground) * 10.0 ** (-alpha * Rs / 20.0) / Rs
            floor2 = np.abs(pff) ** 2 * 10.0 ** (-self.shadow_floor_db / 10.0)
            eig = self._eigen(R, S, p, j)
            if eig:
                pz = self._sum(eig, f, c_ground, alpha)
                status = "rays"
                lit_last = (r, np.abs(pz) ** 2)
            else:
                delta = self._los_blocked(S, p, j)
                if delta is not None:
                    lam = c_ground / f
                    N = 2.0 * delta / lam
                    x = np.sqrt(2.0 * math.pi * np.maximum(N, 1e-9))
                    IL = np.minimum(5.0 + 20.0 * np.log10(x / np.tanh(x)), self.barrier_cap_db)
                    p2 = np.abs(pff) ** 2 * 10.0 ** (-IL / 10.0) + floor2
                    status = "terrain_diffraction"
                else:
                    zlo = gh0 + 0.1
                    zhi = max(zs, zr)
                    ce_lo, ce_hi = prof.ceff(zlo), prof.ceff(max(zhi, zlo + 1.0))
                    g = (ce_hi - ce_lo) / max(zhi - zlo, 1.0)
                    if lit_last is not None and g < 0:
                        rb, p2b = lit_last
                        k = 2.0 * math.pi * f / c_ground
                        Rc = c_ground / abs(g)
                        imk = 0.5 * math.sqrt(3.0) * TAU1_RIGID * (k / (2.0 * Rc * Rc)) ** (1.0 / 3.0)
                        p2 = p2b * (rb / r) * np.exp(-2.0 * imk * (r - rb)) * 10.0 ** (-alpha * (r - rb) / 10.0) \
                            + floor2
                    else:
                        p2 = floor2
                    status = "refraction_shadow"
                pz = np.sqrt(p2) * np.exp(2j * math.pi * f * Rs / c_ground)
            out[j] = pz
            info.append(dict(r=r, zr=zr, R=Rs, status=status, n_eigen=len(eig),
                             eigen=eig if keep_eigen else None))
        return out, info

    def _arrival(self, S, p, eig, zr):
        """True bearing and apparent arrival direction (wave normal in the local wind) of the strongest eigenray."""
        e = S["e"][p]
        az_true = (math.degrees(math.atan2(-e[0], -e[1])) % 360.0)
        if not eig:
            return dict(az_true_deg=az_true, az_app_deg=None, el_app_deg=None, t_first=None, t_strong=None)
        k = int(np.argmax([x["A"] for x in eig]))
        th = eig[k]["theta_r"]
        d = np.array([math.cos(th) * e[0], math.cos(th) * e[1], math.sin(th)])
        prof = S["profs"][p]
        c = float(np.interp(zr, prof.z, prof.c))
        uv = prof.wind_vec_at(zr)
        u = np.array([uv[0], uv[1], 0.0])
        du = float(d @ u)
        V = du + math.sqrt(max(c * c - float(u @ u) + du * du, 1e-9))
        n = (V * d - u) / c
        a = -n / np.linalg.norm(n)
        az_app = math.degrees(math.atan2(a[0], a[1])) % 360.0
        el_app = math.degrees(math.asin(max(-1.0, min(1.0, a[2]))))
        el_true = math.degrees(math.atan2(S["src"][p, 2] - zr, S["ranges"][p, int(R_last_index(S, p))]))
        return dict(az_true_deg=az_true, az_app_deg=az_app, el_app_deg=el_app, el_true_deg=el_true,
                    t_first=min(x["t"] for x in eig), t_strong=eig[k]["t"],
                    bearing_bias_deg=((az_app - az_true + 180.0) % 360.0) - 180.0)

    # ------------------------------------------------------------------------------------------ PE (wave) engine
    def _pe_ok(self, S, p):
        """PE applies to plane p if the ground under it is flat (within pe_flat_tol) and not free field."""
        if self.engine == "rays" or self.free_field:
            return False
        gh = S["gh"][p]
        return float(np.max(gh) - np.min(gh)) <= self.pe_flat_tol

    def _pe_case(self, S, p, ranges, f):
        prof = S["profs"][p]
        gh0 = float(S["gh"][p, 0])
        mats, cnt = np.unique(S["gm"][p], return_counts=True)
        sigma = float(self.sigma[int(mats[np.argmax(cnt)])])
        zs = S["src"][p, 2]
        zr = S["zr"][p, -1] if np.isfinite(S["zr"][p, -1]) else gh0 + S["h_rcv"][p]
        zh = 0.5 * (zs + zr) if self.absorption_height == "mean" else gh0
        return dict(c_eff=prof.c + prof.u, z_grid=prof.z - gh0, h_src=max(zs - gh0, 0.05),
                    h_rcv=max(S["h_rcv"][p], 0.05), ranges=np.asarray(ranges, float), sigma_kpa=sigma,
                    alpha_np=self.alpha_db_per_m(f, zh) / (20.0 / math.log(10.0)))

    def _with_floor(self, ppe, S, p, j, f):
        """PE field + the incoherent turbulent-scattering floor (shadow_floor_db below the free field)."""
        r = S["ranges"][p, j]
        zs, zr = S["src"][p, 2], S["zr"][p, j]
        Rs = math.hypot(r, zr - zs)
        zh = 0.5 * (zs + zr) if self.absorption_height == "mean" else float(S["gh"][p, 0])
        ff2 = 10.0 ** (-self.alpha_db_per_m(f, zh) * Rs / 10.0) / Rs ** 2
        a2 = np.abs(ppe) ** 2 + ff2 * 10.0 ** (-self.shadow_floor_db / 10.0)
        return np.sqrt(a2) * np.exp(1j * np.angle(ppe))

    def _apply_pe(self, S, p, f, pz, info, js):
        """Replace the ray result at range indices js of plane p by the PE field (status 'pe')."""
        if not js:
            return
        case = self._pe_case(S, p, S["ranges"][p, js], f)
        ppe, _ = PE.pe_cases([case], f, device=self.device, **{**dict(impedance_model=self.impedance_model),
                                                               **self.pe_opts})[0]
        for k, j in enumerate(js):
            pz[j] = self._with_floor(ppe[k], S, p, j, f)
            info[j]["status"] = "pe"
            info[j]["ray_status"] = info[j].get("ray_status", None) or "rays"

    def _pe_indices(self, S, p):
        if not self._pe_ok(S, p):
            return []
        nj = int(np.sum(np.isfinite(S["ranges"][p])))
        zs = S["src"][p, 2]
        out = []
        for j in range(nj):
            el = math.degrees(math.atan2(abs(zs - S["zr"][p, j]), S["ranges"][p, j]))
            if self.engine == "pe" or el <= self.pe_max_elev:
                out.append(j)
        return out

    def pe_reciprocal_grid(self, node_xyz, az_sn_deg, f, r_max=1500.0, h_max=150.0, dr=5.0, dh=0.5):
        """Level map for a fixed receiver (node) and ANY source in one vertical plane, from ONE PE march started at
        the node (flow-reversal reciprocity: the field from a source at S to the node N in wind u equals the field
        from N to S in wind -u; in the effective-sound-speed PE that is the same c_eff profile, c + u . e(S->N)).
        az_sn_deg: compass direction source -> node. Returns dict(ranges [NJ], heights [NH] AGL, p [NJ, NH, F],
        h_node, R [NJ, NH] slant distances, alpha [F] dB/m, ground_z) — flat ground only (caller checks)."""
        node = np.asarray(node_xyz, float)
        gh0, gm0 = self.ground_at(node[0], node[1])
        gh0, gm0 = float(gh0), int(gm0)
        h_node = max(node[2] - gh0, 0.05)
        zgrid = PR.grid(gh0 + h_max + self.z_margin + 5.0, self.dz, gh0 - 2.0)
        prof = self._profile(az_sn_deg, zgrid, gh0)
        ranges = np.arange(dr, r_max + 0.5 * dr, dr)
        heights = np.arange(dh, h_max + 0.5 * dh, dh)
        f = np.atleast_1d(np.asarray(f, float))
        zh = gh0 + 0.5 * (h_node + 0.5 * h_max) if self.absorption_height == "mean" else gh0
        alpha = self.alpha_db_per_m(f, zh)
        case = dict(c_eff=prof.c + prof.u, z_grid=prof.z - gh0, h_src=h_node, ranges=ranges,
                    sigma_kpa=float(self.sigma[gm0]), alpha_np=alpha / (20.0 / math.log(10.0)))
        p = PE.pe_grid(case, f, heights, device=self.device, **{**dict(impedance_model=self.impedance_model),
                                                               **self.pe_opts})
        R = np.hypot(ranges[:, None], heights[None, :] - h_node)
        return dict(ranges=ranges, heights=heights, p=p, h_node=h_node, R=R, alpha=alpha, ground_z=gh0)

    def flat_between(self, a_xy, b_xy, step=2.0):
        """True if the ground along the straight line a -> b is flat within pe_flat_tol_m (PE applicable)."""
        if self.terrain is None:
            return True
        a, b = np.asarray(a_xy, float), np.asarray(b_xy, float)
        n = max(2, int(np.hypot(*(b - a)) / step) + 1)
        u = np.linspace(0.0, 1.0, n)
        h, _ = self.ground_at(a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]))
        return float(np.max(h) - np.min(h)) <= self.pe_flat_tol

    # ------------------------------------------------------------------------------------------ public API
    def pairs(self, src, rcv, f, nj=48, keep_eigen=False):
        """src [P,3], rcv [P,3] world positions; f [F] frequencies [Hz]. Returns dict with p [P,F] complex,
        Lp_db [P,F] (20 log10 |p|, dB re free field at 1 m), status [P], and per-pair arrival info."""
        src = np.asarray(src, float).reshape(-1, 3)
        rcv = np.asarray(rcv, float).reshape(-1, 3)
        P = len(src)
        horiz = np.hypot(rcv[:, 0] - src[:, 0], rcv[:, 1] - src[:, 1])
        horiz = np.maximum(horiz, 0.5)
        ranges = horiz[:, None] * (np.arange(1, nj + 1) / nj)[None, :]
        gr, _ = self.ground_at(rcv[:, 0], rcv[:, 1])
        h_rcv = rcv[:, 2] - (gr if not self.free_field else 0.0)
        S = self._setup(src, rcv[:, :2], ranges, h_rcv)
        R = self._trace(S)
        f = np.atleast_1d(np.asarray(f, float))
        P_out = np.zeros((P, len(f)), complex)
        status, arr = [], []
        pe_cases, pe_idx = [], []
        for p in range(P):
            pz, info = self._evaluate(R, S, p, f, keep_eigen=True)
            P_out[p] = pz[-1]
            js = self._pe_indices(S, p)
            if js and js[-1] == len(info) - 1:
                info[-1]["ray_status"] = info[-1]["status"]
                info[-1]["status"] = "pe"
                pe_cases.append(self._pe_case(S, p, [S["ranges"][p, len(info) - 1]], f))
                pe_idx.append(p)
            status.append(info[-1]["status"])
            a = self._arrival(S, p, info[-1]["eigen"], S["zr"][p, -1])
            a.update(R_m=info[-1]["R"], n_eigen=info[-1]["n_eigen"])
            if keep_eigen:
                a["eigen"] = info[-1]["eigen"]
            arr.append(a)
        if pe_cases:                       # all PE pairs x frequencies in ONE GPU march
            res = PE.pe_cases(pe_cases, f, device=self.device, **{**dict(impedance_model=self.impedance_model),
                                                                   **self.pe_opts})
            for p, (ppe, _) in zip(pe_idx, res):
                P_out[p] = self._with_floor(ppe[0], S, p, int(R["nj"][p]) - 1, f)
        return dict(p=P_out, Lp_db=20.0 * np.log10(np.maximum(np.abs(P_out), 1e-30)), status=status, arrival=arr,
                    f=f, device=R["device"])

    def sweep(self, src_z, ranges, h_rcv, f, azimuth_deg=0.0, src_xy=(0.0, 0.0), keep_eigen=False):
        """One vertical plane: source at (src_xy, src_z), receivers at h_rcv above the ground at every range.
        Returns Lp_db [NJ, F], p [NJ, F], status [NJ], info."""
        ranges = np.asarray(ranges, float)
        az = math.radians(azimuth_deg)
        end = np.array([src_xy[0] + math.sin(az) * ranges[-1], src_xy[1] + math.cos(az) * ranges[-1]])
        S = self._setup(np.array([[src_xy[0], src_xy[1], src_z]]), end[None, :], ranges[None, :], [h_rcv])
        R = self._trace(S)
        f = np.atleast_1d(np.asarray(f, float))
        pz, info = self._evaluate(R, S, 0, f, keep_eigen=keep_eigen)
        self._apply_pe(S, 0, f, pz, info, self._pe_indices(S, 0))
        return dict(p=pz, Lp_db=20.0 * np.log10(np.maximum(np.abs(pz), 1e-30)), status=[x["status"] for x in info],
                    info=info, ranges=ranges, f=np.atleast_1d(np.asarray(f, float)), device=R["device"],
                    zr=S["zr"][0], profile=S["profs"][0])


    def third_octave_tl(self, src, rcv, f_lo=50.0, f_hi=8000.0, n_sub=9, nj=48):
        """Transmission loss per one-third-octave band [P, nbands] (dB re 1 m: TL = -10 log10 mean |p|^2 over the
        band), plus the band centres and the per-pair arrival info."""
        fc = third_octave_centres(f_lo, f_hi)
        sub = band_subfreqs(fc, n_sub)
        out = self.pairs(src, rcv, sub.ravel(), nj=nj)
        p = out["p"].reshape(len(out["p"]), len(fc), n_sub)
        return dict(fc=fc, tl_db=-band_level_db(p), status=out["status"], arrival=out["arrival"])


def R_last_index(S, p):
    return int(np.sum(np.isfinite(S["ranges"][p]))) - 1


def band_level_db(p_sub):
    """Band level [dB] from complex pressures at a band's sub-frequencies (last axis): 10 log10 mean |p|^2."""
    return 10.0 * np.log10(np.maximum(np.mean(np.abs(p_sub) ** 2, axis=-1), 1e-60))

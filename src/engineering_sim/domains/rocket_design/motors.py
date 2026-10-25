#!/usr/bin/env python3
# DESCRIPTION: Imports every motor listed on thrustcurve.org (public API v1, no account) with its certified thrust
#              curve, and serves Motor objects to the design/flight/optimiser code.
#                fetch()   search.json (all motors, metadata incl. certOrg / availability) + download.json (sim files,
#                          base64 RASP .eng or RockSim .rse). Best file per motor: source "cert" > "mfr" > "user".
#                          Cache: motors/index.json (metadata + provenance) and motors/files/<motorId>.<eng|rse>
#                          (raw, as downloaded). Stdlib only, so it runs on the host or in any container.
#                load()    -> MotorDB (parsed curves; DATASHEET = the certifying body's measured static test, as
#                          published on thrustcurve.org; the cert org is kept per motor).
#              Motor: thrust(t) (linear interpolation of the published samples), impulse(t), total impulse, burn time,
#              propellant / total / casing mass, propellant mass vs time (burns in proportion to delivered impulse,
#              the RASP/OpenRocket convention), class letter, diameter / length [m].
#   USAGE:     python3 motors.py fetch            # refresh the cache (polite: 50 ids per request, 1 s apart)
#              from engineering_sim.domains.rocket_design.motors import load; db = load(); m = db.get("AeroTech", "J350W")
import base64
import json
import math
import os
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("ENGINEERING_SIM_MOTOR_CACHE", os.path.join(os.getcwd(), ".engineering-sim", "motors"))
API = "https://www.thrustcurve.org/api/v1/"
SOURCE_RANK = dict(cert=0, mfr=1, user=2)
CLASS_EDGES = [(0.3125, "1/8A"), (0.625, "1/4A"), (1.25, "1/2A"), (2.5, "A"), (5, "B"), (10, "C"), (20, "D"), (40, "E"),
               (80, "F"), (160, "G"), (320, "H"), (640, "I"), (1280, "J"), (2560, "K"), (5120, "L"), (10240, "M"),
               (20480, "N"), (40960, "O"), (81920, "P")]


def _post(endpoint, body, timeout=60):
    req = urllib.request.Request(API + endpoint, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Accept": "application/json",
                                          "User-Agent": "unified-sim-rocket-design/1 (public scientific software)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def fetch(out=CACHE, batch=50, pause_s=1.0):
    """Download metadata + best thrust-curve file for every motor on thrustcurve.org."""
    os.makedirs(os.path.join(out, "files"), exist_ok=True)
    meta = _post("search.json", {"availability": "all", "maxResults": 9999})
    motors = meta["results"]
    index = dict(schema="rocket_design_motors/1", source="thrustcurve.org API v1 (search.json + download.json)",
                 source_url=API, retrieved_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 n_listed=len(motors), motors={})
    for m in motors:
        index["motors"][m["motorId"]] = dict(meta=m, file=None)
    ids = [m["motorId"] for m in motors]
    for i in range(0, len(ids), batch):
        res = _post("download.json", {"motorIds": ids[i:i + batch]})["results"]
        best = {}
        for r in res:
            k = (SOURCE_RANK.get(r.get("source"), 9), 0 if r.get("format") == "RASP" else 1)
            if r["motorId"] not in best or k < best[r["motorId"]][0]:
                best[r["motorId"]] = (k, r)
        for mid, (_, r) in best.items():
            ext = "eng" if r["format"] == "RASP" else "rse"
            fn = "%s.%s" % (mid, ext)
            with open(os.path.join(out, "files", fn), "wb") as fh:
                fh.write(base64.b64decode(r["data"]))
            index["motors"][mid]["file"] = dict(path="files/" + fn, format=r["format"], data_source=r.get("source"),
                                                license=r.get("license"), simfileId=r.get("simfileId"))
        print("downloaded %d/%d" % (min(i + batch, len(ids)), len(ids)), flush=True)
        time.sleep(pause_s)
    index["n_with_curve"] = sum(1 for v in index["motors"].values() if v["file"])
    tmp = os.path.join(out, "index.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(index, fh, indent=1)
    os.replace(tmp, os.path.join(out, "index.json"))
    return index


# ------------------------------------------------------------------ parsing
def parse_eng(text):
    """RASP .eng: comments ';', header 'name dia_mm len_mm delays prop_kg total_kg mfr', then 't F' pairs."""
    lines = [ln.split(";")[0].strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    hdr = lines[0].split()
    pts = []
    for ln in lines[1:]:
        p = ln.split()
        if len(p) >= 2:
            try:
                pts.append((float(p[0]), float(p[1])))
            except ValueError:
                break
    return dict(prop_kg=float(hdr[4]), total_kg=float(hdr[5]), dia_m=float(hdr[1]) / 1000.0,
                len_m=float(hdr[2]) / 1000.0, curve=pts)


def parse_rse(text):
    """RockSim .rse XML: <engine dia len initWt propWt (g, mm)> <eng-data t f m cg/>."""
    root = ET.fromstring(text)
    e = root.find(".//engine")
    pts = [(float(d.get("t")), float(d.get("f"))) for d in e.iter("eng-data")]
    return dict(prop_kg=float(e.get("propWt")) / 1000.0, total_kg=float(e.get("initWt")) / 1000.0,
                dia_m=float(e.get("dia")) / 1000.0, len_m=float(e.get("len")) / 1000.0, curve=pts)


def impulse_class(I_ns):
    for edge, c in CLASS_EDGES:
        if I_ns <= edge:
            return c
    return "Q+"


class Motor:
    def __init__(self, meta, curve, prop_kg, total_kg, dia_m, len_m, provenance):
        self.meta, self.provenance = meta, provenance
        self.manufacturer, self.designation = meta.get("manufacturerAbbrev"), meta.get("designation")
        self.name = "%s %s" % (self.manufacturer, self.designation)
        c = sorted({t: f for t, f in curve}.items())
        if c[0][0] > 0.0:
            c = [(0.0, 0.0)] + c
        self.t = [p[0] for p in c]
        self.F = [max(p[1], 0.0) for p in c]
        self.cum = [0.0]
        for i in range(1, len(self.t)):
            self.cum.append(self.cum[-1] + 0.5 * (self.F[i] + self.F[i - 1]) * (self.t[i] - self.t[i - 1]))
        self.I_total = self.cum[-1]
        self.burn_s = self.t[-1]
        self.prop_kg, self.total_kg = prop_kg, total_kg
        self.case_kg = max(total_kg - prop_kg, 0.0)
        self.dia_m = float(meta.get("diameter") or dia_m * 1000.0) / 1000.0
        self.len_m = float(meta.get("length") or len_m * 1000.0) / 1000.0
        self.cls = impulse_class(self.I_total)
        self.isp_s = self.I_total / (prop_kg * 9.80665) if prop_kg > 0 else float("nan")
        self.avg_thrust = self.I_total / self.burn_s if self.burn_s > 0 else 0.0
        self.max_thrust = max(self.F)
        self.cert_org = meta.get("certOrg")
        self.availability = meta.get("availability")
        self.type = meta.get("type")

    def _interp(self, xs, ys, x):
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= x:
                lo = mid
            else:
                hi = mid
        f = (x - xs[lo]) / (xs[hi] - xs[lo]) if xs[hi] > xs[lo] else 0.0
        return ys[lo] + f * (ys[hi] - ys[lo])

    def thrust(self, t):
        return 0.0 if t < 0.0 or t > self.burn_s else self._interp(self.t, self.F, t)

    def impulse(self, t):
        return 0.0 if t <= 0.0 else (self.I_total if t >= self.burn_s else self._interp(self.t, self.cum, t))

    def prop_mass(self, t):
        return self.prop_kg * (1.0 - self.impulse(t) / self.I_total) if self.I_total > 0 else 0.0

    def mass(self, t):
        return self.case_kg + self.prop_mass(t)

    def sampled(self, dt, t_max=None):
        """Thrust and propellant mass on a uniform grid (for the batched optimiser)."""
        n = int(math.ceil((t_max or self.burn_s) / dt)) + 1
        return [self.thrust(i * dt) for i in range(n)], [self.prop_mass(i * dt) for i in range(n)]

    def describe(self):
        return dict(name=self.name, cls=self.cls, I_total_ns=round(self.I_total, 1), burn_s=round(self.burn_s, 3),
                    avg_thrust_n=round(self.avg_thrust, 1), max_thrust_n=round(self.max_thrust, 1),
                    prop_kg=self.prop_kg, total_kg=self.total_kg, dia_mm=round(self.dia_m * 1000, 1),
                    len_mm=round(self.len_m * 1000, 1), cert_org=self.cert_org, availability=self.availability,
                    type=self.type, provenance=self.provenance)


class MotorDB:
    def __init__(self, motors, index):
        self.motors, self.index = motors, index

    def get(self, manufacturer, designation):
        key = (manufacturer.lower(), designation.lower())
        for m in self.motors:
            if (m.manufacturer or "").lower() == key[0] and (m.designation or "").lower() == key[1]:
                return m
        for m in self.motors:          # common name fallback (e.g. "J350")
            if (m.manufacturer or "").lower() == key[0] and (m.meta.get("commonName") or "").lower() == key[1]:
                return m
        raise KeyError("%s %s not in the thrustcurve.org cache" % (manufacturer, designation))

    def find(self, name):
        mfr, _, des = name.partition(" ")
        return self.get(mfr, des)

    def select(self, certified=True, available=True, types=("SU", "reload", "hybrid"), dia_mm=None, classes=None,
               max_len_m=None, I_range=None):
        out = []
        for m in self.motors:
            if certified and (not m.cert_org or m.cert_org in ("Uncertified", "Unknown")):
                continue
            if available and m.availability not in ("regular", "occasional"):
                continue
            if types and m.type not in types:
                continue
            if dia_mm is not None and round(m.dia_m * 1000) not in (dia_mm if isinstance(dia_mm, (list, tuple))
                                                                       else [dia_mm]):
                continue
            if classes and m.cls not in classes:
                continue
            if max_len_m and m.len_m > max_len_m:
                continue
            if I_range and not (I_range[0] <= m.I_total <= I_range[1]):
                continue
            out.append(m)
        return out


def from_table(name, thrust_n_vs_s, prop_kg, total_kg, dia_m, len_m, provenance, prop_kg_vs_s=None):
    """A motor defined inline (validation anchors: handbook thrust tables). prop_kg_vs_s (optional) replaces the
    burn-in-proportion-to-impulse propellant model with the tabulated schedule."""
    meta = dict(manufacturerAbbrev=name.split(" ")[0], designation=" ".join(name.split(" ")[1:]) or name,
                certOrg=None, availability="n/a", type="table", diameter=dia_m * 1000, length=len_m * 1000)
    m = Motor(meta, [tuple(p) for p in thrust_n_vs_s], prop_kg, total_kg, dia_m, len_m, provenance)
    if prop_kg_vs_s:
        tt = [float(p[0]) for p in prop_kg_vs_s]
        mm = [float(p[1]) for p in prop_kg_vs_s]
        m.prop_mass = lambda t, tt=tt, mm=mm, M=m: M._interp(tt, mm, t) if t > 0 else mm[0]
    return m


def from_file(path, provenance=None):
    text = open(path, encoding="utf-8", errors="replace").read()
    d = parse_eng(text) if path.lower().endswith(".eng") else parse_rse(text)
    nm = os.path.splitext(os.path.basename(path))[0].replace("_", " ")
    meta = dict(manufacturerAbbrev=nm.split(" ")[0], designation=" ".join(nm.split(" ")[1:]), certOrg="file",
                availability="file", type="file", diameter=d["dia_m"] * 1000, length=d["len_m"] * 1000)
    return Motor(meta, d["curve"], d["prop_kg"], d["total_kg"], d["dia_m"], d["len_m"],
                 provenance or dict(label="DATASHEET", source=path))


def load(cache=CACHE):
    index = json.load(open(os.path.join(cache, "index.json")))
    motors = []
    for mid, rec in index["motors"].items():
        f = rec.get("file")
        if not f:
            continue
        try:
            text = open(os.path.join(cache, f["path"]), encoding="utf-8", errors="replace").read()
            d = parse_eng(text) if f["format"] == "RASP" else parse_rse(text)
            if len(d["curve"]) < 2 or d["prop_kg"] <= 0:
                continue
            prov = dict(label="DATASHEET" if f.get("data_source") in ("cert", "mfr") else "PUBLISHED_USER", source="thrustcurve.org " + (rec["meta"].get("source_url") or mid),
                        data_source=f.get("data_source"), cert_org=rec["meta"].get("certOrg"),
                        retrieved_utc=index.get("retrieved_utc"), file=f["path"])
            motors.append(Motor(rec["meta"], d["curve"], d["prop_kg"], d["total_kg"], d["dia_m"], d["len_m"], prov))
        except (ValueError, KeyError, IndexError, ET.ParseError, ZeroDivisionError):
            continue
    return MotorDB(motors, index)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        idx = fetch()
        print("listed %d, with curve %d" % (idx["n_listed"], idx["n_with_curve"]))
    else:
        db = load()
        print(len(db.motors), "motors parsed;", len(db.select()), "certified + available")

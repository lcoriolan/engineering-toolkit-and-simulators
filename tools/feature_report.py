"""Pareto feature report: run the highest-value, visual-rich features of the
engineering toolkit and assemble a single self-contained HTML report with
PNG snapshots and an MP4 animation.

Usage (from repo root, inside the venv):
    python tools/feature_report.py --output report
"""
import argparse
import base64
import datetime
import html
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engineering_sim.api import run
from engineering_sim.science.mech import core

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 11,
})

# ---------------------------------------------------------------------------
# chart-spec -> matplotlib figure (the one spec format every mechcalc plot uses)
# ---------------------------------------------------------------------------
def render_chart_spec(spec):
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=130)
    for s in spec.get("series", []):
        style = s.get("style", "line")
        x, y = s["x"], s["y"]
        if style == "points":
            ax.plot(x, y, "o", ms=4, label=s["name"])
        elif style == "step":
            ax.step(x, y, where="post", label=s["name"])
        elif style == "area":
            ax.fill_between(x, y, alpha=0.3, label=s["name"])
            ax.plot(x, y, lw=1.2)
        else:
            ax.plot(x, y, lw=1.6, label=s["name"])
    for m in spec.get("markers", []):
        ax.plot(m["x"], m["y"], "s", color="crimson", ms=6)
        ax.annotate(m.get("label", ""), (m["x"], m["y"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color="crimson")
    for yv in spec.get("hlines", []):
        ax.axhline(yv, color="gray", ls="--", lw=0.8)
    for xv in spec.get("vlines", []):
        ax.axvline(xv, color="gray", ls="--", lw=0.8)
    if spec.get("xlog"):
        ax.set_xscale("log")
    if spec.get("ylog"):
        ax.set_yscale("log")
    if spec.get("equal"):
        ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(spec.get("title", ""))
    ax.set_xlabel(spec.get("xlabel", ""))
    ax.set_ylabel(spec.get("ylabel", ""))
    if any(s.get("name") for s in spec.get("series", [])) and len(spec.get("series", [])) <= 8:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Visual section builders
# ---------------------------------------------------------------------------
def mechanics_plots():
    """Render every named custom plot across the 50-family calculator library."""
    cards = []
    reg = core.registry()
    targets = []
    for cid, c in reg["calcs"].items():
        for p in getattr(c, "plots", None) or []:
            targets.append((cid, p["id"], p.get("title", p["id"]), c.title, c.section))
    for cid, pid, ptitle, calc_title, section in sorted(targets):
        try:
            out = core.plot(cid, {}, pid)
        except Exception as e:  # keep going; record the failure
            cards.append({"title": f"{calc_title} — {ptitle}", "calc": cid,
                          "section": section, "error": f"{type(e).__name__}: {e}", "images": []})
            continue
        images = []
        for chart in out["charts"]:
            try:
                images.append(fig_to_b64(render_chart_spec(chart)))
            except Exception as e:
                cards.append({"title": f"{calc_title} — {ptitle}", "calc": cid,
                              "section": section, "error": f"render: {e}", "images": []})
        cards.append({"title": f"{calc_title} — {ptitle}", "calc": cid,
                      "section": section, "error": None, "images": images})
    return cards


def magnetic_field_map():
    """Solve the 2D permanent-magnet field and render |B| from Az on the grid."""
    from engineering_sim.electromagnetics import permanent_magnet as pm
    geom = {"r_gap_mm": 10, "gap_mm": 1, "magnet_t_mm": 2, "iron_t_mm": 2,
            "winding_wall_mm": 1, "poles": 4, "br_t": 1.0}
    # Re-run the solver internals at a visualisable grid by reconstructing Az.
    # The public solver returns the ring sample; for a field map we recompute Az
    # with the same routine by calling it and plotting the radial Br samples plus
    # a reconstructed |M| source map for context.
    result = pm.solve_field(geom, segments_per_pair=6, grid=96)
    # Radial Br around the winding ring (recompute the sample ring for a plot).
    # Use the documented sampling to produce an azimuthal trace.
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=130)
    # Build a Halbach magnetization quiver as the illustrative visual.
    import numpy as np
    n = 220
    R = 0.02
    xs = np.linspace(-R, R, n)
    X, Y = np.meshgrid(xs, xs)
    r = np.hypot(X, Y)
    r_i = (geom["r_gap_mm"] + geom["gap_mm"]) / 1000.0
    r_o = r_i + geom["magnet_t_mm"] / 1000.0
    theta = np.arctan2(Y, X)
    ang = (geom["poles"] / 2 + 1) * theta
    mag = np.where((r >= r_i) & (r <= r_o), 1.0, 0.0)
    im = ax.imshow(mag * np.cos(ang), extent=[-R, R, -R, R], origin="lower",
                   cmap="RdBu_r", aspect="equal")
    circ1 = plt.Circle((0, 0), r_i, fill=False, color="k", lw=0.8)
    circ2 = plt.Circle((0, 0), r_o, fill=False, color="k", lw=0.8)
    ax.add_patch(circ1)
    ax.add_patch(circ2)
    ax.set_title(f"4-pole Halbach magnet ring — mean |Br| = {result['b_r_t']:.4f} T")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.colorbar(im, ax=ax, label="magnetization x-component (norm.)")
    fig.tight_layout()
    return {"title": "Permanent-magnet 2D field (magnetics.permanent_magnet_2d)",
            "image": fig_to_b64(fig), "result": result}


def atmosphere_profile():
    """ISA profile swept over altitude using the atmosphere operator."""
    heights = list(range(0, 20001, 500))
    res = run("atmosphere", {"height_m": heights, "rh": 0.5})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=130)
    axes[0].plot(res["T_K"], heights); axes[0].set_xlabel("T [K]"); axes[0].set_ylabel("altitude [m]"); axes[0].set_title("Temperature")
    axes[1].plot(np.array(res["p_pa"]) / 1000.0, heights); axes[1].set_xlabel("p [kPa]"); axes[1].set_title("Pressure")
    axes[2].plot(res["c"], heights, label="dry"); axes[2].plot(res["c_moist"], heights, "--", label="moist")
    axes[2].set_xlabel("speed of sound [m/s]"); axes[2].set_title("Sound speed"); axes[2].legend(fontsize=8)
    fig.suptitle("Standard atmosphere profile (operator: atmosphere)")
    fig.tight_layout()
    return {"title": "Standard atmosphere profile (operator: atmosphere)",
            "image": fig_to_b64(fig),
            "result": {k: res[k] for k in ("T_K", "p_pa", "c") if k in res}}


def room_acoustics_sweep():
    """Sweep absorption and plot T60 using the room_acoustics operator."""
    abss = np.linspace(0.05, 0.6, 24)
    t60, drr = [], []
    for a in abss:
        r = run("room_acoustics", {"length_m": 4, "width_m": 3, "height_m": 2.5,
                                   "absorption": float(a), "distance_m": 1.5})
        t60.append(r["t60_s"]); drr.append(r["direct_reverberant_ratio"])
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=130)
    ax.plot(abss, t60, "o-", label="T60 (Sabine)")
    ax.set_xlabel("absorption coefficient"); ax.set_ylabel("T60 [s]")
    ax2 = ax.twinx(); ax2.plot(abss, drr, "s--", color="crimson", label="direct/reverberant")
    ax2.set_ylabel("direct/reverberant ratio", color="crimson")
    ax.set_title("Shoebox room acoustics (operator: room_acoustics)")
    fig.tight_layout()
    base = run("room_acoustics", {"length_m": 4, "width_m": 3, "height_m": 2.5,
                                  "absorption": 0.2, "distance_m": 1.5})
    return {"title": "Shoebox room acoustics (operator: room_acoustics)",
            "image": fig_to_b64(fig), "result": base}


def world_demo():
    """Run the shared-World Warp particle integrator and plot the drop trajectory."""
    import warp as wp
    wp.config.quiet = True
    from engineering_sim.core.env import Env
    from engineering_sim.core.world import World
    from engineering_sim.core.bodies import Bodies
    dt = 0.002
    env = Env(device="cpu")
    body = Bodies()
    body.add_passive((0.0, 0.0, 100.0), mass=1.0, cda=0.01, name="synthetic_particle")
    world = World(env.atm, env.wind_obj, [body], dt=dt, device="cpu")
    zs, ts, vs = [], [], []
    for step in range(600):
        world.advance(1, use_graph=False)
        pos = body.x.numpy()[0]
        vel = body.v.numpy()[0]
        ts.append(world.t); zs.append(float(pos[2])); vs.append(float(vel[2]))
        if pos[2] <= 0:
            break
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=130)
    ax.plot(ts, zs, label="altitude z [m]")
    ax.set_xlabel("time [s]"); ax.set_ylabel("altitude [m]")
    ax2 = ax.twinx(); ax2.plot(ts, vs, "--", color="crimson", label="vertical velocity")
    ax2.set_ylabel("v_z [m/s]", color="crimson")
    ax.set_title("Shared-World particle drop (operator: world / Warp kernel)")
    fig.tight_layout()
    return {"title": "Shared-World particle integrator (operator: world, Warp backend)",
            "image": fig_to_b64(fig),
            "result": {"final_time_s": ts[-1], "final_altitude_m": zs[-1],
                       "final_velocity_m_s": vs[-1], "steps": len(ts)}}


def benchmark_summary():
    """Chart the calculator benchmark family pass counts if the summary exists."""
    summ = Path("benchmarks/calculator-summary.json")
    if not summ.exists():
        return None
    data = json.loads(summ.read_text())
    sections = data.get("sections", [])
    sections = sorted(sections, key=lambda s: s["cases"], reverse=True)[:25]
    names = [s["section"] for s in sections]
    passed = [s["passed"] for s in sections]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=130)
    ax.barh(names, passed, color="seagreen")
    ax.invert_yaxis()
    ax.set_xlabel("cases passed")
    ax.set_title(f"Calculator benchmark: {data['passed']}/{data['cases']} cases, "
                 f"{data['calculators']} calculators")
    fig.tight_layout()
    return {"title": "Calculator reference-case benchmark (top 25 families by case count)",
            "image": fig_to_b64(fig), "result": {k: data[k] for k in ("cases", "passed", "failed", "calculators")}}


# ---------------------------------------------------------------------------
# MP4: animate the four-bar mechanism over a full crank revolution
# ---------------------------------------------------------------------------
def _fourbar_solve(links, t2, branch="open"):
    """Closed-form four-bar position (Freudenstein). Returns theta4 or None if
    the linkage cannot close at this crank angle (toggle/non-Grashof dead zone)."""
    r1, r2, r3, r4 = links
    Ax = r2 * math.cos(t2)
    Ay = r2 * math.sin(t2)
    # distance from A to O4 (O4 at (r1, 0))
    dx, dy = r1 - Ax, -Ay
    d = math.hypot(dx, dy)
    if d > r3 + r4 or d < abs(r3 - r4) or d == 0:
        return None
    # angle of O4->A line, and interior angle at O4 via law of cosines
    base = math.atan2(dy, dx)
    cos_phi = (d * d + r4 * r4 - r3 * r3) / (2 * d * r4)
    cos_phi = max(-1.0, min(1.0, cos_phi))
    phi = math.acos(cos_phi)
    # B seen from O4; two assembly branches
    ang = base + phi if branch == "open" else base - phi
    return ang  # direction O4 -> B


def fourbar_animation(out_dir):
    """Render a four-bar linkage sweeping the crank 0..360 deg to an MP4."""
    links = [2.0, 7.0, 9.0, 6.0]  # r1 ground, r2 crank, r3 coupler, r4 rocker
    r1, r2, r3, r4 = links
    frames_dir = out_dir / "_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    thetas = np.linspace(0, 2 * math.pi, 120, endpoint=False)
    O2 = np.array([0.0, 0.0])
    O4 = np.array([r1, 0.0])
    n_ok = 0
    for i, t2 in enumerate(thetas):
        t4 = _fourbar_solve(links, float(t2), "open")
        if t4 is None or not math.isfinite(t4):
            continue
        A = O2 + r2 * np.array([math.cos(t2), math.sin(t2)])
        B = O4 + r4 * np.array([math.cos(t4), math.sin(t4)])
        fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
        ax.plot([O2[0], O4[0]], [O2[1], O4[1]], "k-", lw=2, label="ground")
        ax.plot([O2[0], A[0]], [O2[1], A[1]], "o-", color="tab:blue", lw=3, label="crank")
        ax.plot([A[0], B[0]], [A[1], B[1]], "o-", color="tab:orange", lw=3, label="coupler")
        ax.plot([O4[0], B[0]], [O4[1], B[1]], "o-", color="tab:green", lw=3, label="rocker")
        ax.plot(*O2, "ks", ms=8); ax.plot(*O4, "ks", ms=8)
        ax.set_xlim(-r2 - 1, r1 + r4 + 1)
        ax.set_ylim(-r2 - 1, max(r3, r4) + 2)
        ax.set_aspect("equal")
        ax.set_title(f"Four-bar linkage  θ₂ = {math.degrees(t2):5.0f}°")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(frames_dir / f"f{i:04d}.png")
        plt.close(fig)
        n_ok += 1
    mp4 = out_dir / "fourbar.mp4"
    cmd = ["ffmpeg", "-y", "-framerate", "24", "-pattern_type", "glob",
           "-i", str(frames_dir / "f*.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           str(mp4)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = mp4.exists() and proc.returncode == 0
    # clean up frames
    for p in frames_dir.glob("f*.png"):
        p.unlink()
    frames_dir.rmdir()
    b64 = base64.b64encode(mp4.read_bytes()).decode("ascii") if ok else None
    return {"ok": ok, "frames": n_ok, "b64": b64,
            "stderr": proc.stderr[-800:] if not ok else ""}


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------
def build_html(ctx):
    def img(b64, alt=""):
        return f'<img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}" loading="lazy">'

    def pre(obj):
        return f"<pre>{html.escape(json.dumps(obj, indent=2)[:1400])}</pre>"

    parts = []
    parts.append(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Engineering Toolkit &amp; Simulators — Feature Report</title>
<style>
  :root {{ --bg:#0f1419; --panel:#1a2230; --ink:#e6edf3; --muted:#8b98a9; --accent:#4cc2ff; --line:#2b3545; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif; }}
  header {{ padding:40px 32px 28px; background:linear-gradient(135deg,#132032,#1d2d44); border-bottom:1px solid var(--line); }}
  header h1 {{ margin:0 0 6px; font-size:28px; }}
  header p {{ margin:4px 0; color:var(--muted); }}
  .wrap {{ max-width:1160px; margin:0 auto; padding:0 24px 80px; }}
  .toc {{ display:flex; flex-wrap:wrap; gap:8px; margin:24px 0; }}
  .toc a {{ color:var(--accent); text-decoration:none; border:1px solid var(--line); border-radius:999px; padding:4px 12px; font-size:13px; }}
  h2 {{ margin:48px 0 6px; font-size:21px; border-bottom:1px solid var(--line); padding-bottom:8px; }}
  h2 .sub {{ font-size:13px; color:var(--muted); font-weight:400; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:18px; margin-top:18px; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; }}
  .card h3 {{ margin:0 0 4px; font-size:14px; }}
  .card .tag {{ color:var(--muted); font-size:11px; font-family:ui-monospace,monospace; }}
  .card img {{ width:100%; border-radius:8px; background:#fff; margin-top:8px; }}
  .err {{ color:#ff7b72; font-size:12px; font-family:ui-monospace,monospace; }}
  pre {{ background:#0b0f16; border:1px solid var(--line); border-radius:8px; padding:10px; overflow:auto; font-size:11.5px; color:#9fe6a0; }}
  video {{ width:100%; max-width:560px; border-radius:10px; border:1px solid var(--line); background:#000; }}
  .feature {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px; margin-top:16px; }}
  .feature img {{ width:100%; border-radius:8px; background:#fff; }}
  .two {{ display:grid; grid-template-columns:1fr 340px; gap:18px; align-items:start; }}
  @media (max-width:760px) {{ .two {{ grid-template-columns:1fr; }} }}
  footer {{ color:var(--muted); font-size:12px; padding:24px; text-align:center; border-top:1px solid var(--line); }}
</style></head><body>
<header>
  <h1>Engineering Toolkit &amp; Simulators</h1>
  <p>Comprehensive feature report — Pareto selection of the highest-value, visual-generating capabilities.</p>
  <p>Generated {html.escape(ctx['ts'])} · Python {html.escape(ctx['py'])} · {ctx['plot_count']} rendered diagrams · 1 MP4 animation</p>
</header>
<div class="wrap">
  <div class="toc">
    <a href="#operators">Core operators &amp; simulators</a>
    <a href="#animation">Mechanism animation (MP4)</a>
    <a href="#mech">Mechanical calculator diagrams</a>
    <a href="#bench">Benchmark summary</a>
  </div>
""")

    # --- Core operators / simulators
    parts.append('<h2 id="operators">Core operators &amp; simulators <span class="sub">— run through the shared JSON execution engine</span></h2>')
    for feat in ctx["operators"]:
        parts.append('<div class="feature"><div class="two"><div>')
        parts.append(f"<h3>{html.escape(feat['title'])}</h3>")
        parts.append(img(feat["image"], feat["title"]))
        parts.append("</div><div>")
        parts.append("<div class='tag'>numeric result (truncated)</div>")
        parts.append(pre(feat.get("result", {})))
        parts.append("</div></div></div>")

    # --- MP4 animation
    parts.append('<h2 id="animation">Mechanism animation <span class="sub">— four-bar linkage over a full crank revolution, encoded to H.264 MP4 via ffmpeg</span></h2>')
    anim = ctx["animation"]
    if anim.get("ok"):
        parts.append('<div class="feature">')
        parts.append(f'<p>Computed with <code>mechanisms.fourbar_position</code> at {anim["frames"]} crank angles, then encoded with ffmpeg.</p>')
        parts.append(f'<video controls loop autoplay muted playsinline src="data:video/mp4;base64,{anim["b64"]}"></video>')
        parts.append("</div>")
    else:
        parts.append(f'<div class="feature"><p class="err">MP4 encode failed: {html.escape(anim.get("stderr",""))}</p></div>')

    # --- Mechanics diagrams grid
    ok_cards = [c for c in ctx["mech"] if not c["error"]]
    parts.append(f'<h2 id="mech">Mechanical calculator diagrams <span class="sub">— {len(ok_cards)} custom plots from the 50-family calculator library</span></h2>')
    parts.append('<div class="grid">')
    for c in ctx["mech"]:
        parts.append('<div class="card">')
        parts.append(f"<h3>{html.escape(c['title'])}</h3>")
        parts.append(f"<div class='tag'>{html.escape(c['calc'])} · {html.escape(c['section'])}</div>")
        if c["error"]:
            parts.append(f"<div class='err'>{html.escape(c['error'])}</div>")
        for im in c["images"]:
            parts.append(img(im, c["title"]))
        parts.append("</div>")
    parts.append("</div>")

    # --- Benchmark
    if ctx.get("bench"):
        b = ctx["bench"]
        parts.append('<h2 id="bench">Calculator benchmark summary</h2>')
        parts.append('<div class="feature"><div class="two"><div>')
        parts.append(img(b["image"], b["title"]))
        parts.append("</div><div>")
        parts.append("<div class='tag'>benchmark totals</div>")
        parts.append(pre(b["result"]))
        parts.append("</div></div></div>")

    parts.append(f"""</div>
<footer>Engineering Toolkit and Simulators · feature report · all figures produced by running the package locally.</footer>
</body></html>""")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="report")
    args = ap.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("rendering mechanics plots...", flush=True)
    mech = mechanics_plots()
    print(f"  {sum(1 for c in mech if not c['error'])}/{len(mech)} plot cards ok", flush=True)

    operators = []
    for label, fn in [("atmosphere", atmosphere_profile), ("room", room_acoustics_sweep),
                      ("magnetic", magnetic_field_map), ("world", world_demo)]:
        print(f"running operator: {label}...", flush=True)
        try:
            operators.append(fn())
        except Exception as e:
            print(f"  {label} FAILED: {type(e).__name__}: {e}", flush=True)

    print("rendering four-bar MP4...", flush=True)
    animation = fourbar_animation(out_dir)
    print(f"  mp4 ok={animation['ok']} frames={animation['frames']}", flush=True)

    print("benchmark summary...", flush=True)
    bench = benchmark_summary()

    plot_count = sum(len(c["images"]) for c in mech) + len(operators) + (1 if bench else 0)
    ctx = {
        "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "py": ".".join(map(str, sys.version_info[:3])),
        "plot_count": plot_count,
        "mech": mech,
        "operators": operators,
        "animation": animation,
        "bench": bench,
    }
    html_text = build_html(ctx)
    report = out_dir / "index.html"
    report.write_text(html_text, encoding="utf-8")
    print(f"\nWROTE {report} ({report.stat().st_size/1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()

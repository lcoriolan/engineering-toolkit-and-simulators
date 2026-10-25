#!/usr/bin/env python3
# DESCRIPTION: Small multiples, one measure per axis (never two y-scales), thin 2 px lines, recessive grid,
#              categorical slot 1 (blue) for MODELED, slot 2 (orange) markers for MEASURED/anchor data, legend
#              whenever two series share an axis. Palette = dataviz reference palette (light surface).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

BLUE, ORANGE, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title, ylabel, xlabel=None):
    ax.set_facecolor(SURF)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def flight_figure(res, path, title):
    tr, cols = res["traj"], res["cols"]
    ix = {c: i for i, c in enumerate(cols)}
    t = tr[:, ix["t"]]
    asc = t <= res["summary"]["t_apogee_s"] * 1.05 + 1
    panels = [("Altitude AGL", "m", tr[:, ix["z"]], np.ones_like(t, bool)),
              ("Mach number", "", tr[:, ix["mach"]], asc),
              ("Static margin (attached stack)", "calibres", tr[:, ix["margin_cal"]], asc & (tr[:, ix["mach"]] > 0.1)),
              ("Dynamic pressure", "kPa", tr[:, ix["q"]] / 1000.0, asc)]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.2), dpi=120, facecolor=SURF)
    for ax, (ttl, unit, y, m) in zip(axs.flat, panels):
        ax.plot(t[m], y[m], color=BLUE, linewidth=2)
        _style(ax, ttl, unit, "time [s]")
    fig.suptitle(title + "  (MODELED)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def validation_figure(cases, path):
    """cases: [(title, M_model, cd_model, M_data, cd_data, data_label)]"""
    fig, axs = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 3.8), dpi=120, facecolor=SURF)
    axs = np.atleast_1d(axs)
    for ax, (ttl, Mm, cm, Md, cdd, lab) in zip(axs, cases):
        ax.plot(Mm, cm, color=BLUE, linewidth=2, label="rocket_design build-up (MODELED)")
        ax.scatter(Md, cdd, s=36, color=ORANGE, edgecolor=SURF, linewidth=1.5, zorder=3, label=lab)
        _style(ax, ttl, "CD (ref. frontal area)", "Mach")
        ax.legend(fontsize=7.5, frameon=False, loc="best", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)


def scatter_landing(points, path, title):
    p = np.asarray(points)
    fig, ax = plt.subplots(figsize=(5.2, 5.0), dpi=120, facecolor=SURF)
    ax.scatter(p[:, 0], p[:, 1], s=14, color=BLUE, alpha=0.7, edgecolor="none")
    ax.scatter([0], [0], s=60, marker="^", color=INK, label="pad")
    ax.set_aspect("equal", adjustable="datalim")
    _style(ax, title, "north [m]", "east [m]")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURF)
    plt.close(fig)

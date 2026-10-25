# DESCRIPTION: Chart-spec builders for custom calculator diagrams (SFD/BMD, Mohr's circle, Bode, T-s,
#              velocity triangles). Specs are plain JSON rendered by web/science_bundle_mech.js and the CLI.
"""Chart spec (the one format every mechcalc plot uses):

{'title', 'xlabel', 'ylabel', 'xlog': bool, 'ylog': bool, 'equal': bool (1:1 aspect, e.g. Mohr),
 'series': [{'name', 'x': [...], 'y': [...], 'style': 'line'|'points'|'step'|'area'}],
 'markers': [{'x', 'y', 'label'}], 'hlines': [y...], 'vlines': [x...]}
"""
import math


def series(name, x, y, style='line'):
    return {'name': name, 'x': [float(v) for v in x], 'y': [float(v) for v in y], 'style': style}


def chart(title, xlabel, ylabel, series_list, markers=None, equal=False, xlog=False, ylog=False,
          hlines=None, vlines=None):
    return {'title': title, 'xlabel': xlabel, 'ylabel': ylabel, 'series': series_list,
            'markers': markers or [], 'equal': bool(equal), 'xlog': bool(xlog), 'ylog': bool(ylog),
            'hlines': hlines or [], 'vlines': vlines or []}


def marker(x, y, label):
    return {'x': float(x), 'y': float(y), 'label': str(label)}


def circle(name, cx, cy, r, n=181):
    """Closed polyline approximating a circle (Mohr's circle, bolt circles)."""
    th = [2 * math.pi * i / (n - 1) for i in range(n)]
    return series(name, [cx + r * math.cos(t) for t in th], [cy + r * math.sin(t) for t in th])


def linspace(a, b, n):
    n = max(2, int(n))
    return [a + (b - a) * i / (n - 1) for i in range(n)]

# DESCRIPTION: Shared involute geometry and declared external/internal tooth thickness conventions.
"""20° spur gears, zero profile shift, millimetres and counterclockwise radians.

Each member contributes half the nominal mesh backlash. The internal cavity is
the complement of inward-facing teeth; it is not a scaled external gear.
Below the external base circle, the profile uses a radial root transition.
"""
import math


def involute(radius, base):
    """Polar involute angle tan(alpha)-alpha; zero below the base circle."""
    if radius <= base:
        return 0.0
    t = math.sqrt((radius/base)**2-1)
    return t-math.atan(t)


def profile(teeth, module, backlash, clearance=.25, internal_cavity=False):
    """Return circle radii and half flank angles; backlash is total per mating mesh."""
    pitch = teeth*module/2
    base = pitch*math.cos(math.radians(20))
    root = pitch-module if internal_cavity else pitch-module*(1+clearance)
    tip = pitch+module*(1+clearance) if internal_cavity else pitch+module
    # Tooth thinning in an external member narrows its material. The same
    # thinning in the ring WIDENS the tooth space represented by this cavity.
    half = math.pi/(2*teeth)+(1 if internal_cavity else -1)*backlash/(4*pitch)
    start = max(root, base)
    def angle(r):
        return half+involute(pitch, base)-involute(r, base)
    if angle(tip) <= .001 or angle(start) >= math.pi/teeth:
        raise ValueError('Tooth tip or root gap collapsed; reduce backlash or increase tooth count.')
    return dict(teeth=teeth, pitch=pitch, base=base, root=root, tip=tip, start=start, angle=angle)


def polar(radius, angle):
    """Convert a radius/angle to XY without a CAD dependency."""
    return radius*math.cos(angle), radius*math.sin(angle)


def flank_radii(g, samples=18):
    """Sample uniformly in involute parameter for stable interpolation near the base circle."""
    lo = math.sqrt(max(0, (g['start']/g['base'])**2-1))
    hi = math.sqrt((g['tip']/g['base'])**2-1)
    return [g['base']*math.sqrt(1+(lo+(hi-lo)*i/samples)**2) for i in range(samples+1)]


def face(teeth, module, backlash, clearance=.25, internal_cavity=False, phase=0):
    """Create one closed planar face using interpolated involutes and exact circular lands."""
    import FreeCAD as App
    import Part
    g = profile(teeth, module, backlash, clearance, internal_cavity)
    radii = flank_radii(g)
    edges = []

    def vector(r, a):
        return App.Vector(*polar(r, a), 0)

    def arc(r, a, b):
        edges.append(Part.Arc(vector(r, a), vector(r, (a+b)/2), vector(r, b)).toShape())

    def curve(vertices):
        spline = Part.BSplineCurve()
        spline.interpolate(vertices)
        edges.append(spline.toShape())

    for index in range(teeth):
        centre = phase+index*math.tau/teeth
        left, right = centre-g['angle'](g['start']), centre+g['angle'](g['start'])
        if g['start'] > g['root']+1e-9:
            edges.append(Part.makeLine(vector(g['root'], left), vector(g['start'], left)))
        curve([vector(r, centre-g['angle'](r)) for r in radii])
        arc(g['tip'], centre-g['angle'](g['tip']), centre+g['angle'](g['tip']))
        curve([vector(r, centre+g['angle'](r)) for r in reversed(radii)])
        if g['start'] > g['root']+1e-9:
            edges.append(Part.makeLine(vector(g['start'], right), vector(g['root'], right)))
        arc(g['root'], right, centre+math.tau/teeth-g['angle'](g['start']))
    result = Part.Face(Part.Wire(edges))
    if not result.isValid() or result.Area <= 0:
        raise ValueError('Involute face is not a valid closed planar region.')
    return result

import math
FAMILIES = ('elliptical', 'straight_le', 'tapered', 'rectangular', 'straight_te', 'delta', 'swept')
CONSTANTS = {'sides': 2.0, 'source': 'Analytical planform and trapezoidal integral definitions'}

def chord_and_le(family, eta, root_chord, tip_chord, sweep_le_deg, half_span):
    """Return chord and leading-edge offset in caller units. Linear, rectangular or truncated elliptical chord; additional leading-edge shear = eta*half_span*tan(sweep)."""
    if family not in FAMILIES:
        raise ValueError(f'unknown planform family {family!r}; expected one of {FAMILIES}')
    if root_chord <= 0.0:
        raise ValueError('root_chord must be > 0')
    if tip_chord < 0.0:
        raise ValueError('tip_chord must be >= 0')
    if half_span <= 0.0:
        raise ValueError('half_span must be > 0')
    if family == 'elliptical':
        chord = math.sqrt(tip_chord * tip_chord + (root_chord * root_chord - tip_chord * tip_chord) * (1.0 - eta * eta))
    elif family == 'rectangular':
        chord = root_chord
    else:
        chord = root_chord + (tip_chord - root_chord) * eta
    if family in ('straight_le', 'rectangular'):
        le_x = 0.0
    elif family == 'straight_te':
        le_x = root_chord - chord
    else:
        le_x = 0.5 * (root_chord - chord)
    le_x += eta * half_span * math.tan(math.radians(sweep_le_deg))
    return (chord, le_x)
chord_and_offset = chord_and_le

def _spacing_values(n, spacing):
    """Spanwise fractions 0..1 for ``n`` stations (``cosine`` or ``uniform``)."""
    if spacing not in ('cosine', 'uniform'):
        raise ValueError(f"spacing must be 'cosine' or 'uniform', got {spacing!r}")
    if n < 2:
        raise ValueError('n must be >= 2 stations')
    if spacing == 'cosine':
        return [0.5 * (1.0 - math.cos(math.pi * i / (n - 1))) for i in range(n)]
    return [i / (n - 1) for i in range(n)]

def stations(family, half_span, root_chord, tip_chord, sweep_le_deg, dihedral_deg, twist_root_deg, twist_tip_deg, root_le_x, root_z, n, spacing, airfoil=None):
    """Contract §2 wing-station dicts for one half-wing, root (y=0) to tip (y=half_span).

    Each station::

      {y, le_x, te_x, chord, z_chord, twist_deg, t_c=None, airfoil, source="generated:<family>"}

    with (mm, degrees):

      y        = half_span * f                          (f from ``spacing``)
      chord    = chord_and_le(...)[0]
      le_x     = root_le_x + chord_and_le(...)[1]       (root LE placed at root_le_x)
      te_x     = le_x + chord
      z_chord  = root_z + y * tan(dihedral_deg)         (chord-line height; straight dihedral)
      twist_deg= twist_root_deg + (twist_tip_deg - twist_root_deg) * f   (linear washout)

    ``spacing`` is "cosine" (stations cluster at root and tip) or "uniform"; elliptical
    planforms need cosine for the tip. ``t_c`` is None (the section library fills it), and
    ``airfoil`` is the caller's profile key (default None). Omissions: dihedral is a
    straight tilt that does not change the projected planform area, and y is the flat
    projected span coordinate (matching measure.py's planform integral).
    """
    out = []
    for f in _spacing_values(n, spacing):
        y = half_span * f
        chord, le_offset = chord_and_le(family, f, root_chord, tip_chord, sweep_le_deg, half_span)
        le_x = root_le_x + le_offset
        out.append(dict(y=y, le_x=le_x, te_x=le_x + chord, chord=chord, z_chord=root_z + y * math.tan(math.radians(dihedral_deg)), twist_deg=twist_root_deg + (twist_tip_deg - twist_root_deg) * f, t_c=None, airfoil=airfoil, source='generated:' + family))
    return out

def area_mac(stations):
    """(area both sides, MAC, MAC LE x, MAC y) by trapezoid integration over the half-span.

    With c(y), le_x(y) sampled at ascending stations and both sides doubled (parametric geometry library
    README "mean aerodynamic chord is ∫c² ds / ∫c ds"):

      area     = 2 * ∫ c dy                  (mm^2)
      MAC      = ∫ c² dy / ∫ c dy            (mm)
      mac_le_x = ∫ c·le_x dy / ∫ c dy        (mm; chord-weighted average leading edge)
      mac_y    = ∫ c·y dy / ∫ c dy           (mm; chord-weighted spanwise centroid)

    Units: mm. Screening: trapezoid rule is exact for a linear taper's area but approximate
    for c² (O(h²)); use enough stations. ``le_x`` is read as the absolute LE x at each
    station (not the root-relative offset), matching the generated record.
    """
    ordered = sorted(stations, key=lambda s: s['y'])
    ys = [s['y'] for s in ordered]
    cs = [s['chord'] for s in ordered]
    les = [s['le_x'] for s in ordered]

    def trap(vals):
        total = 0.0
        for i in range(len(ys) - 1):
            total += 0.5 * (vals[i] + vals[i + 1]) * (ys[i + 1] - ys[i])
        return total
    sc = trap(cs)
    if sc <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    return (CONSTANTS['sides'] * sc, trap([c * c for c in cs]) / sc, trap([c * l for c, l in zip(cs, les)]) / sc, trap([c * y for c, y in zip(cs, ys)]) / sc)

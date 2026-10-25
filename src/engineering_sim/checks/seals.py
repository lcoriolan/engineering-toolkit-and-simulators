import math
def radial_gland_check(cs, bore_d, groove_root_d, groove_w, diam_tol, growth_frac, cs_tolerance,
                       width_tol=0.0, stretch_max=0.0):
    """Squeeze and fill of a static piston gland, nominal and worst case.

    squeeze = (cs − radial gap) / cs, where radial gap = (bore − groove root) / 2
    fill    = O-ring section area / groove area = (π cs² / 4) / (groove_w × radial gap)
    Worst case stacks the ±cs tolerance, a ± process tolerance on both diameters (halved into
    the radial gap) and the moisture growth. Both mating parts are the same material, so growth
    scales the gap by `growth_frac` rather than opening it.
    Optional width tolerance and maximum installed stretch support printed glands.
    The minimum cross-section uses incompressible area scaling, cs/sqrt(1+stretch),
    as a declared screening approximation; it is not an elastomer/contact solve.
    Zero optional allowances preserve historical finished-gland calculations.
    Returns a dict of fractions. All tolerances are bilateral absolute mm.
    """
    gap = (bore_d - groove_root_d) / 2
    tol = float(cs_tolerance)
    gap_max = gap + diam_tol + gap * growth_frac            # both diameters off by tol/2 each, radially
    gap_min = gap - diam_tol
    if (not all(math.isfinite(v) for v in (cs, bore_d, groove_root_d, groove_w,
                                         diam_tol, growth_frac, cs_tolerance, width_tol, stretch_max))
            or min(diam_tol, growth_frac, cs_tolerance, width_tol, stretch_max) < 0
            or groove_w <= width_tol or gap_min <= 0):
        raise ValueError('Gland dimensions and tolerance stack must leave positive width and depth.')
    squeeze = (cs - gap) / cs
    fill = (math.pi * cs ** 2 / 4) / (groove_w * gap)
    cs_min = (cs - tol) / math.sqrt(1 + stretch_max)
    squeeze_min = (cs_min - gap_max) / cs_min
    squeeze_max = (cs + tol - gap_min) / (cs + tol)
    fill_max = (math.pi * (cs + tol) ** 2 / 4) / ((groove_w - width_tol) * gap_min)
    return dict(squeeze=squeeze, fill=fill, squeeze_min=squeeze_min, squeeze_max=squeeze_max, fill_max=fill_max, radial_gap=gap)

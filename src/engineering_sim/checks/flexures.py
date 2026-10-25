import math
import numpy as np

def flexure_arc(x: float, z: float, x_start: float, length: float, angle_deg: float) -> tuple[float, float]:
    """Map a point aft of the flexure start through a circular bend.

    Circular-flexure arc (reused from morphing-airfoil model.py:warp): the
    spine is assumed to bend into an arc of constant curvature that preserves
    arc length after x_start. kappa = angle/length (rad/mm). Positive angle is
    trailing-edge down. This is an imposed kinematic target, not an elastic
    equilibrium shape; skin/rib stiffness and spar reaction are omitted.
    """
    s = max(0.0, x - x_start)
    if not s or abs(angle_deg) < 1e-09:
        return (x, z)
    k = -math.radians(angle_deg) / length
    a = k * s
    return (x_start + math.sin(a) / k - z * math.sin(a), (1.0 - math.cos(a)) / k + z * math.cos(a))

def _crank_angle(center, target, r: float, length: float, servo_limit_deg: float=60.0):
    """Solve the rotary-crank four-bar for the crank angle that closes the loop.

    The crank pin sweeps a circle of radius `r` about `center` with the
    convention pin = center + r*(sin t, cos t), so t = 0 is straight up (+z).
    The pushrod runs pin -> `target` and has a FIXED length `length` (set at the
    neutral target); the crank rotates until the pin lands at that distance from
    `target`. Derivation: |pin - target| = length is a law-of-cosines problem in
    the triangle (center, pin, target); with D = target - center, dist = |D| and
    k = (dist^2 + r^2 - length^2)/(2 r dist), the two branches are
    t = atan2(dz, dx) +- asin(k). The nearest branch wins; overtravel (|k|>1 or
    |t|>servo_limit_deg) raises rather than stretching the rod.
    """
    cx, cz = center
    dx = target[0] - cx
    dz = target[1] - cz
    dist = math.hypot(dx, dz)
    if dist < 1e-09:
        raise ValueError('target coincides with the crank centre')
    k = (dist * dist + r * r - length * length) / (2.0 * r * dist)
    if abs(k) > 1.0:
        raise ValueError('pushrod cannot reach this bend; change crank radius or motion')
    phi = math.atan2(dz, dx)
    a = math.asin(k)
    cands = [a - phi, math.pi - a - phi]
    cands = [(v + math.pi) % (2.0 * math.pi) - math.pi for v in cands]
    theta = min(cands, key=abs)
    if abs(math.degrees(theta)) > servo_limit_deg:
        raise ValueError('actuator travel exceeded by the linkage')
    pin = (cx + r * math.sin(theta), cz + r * math.cos(theta))
    return (math.degrees(theta), pin)

def linkage(center, target0, r: float, angle_deg: float, flexure_start: float, flexure_len: float, servo_limit_deg: float=60.0) -> dict:
    """Solve the servo-crank/pushrod four-bar for one flexure bend angle.

    Reused from morphing-airfoil model.py:linkage. `center` is the servo output
    axis (x, z); `target0` is the rib-horn pin at neutral; `r` is the crank
    radius. The horn moves with the flexure arc; the crank rotates until the
    fixed-length pushrod closes the loop. Returns the nearest valid crank
    branch and raises ValueError on overtravel (never stretches the rod).
    """
    cx, cz = center
    target = flexure_arc(*target0, flexure_start, flexure_len, angle_deg)
    neutral = (cx, cz + r)
    length = math.dist(neutral, target0)
    theta, pin = _crank_angle(center, target, r, length, servo_limit_deg)
    return {'center': center, 'pin': pin, 'target': target, 'rod_length': length, 'angle_deg': theta}

def linkage_external(servo_center, servo_arm_r: float, horn_pivot, horn_r: float, bend_deg: float, servo_limit_deg: float=60.0) -> dict:
    """External horn + pushrod four-bar (stock-aileron style), for the panel-bay servo.

    The zone output horn is a rigid bellcrank pivoting about `horn_pivot` (the
    hinge axis); its pin sits `horn_r` below the pivot at neutral and rotates
    1:1 with the commanded flexure bend `bend_deg` (positive = trailing edge
    down, pin sweeps aft) — the spanwise sync rod carries that rotation to the
    flexible ribs, so the bellcrank angle equals the flexure bend angle. The
    servo sits in the panel bay with its output axis at `servo_center` and its
    arm (`servo_arm_r`) hanging below the centre; a fixed-length pushrod joins
    the arm pin to the horn pin through the existing cover exit. Screening
    kinematics only — no servo torque, rod bending or friction.
    """
    xs, zs = servo_center
    xh, zh = horn_pivot
    arm_neutral = (xs, zs - servo_arm_r)
    horn_neutral = (xh, zh - horn_r)
    length = math.dist(arm_neutral, horn_neutral)
    d = math.radians(bend_deg)
    horn_pin = (xh + horn_r * math.sin(d), zh - horn_r * math.cos(d))
    theta, arm_pin = _crank_angle(servo_center, horn_pin, servo_arm_r, length, servo_limit_deg)
    return {'servo_center': servo_center, 'servo_arm_pin': arm_pin, 'servo_arm_neutral': arm_neutral, 'horn_pivot': horn_pivot, 'horn_pin': horn_pin, 'horn_neutral': horn_neutral, 'rod_length': length, 'servo_angle_deg': theta}

def transmission_angle(pin: tuple, target: tuple, crank_center: tuple) -> float:
    """Four-bar transmission angle: between the pushrod and the crank arm.

    The pushrod runs pin->target; the crank arm runs crank_center->pin. A
    transmission angle near 90 deg gives the best mechanical advantage; the
    declared minimum is 30 deg (classic four-bar rule of thumb). Returns
    degrees in [0, 90].
    """
    rod = (target[0] - pin[0], target[1] - pin[1])
    arm = (pin[0] - crank_center[0], pin[1] - crank_center[1])
    lr = math.hypot(*rod)
    la = math.hypot(*arm)
    if lr < 1e-09 or la < 1e-09:
        return 0.0
    cosang = (rod[0] * arm[0] + rod[1] * arm[1]) / (lr * la)
    cosang = max(-1.0, min(1.0, cosang))
    ang = math.degrees(math.acos(abs(cosang)))
    return min(ang, 180.0 - ang)

def flexure_strain(t_spine: float, flexure_len: float, angle_deg: float) -> float:
    """Nominal spine surface strain for a circular bend.

    strain = kappa * t/2 with kappa = delta/L_f, so strain = delta_rad * t / (2 L_f).
    Reused from morphing-airfoil model.py:calculate (nominal_spine_surface_strain).
    This is thickness x curvature / 2; local posts, horn loading and fatigue
    concentration invalidate it as a life criterion.
    """
    return t_spine * math.radians(angle_deg) / (2.0 * flexure_len)

def rib_stations(y0: float, y1: float, pitch: float, inset: float) -> list[float]:
    """Spanwise rib stations: end ribs `inset` inside the zone ends, then `pitch`.

    Returns ascending y positions. The first and last ribs sit at y0+inset and
    y1-inset; interior ribs are spaced by `pitch`. The end ribs are inset so
    the sync-rod and spar ends have printed material beyond them.
    """
    inner0 = y0 + inset
    inner1 = y1 - inset
    if inner1 <= inner0:
        return [0.5 * (y0 + y1)]
    n = max(1, int(math.floor((inner1 - inner0) / pitch + 0.5)) + 1)
    return [inner0 + (inner1 - inner0) * i / (n - 1) for i in range(n)]

def servo_fit_station(station: dict, servo_h: float, wall: float):
    """Decide where the servo fits inside the local section.

    The servo needs servo_h + 2*wall of local thickness at its chordwise
    centre. Scan the measured thickness samples for the aftmost chord station
    (closest to the hinge) that fits, and return the offset forward of the
    hinge. This is the honest packaging check: the servo is never shrunk.
    """
    hinge = station['hinge_x']
    need = servo_h + 2.0 * wall
    samples = station['thickness']
    fit = [x for x, t in samples if t is not None and t >= need and (x <= hinge + 1e-06)]
    if not fit:
        fit = [x for x, t in samples if t is not None and t >= need]
    if not fit:
        return {'fits': False, 'fits_at_hinge': False, 'x_fit': None, 'offset_mm': None, 'need_mm': need, 'avail_mm': None}
    x_fit = max(fit)
    avail = next((t for x, t in samples if abs(x - x_fit) < 1e-06), None)
    return {'fits': True, 'fits_at_hinge': x_fit >= hinge - 0.5, 'x_fit': x_fit, 'offset_mm': hinge - x_fit, 'need_mm': need, 'avail_mm': avail}

def servo_box_fit(station: dict, servo: dict, wall: float, fairing_mm: float=0.0, aft_limit: float | None=None):
    """Fit the full servo box (l chordwise x h vertical) forward of the aft limit.

    The servo sits in the wing panel's aileron servo bay, entirely forward of
    the aft limit (the aileron hinge for the measured path, or the rear-web
    plane for the generated path), with its long axis chordwise and its `h`
    axis vertical (output shaft down, through the lower-skin / servo-cover
    opening). The shaft face sits on the lower skin, so the section must clear
    `h + wall` between the lower skin and the upper skin over the whole
    chordwise footprint `[x - l/2, x + l/2]`; `fairing_mm` is the extra depth
    the servo cover adds below the lower skin. Returns the AFT-most chord
    centre that fits, plus a pure-airfoil verdict and the fairing depth
    actually required.
    """
    hinge = station['hinge_x']
    aft = aft_limit if aft_limit is not None else hinge
    l, h = (servo['l'], servo['h'])
    need = h + wall
    samples = [s for s in station['thickness'] if s[1] is not None]
    if not samples:
        return {'fits': False, 'fits_airfoil': False, 'x_fit': None, 'offset_mm': None, 'need_mm': need, 'avail_mm': None, 'fairing_needed_mm': None}
    best = None
    for xc in sorted((x for x, t in samples)):
        lo, hi = (xc - l / 2.0, xc + l / 2.0)
        if hi > aft + 1e-06:
            continue
        tmin = None
        for x, t in samples:
            if lo - 1e-06 <= x <= hi + 1e-06 and (tmin is None or t < tmin):
                tmin = t
        if tmin is None:
            continue
        if tmin + fairing_mm >= need and (best is None or xc > best['x_fit']):
            best = {'fits': True, 'x_fit': float(xc), 'avail_mm': float(tmin), 'fits_airfoil': bool(tmin >= need)}
    if best is None:
        cands = [(t, x) for x, t in samples if x + l / 2.0 <= aft + 1e-06]
        if not cands:
            return {'fits': False, 'fits_airfoil': False, 'x_fit': None, 'offset_mm': None, 'need_mm': need, 'avail_mm': None, 'fairing_needed_mm': None}
        tmax, xmax = max(cands)
        return {'fits': False, 'fits_airfoil': False, 'x_fit': float(xmax), 'avail_mm': float(tmax), 'need_mm': float(need), 'offset_mm': float(aft - xmax), 'fairing_needed_mm': float(max(0.0, need - tmax - fairing_mm))}
    best['need_mm'] = float(need)
    best['offset_mm'] = float(aft - best['x_fit'])
    best['fairing_needed_mm'] = float(max(0.0, need - best['avail_mm']))
    return best

import math
def validate(p):
    """Reject coupled interfaces that cannot meet the declared tolerance stack."""
    if p['sprue_diameter']<p['gate_diameter']:raise ValueError('Sprue inlet must be at least the gate diameter.')
    if p['guide_clearance']<2*p['fit_tolerance'] or (p['process']=='injection' and p['ejector_clearance']<2*p['fit_tolerance']):
        raise ValueError('Guide and ejector diametral clearance must cover both ± pin/bore tolerances.')
    if p['margin']<2*p['guide_diameter']+p['minimum_web']:
        raise ValueError('Increase stock margin for guide diameter and minimum steel web.')
    if p['process']=='injection' and p['cooling'] and p['back_wall']<p['cooling_diameter']+2*p['minimum_web']:
        raise ValueError('Back wall must contain the cooling diameter and two steel webs.')
    points=p['casting_vents'] if p['process']=='casting' else p['ejectors']
    if not isinstance(points,list) or len(points)>12:raise ValueError('Use at most 12 vent or ejector XY points.')
    for point in points:
        if not isinstance(point,list) or len(point)!=2 or any(type(v) not in (float,int) or not math.isfinite(v) or abs(v)>.9 for v in point):
            raise ValueError('Vent / ejector positions are [x,y] fractions between −0.9 and 0.9.')
    if len({tuple(x) for x in points})!=len(points):raise ValueError('Vent / ejector positions must be distinct.')

def calculate(p):
    """Expose fit intervals and shrinkage convention before expensive geometry work."""
    result=dict(cavity_scale=1/(1-p['shrink_pct']/100),
      guide_clearance_min_mm=p['guide_clearance']-2*p['fit_tolerance'],
      guide_clearance_max_mm=p['guide_clearance']+2*p['fit_tolerance'],
      ejector_clearance_min_mm=p['ejector_clearance']-2*p['fit_tolerance'],
      ejector_clearance_max_mm=p['ejector_clearance']+2*p['fit_tolerance'])
    if p['process']=='casting':
        # Casting does not report fits for injection hardware that is absent.
        result.pop('ejector_clearance_min_mm');result.pop('ejector_clearance_max_mm')
    return result

import math
def fit_clearance(hole, shaft):
    """(min, max) diametral clearance of a hole/shaft deviation pair (mm)."""
    return hole[0] - shaft[1], hole[1] - shaft[0]

def stack(terms):
    """Bilateral bounds in mm; RSS is a root-sum-square assumption, not a yield probability."""
    values=[abs(float(row['sensitivity'])*float(row['tolerance_mm'])) for row in terms]
    if any(not math.isfinite(v) for v in values):raise ValueError('Finite inputs required.')
    return dict(worst_case_mm=sum(values),rss_mm=math.sqrt(sum(v*v for v in values)))
def thermal_shift(alpha_a,delta_t_a,length_a_mm,alpha_b,delta_t_b,length_b_mm):
    return alpha_a*delta_t_a*length_a_mm-alpha_b*delta_t_b*length_b_mm

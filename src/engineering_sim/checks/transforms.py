import math
import numpy as np
def unit(vector):
    """Return a unit direction, rejecting singular geometry instead of emitting NaNs."""
    vector=np.asarray(vector,dtype=float)
    norm=np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError("singular direction")
    return vector/norm

def transform(axis, point, angle):
    """Rigid rotation about a world-reference hinge line; angles are radians."""
    axis=unit(axis); point=np.asarray(point)
    x,y,z=axis
    skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    rotation=np.eye(3)+math.sin(angle)*skew+(1-math.cos(angle))*(skew@skew)
    matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=point-rotation@point
    return matrix

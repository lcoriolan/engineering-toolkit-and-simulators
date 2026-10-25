"""Linear unit conversion for the generic gimbal simulator; no hardware access."""
import math
class LinearCalibration:
    def __init__(self,units_per_deg,zero_native=0.0):
        if not math.isfinite(units_per_deg) or units_per_deg==0:raise ValueError('Scale must be finite and nonzero.')
        self.units_per_deg=float(units_per_deg);self.zero_native=float(zero_native)
    def to_native(self,degrees):return self.zero_native+float(degrees)*self.units_per_deg
    def to_degrees(self,native):return (float(native)-self.zero_native)/self.units_per_deg

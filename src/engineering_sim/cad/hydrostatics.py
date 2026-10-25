import FreeCAD as App
import Part
from engineering_sim.execution import operation as _simulation_operation

def submerged(shape, z):
    """Intersect the closed displacement envelope below world-Z water level, in mm."""
    b = shape.BoundBox
    if z <= b.ZMin:
        return (None, 0.0)
    if z >= b.ZMax:
        return (shape, shape.Volume)
    box = Part.makeBox(b.XLength + 2, b.YLength + 2, z - b.ZMin + 1, App.Vector(b.XMin - 1, b.YMin - 1, b.ZMin - 1))
    wet = shape.common(box)
    return (wet, wet.Volume)

@_simulation_operation('cad.hydrostatics.equilibrium', kind='static')
def equilibrium(shape, mass, rho):
    """Bisection finds upright heave; over-capacity cases never receive a floating waterline."""
    b = shape.BoundBox
    capacity = shape.Volume / 1000000000.0 * rho
    target = mass / rho * 1000000000.0
    if mass >= capacity:
        return dict(floats=False, capacity_kg=capacity, required_displacement_m3=mass / rho, waterline_mm=None, freeboard_mm=None, reason='Loaded mass reaches or exceeds the sealed envelope displacement capacity.')
    low, high = (b.ZMin, b.ZMax)
    wet = None
    volume = 0
    for _ in range(32):
        z = (low + high) / 2
        wet, volume = submerged(shape, z)
        if abs(volume - target) <= max(0.01, target * 1e-07):
            break
        if volume < target:
            low = z
        else:
            high = z
    center = App.Vector()
    for solid in wet.Solids:
        center += solid.CenterOfMass * solid.Volume
    center /= volume
    return dict(floats=True, capacity_kg=capacity, loaded_mass_kg=mass, displacement_m3=volume / 1000000000.0, waterline_mm=z, draft_mm=z - b.ZMin, freeboard_mm=b.ZMax - z, buoyant_force_n=volume / 1000000000.0 * rho * 9.80665, weight_n=mass * 9.80665, force_residual_n=(volume / 1000000000.0 * rho - mass) * 9.80665, center_of_buoyancy_mm=[center.x, center.y, center.z])

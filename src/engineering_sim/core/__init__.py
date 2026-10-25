import os

__version__ = "0.2.1"   # 0.2.1 (2026-10-15): vapour cold-trapped above tropopause (e>p NaN fix). 0.2.0: one constants module, BL wind above 100 m, Sutherland, actual-T column


def physics_device(device=None):
    """The one place the physics GPU is chosen: explicit arg, else $SIMGPU_PHYSICS_DEVICE, else cpu."""
    return device or os.environ.get("SIMGPU_PHYSICS_DEVICE", "cpu")

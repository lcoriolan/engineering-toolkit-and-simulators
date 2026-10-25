#!/usr/bin/env python3
# DESCRIPTION: Land vehicles and contact bodies now step through the ONE shared rigid-body step,
#              core/rigid6.step_rigid (exact-rotation attitude update; the old first-order quaternion update is gone).
#              This alias only keeps old imports working.
from engineering_sim.core.rigid6 import step_rigid  # noqa: F401

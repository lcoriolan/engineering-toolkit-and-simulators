"""Run with FreeCAD Python: explicit features and hydrostatic displacement."""
import sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(sys.executable).resolve().parent.parent/'lib'))
import FreeCAD as App
import Part
from engineering_sim.cad.features import apply
from engineering_sim.cad.hydrostatics import equilibrium
checks=[]
plate=apply('plate',None,dict(length=10,width=20,thickness=4))
assert plate.isValid() and abs(plate.Volume-800)<1e-8;checks.append('synthetic_plate_volume')
bored=apply('hole_array',plate,dict(dia=1,margin=2,rows=2,cols=2),dict(published=dict(base_len=10,base_wid=20,base_thickness=4)))
expected=800-4*math.pi
assert bored.isValid() and abs(bored.Volume-expected)<1e-8;checks.append('four_hole_volume')
r=equilibrium(Part.makeBox(1000,1000,1000),500,1000)
assert r['floats'] and abs(r['waterline_mm']-500)<1e-4 and abs(r['force_residual_n'])<1e-4;checks.append('upright_hydrostatic_box')
r=equilibrium(Part.makeBox(1000,1000,1000),1000,1000)
assert not r['floats'] and r['waterline_mm'] is None;checks.append('overcapacity_rejected')
print(json.dumps(dict(passed=len(checks),checks=checks,FreeCAD='.'.join(App.Version()[:3]),bored_volume_mm3=bored.Volume,bored_reference_mm3=expected)))

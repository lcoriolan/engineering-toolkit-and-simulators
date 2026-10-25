"""Native geometry checks; run with FreeCAD's Python. Only synthetic solids are used."""
import json,math,sys
from pathlib import Path
sys.path.append(str(Path(sys.executable).resolve().parents[1]/'lib'))
import FreeCAD as App
import Part
from engineering_sim.checks import assembly,mechanisms,mounted_fit
from engineering_sim.cad.gears import face

def main():
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    a=Part.makeBox(2,2,2)
    b=Part.makeBox(2,2,2,App.Vector(2.3,0,0))
    check('clearance_pass',mechanisms.pair_check(a,b,.2)['passed'])
    b.translate(App.Vector(-.2,0,0));check('clearance_fail',not mechanisms.pair_check(a,b,.2)['passed'])
    b.translate(App.Vector(-.2,0,0));r=mechanisms.pair_check(a,b,.2)
    check('positive_interference',r['reason']=='positive_overlap' and abs(r['overlap_mm3']-.4)<1e-9)
    moving=Part.makeBox(2,2,2,App.Vector(5,0,0));obstacle=Part.makeBox(2,2,2,App.Vector(1,0,0))
    result=assembly.check({'moving':moving,'obstacle':obstacle},[dict(step='insert',parts=['moving'],direction=[1,0,0],travel_mm=5)])
    check('sampled_translation_hits_obstacle',not result['ok'])
    clear=assembly.check({'moving':moving,'obstacle':Part.makeBox(2,2,2,App.Vector(1,10,0))},[dict(step='insert',parts=['moving'],direction=[1,0,0],travel_mm=5)])
    check('sampled_translation_clear',clear['ok'])
    ring=Part.makeCylinder(4,5).cut(Part.makeCylinder(2,5))
    bore=mechanisms.bore_datum(ring,[0,0,0],[0,0,1],2)
    check('native_bore_datum',bool(bore))
    src=dict(origin=[0,0,0],axis=[1,0,0],down=[0,0,-1]);dst=dict(origin=[5,6,7],axis=[0,1,0],down=[0,0,-1])
    matrix,R,t=mounted_fit.mounting_matrix(src,dst)
    point=matrix.multVec(App.Vector(1,0,0));check('mount_frame_transform',(point-App.Vector(5,7,7)).Length<1e-10)
    z1,z2,module=24,36,1.;center=module*(z1+z2)/2
    g1=face(z1,module,.1).extrude(App.Vector(0,0,3))
    g2=face(z2,module,.1,phase=math.pi+math.pi/z2).extrude(App.Vector(0,0,3));g2.translate(App.Vector(center,0,0))
    overlaps=[]
    for i in range(13):
        angle=(360/z1)*i/12;a1=g1.copy();b1=g2.copy()
        a1.rotate(App.Vector(),App.Vector(0,0,1),angle);b1.rotate(App.Vector(center,0,0),App.Vector(0,0,1),-angle*z1/z2)
        overlaps.append(max(0.,a1.common(b1).Volume))
    check('gear_motion_sampled_interference',max(overlaps)<1e-5)
    print(json.dumps(dict(passed=len(checks),checks=checks,gear_samples=len(overlaps),max_gear_overlap_mm3=max(overlaps),gear_ratio=-z1/z2,scope='13 poses over one tooth pitch; no continuous collision proof or loaded contact solve')))
if __name__=='__main__':main()

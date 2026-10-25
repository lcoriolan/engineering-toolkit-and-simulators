"""Run with a FreeCAD Python interpreter. All geometry is newly constructed here."""
import importlib.util,json,math,sys,tempfile
from pathlib import Path
sys.path.append(str(Path(sys.executable).resolve().parents[1]/'lib'))
import FreeCAD as App
import Part
from engineering_sim.cad.importer import recognize,convert,equivalent
from engineering_sim.cad.builder import build

def main():
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        shapes={'box':Part.makeBox(10,20,30),'cylinder':Part.makeCylinder(3,10),
                'cone':Part.makeCone(3,1,10),'sphere':Part.makeSphere(3),'torus':Part.makeTorus(6,1)}
        for name,shape in shapes.items():check('recognize_'+name,recognize(shape)['kind']==name)
        moved=shapes['box'].copy();moved.Placement=App.Placement(App.Vector(5,7,9),App.Rotation(App.Vector(1,2,3),23))
        check('recognize_placed_box',recognize(moved)['kind']=='box')
        moved.exportStep(str(root/'input.step'));convert(root/'input.step',root/'converted','synthetic_box')
        report=json.loads((root/'converted/conversion_report.json').read_text())
        check('conversion_geometry',report['all_geometry_verified'] and report['reconstructed']==1)
        path=root/'converted/harness/model.py';spec=importlib.util.spec_from_file_location('synthetic',path)
        model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
        parameters=model.defaults();original=model.build_node('assembly',parameters,{})[0]['shape']
        check('default_exact_match',equivalent(moved,original) is not None)
        key=next(k for k in parameters if k.endswith('_length_mm'));parameters[key]*=1.5
        changed=model.build_node('assembly',parameters,{})[0]['shape']
        check('dimension_edit_changes_volume',abs(changed.Volume/original.Volume-1.5)<1e-10)
        native=model.rebuild_document(parameters,recognized=True,originals=True)
        try:native.saveAs(str(root/'edited.FCStd'))
        finally:App.closeDocument(native.Name)
        convert(root/'edited.FCStd',root/'reimport','edited_box')
        check('native_reimport',json.loads((root/'reimport/conversion_report.json').read_text())['all_geometry_verified'])
        # A union of offset spheres is outside the complete-primitive/extrusion recognizer.
        unsupported=Part.makeSphere(3).fuse(Part.makeSphere(3,App.Vector(3,0,1)))
        check('unsupported_is_retained',recognize(unsupported)['kind']=='brep')
        unsupported.exportBrep(str(root/'compound.brep'));convert(root/'compound.brep',root/'preserved','preserved_solid')
        check('unsupported_conversion_verified',json.loads((root/'preserved/conversion_report.json').read_text())['preserved']==1)
        spec={'nodes':[{'name':'base','kind':'box','parameters':{'length_mm':10,'width_mm':20,'height_mm':30}},
                       {'name':'hole','kind':'cylinder','parameters':{'radius_mm':2,'height_mm':30},'translation_mm':[5,10,0]},
                       {'name':'result','kind':'cut','left':'base','right':'hole'}],'result':'result'}
        result=build(spec,root/'built')
        check('parametric_boolean_volume',abs(result['volume_mm3']-(6000-math.pi*4*30))<1e-7)
        check('parametric_native_and_step',all((root/'built'/n).is_file() for n in ('model.FCStd','model.step','parameters.json')))
        # Repeating conversion from identical bytes has a byte-identical portable source archive.
        convert(root/'input.step',root/'converted2','synthetic_box')
        check('deterministic_source_package',(root/'converted/python_harness.zip').read_bytes()==(root/'converted2/python_harness.zip').read_bytes())
    print(json.dumps({'passed':len(checks),'checks':checks,'freecad':App.Version()[:3]}))
if __name__=='__main__':main()

"""Parametric exact solids from caller data, with native FreeCAD primitive features.

Lengths are millimetres and rotations are axis-angle degrees. No design presets.
"""
import math
from pathlib import Path
import sys
sys.path.append(str(Path(sys.executable).resolve().parents[1]/'lib'))
import FreeCAD as App
import Part

PROPERTIES={
    'box':('Part::Box',{'length_mm':'Length','width_mm':'Width','height_mm':'Height'}),
    'cylinder':('Part::Cylinder',{'radius_mm':'Radius','height_mm':'Height'}),
    'cone':('Part::Cone',{'radius1_mm':'Radius1','radius2_mm':'Radius2','height_mm':'Height'}),
    'sphere':('Part::Sphere',{'radius_mm':'Radius'}),
    'torus':('Part::Torus',{'major_radius_mm':'Radius1','minor_radius_mm':'Radius2'}),
}

def build(spec,output):
    """Build ordered primitive/Boolean nodes and export FCStd, STEP and a report.

    ``spec = {'nodes': [{'name':'block','kind':'box','parameters':{...}}],
    'result':'block'}``. Boolean nodes use kind union/cut/common and left/right
    names. The destination must be new, so existing user files are not replaced.
    """
    return _build(spec,output)

from ..execution import operation
@operation('cad.parametric_build',kind='static')
def _build(spec,output):
    import json,re
    if not isinstance(spec,dict) or not 1<=len(spec.get('nodes',[]))<=512:
        raise ValueError('Supply 1..512 ordered nodes.')
    destination=Path(output);destination.mkdir(parents=True,exist_ok=False)
    doc=App.newDocument('ParametricBuild');nodes={}
    try:
        for row in spec['nodes']:
            name=row['name'];kind=row['kind']
            if not re.fullmatch('[a-z][a-z0-9_]{0,59}',name) or name in nodes:
                raise ValueError('Node names must be unique lowercase identifiers.')
            if kind in PROPERTIES:
                type_name,props=PROPERTIES[kind];values=row['parameters']
                if set(values)!=set(props):raise ValueError('Supply exactly the parameters for '+kind)
                node=doc.addObject(type_name,name)
                for key,property_name in props.items():
                    v=float(values[key])
                    if not math.isfinite(v) or v<0 or (v==0 and not key.startswith('radius')):
                        raise ValueError('Primitive dimensions must be finite and positive (cone tip may be zero).')
                    if v==0 and not (kind=='cone' and key.startswith('radius')):raise ValueError('Zero primitive radius.')
                    setattr(node,property_name,v)
            elif kind in ('union','cut','common'):
                node=doc.addObject({'union':'Part::Fuse','cut':'Part::Cut','common':'Part::Common'}[kind],name)
                node.Base=nodes[row['left']];node.Tool=nodes[row['right']]
            else:raise ValueError('Unsupported node kind: '+str(kind))
            translation=row.get('translation_mm',[0,0,0]);axis_angle=row.get('rotation',[0,0,1,0])
            if len(translation)!=3 or len(axis_angle)!=4 or not all(math.isfinite(float(x)) for x in translation+axis_angle):
                raise ValueError('Invalid placement.')
            node.Placement=App.Placement(App.Vector(*translation),App.Rotation(App.Vector(*axis_angle[:3]),axis_angle[3]))
            nodes[name]=node;doc.recompute()
            if node.Shape.isNull() or not node.Shape.isValid() or not node.Shape.Solids:
                raise ValueError('Invalid or empty exact solid at '+name)
        result=nodes[spec.get('result',spec['nodes'][-1]['name'])]
        doc.saveAs(str((destination/'model.FCStd').resolve()))
        result.Shape.exportStep(str(destination/'model.step'))
        check=Part.Shape();check.read(str(destination/'model.step'))
        from .importer import equivalent
        verification=equivalent(result.Shape,check)
        if verification is None:raise ValueError('STEP round-trip geometry mismatch.')
        report=dict(volume_mm3=result.Shape.Volume,solids=len(result.Shape.Solids),nodes=len(nodes),verification=verification)
        (destination/'parameters.json').write_text(json.dumps(spec,indent=2)+'\n')
        (destination/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:App.closeDocument(doc.Name)

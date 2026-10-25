# DESCRIPTION: Deterministic exact-solid CAD ingestion and verified feature recognition.
"""Runs under FreeCAD only; never executes source CAD macros or native Python proxies."""
import hashlib
import json
import math
import pprint
from pathlib import Path
import sys
import zipfile
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(sys.executable).resolve().parents[1]/'lib'))
import FreeCAD as App
import Part
from .common import ROOT, identity, sha, write_json

VERSION='1.1.0'
CREATED='2026-09-13 17:38 MDT | 2026-09-13 19:38 EDT | 2026-09-13 23:38 Zulu'
MAX_BYTES=128*1024*1024
MAX_PARTS=512


def vector(v):
    """Use full Python float precision; only sort keys, never geometry, are rounded."""
    return [float(v.x),float(v.y),float(v.z)]


def bounds(shape):
    """Use geometric rather than control-point bounds in mm."""
    b=shape.optimalBoundingBox(False,False)
    return dict(min=[b.XMin,b.YMin,b.ZMin],max=[b.XMax,b.YMax,b.ZMax])


def equivalent(original,candidate):
    """Accept reconstructed solids only after independent material and location checks.

    No tolerance-based healing or fitting is applied. The finite comparison
    thresholds accommodate OpenCASCADE arithmetic, not approximate surfaces.
    """
    if candidate.isNull() or not candidate.isValid() or len(candidate.Solids)!=len(original.Solids):return None
    volume_limit=max(1e-8,original.Volume*1e-10);area_limit=max(1e-7,original.Area*1e-10)
    a,b=bounds(original),bounds(candidate);extent=max(1.,*[y-x for x,y in zip(a['min'],a['max'])]);linear_limit=max(1e-7,extent*1e-9)
    if abs(original.Volume-candidate.Volume)>volume_limit or abs(original.Area-candidate.Area)>area_limit:return None
    if any(abs(x-y)>linear_limit for k in ('min','max') for x,y in zip(a[k],b[k])):return None
    try:removed=original.cut(candidate).Volume;added=candidate.cut(original).Volume
    except Exception:return None
    if max(removed,added)>volume_limit:return None
    return dict(removed_mm3=removed,added_mm3=added,volume_tolerance_mm3=volume_limit,linear_tolerance_mm=linear_limit,area_tolerance_mm2=area_limit)


def native_shapes(path,work):
    """Read stored native BREP and parent transforms without restoring executable objects."""
    with zipfile.ZipFile(path) as archive:
        if len(archive.infolist())>20000 or sum(x.file_size for x in archive.infolist())>MAX_BYTES*4:raise ValueError('Native CAD archive exceeds the import limits.')
        if len(set(archive.namelist()))!=len(archive.namelist()):raise ValueError('Native CAD has duplicate archive members.')
        root=ET.fromstring(archive.read('Document.xml'))
        types={o.get('name'):o.get('type') for o in root.findall('Objects/Object')}
        objects={o.get('name'):o for o in root.findall('ObjectData/Object')}
        props={name:{p.get('name'):p for p in o.findall('Properties/Property')} for name,o in objects.items()}
        parents={}
        for name,values in props.items():
            if types.get(name) in ('App::Part','PartDesign::Body'):
                for link in values.get('Group',ET.Element('empty')).findall('LinkList/Link'):
                    child=link.get('value')
                    if child in parents:raise ValueError('Native component has ambiguous assembly parents; export STEP first.')
                    parents[child]=name
        def visible(name):
            """Apply explicit object and container visibility while ignoring GUI state."""
            seen=set()
            while name:
                if name in seen:raise ValueError('Cyclic native assembly.')
                seen.add(name);value=props[name].get('Visibility',ET.Element('empty')).find('Bool')
                if value is not None and value.get('value')=='false':return False
                name=parents.get(name)
            return True
        def placement(name):
            """Stored BREP already includes the object's own placement; add only parents."""
            result=App.Placement();seen=set()
            while name:
                if name in seen:raise ValueError('Cyclic native placement.')
                seen.add(name);element=props[name].get('Placement',ET.Element('empty')).find('PropertyPlacement')
                if element is not None:
                    xyz=[float(element.get('P'+axis,'0')) for axis in 'xyz'];q=[float(element.get('Q'+str(i),'1' if i==3 else '0')) for i in range(4)]
                    result=App.Placement(App.Vector(*xyz),App.Rotation(*q)).multiply(result)
                name=parents.get(name)
            return result
        candidates={name for name,v in props.items() if 'Shape' in v and visible(name)}
        # A Body/Compound's final shape already includes its construction objects.
        referenced={d.get('Name') for dep in root.findall('Objects/ObjectDeps') if dep.get('Name') in candidates for d in dep.findall('Dep')}
        selected=sorted(candidates-referenced)
        for name in types:
            if types[name].startswith('App::Link') and visible(name):raise ValueError('Native linked instances require STEP export before deterministic import.')
        result=[]
        for index,name in enumerate(selected):
            entry=props[name]['Shape'].find('Part')
            if entry is None or not entry.get('file'):continue
            filename=entry.get('file')
            if filename not in archive.namelist() or not filename.endswith(('.brp','.brep')):raise ValueError('Missing native BREP payload.')
            target=work/('native_'+str(index)+'.brep');target.write_bytes(archive.read(filename));shape=Part.Shape();shape.read(str(target))
            if shape.isNull():continue
            if not shape.Solids:raise ValueError('Native CAD contains an unsupported final open surface or wire; export only closed solids first.')
            ancestor=parents.get(name)
            # Children of a selected Body would duplicate its complete solid.
            if ancestor in candidates:continue
            if ancestor:shape.Placement=placement(ancestor).multiply(shape.Placement)
            label=props[name].get('Label',ET.Element('empty')).find('String')
            result.append((label.get('value',name) if label is not None else name,shape))
        return result


def load_solids(path,work):
    """Normalize units through the CAD reader and reject loss of non-solid geometry."""
    if path.suffix.lower()=='.fcstd':items=native_shapes(path,work)
    else:
        shape=Part.Shape();shape.read(str(path));items=[('Component',shape)]
    def closed_components(shape):
        """Reject loose wires/vertices as well as surfaces instead of silently omitting them."""
        if shape.ShapeType=='Solid':return True
        if shape.ShapeType not in ('Compound','CompSolid'):return False
        children=shape.childShapes()
        return bool(children) and all(closed_components(child) for child in children)
    solids=[]
    for label,shape in items:
        if shape.isNull() or not shape.isValid() or not shape.Solids:raise ValueError('Input must contain valid closed solids; open surfaces are not converted or healed.')
        if not closed_components(shape):raise ValueError('Input includes non-solid geometry; importing only its solids would lose geometry.')
        if len(shape.Faces)>sum(len(s.Faces) for s in shape.Solids):raise ValueError('Input includes loose surfaces; importing only its solids would lose geometry.')
        for solid in shape.Solids:
            if solid.Volume<=0:raise ValueError('Input contains non-positive material volume.')
            solids.append((label,solid))
    if not solids or len(solids)>MAX_PARTS:raise ValueError('Import requires 1–512 closed solid components.')
    # Geometric order is independent of source filenames and output directories.
    return sorted(solids,key=lambda item:(*[round(x,10) for x in vector(item[1].CenterOfMass)],round(item[1].Volume,10),item[0]))


def profile(face,normal):
    """Serialize exact planar lines/circles/arcs into local XY Sketcher geometry."""
    origin=face.CenterOfMass;rotation=App.Rotation(App.Vector(0,0,1),normal)
    transform=App.Placement(origin,rotation).inverse();local=face.copy();local.Placement=transform.multiply(local.Placement)
    rows=[]
    for wire in local.Wires:
        for edge in wire.OrderedEdges:
            curve=edge.Curve;kind=type(curve).__name__
            if kind in ('Line','LineSegment'):
                rows.append(dict(kind='line',start=vector(edge.Vertexes[0].Point),end=vector(edge.Vertexes[-1].Point)))
            elif kind=='Circle':
                if edge.isClosed():rows.append(dict(kind='circle',center=vector(curve.Center),radius=float(curve.Radius)))
                else:rows.append(dict(kind='arc',start=vector(edge.valueAt(edge.FirstParameter)),middle=vector(edge.valueAt((edge.FirstParameter+edge.LastParameter)/2)),end=vector(edge.valueAt(edge.LastParameter))))
            else:raise ValueError('Profile includes unsupported exact curve: '+kind)
    if not rows or len(rows)>200:raise ValueError('Profile exceeds the recognition limit.')
    # Weld only already-coincident line/arc endpoints; this does not fit curves.
    endpoints=[];coincident=[]
    for i,row in enumerate(rows):
        if row['kind']=='circle':continue
        for position,key in ((1,'start'),(2,'end')):
            for j,jp,point in endpoints:
                if j!=i and sum((a-b)**2 for a,b in zip(point,row[key]))<1e-16:
                    coincident.append([j,jp,i,position]);break
            endpoints.append((i,position,row[key]))
    return dict(origin=vector(origin),quaternion=list(rotation.Q),profile=rows,coincident=coincident)


def primitive(shape):
    """Recognize complete analytic solids by class and exact material equivalence, never fitting."""
    faces=shape.Faces;kinds=[type(f.Surface).__name__ for f in faces]
    def axial(axis):
        """Choose a deterministic sign for the same geometric axis in source files."""
        axis=App.Vector(axis);axis.normalize()
        for value in (axis.z,axis.x,axis.y):
            if abs(value)>1e-10:
                if value<0:axis=-axis
                break
        return axis
    def accept(candidate,kind,controls,origin,rotation):
        """Return editable controls only when volume, area, bounds and Boolean differences agree."""
        check=equivalent(shape,candidate)
        return dict(kind=kind,controls=controls,origin=vector(origin),quaternion=list(rotation.Q),verification=check,**controls) if check else None
    if len(faces)==1 and kinds[0]=='Sphere':
        s=faces[0].Surface
        return accept(Part.makeSphere(s.Radius,s.Center),'sphere',dict(radius_mm=s.Radius),s.Center,App.Rotation())
    if len(faces)==1 and kinds[0]=='Toroid':
        s=faces[0].Surface;axis=axial(s.Axis)
        return accept(Part.makeTorus(s.MajorRadius,s.MinorRadius,s.Center,axis),'torus',
                      dict(major_radius_mm=s.MajorRadius,minor_radius_mm=s.MinorRadius),s.Center,App.Rotation(App.Vector(0,0,1),axis))
    # A single cylindrical side and two disks cannot silently become a tube;
    # final equivalence rejects holes, sectors or additional cuts.
    if kinds.count('Cylinder')==1 and kinds.count('Plane')==2 and len(faces)==3:
        s=faces[kinds.index('Cylinder')].Surface;axis=axial(s.Axis)
        ends=sorted([f.CenterOfMass for f,k in zip(faces,kinds) if k=='Plane'],key=lambda v:v.dot(axis));origin=ends[0];height=(ends[1]-origin).dot(axis)
        return accept(Part.makeCylinder(s.Radius,height,origin,axis),'cylinder',dict(radius_mm=s.Radius,height_mm=height),origin,App.Rotation(App.Vector(0,0,1),axis))
    if kinds.count('Cone')==1 and kinds.count('Plane') in (1,2) and len(faces)==kinds.count('Plane')+1:
        s=faces[kinds.index('Cone')].Surface;axis=axial(s.Axis);ends=[]
        for face,kind in zip(faces,kinds):
            if kind!='Plane':continue
            circles=[e.Curve for e in face.Edges if type(e.Curve).__name__=='Circle']
            if len(circles)!=1:return None
            ends.append((circles[0].Center,circles[0].Radius))
        if len(ends)==1:ends.append((s.Apex,0.))
        ends.sort(key=lambda row:row[0].dot(axis));origin,r1=ends[0];end,r2=ends[1];height=(end-origin).dot(axis)
        return accept(Part.makeCone(r1,r2,height,origin,axis),'cone',dict(radius_1_mm=r1,radius_2_mm=r2,height_mm=height),origin,App.Rotation(App.Vector(0,0,1),axis))
    if len(faces)==6 and all(k=='Plane' for k in kinds) and len(shape.Vertexes)==8 and len(shape.Edges)==12:
        origin=min((v.Point for v in shape.Vertexes),key=lambda v:(v.x,v.y,v.z));edges=[]
        for edge in shape.Edges:
            if type(edge.Curve).__name__ not in ('Line','LineSegment'):return None
            points=[v.Point for v in edge.Vertexes]
            if len(points)==2:
                if (points[0]-origin).Length<1e-8:edges.append(points[1]-origin)
                elif (points[1]-origin).Length<1e-8:edges.append(points[0]-origin)
        if len(edges)!=3:return None
        edges.sort(key=lambda v:(-abs(v.x/v.Length),-abs(v.y/v.Length),-abs(v.z/v.Length)))
        x,y,z=edges
        if max(abs(x.dot(y)/(x.Length*y.Length)),abs(x.dot(z)/(x.Length*z.Length)),abs(y.dot(z)/(y.Length*z.Length)))>1e-10:return None
        if x.cross(y).dot(z)<0:y,z=z,y
        rotation=App.Rotation(x,y,z,'ZXY');candidate=Part.makeBox(x.Length,y.Length,z.Length);candidate.Placement=App.Placement(origin,rotation)
        return accept(candidate,'box',dict(length_mm=x.Length,width_mm=y.Length,height_mm=z.Length),origin,rotation)
    return None


def recognize(shape):
    """Try a fixed ordered set of exact feature explanations, preserving unsupported solids."""
    attempts=[];faces=[]
    try:
        found=primitive(shape)
        if found:return found
    except Exception as error:attempts.append('Primitive rejected: '+str(error))
    for index,face in enumerate(shape.Faces):
        if type(face.Surface).__name__!='Plane' or len(face.Edges)>200:continue
        normal=face.normalAt(0,0);origin=face.CenterOfMass
        offsets=[(v.Point-origin).dot(normal) for v in shape.Vertexes]
        if not offsets:continue
        if max(offsets)<1e-6:normal=-normal;offsets=[-v for v in offsets]
        if min(offsets)<-1e-6:continue
        depth=max(offsets)
        if depth<1e-6:continue
        # Prefer a horizontal sketch, then other axes, and larger end faces.
        key=(-round(normal.z,8),-round(normal.x,8),-round(normal.y,8),-round(face.Area,8),index)
        faces.append((key,face,normal,depth))
    for _,face,normal,depth in sorted(faces,key=lambda x:x[0])[:12]:
        try:
            detail=profile(face,normal);candidate=face.extrude(normal*depth)
            check=equivalent(shape,candidate)
            if check:return dict(kind='pad',depth_mm=depth,controls=dict(depth_mm=depth,profile_scale=1.),verification=check,**detail)
        except Exception as error:attempts.append(str(error))
    return dict(kind='brep',controls={},reason='No supported complete primitive or exact extrusion passed; original solid retained.',recognition_notes=sorted(set(attempts))[:4])


def deterministic_zip(folder,target):
    """Canonical order and fixed ZIP metadata make repeated packages byte-identical."""
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in sorted(p for p in folder.rglob('*') if p.is_file()):
            info=zipfile.ZipInfo(path.relative_to(folder).as_posix(),(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16;info.create_system=3;archive.writestr(info,path.read_bytes(),compresslevel=9)


def convert(source,destination,name):
    """Generate and verify a deterministic independent Python harness before publication."""
    import re
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,59}', name):raise ValueError('Use a lowercase Python identifier for the model name.')
    source=Path(source);destination=Path(destination)
    if source.suffix.lower() not in ('.step','.stp','.brep','.brp','.fcstd'):raise ValueError('Unsupported exact-CAD format.')
    if source.stat().st_size > MAX_BYTES:raise ValueError('Input exceeds the size limit.')
    destination.mkdir(parents=True,exist_ok=False)
    work=destination/'scratch';work.mkdir();plugin=destination/'harness';plugin.mkdir();(plugin/'geometry').mkdir()
    parts=[]
    for index,(label,shape) in enumerate(load_solids(source,work),1):
        filename=f'geometry/component_{index:03d}.brep';shape.exportBrep(str(plugin/filename))
        restored=Part.Shape();restored.read(str(plugin/filename))
        check=equivalent(shape,restored)
        if check is None:raise ValueError('Stored original BREP differs from input geometry.')
        part=dict(name=f'component_{index:03d}',label=label if label!='Component' else f'Component {index}',brep=filename,color='#78bcb5',
                  volume_mm3=shape.Volume,bounds_mm=bounds(shape),pivot_mm=vector(shape.CenterOfMass),original_sha256=sha(plugin/filename),**recognize(shape))
        parts.append(part)
    data=dict(converter=VERSION,created=CREATED,input_sha256=sha(source),format=source.suffix.lower(),
              kernel=dict(freecad='.'.join(App.Version()[:3]),occ=Part.OCC_VERSION),units='mm',parts=parts)
    template=(ROOT/'cad/rebuild_template.py.txt').read_text()
    def write_model():
        """Regenerate fixed Python after any sketch that fails native reconstruction falls back."""
        code=template.replace('__MODEL_ID__',name).replace('__MODEL_TITLE__',name.replace('_',' ').title()).replace('__CREATED__',CREATED).replace('__MODEL_DATA__',pprint.pformat(data,sort_dicts=True,width=100))
        (plugin/'model.py').write_text(code)
    write_model()
    # Validate the actual Sketcher/Pad document, not only a surface extrusion.
    from .common import Model
    model=Model(plugin/'model.py')
    for part in parts:
        # Validate each reconstructed module independently. A solver rejection
        # preserves this module's BREP without discarding other valid features.
        model.module.DATA=dict(data,parts=[part]);doc=None;check=None
        try:
            # Force the proposed editable representation. Default preview keeps
            # originals, which would otherwise make this verification tautological.
            doc=model.module.rebuild_document(model.module.defaults(),recognized=True)
            original=Part.Shape();original.read(str(plugin/part['brep']))
            check=equivalent(original,doc.getObject(part['name']).Shape)
        except Exception:pass
        finally:
            if doc:App.closeDocument(doc.Name)
        if check is None:
            part.update(kind='brep',controls={},reason='Native feature reconstruction did not match; exact original retained.')
            for key in ('profile','coincident','quaternion','origin','depth_mm','radius_mm','center','verification'):part.pop(key,None)
        else:part['verification']=check
    write_model()
    model=Model(plugin/'model.py');rebuilt=model.module.build_node('assembly',model.normalize(),{})
    for part,record in zip(rebuilt,parts):
        original=Part.Shape();original.read(str(plugin/record['brep']))
        if equivalent(original,part['shape']) is None:raise ValueError('Final reconstructed component differs from the input.')
    result=dict(PROJECT='Engineering Harness CAD import',CREATED=CREATED,DESCRIPTION='Deterministic conversion coverage and original geometry checks',
                converter=VERSION,input_sha256=data['input_sha256'],kernel=data['kernel'],units='mm',model=name,parts=parts,
                reconstructed=sum(p['kind']!='brep' for p in parts),preserved=sum(p['kind']=='brep' for p in parts),all_geometry_verified=True,original_components=len(parts),
                policy='Retain original BREP at default dimensions; expose verified primitives/extrusions and independent component transforms.',
                determinism='Identical input bytes, model identifier, converter and kernel produce identical Python/package bytes. Native FCStd/STEP export container metadata is not part of this guarantee.')
    write_json(plugin/'conversion.json',result)
    recipe=model.recipe();recipe['CREATED']=CREATED;write_json(plugin/'example.json',recipe)
    header=f'# PROJECT: {name}\n# CREATED: {CREATED}\n# DESCRIPTION: Independently editable reconstructed CAD harness.\n'
    (plugin/'VERSION').write_text(header+'1.1.0\n')
    (plugin/'rebuild.py').write_text(header+'''import argparse
import importlib.util
import json
from pathlib import Path
import sys

# FreeCAD's bundled interpreter exposes its modules from the adjacent lib folder.
sys.path.append(str(Path(sys.executable).resolve().parents[1]/'lib'))
import FreeCAD as App
ROOT=Path(__file__).resolve().parent

def main():
    """Regenerate native editable CAD from Python, optionally using a studio recipe."""
    parser=argparse.ArgumentParser();parser.add_argument('--recipe',type=Path);parser.add_argument('--output',type=Path,default=Path('reconstructed.FCStd'));args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('reconstructed_harness',ROOT/'model.py');model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
    raw=json.loads(args.recipe.read_text()) if args.recipe else {};parameters=raw.get('parameters',raw)
    doc=model.rebuild_document(parameters,recognized=True,originals=True)
    try:doc.saveAs(str(args.output.resolve()));print('Rebuilt '+str(args.output))
    finally:App.closeDocument(doc.Name)

if __name__=='__main__':main()
''')
    (plugin/'README.md').write_text(f'<!-- PROJECT: {name}\nCREATED: {CREATED}\nDESCRIPTION: Verified deterministic CAD reconstruction. -->\n\n# {name.replace("_"," ").title()}\n\n'
        f'{result["reconstructed"]} components reconstructed as editable native features; {result["preserved"]} retained as exact BREP. Every component passed original-geometry comparisons. See `conversion.json` for coverage and numerical comparison limits.\n\n'
        'Default builds read the retained original solids. Changing a recognized dimension explicitly builds the verified primitive or extrusion. Every component supports independent move, rotate and scale; world-mm translations are applied after scale. Native exports expose recognized features with original master files retained. `model.py` contains the actual primitive dimensions, profile curves, placements, feature controls and Python builders. `geometry/` retains the original exact solids for verification and unsupported features. The original design history and constraints are not inferred.\n\n'
        'Run `rebuild.py` with a Python interpreter that provides FreeCAD, Part and Sketcher, optionally `--recipe example.json --output reconstructed.FCStd`. Use the FreeCAD interpreter selected for this release. The generated model is independent of the original application.\n\n'
        'All geometry is millimetres. Assembly scale applies to all modules about the shared world origin. Component edits invalidate assumed mating fit; hardware standards, clearances, loads and physical scaling require engineering review. Original geometry is the default configuration.\n')
    deterministic_zip(plugin,destination/'python_harness.zip')
    write_json(destination/'conversion_report.json',result)
    write_json(destination/'report.json',dict(result,state='complete',files=['python_harness.zip','conversion_report.json'],
        artifacts={f.relative_to(destination).as_posix():sha(f) for f in [destination/'python_harness.zip',destination/'conversion_report.json',*sorted(p for p in plugin.rglob('*') if p.is_file())]}))


if __name__=='__main__':convert(*sys.argv[1:])

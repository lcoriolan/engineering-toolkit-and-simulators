import base64,csv,hashlib,json,math,os,re,signal,subprocess,threading,time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from ..execution import operation as _simulation_operation,report_failure as _simulation_failure
ROOT=Path(os.environ.get('ENGINEERING_SIM_DATA') or Path.cwd()/'.engineering-sim')
EXE=Path(os.environ.get('SU2_CFD') or 'SU2_CFD')
RUNS={};LOCK=threading.Lock()
class _Metadata:
    @staticmethod
    def stamp():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    @staticmethod
    def metadata(description):return dict(description=description,created=_Metadata.stamp())
model=_Metadata()
def normalize(raw):
    """Validate physical inputs and bounded volume data; never accept file paths."""
    r={}
    for key,lo,hi,default in [('speed_mph',1,2000,500),('incidence_deg',-90,90,0),('sideslip_deg',-90,90,0),('temperature_k',180,330,288.15),('pressure_pa',1000,110000,101325),('iterations',20,5000,300)]:
        value=raw.get(key,default)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:raise ValueError(f'{key} must be {lo}–{hi}')
        r[key]=value
    r['iterations']=int(r['iterations']);r['geometry_id']='user_mask';r['state']='supplied';entry={'entry':{'name':'User-supplied solid mask'}}
    grid=raw['grid']
    if len(grid)!=3 or any(type(v) is not int or not 16<=v<=160 for v in grid) or math.prod(grid)>1100000:raise ValueError('Unsupported volume grid')
    mask=base64.b64decode(raw['mask_base64'],validate=True)
    if len(mask)!=math.prod(grid) or any(v not in [0,255] for v in mask):raise ValueError('Mask size/content mismatch')
    a=np.frombuffer(mask,dtype=np.uint8).reshape(grid[2],grid[1],grid[0])
    if not a.any() or a.all() or any(np.any(v) for v in [a[0],a[-1],a[:,0],a[:,-1],a[:,:,0],a[:,:,-1]]):raise ValueError('Airframe must be fully inside the flow volume')
    r.update(grid=grid,cell_mm=float(raw['cell_mm']),length_mm=float(raw['length_mm']),mask_sha256=hashlib.sha256(mask).hexdigest(),geometry=entry['entry'])
    if not .01<r['cell_mm']<1000 or not 1<r['length_mm']<10000:raise ValueError('Invalid physical mesh dimensions')
    r['speed_m_s']=r['speed_mph']*.44704;r['mach']=r['speed_m_s']/math.sqrt(1.4*287.058*r['temperature_k'])
    r['density_kg_m3']=r['pressure_pa']/(287.058*r['temperature_k'])
    return r,a

def write_mesh(folder,r,mask):
    """Build positive-volume hexahedra in metres with Euler-wall/farfield markers.

    Eight geometrically graded layers per face have widths 2dx..256dx.
    Only vertices used by fluid cells are written; sealed solid interiors have
    no zero-volume/isolated nodes. The browser cell map is saved for review.
    """
    pad=8;dx=r['cell_mm']/1000;nx0,ny0,nz0=r['grid'];body=np.pad(mask!=0,pad,constant_values=False);nz,ny,nx=body.shape
    z,y,x=np.nonzero(~body);base=x+(nx+1)*(y+(ny+1)*z)
    offsets=np.array([0,1,nx+2,nx+1,(nx+1)*(ny+1),(nx+1)*(ny+1)+1,(nx+1)*(ny+1)+nx+2,(nx+1)*(ny+1)+nx+1])
    old=base[:,None]+offsets;used=np.unique(old);mapping=np.full((nx+1)*(ny+1)*(nz+1),-1,dtype=np.int64);mapping[used]=np.arange(len(used));cells=mapping[old]
    widths=dx*2.**np.arange(1,pad+1)
    def axis(count):
        """Node coordinates preserve the unpadded, uniform display grid exactly."""
        return np.r_[-np.cumsum(widths)[::-1],np.arange(count+1)*dx,count*dx+np.cumsum(widths)]
    ax,ay,az=axis(nx0),axis(ny0),axis(nz0)
    points=np.c_[ax[used%(nx+1)],ay[(used//(nx+1))%(ny+1)],az[used//((nx+1)*(ny+1))]]
    faces=[(-1,0,0,[0,4,7,3]),(1,0,0,[1,2,6,5]),(0,-1,0,[0,1,5,4]),(0,1,0,[3,7,6,2]),(0,0,-1,[0,3,2,1]),(0,0,1,[4,5,6,7])]
    walls=[];far=[]
    for sx,sy,sz,ids in faces:
        xx,yy,zz=x+sx,y+sy,z+sz;outside=(xx<0)|(xx>=nx)|(yy<0)|(yy>=ny)|(zz<0)|(zz>=nz)
        inside=np.clip(xx,0,nx-1);occupied=body[np.clip(zz,0,nz-1),np.clip(yy,0,ny-1),inside]&~outside
        far.append(cells[outside][:,ids]);walls.append(cells[occupied][:,ids])
    wall=np.concatenate(walls);far=np.concatenate(far)
    with (folder/'mesh.su2').open('w') as f:
        f.write('% Generic voxel-flow case\n% CREATED: '+model.stamp()+'\n% DESCRIPTION: Conservative Cartesian full-airframe fluid mesh in metres\nNDIME= 3\nNELEM= '+str(len(cells))+'\n')
        np.savetxt(f,np.c_[np.full(len(cells),12),cells],fmt='%d')
        f.write('NPOIN= '+str(len(points))+'\n');np.savetxt(f,points,fmt='%.12g')
        f.write('NMARK= 2\n')
        for name,faceset in [('airframe',wall),('farfield',far)]:f.write('MARKER_TAG= '+name+'\nMARKER_ELEMS= '+str(len(faceset))+'\n');np.savetxt(f,np.c_[np.full(len(faceset),9),faceset],fmt='%d')
    # Store the original volume's eight point IDs, including -1 only for solid
    # cells; these have no displayed fluid value and are explicitly masked.
    zz,yy,xx=np.indices((nz0,ny0,nx0));base=(xx+pad)+(nx+1)*((yy+pad)+(ny+1)*(zz+pad));display=mapping[base[...,None]+offsets]
    np.save(folder/'display_point_ids.npy',display)
    np.save(folder/'mesh_points.npy',points)
    report=dict(model.metadata('CFD mesh quality and discretization disclosure'),fluid_cells=len(cells),nodes=len(points),wall_faces=len(wall),farfield_faces=len(far),cell_mm=r['cell_mm'],length_cells=r['length_mm']/r['cell_mm'],
        positive_orthogonal_cells=bool(np.all(np.diff(ax)>0) and np.all(np.diff(ay)>0) and np.all(np.diff(az)>0)),farfield_padding_lengths=float(sum(widths)/(r['length_mm']/1000)),adjacent_spacing_ratio_max=2.,
        boundary_representation='Staircase surface, conservatively thickened thin features',boundary_layer_mesh=False,mesh_independence_demonstrated=False,production_mesh_qualified=False)
    (folder/'MESH_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');return report

def configuration(r):
    """Set physical compressible flow with a stationary, already-rotated body.

    AOA/SIDESLIP are zero in SU2 because pose is embedded in the supplied mask.
    The original requested angles remain in CASE.json. Euler cannot establish
    viscous drag, high-incidence separated flow or thermal material suitability.
    """
    return '\n'.join(['% Generic voxel-flow case','% CREATED: '+model.stamp(),'% DESCRIPTION: Full-airframe compressible Euler experiment, not production-qualified',
        'SOLVER= EULER','MATH_PROBLEM= DIRECT','RESTART_SOL= NO','MACH_NUMBER= '+str(r['mach']),'AOA= 0','SIDESLIP_ANGLE= 0','FREESTREAM_TEMPERATURE= '+str(r['temperature_k']),
        'FREESTREAM_PRESSURE= '+str(r['pressure_pa']),'GAMMA_VALUE= 1.4','GAS_CONSTANT= 287.058','REF_DIMENSIONALIZATION= DIMENSIONAL',
        'REF_ORIGIN_MOMENT_X= '+str(r['grid'][0]*.45*r['cell_mm']/1000),'REF_ORIGIN_MOMENT_Y= '+str(r['grid'][1]*.5*r['cell_mm']/1000),'REF_ORIGIN_MOMENT_Z= '+str(r['grid'][2]*.5*r['cell_mm']/1000),'REF_LENGTH= '+str(r['length_mm']/1000),'REF_AREA= '+str((r['length_mm']/1000)**2),'MARKER_EULER= (airframe)','MARKER_FAR= (farfield)','MARKER_MONITORING= (airframe)','MARKER_PLOTTING= (airframe)',
        'NUM_METHOD_GRAD= WEIGHTED_LEAST_SQUARES','CFL_NUMBER= 2.0','CFL_ADAPT= NO','ITER= '+str(r['iterations']),'LINEAR_SOLVER= FGMRES','LINEAR_SOLVER_PREC= ILU','LINEAR_SOLVER_ERROR= 0.01','LINEAR_SOLVER_ITER= 10',
        'MGLEVEL= 0','CONV_NUM_METHOD_FLOW= HLLC','MUSCL_FLOW= YES','SLOPE_LIMITER_FLOW= VENKATAKRISHNAN','TIME_DISCRE_FLOW= EULER_IMPLICIT','CONV_FIELD= RMS_DENSITY','CONV_RESIDUAL_MINVAL= -6','CONV_STARTITER= 10',
        'MESH_FILENAME= mesh.su2','MESH_FORMAT= SU2','TABULAR_FORMAT= CSV','CONV_FILENAME= history','VOLUME_FILENAME= volume','SURFACE_FILENAME= surface','OUTPUT_FILES= (PARAVIEW_ASCII, SURFACE_PARAVIEW_ASCII)',
        'VOLUME_OUTPUT= (COORDINATES, SOLUTION, PRIMITIVE)','SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, LIFT, DRAG)','HISTORY_OUTPUT= (ITER, RMS_RES, AERO_COEFF)','OUTPUT_WRT_FREQ= '+str(r['iterations']), ''])

def read_vtk(path):
    """Read SU2 ASCII legacy VTK or XML VTU, validating every field count.

    SU2 8.5 PARAVIEW_ASCII writes legacy .vtk despite the XML format supported
    by its binary writer. Explicit headers avoid interpreting connectivity as data.
    """
    arrays={}
    if path.suffix=='.vtu':
        tree=ET.parse(path);points=np.fromstring(tree.find('.//Points/DataArray').text,sep=' ').reshape(-1,3)
        for a in tree.findall('.//PointData/DataArray'):
            if a.get('format','ascii')!='ascii':raise ValueError('Expected ASCII VTU')
            v=np.fromstring(a.text or '',sep=' ');components=int(a.get('NumberOfComponents','1'));arrays[a.get('Name')]=v.reshape(-1,components) if components>1 else v
    else:
        text=path.read_text();match=re.search(r'POINTS (\d+) \w+\s*\n(.*?)\nCELLS',text,re.S)
        if not match:raise ValueError('Missing VTK points')
        count=int(match[1]);points=np.fromstring(match[2],sep=' ').reshape(count,3)
        headers=list(re.finditer(r'^(SCALARS|VECTORS) ([^\n]+)\n',text,re.M))
        for i,h in enumerate(headers):
            raw=text[h.end():headers[i+1].start() if i+1<len(headers) else len(text)]
            if raw.startswith('LOOKUP_TABLE'):raw=raw.split('\n',1)[1]
            v=np.fromstring(raw,sep=' ');components=3 if h[1]=='VECTORS' else 1
            if v.size!=count*components:raise ValueError('VTK field size mismatch')
            arrays[h[2].split()[0]]=v.reshape(count,components) if components>1 else v
    return points,arrays

def parse_volume(folder,r,mask):
    """Average actual SU2 nodal primitives onto the browser's cubic display grid."""
    path=next(iter(sorted(folder.glob('volume*.vtk'))+sorted(folder.glob('volume*.vtu'))))
    points,arrays=read_vtk(path)
    # SU2 output must preserve the known input point ordering before nodal data
    # can be interpolated by connectivity. Reject reordered/mismatched output.
    expected_path=folder/'mesh_points.npy'
    if expected_path.exists():expected=np.load(expected_path)
    else:
        match=re.search(r'NPOIN=\s*(\d+)\n(.*?)\nNMARK', (folder/'mesh.su2').read_text(),re.S)
        expected=np.fromstring(match[2],sep=' ').reshape(int(match[1]),3)
    if points.shape!=expected.shape:raise ValueError('VTK/input mesh node count mismatch')
    point_error=float(np.max(np.abs(points-expected)))
    if point_error>1e-5*max(1,float(np.max(np.abs(expected)))):raise ValueError('SU2 point order differs from interpolation map')
    ids=np.load(folder/'display_point_ids.npy');fluid=mask==0;sample=ids[fluid]
    if np.any(sample<0):raise ValueError('Fluid interpolation contains an absent mesh point')
    def field(name):
        """Exact named primitive fields avoid guessing pressure from density."""
        return arrays[name][sample].mean(axis=1)
    rho=field('Density');pressure=field('Pressure')
    if 'Velocity' in arrays:velocity=field('Velocity')
    else:
        mom=field('Momentum') if 'Momentum' in arrays else np.c_[field('Momentum_x'),field('Momentum_y'),field('Momentum_z')]
        velocity=mom/rho[:,None]
    if not np.isfinite(velocity).all() or not np.isfinite(rho).all() or not np.isfinite(pressure).all() or np.any(rho<=0) or np.any(pressure<=0):raise ValueError('Nonphysical/nonfinite CFD primitive field; result rejected')
    mach=np.linalg.norm(velocity,axis=1)/np.sqrt(1.4*pressure/rho)
    display=np.zeros((*mask.shape,4),dtype=np.float32);display[fluid,:3]=velocity/r['speed_m_s']*.04;display[fluid,3]=pressure/r['pressure_pa'];display[~fluid,3]=1
    display.astype('<f4').tofile(folder/'field.f32')
    return dict(point_order_max_error_m=point_error,available_fields=list(arrays),finite=bool(np.isfinite(display).all()),min_pressure_pa=float(pressure.min()),max_pressure_pa=float(pressure.max()),min_density=float(rho.min()),max_mach=float(mach.max()),field_file='field.f32')

def runtime_provenance(folder):
    """Hash the executable and immutable inputs immediately before solver launch.

    The record identifies this exact local build and mesh/configuration; solver
    version strings alone cannot distinguish compiler or configuration changes.
    """
    return dict(model.metadata('Exact CFD executable and input hashes before launch'),
                solver_version='8.5.0',binary_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest(),
                numpy_version=np.__version__,inputs={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                for name in ['CASE.json','solid_mask.u8','mesh.su2','case.cfg']})

@_simulation_operation("flow.su2.case",kind="static")
def run_job(job,r,mask):
    """Capture one isolated SU2 run and its true convergence state, even on failure."""
    folder=ROOT/'cfd_runs'/job;folder.mkdir(parents=True,exist_ok=False)
    (folder/'CASE.json').write_text(json.dumps(dict(model.metadata('Complete airframe CFD case'),**r),indent=2)+'\n');mask.tofile(folder/'solid_mask.u8')
    try:
        with LOCK:RUNS[job]['phase']='Meshing'
        mesh=write_mesh(folder,r,mask);(folder/'case.cfg').write_text(configuration(r))
        provenance=runtime_provenance(folder);(folder/'RUN_MANIFEST.json').write_text(json.dumps(provenance,indent=2)+'\n')
        with (folder/'solver.log').open('w') as log:
            proc=subprocess.Popen([str(EXE),'case.cfg','-t','1'],cwd=folder,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with LOCK:RUNS[job].update(phase='Solving compressible Euler',process=proc)
            code=proc.wait(timeout=14400)
        if code:raise RuntimeError('SU2 stopped with exit code '+str(code)+'; inspect solver.log')
        field=parse_volume(folder,r,mask)
        with (folder/'history.csv').open() as f:history=list(csv.DictReader(f))
        last={k.strip().strip('"'):v.strip() for k,v in history[-1].items()};density_key=next(k for k in last if 'rms' in k.lower() and ('rho' in k.lower() or 'density' in k.lower()))
        residual=float(last[density_key]);converged=residual<=-6
        result=dict(model.metadata('SU2 full-airframe result and qualification status'),case=r,mesh=mesh,field=field,final_history=last,density_residual_log10=residual,residual_target_met=converged,
            status='Residual target met; mesh/physics qualification still pending' if converged else 'Iteration limit reached; result is unconverged',
            solver_version='8.5.0',runtime=provenance,production_qualified=False,history_trace=history_trace(folder),
            force_axes='Flow-frame +X drag, +Y upward, +Z lateral. SU2 CL/CSF labels follow SU2 axes; use CFx/CFy/CFz with this mapping.',
            limitations=['Coarse staircase Euler walls; no viscosity, boundary-layer mesh, turbulence or thermal conduction.','Incidence up to 90 degrees is accepted geometry; separated-flow accuracy is not established.','No mesh/time/domain independence or experimental full-airframe validation.','Force coefficient reference area is length squared; coefficients are diagnostic, not design ratings.'])
        (folder/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
        with LOCK:RUNS[job].update(state='complete',phase='Finished',result=result)
    except Exception as error:
        _simulation_failure(error)  # The existing asynchronous failure remains a failure centrally.
        with LOCK:
            proc=RUNS[job].get('process')
            if proc and proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            RUNS[job].update(state='failed',error=str(error))
        (folder/'ERROR.json').write_text(json.dumps(dict(model.metadata('CFD run failure'),error=str(error)),indent=2)+'\n')

def start(raw):
    """Serialize heavy CFD jobs and retain each run under a unique identifier."""
    r,mask=normalize(raw)
    if not EXE.is_file():raise ValueError('Set SU2_CFD to an installed SU2_CFD executable.')
    with LOCK:
        if any(v['state']=='running' for v in RUNS.values()):raise ValueError('A CFD case is running; finish or stop it first.')
        job=time.strftime('%Y%m%d_%H%M%S')+'_'+r['mask_sha256'][:8]+'_'+str(time.time_ns())[-6:];RUNS[job]=dict(job=job,state='running',phase='Preparing',created=model.stamp())
    threading.Thread(target=run_job,args=(job,r,mask),daemon=True).start();return dict(job=job,state='running')

def history_trace(folder):
    """Read a bounded residual trace, tolerating an incomplete actively written row."""
    path=folder/'history.csv'
    if not path.exists():return []
    rows=[]
    with path.open() as f:
        for raw in csv.DictReader(f):
            try:
                row={k.strip().strip('"'):v.strip() for k,v in raw.items() if k and v}
                x=float(row['Inner_Iter']);key=next(k for k in row if 'rms' in k.lower() and ('rho]' in k.lower() or 'density' in k.lower()));y=float(row[key])
                if math.isfinite(x) and math.isfinite(y):rows.append([x,y])
            except (ValueError,KeyError,StopIteration):continue
    return rows[::max(1,len(rows)//200)]+(rows[-1:] if rows else [])

def status(job):
    """Return status and a bounded log tail, keeping process objects private."""
    with LOCK:state={k:v for k,v in RUNS.get(job,{}).items() if k!='process'}
    if not state:raise ValueError('Unknown CFD job')
    log=ROOT/'cfd_runs'/job/'solver.log';state['log_tail']=log.read_text(errors='replace')[-7000:] if log.exists() else '';state['history_trace']=history_trace(log.parent);return state

def cancel(job):
    """Terminate only the selected process that this runner started."""
    with LOCK:
        proc=RUNS.get(job,{}).get('process')
        if proc and proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM);return dict(stopped=True)
    return dict(stopped=False)

def saved_cases():
    """Index only complete local results with their observed convergence status."""
    cases=[]
    for path in sorted((ROOT/'cfd_runs').glob('*/RESULT.json'),reverse=True):
        r=json.loads(path.read_text());c=r['case'];cases.append(dict(id=path.parent.name,name=c['geometry']['name']+' · '+str(c['speed_mph'])+' mph · '+str(c['incidence_deg'])+'°',status=r['status']))
    return dict(model.metadata('Saved complete CFD result catalogue'),cases=cases)

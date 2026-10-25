"""Public JSON execution entry point; every numerical request enters the shared engine."""
import importlib
from .execution import operation
OPERATORS = {
    'science': ('science.worker','_run'),
    'operating_point': ('operating_points','solve'),
    'magnetic_field': ('electromagnetics.permanent_magnet','solve_field'),
    'magnetic_bearing': ('electromagnetics.magnetic_bearing','simulate'),
    'reluctance_drive': ('api','_drive'),
    'atmosphere': ('api','_atmosphere'),
    'room_acoustics': ('api','_room'),
    'world': ('demo','run_world'),
    'cfd_start': ('flow.su2','start'),
    'cfd_status': ('flow.su2','status'),
}
def plain(value):
    import numpy as np
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value
@operation('public.dispatch',kind='orchestration')
def run(operator,arguments):
    if operator not in OPERATORS:raise ValueError('Unknown operator: '+str(operator))
    if not isinstance(arguments,dict):raise ValueError('Arguments must be an object.')
    module,name=OPERATORS[operator]
    function=getattr(importlib.import_module('engineering_sim.'+module),name)
    return plain(function(**arguments))
def _drive(report,**settings):
    from .electromagnetics.reluctance_drive import PhaseMap,simulate
    return simulate(PhaseMap(report),**settings)
def _atmosphere(height_m=0.0,**conditions):
    from .core.air_np import ref_state
    return ref_state(height_m,**conditions)
def _room(length_m,width_m,height_m,absorption,sound_speed_m_s=None,distance_m=1.0):
    from .domains.room_acoustics.room import ShoeboxRoom
    r=ShoeboxRoom(length_m,width_m,height_m,absorption,sound_speed_m_s)
    return dict(volume_m3=r.volume,t60_s=r.t60_sabine(),critical_distance_m=r.critical_distance(),direct_reverberant_ratio=r.drr_diffuse(distance_m))

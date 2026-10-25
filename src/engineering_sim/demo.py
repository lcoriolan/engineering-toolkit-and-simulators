"""Synthetic particle example executed by the actual shared World and Warp body kernel."""
from .execution import operation
@operation('demo.world',kind='transient')
def run_world(duration_s=.1,dt_s=.001,device='cpu',output=None):
    import warp as wp
    wp.config.quiet=True
    from .core.env import Env
    from .core.world import World
    from .core.bodies import Bodies
    if duration_s<=0 or dt_s<=0 or duration_s/dt_s>1000000:raise ValueError('Invalid duration/time step.')
    env=Env(device=device);body=Bodies();body.add_passive((0.,0.,10.),mass=1.,cda=0.,name='synthetic_particle')
    world=World(env.atm,env.wind_obj,[body],dt=dt_s,device=device);world.advance(round(duration_s/dt_s),use_graph=False)
    result=dict(time_s=world.t,steps=world.k,position_m=body.x.numpy()[0].tolist(),velocity_m_s=body.v.numpy()[0].tolist())
    if output:
        import json
        from pathlib import Path
        p=Path(output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
    return result

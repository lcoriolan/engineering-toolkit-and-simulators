import math,copy,hashlib,json
from fractions import Fraction
from functools import lru_cache
import numpy as np
from scipy.interpolate import PchipInterpolator,CubicSpline
from ..execution import operation
ALPHA_CU=0.00393
def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
class PhaseMap:
    """Interpolate one immutable recipe's FE samples; preserve monotone current/flux behavior."""
    def __init__(self,report,parameters=None):
        self.parameters=copy.deepcopy(report['parameters']);rows=report['rows']
        self.recipe_identity=identity(self.parameters)
        topology=report['topology']
        self.phases=int(topology['phases']);self.rotor_poles=int(topology['rotor_poles']);self.stator_poles=int(topology['stator_poles'])
        self.clock_offsets_deg=np.asarray(topology.get('clock_offsets_deg',[0.0]),float)
        self.cartridges=len(self.clock_offsets_deg);self.period_deg=360/self.rotor_poles;self.half_period_deg=self.period_deg/2
        if parameters is not None and identity(parameters)!=self.recipe_identity:
            raise ValueError('Recipe differs from field map; regenerate the field samples.')
        if any(r['state']!='complete' for r in rows):raise ValueError('Incomplete field map.')
        self.angles=np.array(sorted({r['angle'] for r in rows}),float)
        self.currents=np.array([0]+sorted({r['result']['current_a'] for r in rows}))
        pairs={(r['angle'],r['result']['current_a']) for r in rows}
        if len(pairs)!=len(rows) or len(rows)!=len(self.angles)*(len(self.currents)-1):
            raise ValueError('Field map must have one sample at every angle/current pair.')
        if self.angles[0]!=0 or abs(self.angles[-1]-self.half_period_deg)>1e-8:
            raise ValueError('Phase map must cover alignment through half the rotor pitch.')
        values=np.zeros((len(self.angles),len(self.currents)))
        energies=values.copy()
        for r in rows:
            j=np.searchsorted(self.angles,r['angle']);k=np.searchsorted(self.currents,r['result']['current_a'])
            values[j,k]=r['result']['flux_linkage_wb_turn'];energies[j,k]=r['result']['coenergy_j']
        if np.any(np.diff(values,axis=1)<=0):raise ValueError('Field map has non-positive incremental inductance.')
        # Integrating lambda over current gives coenergy, enforcing the same
        # energy potential for circuit torque and electromechanical balance.
        self.flux_i=PchipInterpolator(self.currents,values,axis=1)
        self.energy_i=self.flux_i.antiderivative()
        integrated=self.energy_i(self.currents)-self.energy_i(0)[:,None]
        self.coenergy_integral_relative=float(np.max(abs(integrated[:,1:]/energies[:,1:]-1)))
        self.peak_current=float(report['peak_current_a']);self.r20=float(report['resistance_20c_ohm'])
        if self.r20<=0 or self.peak_current<=0:raise ValueError('Resistance and current limit must be positive.')

    @lru_cache(maxsize=4)
    def dense(self,step_deg=.25):
        """Full periodic map on a uniform mechanical-angle grid, current in amperes."""
        if step_deg<=0 or any(abs(v/step_deg-round(v/step_deg))>1e-8 for v in (self.period_deg/self.phases,self.period_deg)):
            raise ValueError('Angle step must divide phase spacing and rotor pitch exactly.')
        current=np.linspace(0,self.currents[-1],241)
        flux=self.flux_i(current);energy=self.energy_i(current)-self.energy_i(0)[:,None]
        radians=np.deg2rad(self.angles)
        f=CubicSpline(radians,flux,axis=0,bc_type=((1,np.zeros(len(current))),(1,np.zeros(len(current)))))
        w=CubicSpline(radians,energy,axis=0,bc_type=((1,np.zeros(len(current))),(1,np.zeros(len(current)))))
        phase=np.arange(round(self.period_deg/step_deg))*step_deg
        relative=(self.half_period_deg-phase)%self.period_deg-self.half_period_deg
        # One phase bridge feeds equal-current series windings across the
        # cartridges. Sum their flux linkages and coenergies at the recipe's
        # rotor clock offsets. Cross-cartridge axial magnetic coupling is omitted.
        clocks=self.clock_offsets_deg.copy()
        clocks-=clocks.mean()
        linkage=np.zeros((len(phase),len(current)));coenergy=linkage.copy();torque=linkage.copy()
        for clock in clocks:
            local=(relative+clock+self.half_period_deg)%self.period_deg-self.half_period_deg
            distance=np.deg2rad(abs(local))
            linkage+=f(distance);coenergy+=w(distance)
            torque+=-np.sign(local)[:,None]*w(distance,1)
        if np.any(np.diff(linkage,axis=1)<=0):raise ValueError('Angular interpolation lost positive incremental inductance.')
        return dict(current=current,flux=linkage,coenergy=coenergy,torque=torque,relative_deg=relative,step_deg=step_deg)

@operation("motors.reluctance.drive",kind="transient")
def simulate(mapping,voltage,rpm,on_deg=None,off_deg=None,current_scale=1.,temperature_c=100.,step_deg=.25,cycles=7,pwm_hz=None):
    """Periodic fixed-speed drive, SI. Return the settled final electrical period.

    Averaged current control is voltage-limited. Optional PWM uses fixed carrier
    duty calculated from the same regulator; it resolves switching only when the
    time step is sufficiently smaller than its carrier period.
    """
    on_deg=mapping.half_period_deg*7/9 if on_deg is None else on_deg
    off_deg=mapping.half_period_deg/9 if off_deg is None else off_deg
    if cycles<2 or not 0<current_scale or rpm<=0 or voltage<=2 or not 0<=off_deg<on_deg<=mapping.half_period_deg:raise ValueError('Invalid drive operating point.')
    p=mapping.parameters
    if identity(p)!=mapping.recipe_identity:raise ValueError('Field-map recipe changed after loading.')
    q=mapping.phases
    table=mapping.dense(step_deg);n=len(table['relative_deg']);dt=step_deg/(6*rpm);omega=rpm*2*math.pi/60
    if pwm_hz and dt>1/(pwm_hz*15):raise ValueError('PWM needs at least 15 integration steps per switching period.')
    current_grid=table['current'];linkage=table['flux'];energy=table['coenergy'];torque=table['torque']
    resistance=mapping.r20*(1+ALPHA_CU*(temperature_c-20));imax=mapping.peak_current*current_scale
    if imax>current_grid[-1]*.95:raise ValueError('Current command exceeds the sampled field envelope.')
    # PWM and electrical angle can have different periods. Average their
    # common cycle and compare flux at equal carrier phase.
    periods=Fraction(float(pwm_hz)/(rpm*mapping.rotor_poles/60)).limit_denominator(100).denominator if pwm_hz else 1
    if cycles<2*periods:raise ValueError('Need at least two common electrical/PWM cycles.')
    # Phase k belongs to stator pole k (and k+q, ...). Use its physical
    # angular position, preserving the winding-sheet phase identity.
    offsets=np.rint(((np.arange(q)*360/mapping.stator_poles)%mapping.period_deg)/step_deg).astype(int);state=np.zeros(q);end_states=[];records=[]
    for step in range(n*cycles):
        index=(step+offsets)%n;following=(index+1)%n
        amps=np.array([np.interp(state[j],linkage[index[j]],current_grid) for j in range(q)])
        target=np.array([np.interp(imax,current_grid,linkage[following[j]]) for j in range(q)])
        active=(table['relative_deg'][index]>=off_deg)&(table['relative_deg'][index]<=on_deg)
        requested=(target-state)/dt+resistance*amps
        # Two conducting devices give a declared total 2 V drop. The
        # asymmetric bridge connects +bus, freewheel, or -bus; demagnetizing
        # terminal voltage is -bus-2, and freewheel is -2 while current flows.
        bus_fraction=np.where(active,np.clip((requested+2)/voltage,-1,1),-1)
        if pwm_hz:
            carrier=(step*dt*pwm_hz)%1
            bus_fraction=np.sign(bus_fraction)*(carrier<abs(bus_fraction))
        applied=voltage*bus_fraction-2
        applied=np.where((state<=0)&(applied<0),0,applied)
        next_state=np.maximum(0,state+dt*(applied-resistance*amps))
        if any(next_state[j]>linkage[following[j],-1] for j in range(q)):raise ValueError('Drive flux exceeded the FE map; extend the current samples.')
        next_amps=np.array([np.interp(next_state[j],linkage[following[j]],current_grid) for j in range(q)])
        mid=.5*(amps+next_amps)
        phase_torque=np.array([.5*(np.interp(amps[j],current_grid,torque[index[j]])+np.interp(next_amps[j],current_grid,torque[following[j]])) for j in range(q)])
        stored=sum(amps[j]*state[j]-np.interp(amps[j],current_grid,energy[index[j]]) for j in range(q))
        stored_next=sum(next_amps[j]*next_state[j]-np.interp(next_amps[j],current_grid,energy[following[j]]) for j in range(q))
        # Clamp-at-zero can finish demagnetizing before the step ends. Use the
        # realized linkage change to record its effective terminal voltage.
        realized=(next_state-state)/dt+resistance*mid
        terminal=float(realized@mid);copper=float(resistance*(mid@mid));mechanical=float(sum(phase_torque)*omega)
        converter=float(2*np.sum(mid))
        if step>=n*(cycles-periods):records.append(dict(angle_deg=(step%n)*step_deg,current_a=amps.tolist(),torque_nm=float(sum(phase_torque)),
            copper_w=copper,converter_w=converter,dc_power_w=terminal+converter,magnetic_power_w=(stored_next-stored)/dt,
            energy_residual_w=terminal-copper-mechanical-(stored_next-stored)/dt))
        if step%n==n-1:
            periodic=float(np.max(abs(next_state-end_states[-periods]))) if len(end_states)>=periods else float('inf')
            end_states.append(next_state.copy())
        state=next_state
    def mean(key):return float(np.mean([r[key] for r in records]))
    t=np.array([r['torque_nm'] for r in records]);mean_t=float(t.mean())
    return dict(voltage_v=voltage,rpm=rpm,on_deg=on_deg,off_deg=off_deg,current_limit_a=imax,temperature_c=temperature_c,
        phase_resistance_ohm=resistance,connection='series phase windings across cartridges; shared bridge per phase',mean_torque_nm=mean_t,gross_mechanical_w=mean_t*omega,
        ripple_pp_over_mean=float(np.ptp(t)/mean_t) if mean_t>0 else None,peak_current_a=float(np.max([r['current_a'] for r in records])),
        copper_w=mean('copper_w'),converter_w=mean('converter_w'),dc_power_w=mean('dc_power_w'),
        energy_balance_relative=abs(mean('energy_residual_w'))/max(abs(mean_t*omega),1),periodic_flux_error_wb_turn=periodic,
        step_deg=step_deg,pwm_hz=pwm_hz,averaged_electrical_periods=periods,rows=records,
        qualification='FE-map drive simulation; hypothetical steel; independent phases/cartridges; core, mechanical and cooling auxiliaries excluded from gross power.')

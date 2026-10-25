import math
from collections import deque
from ..execution import operation
MU0=4*math.pi*1e-7
def _allocate(force, lower_z, upper_z):
    """Map COM [Fx,Fy,Fz,Mx,My] to [lower X,Y,upper X,Y,thrust] force.

    For small shaft tilts: x(z)=x+z*theta_y and y(z)=y-z*theta_x.
    Hence Mx=-sum(z*Fy), My=sum(z*Fx); this map includes CG asymmetry.
    """
    fx, fy, fz, mx, my = force
    span = upper_z-lower_z
    return [(upper_z*fx-my)/span, (upper_z*fy+mx)/span,
            (my-lower_z*fx)/span, (-mx-lower_z*fy)/span, fz]

def nonlinear_axis_force(actuator_info, plus_a, minus_a, displacement_m, flux_limit_t=None):
    """Ideal two-gap force for derivative checks and nonlinear cross-checks.

    Flux clipping represents a declared engineering ceiling, not a B-H curve.
    Both pole faces of one U carry equal flux; their normal components add.
    """
    a = actuator_info
    projected = displacement_m*a['pole_normal_cosine']
    gaps = (a['gap_m']-projected, a['gap_m']+projected)
    if min(gaps) <= 0:
        raise ValueError('An ideal magnetic gap closed.')
    fluxes = [MU0*a['turns']*current/(2*gap) for current, gap in zip((plus_a, minus_a), gaps)]
    if flux_limit_t is not None:
        fluxes = [min(flux_limit_t, value) for value in fluxes]
    return a['pole_face_area_mm2']*1e-6*a['pole_normal_cosine']/MU0*(fluxes[0]**2-fluxes[1]**2)

@operation("motors.magnetic_bearing.transient",kind="transient")
def simulate(controls, sizing, scenario='bounded_disturbance', force_model='linearized'):
    """Run a finite-time 5-DOF sampled rigid-rotor feedback experiment.

    All ten coils have current/flux/voltage limits and first-order current
    tracking. Position noise, a filtered derivative and whole-sample command
    delay are explicit. Gravity and fixed external/magnetic loads are trimmed;
    imbalance rotates at shaft speed and force/torque pulses remain untrimmed.
    Startup from touchdown is deliberately excluded from this local model.
    """
    if scenario not in ('bounded_disturbance', 'loss_of_power', 'overload'):
        raise ValueError('Choose bounded_disturbance, loss_of_power or overload.')
    if force_model not in ('linearized', 'nonlinear'):
        raise ValueError('Choose linearized or nonlinear magnetic force.')
    # A first-order current expansion around nonzero trim cannot represent
    # zero power: extrapolating to zero current can reverse the calculated force.
    # Always use the ideal quadratic/gap law for this large electrical excursion.
    if scenario == 'loss_of_power':
        force_model = 'nonlinear'
    p = dict(controls)
    radial, thrust, rotor = sizing['radial'], sizing['thrust'], sizing['rotor']
    load = rotor['load_case']
    m, jt, jp = rotor['mass_kg'], rotor['transverse_inertia_kg_m2'], rotor['polar_inertia_kg_m2']
    cg, a = rotor['axial_cg_m'], sizing['radial_half_span_m']
    zl, zu = -a-cg, a-cg
    speed = p['rpm']*2*math.pi/60
    imbalance = load['imbalance_force_n']
    static_direction = 1 if load['radial_gravity_n']+load.get('external_radial_force_n', 0) >= 0 else -1
    static_fx = load['radial_force_n']-imbalance*static_direction
    static_my = load.get('radial_force_moment_about_stack_center_nm', 0)-imbalance*static_direction*cg-static_fx*cg
    static_load = [static_fx, 0, load['axial_force_n'], 0, static_my]
    equilibrium_forces = _allocate([-value for value in static_load], zl, zu)
    axes = [radial]*4+[thrust]
    trim = [equilibrium_forces[i]/axes[i]['current_stiffness_n_per_a'] for i in range(5)]
    # Linearize position stiffness about the actual gravity-loaded currents.
    # The u_trim^2 term matters once gravity consumes significant current.
    loaded_kx = [axes[i]['negative_position_stiffness_n_per_m']*
                 (1+(trim[i]/axes[i]['bias_current_a'])**2) for i in range(5)]
    currents = [[max(0, min(ax['allowed_coil_current_a'], ax['bias_current_a']+sign*trim[i]))
                 for sign in (1, -1)] for i, ax in enumerate(axes)]
    original_currents = [[ax['bias_current_a']+trim[i], ax['bias_current_a']-trim[i]] for i, ax in enumerate(axes)]
    filtered_forces = equilibrium_forces[:]
    # Begin near an established levitated equilibrium. This is not a simulated
    # capture from the backup bearings, which requires a nonlinear contact model.
    q = [10e-6, -7e-6, 5e-6, 12e-6, -9e-6]
    velocity = [0.0]*5
    observed_velocity = [0.0]*5
    last_observation = q[:]
    natural = 2*math.pi*p['amb_position_bandwidth_hz']
    modal_mass = [m, m, m, jt, jt]
    dt = 1/p['amb_sample_hz']
    steps = math.ceil(p['amb_simulation_duration_s']/dt)
    derivative_alpha = 1-math.exp(-2*math.pi*3*p['amb_position_bandwidth_hz']*dt)
    queue = deque([trim[:]]*int(p['amb_delay_samples']))
    saturated_samples, voltage_limited_samples = 0, 0
    maximum_coil_current = 0.0
    peak_radial, peak_axial, peak_axial_catcher, peak_motor_closure = 0.0, 0.0, 0.0, 0.0
    peak_ratio = 0.0
    peak_current_perturbation_ratio = 0.0
    trace = []
    termination = 'completed'
    final_time = 0.0
    for step in range(steps):
        t = step*dt
        # Five physical sensor channels are reconstructed at COM. The axial
        # probe is assumed calibrated for its known off-axis tilt contribution.
        noise = p['amb_sensor_noise_um']*1e-6
        sensors = [q[0]+zl*q[4], q[1]-zl*q[3], q[0]+zu*q[4], q[1]-zu*q[3], q[2]]
        sensed = [value+noise*math.sin(2*math.pi*(83+17*i)*t+i) for i, value in enumerate(sensors)]
        observed = [(zu*sensed[0]-zl*sensed[2])/(zu-zl),
                    (zu*sensed[1]-zl*sensed[3])/(zu-zl), sensed[4],
                    -(sensed[3]-sensed[1])/(zu-zl), (sensed[2]-sensed[0])/(zu-zl)]
        for i in range(5):
            observed_velocity[i] += derivative_alpha*((observed[i]-last_observation[i])/dt-observed_velocity[i])
        last_observation = observed
        desired = [-modal_mass[i]*(natural**2*observed[i]+2*p['amb_damping_ratio']*natural*observed_velocity[i])-static_load[i]
                   for i in range(5)]
        # Counter the nominal skew gyroscopic coupling using measured spin and
        # estimated tilt rates; flexible modes and speed-estimation error remain.
        desired[3] += jp*speed*observed_velocity[4]
        desired[4] -= jp*speed*observed_velocity[3]
        desired_axis = _allocate(desired, zl, zu)
        observed_axis = [observed[0]+zl*observed[4], observed[1]-zl*observed[3],
                         observed[0]+zu*observed[4], observed[1]-zu*observed[3], observed[2]]
        command = [(desired_axis[i]-loaded_kx[i]*observed_axis[i])/axes[i]['current_stiffness_n_per_a'] for i in range(5)]
        queue.append(command)
        delayed = queue.popleft()
        actual_axis = [q[0]+zl*q[4], q[1]-zl*q[3], q[0]+zu*q[4], q[1]-zu*q[3], q[2]]
        forces = []
        any_saturation, any_voltage_limit = False, False
        powered = not (scenario == 'loss_of_power' and t >= .25*p['amb_simulation_duration_s'])
        for i, ax in enumerate(axes):
            bias = ax['bias_current_a']
            clipped = max(-ax['differential_current_limit_a'], min(ax['differential_current_limit_a'], delayed[i]))
            any_saturation |= abs(clipped-delayed[i]) > 1e-9
            for pole, sign in enumerate((1, -1)):
                target = bias+sign*clipped if powered else 0.0
                current = currents[i][pole]
                tracked = current+(1-math.exp(-2*math.pi*p['amb_current_bandwidth_hz']*dt))*(target-current)
                # Bipolar bridge voltage enables active current decay. Power
                # loss uses a modeled passive resistive decay; no hold-up claimed.
                resistance, inductance = ax['resistance_hot_ohm'], ax['inductance_ideal_h']
                bus = p.get('voltage', 48) if powered else 0.0
                # Exact RL and first-order updates avoid a false unstable
                # current loop when an allowed sample step exceeds its tau.
                decay = math.exp(-resistance*dt/inductance)
                lower_current = current*decay-bus/resistance*(1-decay)
                upper_current = current*decay+bus/resistance*(1-decay)
                limited = max(lower_current, min(upper_current, tracked))
                any_voltage_limit |= abs(limited-tracked) > 1e-9
                currents[i][pole] = max(0, min(ax['allowed_coil_current_a'], limited))
                maximum_coil_current = max(maximum_coil_current, currents[i][pole])
                peak_current_perturbation_ratio = max(peak_current_perturbation_ratio,
                    abs(currents[i][pole]-original_currents[i][pole])/bias)
            # Taylor expansion in each coil current about loaded equilibrium.
            # It reduces to Ki*delta(u) for a complementary current change.
            if force_model == 'nonlinear':
                fi = nonlinear_axis_force(ax, *currents[i], actual_axis[i], p['amb_core_flux_limit_t'])
            else:
                k = ax['force_constant_n_m2_a2']/ax['gap_m']**2
                fi = equilibrium_forces[i]+loaded_kx[i]*actual_axis[i]
                fi += 2*k*original_currents[i][0]*(currents[i][0]-original_currents[i][0])
                fi -= 2*k*original_currents[i][1]*(currents[i][1]-original_currents[i][1])
            # Current bandwidth does not guarantee force bandwidth. This
            # separate lag is an explicit surrogate for magnetic diffusion;
            # a measured frequency response may require a higher-order model.
            force_bandwidth = p['amb_radial_force_bandwidth_hz'] if i < 4 else p['amb_thrust_force_bandwidth_hz']
            filtered_forces[i] += (1-math.exp(-2*math.pi*force_bandwidth*dt))*(fi-filtered_forces[i])
            forces.append(filtered_forces[i])
        saturated_samples += any_saturation
        voltage_limited_samples += any_voltage_limit
        external = static_load[:]
        external[0] += imbalance*math.cos(speed*t)
        external[1] += imbalance*math.sin(speed*t)
        if .25*p['amb_simulation_duration_s'] <= t < .5*p['amb_simulation_duration_s']:
            pulse = p['amb_step_disturbance_n']*(100 if scenario == 'overload' else 1)
            external[0] += pulse
            external[2] += pulse
            external[3] += p['amb_step_torque_nm']
            external[4] -= p['amb_step_torque_nm']
        generalized = [forces[0]+forces[2]+external[0], forces[1]+forces[3]+external[1], forces[4]+external[2],
                       -zl*forces[1]-zu*forces[3]+external[3], zl*forces[0]+zu*forces[2]+external[4]]
        acceleration = [generalized[0]/m, generalized[1]/m, generalized[2]/m,
                        (generalized[3]-jp*speed*velocity[4])/jt,
                        (generalized[4]+jp*speed*velocity[3])/jt]
        # Semi-implicit Euler is repeated at the actual control sample step.
        # The test suite compares finer sampling and an intentionally failed case.
        for i in range(5):
            velocity[i] += acceleration[i]*dt
            q[i] += velocity[i]*dt
        backup_z = sizing['touchdown_half_span_m']
        radial_displacement = max(math.hypot(q[0]+z*q[4], q[1]-z*q[3]) for z in (-backup_z-cg, backup_z-cg))
        axial_displacement = abs(q[2])
        axial_catcher_closure = axial_displacement+sizing['thrust_collar_radius_m']*math.hypot(q[3], q[4])
        face_closure = abs(q[2])+sizing['motor_outer_radius_m']*math.hypot(q[3],q[4])
        local_ratio = max(abs(value)/axes[i]['gap_m'] for i, value in enumerate(actual_axis))
        peak_radial, peak_axial = max(peak_radial,radial_displacement),max(peak_axial,axial_displacement)
        peak_motor_closure, peak_ratio = max(peak_motor_closure,face_closure),max(peak_ratio,local_ratio)
        peak_axial_catcher = max(peak_axial_catcher, axial_catcher_closure)
        final_time = t+dt
        if step % max(1, steps//300) == 0:
            trace.append(dict(t_s=final_time, x_um=q[0]*1e6, y_um=q[1]*1e6, z_um=q[2]*1e6,
                              theta_x_urad=q[3]*1e6, theta_y_urad=q[4]*1e6,
                              max_coil_current_a=max(max(pair) for pair in currents)))
        if radial_displacement >= sizing['radial_position_limit_m'] or axial_catcher_closure >= sizing['axial_position_limit_m'] or face_closure >= sizing['motor_face_gap_m']:
            termination = 'position_bound_reached'
            break
    return dict(model='5-DOF rigid rotor with sampled PD, gyroscopic compensation and bounded coil dynamics',
        magnetic_force_model=force_model,
        scenario=scenario, completed=termination=='completed', termination=termination,
        candidate_checks_pass=sizing['all_candidate_checks_pass'],
        simulated_time_s=final_time, sample_hz=p['amb_sample_hz'], delay_samples=int(p['amb_delay_samples']),
        static_gravity_and_load_trim_a=trim, maximum_coil_current_a=maximum_coil_current,
        saturated_samples=saturated_samples, voltage_limited_samples=voltage_limited_samples,
        peak_radial_backup_displacement_um=peak_radial*1e6, peak_axial_displacement_um=peak_axial*1e6,
        peak_axial_catcher_closure_um=peak_axial_catcher*1e6,
        peak_motor_face_closure_um=peak_motor_closure*1e6, peak_displacement_over_magnetic_gap=peak_ratio,
        stayed_in_small_displacement_region=peak_ratio <= .1,
        peak_current_perturbation_over_bias=peak_current_perturbation_ratio,
        stayed_in_small_current_perturbation_region=peak_current_perturbation_ratio <= .1,
        final_position_um=[value*1e6 for value in q[:3]], final_tilt_urad=[value*1e6 for value in q[3:]],
        notes='Initialized near levitated trim. No hardware calibration, flexible modes, contact/landing dynamics, heating transients or disturbance qualification.',
        trace=trace)

using LinearAlgebra
using SatelliteDynamics
using Distributions

"""
Simple Keplerian orbital propagation.

# Arguments
- `eci`: State vector [x, y, z, vx, vy, vz] in ECI frame (km, km/s)
- `n_steps`: Number of steps to propagate
- `cdmscale`: If true, n_steps in units of 1/3 days (8 hours). If false, in seconds.
- `returnosc`: If true, return osculating elements. If false, return Cartesian state.

# Returns
- Propagated state vector or osculating elements
"""
function kepler_propagate(eci, n_steps; cdmscale=true, returnosc=false)
    # if cdmscale, n_steps in units of 1/3 days
    # if not cdmscale, n_steps in units of seconds
    if cdmscale
        t = n_steps * 60 * 60 * 8
    else
        t = n_steps
    end
    oe0 = sCARTtoOSC(eci)
    P = orbit_period(oe0[1])
    orbit_frac_to_advance = (t % P) / P
    radians_to_shift = 2π * orbit_frac_to_advance
    new_mean_anomaly = oe0[6] + radians_to_shift

    # Normalize mean anomaly to [0, 2π]
    while new_mean_anomaly > 2π
        new_mean_anomaly -= 2π
    end
    while new_mean_anomaly < 0
        new_mean_anomaly += 2π
    end
    oe0[6] = new_mean_anomaly
    if returnosc
        return oe0
    else
        return sOSCtoCART(oe0)
    end
end

"""
Apply thrust to a state vector.

# Arguments
- `eci`: State vector [x, y, z, vx, vy, vz] in ECI frame (km, km/s)
- `thrust_magnitude`: Magnitude of thrust (km/s)
- `thrust_direction`: Direction of thrust (`:along_track` or `:anti_along_track`)

# Returns
- New state vector with updated velocity
"""
function apply_thrust(eci, thrust_magnitude, thrust_direction)
    x = eci[1:3]
    v = eci[4:6]
    if thrust_direction == :along_track
        tdir = 1
    else 
        tdir = -1
    end
    thrust = thrust_magnitude * v / norm(v) * tdir
    return vcat(x, v + thrust)
end

"""
Compute evasive maneuver for a given state.

Finds the minimum thrust required to reduce collision probability below threshold
by iteratively increasing thrust magnitude until safe.

# Arguments
- `state::SpacecraftCAState`: Current state
- `unit_dv`: Unit delta-V increment (km/s) for iterative search (default: 1e-3)
- `prebackprop`: Pre-computed backpropagated state (optional)

# Returns
- `(cost, new_state)`: Tuple of maneuver cost (delta-V in km/s) and new state at TCA=0
"""
function evasive_maneuver(state::SpacecraftCAState; unit_dv=1e-3, prebackprop=false)
    # dv is in km/s
    # Cost of maneuvering
    current_Pc = fosterPcState(state)
    
    if state.TCA == 0
        return 0.0, state
    end
    if current_Pc < 1e-5
        # we are already safe, but there should be some cost to maneuvering
        return unit_dv, state
    end

    if prebackprop !== false
        backprop_eci = prebackprop
    else
        backprop_eci = kepler_propagate(state.xs, -state.TCA)
    end

    i = 1
    
    unsafe = true
    new_Pc = current_Pc
    new_eci = state.xs
    
    eci_at_thrust = apply_thrust(backprop_eci, unit_dv, :along_track)
    Σs_rtn = getRTNCovariance(eci_at_thrust, state.Σs)
    Σd_rtn = getRTNCovariance(state.xd, state.Σd)
    Σs_6x6 = zeros(6, 6)
    Σd_6x6 = zeros(6, 6)
    Σs_6x6[1:3, 1:3] = Σs_rtn
    Σd_6x6[1:3, 1:3] = Σd_rtn
    pc_at = fosterPcAnalytical(eci_at_thrust, Σs_6x6, state.xd, Σd_6x6, 
                                object1_radius=state.rs, object2_radius=state.rd)
    eci_aat_thrust = apply_thrust(backprop_eci, unit_dv, :anti_along_track)
    Σs_rtn_aat = getRTNCovariance(eci_aat_thrust, state.Σs)
    Σs_6x6_aat = zeros(6, 6)
    Σs_6x6_aat[1:3, 1:3] = Σs_rtn_aat
    pc_aat = fosterPcAnalytical(eci_aat_thrust, Σs_6x6_aat, state.xd, Σd_6x6, 
                                 object1_radius=state.rs, object2_radius=state.rd)

    thrust_dir = nothing
    if pc_at < pc_aat
        thrust_dir = :along_track
        new_Pc = pc_at
    else
        thrust_dir = :anti_along_track
        new_Pc = pc_aat
    end

    while unsafe
        i += 1
        
        eci_after_thrust = apply_thrust(backprop_eci, i * unit_dv, thrust_dir)
        new_eci = kepler_propagate(eci_after_thrust, state.TCA)
        Σs_rtn_new = getRTNCovariance(new_eci, state.Σs)
        Σd_rtn_new = getRTNCovariance(state.xd, state.Σd)
        Σs_6x6_new = zeros(6, 6)
        Σd_6x6_new = zeros(6, 6)
        Σs_6x6_new[1:3, 1:3] = Σs_rtn_new
        Σd_6x6_new[1:3, 1:3] = Σd_rtn_new
        new_Pc = fosterPcAnalytical(new_eci, Σs_6x6_new, state.xd, Σd_6x6_new, 
                                    object1_radius=state.rs, object2_radius=state.rd)
        
        if new_Pc < 1e-5
            unsafe = false
        end
    end
    
    # Returns new cost to maneuver, and new, modified state, stepped to TCA = 0
    new_state = SpacecraftCAState(
        0,              # TCA = 0 (maneuver complete)
        new_eci,         # New satellite state at TCA
        state.xd,        # Debris state unchanged
        state.Σs,        # Satellite covariance unchanged
        state.Σd,        # Debris covariance unchanged
        state.rs,        # Satellite radius unchanged
        state.rd         # Debris radius unchanged
    )
    return i * unit_dv, new_state
end


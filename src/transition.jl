using POMDPs
using POMDPTools
using Distributions
using LinearAlgebra
using Random
using GaussianFilters
using SatelliteDynamics

function kepler_propagate(eci, n_steps; cdmscale=true)
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
    
    while new_mean_anomaly > 2π
        new_mean_anomaly -= 2π
    end
    while new_mean_anomaly < 0
        new_mean_anomaly += 2π
    end
    oe0[6] = new_mean_anomaly
    return sOSCtoCART(oe0)
end

function apply_thrust_maneuver(eci, thrust_magnitude, thrust_direction)
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

function evasive_maneuver(state::SpacecraftCAState; unit_dv=1e-3, prebackprop=false)
    current_Pc = fosterPcState(state)
    
    if state.TCA == 0
        return 0.0, state
    end
    if current_Pc < 1e-5
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
    
    eci_at_thrust = apply_thrust_maneuver(backprop_eci, unit_dv, :along_track)
    Σs_rtn = getRTNCovariance(eci_at_thrust, state.Σs)
    Σd_rtn = getRTNCovariance(state.xd, state.Σd)
    Σs_6x6 = zeros(6, 6)
    Σd_6x6 = zeros(6, 6)
    Σs_6x6[1:3, 1:3] = Σs_rtn
    Σd_6x6[1:3, 1:3] = Σd_rtn
    pc_at = fosterPcAnalytical(eci_at_thrust, Σs_6x6, state.xd, Σd_6x6, 
                                object1_radius=state.rs, object2_radius=state.rd)
    eci_aat_thrust = apply_thrust_maneuver(backprop_eci, unit_dv, :anti_along_track)
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
        
        eci_after_thrust = apply_thrust_maneuver(backprop_eci, i * unit_dv, thrust_dir)
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
    
    new_state = SpacecraftCAState(
        0,
        new_eci,
        state.xd,
        state.Σs,
        state.Σd,
        state.rs,
        state.rd
    )
    return i * unit_dv, new_state
end

function POMDPs.transition(pomdp::SpacecraftCAPOMDP, s::SpacecraftCAState, a::Symbol)
    @assert s.TCA >= 0 "TCA must be non-negative"
    
    if a == :wait
        return wait_transition(pomdp, s)
    elseif a == :maneuver
        return maneuver_transition(pomdp, s)
    else
        error("Unknown action: $a. Must be :wait or :maneuver")
    end
end

function wait_transition(pomdp::SpacecraftCAPOMDP, s::SpacecraftCAState)
    if s.TCA <= 0
        return Deterministic(s) 
    end
    
    new_TCA = s.TCA - 1
    
    dt_seconds = pomdp.dt_seconds  
    epc_str = pomdp.current_epoch_str 
    epc0 = spaceXEpoch(epc_str)
    
    Ws = dt_seconds * 0.05 * Matrix{Float64}(I, 6, 6)
    Wd = dt_seconds * 0.05 * Matrix{Float64}(I, 6, 6)
    
    dmodel_s = NonlinearDynamicsModel(step, Ws)
    dmodel_d = NonlinearDynamicsModel(step, Wd)
    omodel = NonlinearObservationModel(observe, zeros(6,6))
    ukf_s = UnscentedKalmanFilter(dmodel_s, omodel)
    ukf_d = UnscentedKalmanFilter(dmodel_d, omodel)
    
    b0_s = GaussianBelief(s.xs, Symmetric(s.Σs))
    b0_d = GaussianBelief(s.xd, Symmetric(s.Σd))
    
    u = [0.0]
    bp_s = predictEpc(ukf_s, b0_s, u, epc0, dt_seconds)
    bp_d = predictEpc(ukf_d, b0_d, u, epc0, dt_seconds)
    
    propagated_xs = bp_s.μ
    propagated_Σs = bp_s.Σ
    propagated_xd = bp_d.μ
    propagated_Σd = bp_d.Σ
    
    new_Σs = propagated_Σs * (1 - pomdp.satellite_scale_factor)
    debris_scale_factor = rand(Distributions.Uniform(pomdp.debris_scale_range...))
    new_Σd = propagated_Σd * (1 - debris_scale_factor)
    
    new_state = SpacecraftCAState(
        new_TCA,
        propagated_xs,
        propagated_xd,
        new_Σs,
        new_Σd,
        s.rs,
        s.rd
    )
    
    return Deterministic(new_state)
end
function maneuver_transition(pomdp::SpacecraftCAPOMDP, s::SpacecraftCAState)
    if s.TCA <= 0
        return Deterministic(s) 
    end
    
    _, new_state = evasive_maneuver(s, unit_dv=pomdp.unit_dv)
    
    return Deterministic(new_state)
end


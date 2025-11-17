using POMDPs
using POMDPTools
using Distributions
using LinearAlgebra
using Random
using GaussianFilters

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


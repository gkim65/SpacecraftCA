using POMDPs
using POMDPTools
using Distributions
using LinearAlgebra
using Random
using GaussianFilters
using SatelliteDynamics

function ensure_positive_definite(Σ::Matrix{Float64}, eps=1e-10)
    Σ_sym = Symmetric(Σ)
    eigenvals = eigvals(Σ_sym)
    min_eigenval = minimum(eigenvals)
    if min_eigenval <= eps
        Σ_corrected = Σ_sym + (eps - min_eigenval + 1e-12) * Matrix{Float64}(I, size(Σ, 1), size(Σ, 2))
        return Symmetric(Σ_corrected)
    end
    return Σ_sym
end

function unscented_kalman_filter(pomdp::SpacecraftCAPOMDP, x::Vector{Float64}, C_eci::Matrix{Float64}, u::AbstractVector{<:Number})
    C_eci_pd = ensure_positive_definite(C_eci)
    b0 = GaussianBelief(Float64.(x), C_eci_pd)

    W = diagm([1e-6, 1e-6, 1e-6, 1e-12, 1e-12, 1e-12])
    dmodel = NonlinearDynamicsModel(step, W)
    V = diagm([1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9])
    omodel = NonlinearObservationModel(observe, V)

    ukf = UnscentedKalmanFilter(dmodel, omodel)
    
    N = 2
    T = pomdp.dt_seconds / N
    epc0 = spaceXEpoch(pomdp.current_epoch_str)
    
    predictions = Vector{GaussianBelief}(undef, N)
    predictions[1] = b0

    current_epc = epc0
    for k in 2:N
        predictions[k] = predictEpc(ukf, predictions[k-1], u, current_epc, T)
        current_epc = current_epc + T
    end
    
    return predictions[N]
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
    
    bp_s = unscented_kalman_filter(pomdp, s.xs, s.Σs, [0.0])
    bp_d = unscented_kalman_filter(pomdp, s.xd, s.Σd, [0.0])
    
    propagated_xs = bp_s.μ
    propagated_Σs = Matrix(bp_s.Σ)
    propagated_xd = bp_d.μ
    propagated_Σd = Matrix(bp_d.Σ)
    
    new_Σs = ensure_positive_definite(propagated_Σs * (1 - pomdp.satellite_scale_factor))
    debris_scale_factor = rand(Distributions.Uniform(pomdp.debris_scale_range...))
    new_Σd = ensure_positive_definite(propagated_Σd * (1 - debris_scale_factor))
    
    new_state = SpacecraftCAState(
        new_TCA,
        propagated_xs,
        propagated_xd,
        Matrix(new_Σs),
        Matrix(new_Σd),
        s.rs,
        s.rd
    )
    
    return Deterministic(new_state)
end

function maneuver_transition(pomdp::SpacecraftCAPOMDP, s::SpacecraftCAState)
    if s.TCA <= 0
        return Deterministic(s)
    end
    
    new_TCA = s.TCA - 1
    
    bp_s = unscented_kalman_filter(pomdp, s.xs, s.Σs, [1.0])
    bp_d = unscented_kalman_filter(pomdp, s.xd, s.Σd, [0.0])
    
    propagated_xs = bp_s.μ
    propagated_Σs = Matrix(bp_s.Σ)
    propagated_xd = bp_d.μ
    propagated_Σd = Matrix(bp_d.Σ)
    
    new_Σs = ensure_positive_definite(propagated_Σs * (1 - pomdp.satellite_scale_factor))
    debris_scale_factor = rand(Distributions.Uniform(pomdp.debris_scale_range...))
    new_Σd = ensure_positive_definite(propagated_Σd * (1 - debris_scale_factor))
    
    new_state = SpacecraftCAState(
        new_TCA,
        propagated_xs,
        propagated_xd,
        Matrix(new_Σs),
        Matrix(new_Σd),
        s.rs,
        s.rd
    )
    
    return Deterministic(new_state)
end

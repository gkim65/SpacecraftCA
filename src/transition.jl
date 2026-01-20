using POMDPs
using POMDPTools
using Distributions
using LinearAlgebra
using Random
using GaussianFilters

"""
STM-based state and covariance propagation
Uses state transition matrix from Python brahe

Parameters:
- pomdp: SpacecraftCAPOMDP object
- x: State vector [pos; vel] in km and km/s
- C_eci: Covariance matrix (6x6)
- u: Control input [thrust_direction]
- T_total: Total propagation time in seconds

Returns:
- GaussianBelief with propagated mean and covariance
"""
function stm_propagate(pomdp::SpacecraftCAPOMDP, x::Vector{Float64}, C_eci::Matrix{Float64}, 
                       u::AbstractVector{<:Number}, T_total::Float64=pomdp.dt_seconds)
    # Safety check for NaN/Inf in inputs
    if any(isnan.(x)) || any(isinf.(x)) || any(isnan.(C_eci)) || any(isinf.(C_eci))
        return GaussianBelief(x, Symmetric(C_eci))
    end
    
    C_eci_pd = ensure_positive_definite(C_eci)
    epc0_brahe = spaceXEpoch_brahe(pomdp.current_epoch_str)  # Use brahe Epoch
    
    # Use STM-based propagation with brahe (much faster than UKF)
    x_final, Σ_final = propagate_state_cov_stm(x, Matrix(C_eci_pd), epc0_brahe, T_total, Float64.(u))
    
    # Safety check after propagation
    if any(isnan.(x_final)) || any(isinf.(x_final)) ||
       any(isnan.(Σ_final)) || any(isinf.(Σ_final))
        return GaussianBelief(x, C_eci_pd)
    end
    
    # Add process noise
    W = diagm([1e-6, 1e-6, 1e-6, 1e-12, 1e-12, 1e-12])
    Σ_final_with_noise = Σ_final .+ W * T_total / pomdp.dt_seconds
    
    return GaussianBelief(x_final, ensure_positive_definite(Σ_final_with_noise))
end

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

function unscented_kalman_filter(pomdp::SpacecraftCAPOMDP, x::Vector{Float64}, C_eci::Matrix{Float64}, u::AbstractVector{<:Number}, T_total::Float64=pomdp.dt_seconds)
    # Safety check for NaN/Inf in inputs
    if any(isnan.(x)) || any(isinf.(x)) || any(isnan.(C_eci)) || any(isinf.(C_eci))
        # Return original state if inputs are corrupted
        return GaussianBelief(x, Symmetric(C_eci))
    end
    
    C_eci_pd = ensure_positive_definite(C_eci)
    b0 = GaussianBelief(Float64.(x), C_eci_pd)

    W = diagm([1e-6, 1e-6, 1e-6, 1e-12, 1e-12, 1e-12])
    dmodel = NonlinearDynamicsModel(step, W)
    V = diagm([1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9])
    omodel = NonlinearObservationModel(observe, V)

    ukf = UnscentedKalmanFilter(dmodel, omodel)
    
    N = max(1, Int(ceil(T_total / pomdp.dt_seconds)))
    T = pomdp.dt_seconds
    epc0 = spaceXEpoch_brahe(pomdp.current_epoch_str)
    
    current_belief = b0
    current_epc = epc0
    
    for k in 1:N
        if k == N && T_total < N * pomdp.dt_seconds
            T_step = T_total - (N - 1) * pomdp.dt_seconds
        else
            T_step = T
        end
        
        current_belief = predictEpc(ukf, current_belief, u, current_epc, T_step)
        
        # Safety check after propagation
        if any(isnan.(current_belief.μ)) || any(isinf.(current_belief.μ)) ||
           any(isnan.(current_belief.Σ)) || any(isinf.(current_belief.Σ))
            # Return previous belief if propagation failed
            return b0
        end
        
        current_epc = current_epc + T_step
    end
    
    return current_belief
end

function POMDPs.transition(pomdp::SpacecraftCAPOMDP, s::SpacecraftCAState, a::Symbol)
    @assert s.TCA >= 0 "TCA must be non-negative"
    
    # Use STM-based propagation by default for better performance (10-50x faster)
    # To use UKF-based propagation instead, set ENV["USE_UKF_PROPAGATION"] = "true"
    use_stm = get(ENV, "USE_UKF_PROPAGATION", "false") != "true"
    
    if a == :wait
        return use_stm ? wait_transition_stm(pomdp, s) : wait_transition(pomdp, s)
    elseif a == :maneuver
        return use_stm ? maneuver_transition_stm(pomdp, s) : maneuver_transition(pomdp, s)
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

# ==================== STM-BASED TRANSITION FUNCTIONS (FAST) ====================
# These use state transition matrices and are 10-50x faster than UKF

function wait_transition_stm(pomdp::SpacecraftCAPOMDP, s::SpacecraftCAState)
    if s.TCA <= 0
        return Deterministic(s)
    end
    
    new_TCA = s.TCA - 1
    
    bp_s = stm_propagate(pomdp, s.xs, s.Σs, [0.0])
    bp_d = stm_propagate(pomdp, s.xd, s.Σd, [0.0])
    
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

function maneuver_transition_stm(pomdp::SpacecraftCAPOMDP, s::SpacecraftCAState)
    if s.TCA <= 0
        return Deterministic(s)
    end
    
    new_TCA = s.TCA - 1
    
    bp_s = stm_propagate(pomdp, s.xs, s.Σs, [1.0])
    bp_d = stm_propagate(pomdp, s.xd, s.Σd, [0.0])
    
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

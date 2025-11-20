__precompile__(false)

module SpacecraftCA

using POMDPs
using POMDPTools
using LinearAlgebra
using Distributions
using Random
using StaticArrays
using SatelliteDynamics
using HCubature
using Dates, Printf
using GaussianFilters

export SpacecraftCAPOMDP, SpacecraftCAState, fosterPcState

include("states.jl")
function covECItoRTN(x, covariance)
    R = rRTNtoECI(x)
    R_inv = R'
    return [R_inv zeros(3,3); zeros(3,3) R_inv] * covariance * [R_inv' zeros(3,3); zeros(3,3) R_inv']
end

function getRTNCovariance(x, Σ_eci)
    Σ_rtn = covECItoRTN(x, Σ_eci)
    return Σ_rtn[1:3, 1:3]
end

function fosterPcState(state::SpacecraftCAState)
    Σs_rtn = getRTNCovariance(state.xs, state.Σs)
    Σd_rtn = getRTNCovariance(state.xd, state.Σd)
    Σs_6x6 = zeros(6, 6)
    Σd_6x6 = zeros(6, 6)
    Σs_6x6[1:3, 1:3] = Σs_rtn
    Σd_6x6[1:3, 1:3] = Σd_rtn
    return fosterPcAnalytical(
        state.xs, Σs_6x6, 
        state.xd, Σd_6x6,
        object1_radius=state.rs,
        object2_radius=state.rd
    )
end

include("utils/probabilityCollision.jl")
include("utils/genConjunctions.jl")
include("utils/propCovariance.jl")

"""
Spacecraft Collision Avoidance POMDP.

A Partially Observable Markov Decision Process for satellite collision avoidance
with debris objects. The agent must decide when to execute evasive maneuvers based
on noisy observations of the debris state.

# Fields
- `discount_factor`: Discount factor γ for future rewards (default: 0.95)
- `satellite_scale_factor`: Deterministic scaling factor for satellite covariance (default: 0.05)
- `debris_scale_range`: Range for stochastic debris covariance scaling (default: (0.05, 0.3))
- `unit_dv`: Unit delta-V increment for maneuver search (default: 1e-3 km/s)
- `dt_seconds`: Time step in seconds (default: 28800 = 8 hours)
- `current_epoch_str`: Current epoch string "YYYYJJJHHMMSS.fff" (default: "2024001000000.000")
- `observation_noise`: Observation noise covariance matrix V (optional, default: 0.1*I)
- `seed`: Random seed for reproducibility (optional)
"""
struct SpacecraftCAPOMDP <: POMDP{SpacecraftCAState, Symbol, Vector{Float64}}
    discount_factor::Float64
    satellite_scale_factor::Float64
    debris_scale_range::Tuple{Float64, Float64}
    unit_dv::Float64
    dt_seconds::Float64
    current_epoch_str::String
    observation_noise::Matrix{Float64}
    seed::Union{Int, Nothing}
end

function SpacecraftCAPOMDP(;
    discount_factor=0.95,
    satellite_scale_factor=0.05,
    debris_scale_range=(0.05, 0.3),
    unit_dv=1e-3,
    dt_seconds=28800.0,  # 8 hours
    current_epoch_str="2024001000000.000",
    observation_noise=0.1 * Matrix{Float64}(I, 6, 6),
    seed=nothing
)
    return SpacecraftCAPOMDP(
        discount_factor,
        satellite_scale_factor,
        debris_scale_range,
        unit_dv,
        dt_seconds,
        current_epoch_str,
        observation_noise,
        seed
    )
end

POMDPs.discount(pomdp::SpacecraftCAPOMDP) = pomdp.discount_factor

include("states.jl")

include("actions.jl")
include("observation.jl")
include("transition.jl")
include("rewards.jl")

end

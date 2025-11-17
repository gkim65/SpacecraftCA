using POMDPs
using Distributions
using LinearAlgebra

POMDPs.observations(pomdp::SpacecraftCAPOMDP) = pomdp

function POMDPs.obsindex(pomdp::SpacecraftCAPOMDP, o::Vector{Float64})
    return hash(o)
end

function POMDPs.observation(pomdp::SpacecraftCAPOMDP, a::Symbol, sp::SpacecraftCAState)
    V = pomdp.observation_noise
    obs_mean = sp.xd
    return MvNormal(obs_mean, V)
end


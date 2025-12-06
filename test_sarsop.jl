using Pkg
Pkg.activate(".")

include("src/SpacecraftCA.jl")
using .SpacecraftCA
using POMDPs
using POMDPTools
using ParticleFilters
using Random

struct ThresholdPolicy <: Policy
    pomdp::SpacecraftCAPOMDP
    threshold_multiplier::Float64
end

function POMDPs.action(policy::ThresholdPolicy, b)
    if typeof(b) <: SpacecraftCAState
        s = b
    elseif hasmethod(rand, (typeof(b),))
        s = rand(b)
    else
        s0_dist = initialstate(policy.pomdp)
        s = rand(s0_dist)
    end
    pc = fosterPcState(policy.pomdp, s)
    if pc > policy.pomdp.collision_threshold * policy.threshold_multiplier
        return :maneuver
    else
        return :wait
    end
end

POMDPs.updater(policy::ThresholdPolicy) = BootstrapFilter(policy.pomdp, 100)

seed = 42
Random.seed!(seed)

pomdp = SpacecraftCAPOMDP(seed=seed)
policy = ThresholdPolicy(pomdp, 10.0)

hr = HistoryRecorder(max_steps=20, rng=MersenneTwister(seed), show_progress=false)
up = updater(policy)
hist = simulate(hr, pomdp, policy, up)

println("Total discounted reward: ", discounted_reward(hist))

for (t, step) in enumerate(eachstep(hist, "(s, a, o, r)"))
    pc = fosterPcState(pomdp, step.s)
    println("t=$t TCA=$(step.s.TCA) Pc=$(round(pc, sigdigits=4)) a=$(step.a) r=$(round(step.r, sigdigits=4))")
end


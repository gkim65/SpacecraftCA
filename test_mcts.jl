using Pkg
Pkg.activate(".")

try
    using POMCPOW
catch
    Pkg.add("POMCPOW")
    using POMCPOW
end

try
    using MCTS
catch
    Pkg.add("MCTS")
    using MCTS
end

using SpacecraftCA
using POMDPs
using POMDPTools
using ParticleFilters
using Random

seed = 42
Random.seed!(seed)

pomdp = SpacecraftCAPOMDP(seed=seed)

solver = POMCPOWSolver(
    tree_queries=1000,      # Increased from 2: more simulations = better exploration
    max_depth=20,           # Increased from 4: match simulation horizon
    criterion=MaxUCB(1.0),  # Exploration-exploitation tradeoff       
    k_observation=10.0,     # Observation widening parameter
    alpha_observation=0.1,  # Observation action pair widening
    estimate_value=FORollout(RandomPolicy(pomdp)),  # Rollout policy for value estimation
    next_action=RandomActionGenerator()        # Action generator
)

policy = solve(solver, pomdp)

# Create belief updater for simulation
updater = BootstrapFilter(pomdp, 100)

hr = HistoryRecorder(max_steps=20, rng=MersenneTwister(seed))
hist = simulate(hr, pomdp, policy, updater)

println("Total discounted reward: ", discounted_reward(hist))

for (t, step) in enumerate(eachstep(hist, "(s, a, o, r)"))
    pc = fosterPcState(pomdp, step.s)
    println("t=$t TCA=$(step.s.TCA) Pc=$(round(pc, sigdigits=4)) a=$(step.a) r=$(round(step.r, sigdigits=4))")
end


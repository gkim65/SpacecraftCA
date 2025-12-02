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
using Random

seed = 42
Random.seed!(seed)

pomdp = SpacecraftCAPOMDP(seed=seed)
rollout_policy = RandomPolicy(pomdp)

solver = POMCPOWSolver(
    tree_queries=50,
    max_depth=6,
    criterion=MaxUCB(5.0),
    estimate_value=MCTS.RolloutEstimator(rollout_policy; max_depth=3)
)

policy = solve(solver, pomdp)

hr = HistoryRecorder(max_steps=20, rng=MersenneTwister(seed))
hist = simulate(hr, pomdp, policy)

println("Total discounted reward: ", discounted_reward(hist))

for (t, step) in enumerate(eachstep(hist, "(s, a, o, r)"))
    pc = fosterPcState(pomdp, step.s)
    println("t=$t TCA=$(step.s.TCA) Pc=$(round(pc, sigdigits=4)) a=$(step.a) r=$(round(step.r, sigdigits=4))")
end


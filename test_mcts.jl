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
using Printf

seed = 42
Random.seed!(seed)

pomdp = SpacecraftCAPOMDP(seed=seed)

solver = POMCPOWSolver(
    tree_queries=4,      # Increased from 2: more simulations = better exploration
    max_depth=4,           # Increased from 4: match simulation horizon
    criterion=MaxUCB(1.0),  # Exploration-exploitation tradeoff       
    k_observation=10.0,     # Observation widening parameter
    alpha_observation=0.1,  # Observation action pair widening
    estimate_value=FORollout(RandomPolicy(pomdp)),  # Rollout policy for value estimation
    next_action=RandomActionGenerator()        # Action generator
)

policy = solve(solver, pomdp)

println("=" ^ 60)
println("Policy Extraction")
println("=" ^ 60)

# Create belief updater for simulation
updater = BootstrapFilter(pomdp, 100)

# Extract initial belief
initial_belief = initialstate(pomdp)
initial_state = rand(initial_belief)
initial_belief_particles = [rand(initial_belief) for _ in 1:100]
initial_belief_dist = ParticleCollection(initial_belief_particles)

println("\nInitial belief state:")
println("  TCA: $(initial_state.TCA)")
pc_initial = fosterPcState(pomdp, initial_state)
println("  Collision Probability: $(round(pc_initial, sigdigits=4))")

# Query policy for initial belief
initial_action = action(policy, initial_belief_dist)
println("  Policy action: $initial_action")

# Run simulation to extract policy decisions
hr = HistoryRecorder(max_steps=20, rng=MersenneTwister(seed))
hist = simulate(hr, pomdp, policy, updater)

println("\n" * "=" ^ 60)
println("Policy Decisions During Simulation")
println("=" ^ 60)
println("Total discounted reward: ", discounted_reward(hist))
println()

# Extract policy decisions from history
policy_decisions = []
for (t, step) in enumerate(eachstep(hist, "(s, a, o, r)"))
    pc = fosterPcState(pomdp, step.s)
    println("t=$t TCA=$(step.s.TCA) Pc=$(round(pc, sigdigits=4)) a=$(step.a) r=$(round(step.r, sigdigits=4))")
    push!(policy_decisions, (t=t, TCA=step.s.TCA, Pc=pc, action=step.a, reward=step.r))
end

println("\n" * "=" ^ 60)
println("Policy Summary")
println("=" ^ 60)
println("Total steps: $(length(policy_decisions))")
maneuver_count = count(d -> d.action == :maneuver, policy_decisions)
wait_count = count(d -> d.action == :wait, policy_decisions)
println("Maneuver actions: $maneuver_count")
println("Wait actions: $wait_count")
println("Policy decisions:")
for d in policy_decisions
    println("  t=$(d.t), TCA=$(d.TCA), Pc=$(round(d.Pc, sigdigits=4)) -> $(d.action)")
end

# Write policy to file
policy_filename = "policy_mcts_seed$(seed).txt"
csv_filename = "policy_mcts_seed$(seed).csv"

open(policy_filename, "w") do io
    println(io, "=" ^ 70)
    println(io, "POMCPOW Policy Output")
    println(io, "=" ^ 70)
    println(io, "\nSolver Configuration:")
    println(io, "  tree_queries: $(solver.tree_queries)")
    println(io, "  max_depth: $(solver.max_depth)")
    println(io, "  criterion: $(solver.criterion)")
    println(io, "  k_observation: $(solver.k_observation)")
    println(io, "  alpha_observation: $(solver.alpha_observation)")
    println(io, "\nSimulation Configuration:")
    println(io, "  Seed: $seed")
    println(io, "  Max steps: 20")
    println(io, "  Particle filter size: 100")
    
    println(io, "\n" * "=" ^ 70)
    println(io, "Initial Belief State")
    println(io, "=" ^ 70)
    println(io, "  TCA: $(initial_state.TCA)")
    println(io, "  Collision Probability: $(round(pc_initial, sigdigits=4))")
    println(io, "  Policy action: $initial_action")
    
    println(io, "\n" * "=" ^ 70)
    println(io, "Policy Decisions During Simulation")
    println(io, "=" ^ 70)
    println(io, "Total discounted reward: $(discounted_reward(hist))")
    println(io, "\nStep-by-step decisions:")
    println(io, @sprintf("%-4s | %6s | %12s | %-10s | %10s", "t", "TCA", "Pc", "Action", "Reward"))
    println(io, "-" ^ 70)
    for d in policy_decisions
        println(io, @sprintf("%-4d | %6d | %12.6f | %-10s | %10.6f", 
            d.t, d.TCA, d.Pc, d.action, d.reward))
    end
    
    println(io, "\n" * "=" ^ 70)
    println(io, "Policy Summary")
    println(io, "=" ^ 70)
    println(io, "Total steps: $(length(policy_decisions))")
    println(io, "Maneuver actions: $maneuver_count")
    println(io, "Wait actions: $wait_count")
    println(io, "Maneuver percentage: $(round(maneuver_count / length(policy_decisions) * 100, sigdigits=2))%")
    
    println(io, "\nPolicy Decision Mapping:")
    println(io, "State (TCA, Pc) -> Action")
    println(io, "-" ^ 70)
    for d in policy_decisions
        println(io, @sprintf("  t=%d, TCA=%d, Pc=%.6f -> %s", 
            d.t, d.TCA, d.Pc, d.action))
    end
    
    # Check for collisions
    final_state = last([step.s for step in eachstep(hist, "s")])
    final_pc = fosterPcState(pomdp, final_state)
    collision_occurred = final_pc >= pomdp.collision_threshold
    println(io, "\n" * "=" ^ 70)
    println(io, "Terminal State Analysis")
    println(io, "=" ^ 70)
    println(io, "Final TCA: $(final_state.TCA)")
    println(io, "Final collision probability: $(round(final_pc, sigdigits=6))")
    println(io, "Collision threshold: $(pomdp.collision_threshold)")
    println(io, "Collision occurred: $collision_occurred")
end

# Write CSV file for easy import into papers/tables
open(csv_filename, "w") do io
    println(io, "t,TCA,Pc,action,reward")
    for d in policy_decisions
        println(io, "$(d.t),$(d.TCA),$(d.Pc),$(d.action),$(d.reward)")
    end
end

println("\n" * "=" ^ 60)
println("Policy Output")
println("=" ^ 60)
println("Policy saved to: $policy_filename")
println("CSV data saved to: $csv_filename")

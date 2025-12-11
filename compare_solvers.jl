using Pkg
Pkg.activate(".")

try
    using POMCPOW
catch
    Pkg.add("POMCPOW")
    using POMCPOW
end

using SpacecraftCA
using POMDPs
using POMDPTools
using ParticleFilters
using Random
using Printf
using Statistics

# Threshold policy (baseline)
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

# Function to run a single simulation and collect metrics
function run_one_simulation(pomdp, policy, updater, seed, max_steps=20)
    Random.seed!(seed)
    hr = HistoryRecorder(max_steps=max_steps, rng=MersenneTwister(seed), show_progress=false)
    
    # Time the simulation
    start_time = time()
    hist = simulate(hr, pomdp, policy, updater)
    runtime = time() - start_time
    
    # Collect metrics
    total_reward = discounted_reward(hist)
    maneuver_count = count(step -> step.a == :maneuver, eachstep(hist, "(s, a, o, r)"))
    wait_count = count(step -> step.a == :wait, eachstep(hist, "(s, a, o, r)"))
    
    # Check for collision at terminal state
    steps = collect(eachstep(hist, "(s, a, o, r)"))
    final_state = isempty(steps) ? nothing : last(steps).s
    if final_state !== nothing
        final_pc = fosterPcState(pomdp, final_state)
        collision_occurred = final_pc >= pomdp.collision_threshold
    else
        final_pc = 0.0
        collision_occurred = false
    end
    
    return (
        reward=total_reward,
        runtime=runtime,
        maneuver_count=maneuver_count,
        wait_count=wait_count,
        collision_occurred=collision_occurred,
        final_pc=final_pc
    )
end

# Function to evaluate a policy across multiple seeds
function evaluate_policy(policy_name, policy_factory, pomdp_factory, seeds, max_steps=20)
    println("\nEvaluating $policy_name...")
    results = []
    
    for (i, seed) in enumerate(seeds)
        print("  Run $(i)/$(length(seeds))... ")
        flush(stdout)
        
        pomdp = pomdp_factory(seed)
        policy = policy_factory(pomdp)
        belief_updater = POMDPs.updater(policy)
        
        result = run_one_simulation(pomdp, policy, belief_updater, seed, max_steps)
        push!(results, result)
        println("Reward: $(round(result.reward, sigdigits=4)), Runtime: $(round(result.runtime, sigdigits=3))s")
    end
    
    # Compute statistics
    rewards = [r.reward for r in results]
    runtimes = [r.runtime for r in results]
    maneuver_counts = [r.maneuver_count for r in results]
    collision_rate = mean([r.collision_occurred for r in results])
    
    return (
        name=policy_name,
        reward_mean=mean(rewards),
        reward_std=std(rewards),
        runtime_mean=mean(runtimes),
        runtime_std=std(runtimes),
        maneuver_mean=mean(maneuver_counts),
        collision_rate=collision_rate,
        results=results
    )
end

# Main execution
println("=" ^ 70)
println("Satellite Collision Avoidance - Algorithm Comparison")
println("=" ^ 70)

# Configuration
num_runs = 10
seeds = collect(1:num_runs)
max_steps = 20

println("\nConfiguration:")
println("  Number of runs per solver: $num_runs")
println("  Seeds: $(seeds[1]) to $(seeds[end])")
println("  Max steps per simulation: $max_steps")

# Define policy factories
pomdp_factory = (seed) -> SpacecraftCAPOMDP(seed=seed)

threshold_policy_factory = (pomdp) -> ThresholdPolicy(pomdp, 10.0)

pomcpow_policy_factory = (tree_queries) -> (pomdp) -> begin
    solver = POMCPOWSolver(
        tree_queries=tree_queries,
        max_depth=4,
        criterion=MaxUCB(1.0),
        k_observation=10.0,
        alpha_observation=0.1,
        estimate_value=FORollout(RandomPolicy(pomdp)),
        next_action=RandomActionGenerator()
    )
    return solve(solver, pomdp)
end

# Evaluate all policies
println("\n" * "=" ^ 70)
println("Running Simulations")
println("=" ^ 70)

threshold_results = evaluate_policy(
    "Threshold Baseline (10x)",
    threshold_policy_factory,
    pomdp_factory,
    seeds,
    max_steps
)

pomcpow_q2_results = evaluate_policy(
    "POMCPOW (tree_queries=2)",
    pomcpow_policy_factory(2),
    pomdp_factory,
    seeds,
    max_steps
)

pomcpow_q4_results = evaluate_policy(
    "POMCPOW (tree_queries=4)",
    pomcpow_policy_factory(4),
    pomdp_factory,
    seeds,
    max_steps
)

pomcpow_q10_results = evaluate_policy(
    "POMCPOW (tree_queries=10)",
    pomcpow_policy_factory(10),
    pomdp_factory,
    seeds,
    max_steps
)

# Collect all results
all_results = [threshold_results, pomcpow_q2_results, pomcpow_q4_results, pomcpow_q10_results]

# Print comparison table
println("\n" * "=" ^ 70)
println("TABLE I: COMPARISON OF ALGORITHM PERFORMANCE")
println("=" ^ 70)
println()
println(@sprintf("%-30s | %12s | %12s | %12s", 
    "Solver", "Reward μ", "Reward σ", "Runtime μ (s)"))
println("-" ^ 70)

for r in all_results
    println(@sprintf("%-30s | %12.2f | %12.2f | %12.3f",
        r.name, r.reward_mean, r.reward_std, r.runtime_mean))
end

# Print parameter tuning table
println("\n" * "=" ^ 70)
println("TABLE II: POMCPOW PARAMETER TUNING")
println("=" ^ 70)
println()
println(@sprintf("%-15s | %12s | %12s | %12s", 
    "Tree Queries", "Reward μ", "Reward σ", "Runtime μ (s)"))
println("-" ^ 70)

pomcpow_variants = [pomcpow_q2_results, pomcpow_q4_results, pomcpow_q10_results]
for r in pomcpow_variants
    queries = match(r"tree_queries=(\d+)", r.name).captures[1]
    println(@sprintf("%-15s | %12.2f | %12.2f | %12.3f",
        queries, r.reward_mean, r.reward_std, r.runtime_mean))
end

# Generate LaTeX table format
println("\n" * "=" ^ 70)
println("LATEX TABLE FORMAT")
println("=" ^ 70)
println()
println("% Table I: Comparison of Algorithm Performance")
println("\\begin{table}[h]")
println("\\centering")
println("\\begin{tabular}{|l|r|r|r|}")
println("\\hline")
println("Solver & Reward \$\\mu\$ & Reward \$\\sigma\$ & Runtime \$\\mu\$ (s) \\\\")
println("\\hline")

for r in all_results
    name = replace(r.name, "POMCPOW" => "POMCPOW", "Threshold Baseline" => "Threshold")
    println(@sprintf("%s & %.2f & %.2f & %.3f \\\\",
        name, r.reward_mean, r.reward_std, r.runtime_mean))
end

println("\\hline")
println("\\end{tabular}")
println("\\caption{Comparison of algorithm performance across $num_runs randomly generated scenarios.}")
println("\\label{tab:algorithm_comparison}")
println("\\end{table}")

println("\n% Table II: POMCPOW Parameter Tuning")
println("\\begin{table}[h]")
println("\\centering")
println("\\begin{tabular}{|r|r|r|r|}")
println("\\hline")
println("Tree Queries & Reward \$\\mu\$ & Reward \$\\sigma\$ & Runtime \$\\mu\$ (s) \\\\")
println("\\hline")

for r in pomcpow_variants
    queries = match(r"tree_queries=(\d+)", r.name).captures[1]
    println(@sprintf("%s & %.2f & %.2f & %.3f \\\\",
        queries, r.reward_mean, r.reward_std, r.runtime_mean))
end

println("\\hline")
println("\\end{tabular}")
println("\\caption{POMCPOW parameter tuning: effect of tree_queries on performance.}")
println("\\label{tab:pomcpow_tuning}")
println("\\end{table}")

# Save detailed results to CSV
csv_filename = "results_comparison.csv"
open(csv_filename, "w") do io
    println(io, "solver,run,reward,runtime,maneuvers,collision")
    for r in all_results
        for (i, res) in enumerate(r.results)
            println(io, "$(r.name),$i,$(res.reward),$(res.runtime),$(res.maneuver_count),$(res.collision_occurred)")
        end
    end
end

println("\n" * "=" ^ 70)
println("Detailed results saved to: $csv_filename")
println("=" ^ 70)

# Print additional statistics
println("\nAdditional Statistics:")
println("-" ^ 70)
for r in all_results
    println("\n$(r.name):")
    println("  Maneuvers per run (mean): $(round(r.maneuver_mean, sigdigits=2))")
    println("  Collision rate: $(round(r.collision_rate * 100, sigdigits=2))%")
    println("  Runtime std: $(round(r.runtime_std, sigdigits=3))s")
end


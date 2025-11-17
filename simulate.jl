using Pkg
Pkg.activate(".")
using SpacecraftCA
using POMDPs
using POMDPTools

const fosterPcState = SpacecraftCA.fosterPcState

println("=" ^ 60)
println("Spacecraft Collision Avoidance POMDP Simulation")
println("=" ^ 60)

pomdp = SpacecraftCAPOMDP()
println("\nPOMDP created with:")
println("  Discount factor: ", discount(pomdp))
println("  Time step: ", pomdp.dt_seconds / 3600, " hours")
println("  Unit delta-V: ", pomdp.unit_dv, " km/s")

println("\n" * "=" ^ 60)
println("Initial State")
println("=" ^ 60)

s0_dist = initialstate(pomdp)
s0 = rand(s0_dist)
println("TCA: ", s0.TCA, " time steps")
println("Satellite position (km): ", s0.xs[1:3])
println("Debris position (km): ", s0.xd[1:3])
println("Collision probability: ", fosterPcState(s0))

println("\n" * "=" ^ 60)
println("Simulation Run")
println("=" ^ 60)

let
    current_state = s0
    total_reward = 0.0
    step_count = 0
    max_steps = min(s0.TCA, 5)

    for step in 1:max_steps
        println("\n--- Step $step ---")
        println("Current TCA: ", current_state.TCA)
        
        pc = fosterPcState(current_state)
        println("Current collision probability: ", pc)
        
        if current_state.TCA <= 0
            println("TCA reached! Episode complete.")
            break
        end
        
        action = pc > 1e-4 ? :maneuver : :wait
        println("Selected action: ", action)
        
        sp_dist = transition(pomdp, current_state, action)
        sp = rand(sp_dist)
        
        r = reward(pomdp, current_state, action)
        total_reward += discount(pomdp)^(step-1) * r
        println("Reward: ", r)
        println("Discounted total reward: ", total_reward)
        
        o_dist = observation(pomdp, action, sp)
        o = rand(o_dist)
        println("Observation (debris state estimate, first 3 elements): ", o[1:3])
        
        current_state = sp
        step_count = step
    end

    println("\n" * "=" ^ 60)
    println("Simulation Complete")
    println("=" ^ 60)
    println("Total steps: ", step_count)
    println("Final TCA: ", current_state.TCA)
    println("Final collision probability: ", fosterPcState(current_state))
    println("Total discounted reward: ", total_reward)
end



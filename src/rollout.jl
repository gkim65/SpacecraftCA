using POMDPs
using POMDPTools
using Random

function rollout(pomdp::SpacecraftCAPOMDP; max_steps=100, seed=nothing, verbose=false)
    if seed !== nothing
        Random.seed!(seed)
    end
    
    s0_dist = initialstate(pomdp)
    s0 = rand(s0_dist)
    
    trajectory = []
    total_reward = 0.0
    current_state = s0
    
    for step in 1:max_steps
        if isterminal(pomdp, current_state)
            break
        end
        
        pc = fosterPcState(pomdp, current_state)
        action = pc > pomdp.collision_threshold * 10 ? :maneuver : :wait
        
        sp_dist = transition(pomdp, current_state, action)
        sp = rand(sp_dist)
        
        r = reward(pomdp, current_state, action)
        total_reward += discount(pomdp)^(step-1) * r
        
        o_dist = observation(pomdp, action, sp)
        o = rand(o_dist)
        
        push!(trajectory, (state=current_state, action=action, reward=r, next_state=sp, observation=o))
        
        if verbose
            println("Step $step: TCA=$(current_state.TCA), Pc=$(pc), action=$action, reward=$r")
        end
        
        current_state = sp
    end
    
    return (trajectory=trajectory, total_reward=total_reward, final_state=current_state)
end


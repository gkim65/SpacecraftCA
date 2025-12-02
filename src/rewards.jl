using POMDPs

function evasive_maneuver_cost(pomdp::SpacecraftCAPOMDP, state::SpacecraftCAState, unit_dv::Float64)
    return pomdp.maneuver_cost
end

function POMDPs.reward(pomdp::SpacecraftCAPOMDP, s::SpacecraftCAState, a::Symbol)
    current_Pc = fosterPcState(pomdp, s)
    
    if s.TCA <= 0
        if a == :maneuver
            return pomdp.crash_cost
        else
            if current_Pc < pomdp.collision_threshold
                return 0.0
            else
                return pomdp.crash_cost
            end
        end
    else
        if a == :wait
            return 0.0
        else
            cost = evasive_maneuver_cost(pomdp, s, pomdp.unit_dv)
            return -cost
        end
    end
end


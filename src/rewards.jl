using POMDPs

const COLLISION_THRESHOLD = 1e-5  
const CRASH_COST = -0.5
const MANEUVER_COST = 0.01

function evasive_maneuver_cost(state::SpacecraftCAState, unit_dv::Float64)
    return MANEUVER_COST
end

function POMDPs.reward(pomdp::SpacecraftCAPOMDP, s::SpacecraftCAState, a::Symbol)
    current_Pc = fosterPcState(s)
    
    if s.TCA <= 0
        if a == :maneuver
            return CRASH_COST
        else
            if current_Pc < COLLISION_THRESHOLD
                return 0.0
            else
                return CRASH_COST
            end
        end
    else
        if a == :wait
            return 0.0
        else
            cost = evasive_maneuver_cost(s, pomdp.unit_dv)
            return -cost
        end
    end
end


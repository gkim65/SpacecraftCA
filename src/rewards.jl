using POMDPs

const COLLISION_THRESHOLD = 1e-5  
const CRASH_COST = -0.5

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
            cost, _ = evasive_maneuver(s, unit_dv=pomdp.unit_dv)
            return -cost
        end
    end
end


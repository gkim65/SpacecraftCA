using POMDPs

POMDPs.actions(pomdp::SpacecraftCAPOMDP) = [:wait, :maneuver]

function POMDPs.actionindex(pomdp::SpacecraftCAPOMDP, a::Symbol)
    if a == :wait
        return 1
    elseif a == :maneuver
        return 2
    else
        error("Unknown action: $a. Must be :wait or :maneuver")
    end
end






### layout of spacecraftCA POMDP... Maybe is it better to keep it MDP initially? might be

const ObjectPos = SVector{3, Int64}

struct SpacecraftCA
    t::Int64        # Time until closest approach between two objects
    object1::ObjectPos
    object2::ObjectPos
end


# beliefs >> update with unscented kalman filter?
# 
# Reward func
    # if we go over probability of collision threshold >> huge crash cost 
    # if we use thrust (small penalty)

# actions >> thrust in direction versus not


# need to calculate probability of collision (could be monte carlo, could be something else)
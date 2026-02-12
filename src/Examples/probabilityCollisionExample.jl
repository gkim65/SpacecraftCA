# using Pkg
# Pkg.activate("SpacecraftCA")
# Pkg.instantiate()

using Pkg
Pkg.activate(".")
using SpacecraftCA
using POMDPs
using POMDPTools


pomdp = SpacecraftCAPOMDP()
s0_dist = initialstate(pomdp)
s0 = rand(s0_dist)
transition(pomdp, s0, :wait)


# need to propagate the objects forward to the correct tca 
# then check how the probability of collision is there versus before
# @vedant i think this example doesn't check the propagation then the state yet
# but I would make this check

println("TCA: ", s0.TCA, " time steps")
println("Satellite position (km): ", s0.xs[1:3])
println("Debris position (km): ", s0.xd[1:3])
println("Collision probability: ", fosterPcState(s0))
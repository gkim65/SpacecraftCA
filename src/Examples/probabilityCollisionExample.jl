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

# propagate them foward
# Then check for probability of collision


# Takes object1_Σ and object2_Σ as they are.
# Assumes objects are already in the encounter geometry frame (RTN) and already propagated to the encounter time.
fosterPcAnalytical(cdm_rand.xc, cdm_rand.Σc, cdm_rand.xd, cdm_rand.Σd, object1_radius = cdm_rand.rc, object2_radius = cdm_rand.rd)

println("TCA: ", s0.TCA, " time steps")
println("Satellite position (km): ", s0.xs[1:3])
println("Debris position (km): ", s0.xd[1:3])
println("Collision probability: ", fosterPcState(s0))
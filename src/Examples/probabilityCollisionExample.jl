using Pkg
Pkg.activate("SpacecraftCA")
Pkg.instantiate()

include("../utils/genConjunctions.jl")
include("../utils/probabilityCollision.jl")


cdm_rand = generate_one_CDM()

# Takes object1_Σ and object2_Σ as they are.
# Assumes objects are already in the encounter geometry frame (RTN) and already propagated to the encounter time.
fosterPcAnalytical(cdm_rand.xc, cdm_rand.Σc, cdm_rand.xd, cdm_rand.Σd, object1_radius = cdm_rand.rc, object2_radius = cdm_rand.rd)
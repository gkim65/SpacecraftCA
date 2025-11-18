using Distributions

mutable struct SpacecraftCAState
    TCA::Int                    # Time to Closest Approach (time steps until TCA)
    xs::Vector{Float64}         # Satellite state at TCA: [x, y, z, vx, vy, vz] (km, km/s)
    xd::Vector{Float64}         # Debris TRUE state at TCA: [x, y, z, vx, vy, vz] (km, km/s)
    Σs::Matrix{Float64}         # Satellite covariance matrix at TCA (6x6)
    Σd::Matrix{Float64}         # Debris covariance matrix at TCA (6x6)
    rs::Float64                 # Satellite hardbody radius (meters)
    rd::Float64                 # Debris hardbody radius (meters)
end

if @isdefined(SpacecraftCAPOMDP)
    POMDPs.states(pomdp::SpacecraftCAPOMDP) = pomdp

    function Base.iterate(pomdp::SpacecraftCAPOMDP, i::Int=1)
        return nothing
    end

    Base.length(pomdp::SpacecraftCAPOMDP) = typemax(Int) 

    function POMDPs.stateindex(pomdp::SpacecraftCAPOMDP, s::SpacecraftCAState)
        return hash((s.TCA, hash(s.xs), hash(s.xd)))
    end

    function POMDPs.initialstate(pomdp::SpacecraftCAPOMDP)
        seed = pomdp.seed
        cdm = generate_one_CDM(seed=seed !== nothing ? seed : false)
        
        Σs_6x6 = zeros(6, 6)
        Σd_6x6 = zeros(6, 6)
        Σs_6x6[1:3, 1:3] = cdm.Σc
        Σd_6x6[1:3, 1:3] = cdm.Σd
        
        initial_state = SpacecraftCAState(
            cdm.TCA,
            cdm.xc,
            cdm.xd,
            Σs_6x6,
            Σd_6x6,
            cdm.rc,
            cdm.rd
        )
        return Deterministic(initial_state)
    end

    function POMDPs.isterminal(pomdp::SpacecraftCAPOMDP, s::SpacecraftCAState)
        return s.TCA <= 0
    end
end


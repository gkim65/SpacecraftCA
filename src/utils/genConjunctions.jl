function eci2orb(eci; isfullstate = false, delta_t = 1, epoch = Epoch(2024, 1, 1, 12, 0, 0, 0.0), sat_mass = 1.0)
    orb = EarthInertialState(epoch, eci, dt=delta_t,
	            mass=sat_mass, n_grav=0, m_grav=0,
	            drag=isfullstate, srp=isfullstate,
	            moon=isfullstate, sun=isfullstate,
	            relativity=isfullstate)
    return orb
end

function random_eci(; seed=false)
    """
    Using a reasonable set of orbital parameters for LEO, generate an initial ephemeris position
    """
    if seed != false
        Random.seed!(seed)
    end
    epc = Epoch(2024, 1, 1, 12, 0, 0, 0.0)
    R = R_EARTH + 400e3 + 200e3*rand()
    e = 0.01 + 0.1*rand()
    i = 75.0 + 15.0*rand()
    Ω = 45.0 + 45.0*rand()
    ω = 30.0 + 30.0*rand()
    M = 360.0*rand()
    new_eci = sOSCtoCART([R, e, i, Ω, ω, M])
    return new_eci
end

function random_debris(eci_spacecraft; rtn_cov_diag = [1e1, 1e4, 1e4], seed=false)
    """
    Generate a random debris orbit
    """
    if seed != false
        Random.seed!(seed)
    end
   
    # use the vector and the arbitrary covariance to generate a random debris position

    rtn_offset_distribution = Distributions.MvNormal([0, 0, 0], Diagonal(rtn_cov_diag))
    rtn_offset = rand(rtn_offset_distribution)
    rvec = eci_spacecraft[1:3]/norm(eci_spacecraft[1:3]) # radial direction
    tvec = eci_spacecraft[4:6]/norm(eci_spacecraft[4:6]) # along-track direction
    nvec = cross(rvec, tvec) # cross-track direction
    nvec = nvec/norm(nvec)
    R = hcat(rvec, tvec, nvec)
    debris_x = eci_spacecraft[1:3] + R*rtn_offset

    # For the debris to be at TCA, it must meet the following conditions:
    # 1. The debris velociy must be perpendicular to the vector from the debris to the chief. 
    # 2. The debris must be in a valid orbit.
    #    - The debris velocity must be mostly perpendicular to the vector from the earth to the debris.
    #    - The debris velocity must have a magnitude that is reasonable for LEO. (Starting with just mimicing the Chief velocity)
    # 3. These conditions constrain 2 dimentsion of the 3 dimensionsal velocity vector. We just use rand to pick a random direction for the third dimension from -1 to 1.

    relative_vector = debris_x - eci_spacecraft[1:3]
    A = transpose(hcat(debris_x, relative_vector, [0, 0, 1]))
    b = [0, 0, rand(-1:1e-18:1)]
    debris_v_dir = A\b
    debris_v = debris_v_dir/norm(debris_v_dir) * norm(eci_spacecraft[4:6])

    return vcat(debris_x, debris_v)
end

function generate_conjunction(;seed=false, orbit = false, debris_cov_diag = [1e1, 1e5, 1e6])
    if seed != false
        Random.seed!(seed)
    end

    eci_spacecraft = random_eci()
    eci_debris = random_debris(eci_spacecraft, rtn_cov_diag=debris_cov_diag)

    if orbit
        eci_spacecraft = eci2orb(eci_spacecraft)
        eci_debris = eci2orb(eci_debris)
    end

    return eci_spacecraft, eci_debris
end

function generate_one_CDM(;seed = false, debris_cov_diag = [1e1, 1e4, 1e4], syntheticTCA = 9, rc_range = (5.,10.), rd_range = (5.,10.))
    if seed != false
        Random.seed!(seed)
    end
    spacecraft, debris = generate_conjunction(;seed=seed, debris_cov_diag=debris_cov_diag)
    Σc, Σd = sample_covariance(;seed=seed)
    rc = rand(Distributions.Uniform(rc_range...)) 
    rd = rand(Distributions.Uniform(rd_range...))
    return CDM(syntheticTCA, spacecraft, debris, Σc, Σd, rc, rd)
end

# TODO: Revisit these sample covariance values
function sample_covariance(;seed=false, Σc_μ = [10^1,10^3,10^1], Σd_μ = [10^2,10^6,10^2], Σc_σ = [2,20,2], Σd_σ = [20,200,20])
    if seed != false
        Random.seed!(seed)
    end
    # Generate Covariance distributions based on the examples from spacetrack.org - which happen at TCA = 3 - 1 day in advance.
    Σc_distribution = Distributions.MvNormal(Σc_μ, Diagonal(Σc_σ))
    Σd_distribution = Distributions.MvNormal(Σd_μ, Diagonal(Σd_σ))

    # Generate Covariances
    Σc = Diagonal(rand(Σc_distribution))
    Σd = Diagonal(rand(Σd_distribution))
    return Σc, Σd
end


mutable struct CDM 
    # Conjunction Data Message
    # This is the state vector at the time of closest approach.
    TCA::Int # Time of Closest Approach - relative to current time
    xc::Vector # Position of the spacecraft at TCA
    xd::Vector # Position of the debris at TCA
    Σc::Matrix # Covariance matrix of the spacecraft at TCA
    Σd::Matrix # Covariance matrix of the debris at TCA
    rc::Float64 # Hardbody radius of the spacecraft
    rd::Float64 # Hardbody radius of the debris
end

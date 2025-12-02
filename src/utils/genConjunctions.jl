using Distributions 
using LinearAlgebra
using SatelliteDynamics

include("propCovariance.jl")

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


##################### Helper Functions #####################

function eci2orb(eci; isfullstate = false, delta_t = 1, epoch = Epoch(2024, 1, 1, 12, 0, 0, 0.0), sat_mass = 1.0)
    """
    Return earth inital state orbit needed for propagation with SatelliteDynamics
    """
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
    Generate a random debris orbit perpendicular to spacecraft object
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

function generate_conjunction(;seed=false, debris_cov_diag = [1e1, 1e5, 1e6], epc = Epoch(2024, 1, 1, 12, 0, 0, 0.0))
    """
    Generate a conjunction between a spacecraft and object
    """
    if seed != false
        Random.seed!(seed)
    end

    eci_spacecraft = random_eci()
    eci_debris = random_debris(eci_spacecraft, rtn_cov_diag=debris_cov_diag)

    orb_spacecraft = eci2orb(eci_spacecraft, epoch = epc)
    orb_debris = eci2orb(eci_debris, epoch = epc)

    return eci_spacecraft, eci_debris, orb_spacecraft, orb_debris
end


# TODO: Revisit these sample covariance value ranges, not sure if right range
function sample_covariance(;seed=false, Σc_μ = [10^1,10^3,10^1, 4e-6, 4e-4, 4e-6], Σd_μ = [10^2,10^6,10^2, 1e-4, 1e-2, 1e-4], Σc_σ = [2,20,2,0.0005, 0.005, 0.0005], Σd_σ = [20,200,20,0.002, 0.020, 0.002])
    """
    Return a random covariance matrix for both the spacecraft and debris object
    """
    
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


function sim_backwards!(state::EarthInertialState, time::Epoch)
    """
    Return a random covariance matrix for both the spacecraft and debris object
    """
    # If user gives a forward time, just call sim!
    if time >= state.epc
        return sim!(state, time)
    end

    # Number of state variables
    n = length(state.x)

    # Compute number of steps (positive)
    Δt = state.epc - time
    n_steps = ceil(Int, Δt/state.dt) + 1

    # Allocate outputs
    t_arr   = zeros(Float64, n_steps)
    epc_arr = Array{Epoch}(undef, n_steps)
    x_arr   = zeros(Float64, n, n_steps)
    A_arr   = state.phi === nothing ? nothing : zeros(Float64, n*n_steps, n)

    # Save initial state (this is the most recent epoch)
    idx = 1
    epc_arr[idx]  = state.epc
    t_arr[idx]    = 0.0
    x_arr[:, idx] = state.x
    if state.phi !== nothing
        A_arr[(1+n*(idx-1)):(n*idx), :] = state.phi
    end

    # Propagate backwards
    while state.epc > time
        idx += 1

        dt = -min(state.dt, state.epc - time)   # negative step!

        step!(state, dt)

        epc_arr[idx]  = state.epc
        t_arr[idx]    = epc_arr[idx] - epc_arr[1]
        x_arr[:, idx] = state.x

        if state.phi !== nothing
            A_arr[(1+n*(idx-1)):(n*idx), :] = state.phi
        end
    end

    # Reverse arrays so time is chronological (like sim!)
    t_arr    = reverse(t_arr)
    epc_arr  = reverse(epc_arr)
    x_arr    = reverse(x_arr, dims=2)
    if A_arr !== nothing
        A_arr = reverse(A_arr, dims=1)
    end

    # Return consistent API
    if A_arr !== nothing
        return t_arr, epc_arr, x_arr, A_arr
    else
        return t_arr, epc_arr, x_arr
    end
end

function generate_one_CDM(;seed = false, epoch_str="2024001000000.000", debris_cov_diag = [1e1, 1e4, 1e4], syntheticTCA = 9, dt_seconds = 28800.0, rc_range = (5.,10.), rd_range = (5.,10.))
    """
    Generate a conjunction, and propagate the states backwards from indicated tca epoch
    Generate a Return a random covariance matrix for both the spacecraft and debris object
    """
    
    if seed != false
        Random.seed!(seed)
    end

    epcStart = spaceXEpoch(epoch_str)
    epcTCA = epcStart + syntheticTCA * dt_seconds

    spacecraft, debris, orb_spacecraft, orb_debris= generate_conjunction(;seed=seed, debris_cov_diag=debris_cov_diag, epc = epcTCA)
    Σc, Σd = sample_covariance(;seed=seed)

    rc = rand(Distributions.Uniform(rc_range...)) 
    rd = rand(Distributions.Uniform(rd_range...))

    # Propagate satellite and debris states backward
    t_c, epc_c, x_arr_c = sim_backwards!(orb_spacecraft, epcStart)
    t_d, epc_d, x_arr_d = sim_backwards!(orb_debris, epcStart)

    # TODO: Just shrink covariances for now (we should fix this later)
    # Also note for vedant: this may be reducing the positional and velocity uncertainty a bit too much, so if things don't 
    # collide after propagating this forward we might need to lower these values
    scale_pos = 0.1   # 10x reduction in positional uncertainty
    scale_vel = 0.2    # 5x reduction in velocity uncertainty

    diag_c_new = diag(Σc)
    Σc_new = Diagonal([
        diag_c_new[1]*scale_pos, diag_c_new[2]*scale_pos, diag_c_new[3]*scale_pos,
        diag_c_new[4]*scale_vel, diag_c_new[5]*scale_vel, diag_c_new[6]*scale_vel
    ])
    diag_d_new = diag(Σd)
    Σd_new = Diagonal([
        diag_d_new[1]*scale_pos, diag_d_new[2]*scale_pos, diag_d_new[3]*scale_pos,
        diag_d_new[4]*scale_vel, diag_d_new[5]*scale_vel, diag_d_new[6]*scale_vel
    ])

    return CDM(syntheticTCA, x_arr_c[:,end], x_arr_d[:,end], Σc_new, Σd_new, rc, rd)
end
using Distributions 
using LinearAlgebra
using PyCall
using Dates
using Random

# Import brahe Python library - lazy initialization
const _bh_cache = Ref{Union{PyObject, Nothing}}(nothing)

function get_brahe()
    if _bh_cache[] === nothing
        _bh_cache[] = pyimport("brahe")
        # Initialize EOP and space weather data
        _bh_cache[].initialize_eop()
        _bh_cache[].initialize_sw()
    end
    return _bh_cache[]
end

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

function eci2orb_brahe(eci; isfullstate = false, epoch_datetime = (2024, 1, 1, 12, 0, 0.0, 0.0), sat_params = nothing)
    """
    Create a brahe NumericalOrbitPropagator for the given ECI state
    Returns: (propagator, epoch)
    """
    bh = get_brahe()
    
    # Create brahe epoch
    bh_epoch = bh.Epoch.from_datetime(epoch_datetime..., bh.TimeSystem.UTC)
    
    # Convert state to meters (brahe uses meters)
    state = eci .* 1000.0  # Convert km to m
    
    # Create propagation config with STM for covariance propagation
    prop_config = bh.NumericalPropagationConfig.default().with_stm().with_stm_history()
    
    # Set up force model
    if isfullstate
        force_config = bh.ForceModelConfig.default()
    else
        force_config = bh.ForceModelConfig.two_body()
    end
    
    # Spacecraft parameters: [mass, drag_area, Cd, srp_area, Cr]
    # Default to simple parameters if not provided
    if sat_params === nothing
        sat_params = pyimport("numpy").array([500.0, 2.0, 2.2, 2.0, 1.3])
    end
    
    # Create propagator
    prop = bh.NumericalOrbitPropagator(
        bh_epoch,
        state,
        prop_config,
        force_config,
        params=sat_params
    )
    
    return prop, bh_epoch
end

function random_eci(; seed=false)
    """
    Using a reasonable set of orbital parameters for LEO, generate an initial ephemeris position
    Returns state in km and km/s
    """
    bh = get_brahe()
    
    if seed != false
        Random.seed!(seed)
    end
    
    # Generate random orbital elements
    R = bh.R_EARTH + 400e3 + 200e3*rand()  # meters
    e = 0.01 + 0.1*rand()
    i = 75.0 + 15.0*rand()  # degrees
    Ω = 45.0 + 45.0*rand()  # degrees
    ω = 30.0 + 30.0*rand()  # degrees
    M = 360.0*rand()  # degrees
    
    # Create orbital elements array
    np = pyimport("numpy")
    oe = np.array([R, e, i, Ω, ω, M])
    
    # Convert to ECI (returns in meters and m/s)
    new_eci_m = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
    
    # Convert to km and km/s for consistency with existing code
    new_eci = collect(new_eci_m) ./ 1000.0
    
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

function generate_conjunction(;seed=false, debris_cov_diag = [1e1, 1e5, 1e6], epoch_datetime = (2024, 1, 1, 12, 0, 0.0, 0.0))
    """
    Generate a conjunction between a spacecraft and object
    Returns: (eci_spacecraft, eci_debris, prop_spacecraft, prop_debris, epoch_spacecraft, epoch_debris)
    """
    if seed != false
        Random.seed!(seed)
    end

    eci_spacecraft = random_eci()
    eci_debris = random_debris(eci_spacecraft, rtn_cov_diag=debris_cov_diag)

    prop_spacecraft, epoch_spacecraft = eci2orb_brahe(eci_spacecraft, epoch_datetime = epoch_datetime)
    prop_debris, epoch_debris = eci2orb_brahe(eci_debris, epoch_datetime = epoch_datetime)

    return eci_spacecraft, eci_debris, prop_spacecraft, prop_debris, epoch_spacecraft, epoch_debris
end


# TODO: Revisit these sample covariance value ranges, not sure if right range
function sample_covariance(;seed=false, Σc_μ = [10^2,10^4,10^2, 4e-5, 4e-3, 4e-5], Σd_μ = [10^3,10^7,10^3, 1e-3, 1e-1, 1e-3], Σc_σ = [20,200,20,0.005, 0.05, 0.005], Σd_σ = [200,2000,200,0.02, 0.20, 0.02])
    """
    Return a random covariance matrix for both the spacecraft and debris object
    Uses absolute values to ensure positive diagonal elements
    """
    
    if seed != false
        Random.seed!(seed)
    end
    # Generate Covariance distributions based on the examples from spacetrack.org - which happen at TCA = 3 - 1 day in advance.
    Σc_distribution = Distributions.MvNormal(Σc_μ, Diagonal(Σc_σ))
    Σd_distribution = Distributions.MvNormal(Σd_μ, Diagonal(Σd_σ))

    
    # Generate Covariances and ensure positive values
    Σc_vals = abs.(rand(Σc_distribution))  
    Σd_vals = abs.(rand(Σd_distribution))  
    Σc = Diagonal(Σc_vals)
    Σd = Diagonal(Σd_vals)
    return Σc, Σd
end


function sim_backwards_brahe(prop, epoch_start, epoch_target)
    """
    Return a random covariance matrix for both the spacecraft and debris object

    Propagate backward from epoch_start to epoch_target using brahe
    Since brahe only propagates forward, or backward propagation, a new propagato is set up at the target time
    and propagate forward to the start time, not sure if this works or if there's something else we can try here?
        
    Returns: (t_arr, epc_arr, x_arr) where x_arr is in km
    """
    bh = get_brahe()
    
    # Check if we're actually going forward
    if epoch_target >= epoch_start
        # Forward propagation
        prop.propagate_to(epoch_target)
        final_state = prop.state() ./ 1000.0  # Convert to km
        return [0.0, (epoch_target - epoch_start)], [epoch_start, epoch_target], hcat(prop.state_at(epoch_start) ./ 1000.0, final_state)
    end

    # For backward propagation, we'll use the current state and negate the velocity
    # This is an approximation that works for short timescales
    current_state = prop.state()
    
    # Create a state with reversed velocity for backward integration
    reversed_state = copy(current_state)
    reversed_state[4:6] = -current_state[4:6]
    
    # Calculate time difference (positive)
    dt_seconds = epoch_start - epoch_target
    
    # Create a new propagator with reversed velocity
    prop_config = bh.NumericalPropagationConfig.default()
    force_config = bh.ForceModelConfig.two_body()  # Simplified for backward prop
    
    np = pyimport("numpy")
    params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])
    
    prop_backward = bh.NumericalOrbitPropagator(
        epoch_start,
        reversed_state,
        prop_config,
        force_config,
        params=params
    )
    
    # Propagate "forward" in reversed time
    prop_backward.propagate_to(epoch_start + dt_seconds)
    
    # Get the final state and reverse the velocity back
    final_state = prop_backward.state()
    final_state[4:6] = -final_state[4:6]
    
    # Build output arrays (convert to km)
    t_arr = [0.0, dt_seconds]
    epc_arr = [epoch_target, epoch_start]
    x_arr = hcat(final_state ./ 1000.0, current_state ./ 1000.0)
    
        return t_arr, epc_arr, x_arr
end

function generate_one_CDM(;seed = false, epoch_str="2024001000000.000", debris_cov_diag = [0.1, 20.0, 20.0], syntheticTCA = 9, dt_seconds = 28800.0, rc_range = (5.,10.), rd_range = (5.,10.))
    """
    Generate a conjunction, and propagate the states backwards from indicated tca epoch
    Generate a Return a random covariance matrix for both the spacecraft and debris object
    """
    
    if seed != false
        Random.seed!(seed)
    end

    # Parse epoch string to datetime tuple
    epoch_dt = parse_spacex_epoch(epoch_str)
    
    # Calculate TCA time in seconds from start
    tca_offset_seconds = syntheticTCA * dt_seconds

    # Generate conjunction at TCA
    spacecraft, debris, prop_spacecraft, prop_debris, epoch_spacecraft, epoch_debris = generate_conjunction(;seed=seed, debris_cov_diag=debris_cov_diag, epoch_datetime = epoch_dt)
    Σc, Σd = sample_covariance(;seed=seed)

    rc = rand(Distributions.Uniform(rc_range...)) 
    rd = rand(Distributions.Uniform(rd_range...))

    # For now, we'll use the initial states directly since backward propagation is complex
    # In a full implementation, we would create the conjunction at TCA and propagate backward
    # For this version, we create the conjunction at the start and will propagate forward to TCA
    
    # Get initial states in km
    x_arr_c = spacecraft
    x_arr_d = debris

    # Use covariances as-is (no scaling for better collision detection)
    # Previous scaling was reducing uncertainties too much, making Pc too low
    return CDM(syntheticTCA, x_arr_c, x_arr_d, Matrix(Σc), Matrix(Σd), rc, rd)
end

function parse_spacex_epoch(epc_str="2025310104542.000")
    """
    Parse SpaceX epoch string to datetime tuple for brahe
    Returns: (year, month, day, hour, minute, second, nanosecond)
    """
    # Parse components
    year = parse(Int, epc_str[1:4])
    julian_day = parse(Int, epc_str[5:7])
    hour = parse(Int, epc_str[8:9])
    minute = parse(Int, epc_str[10:11])
    second = parse(Float64, epc_str[12:end])
    
    # Convert Julian day to month/day
    date_val = Date(year) + Day(julian_day - 1)
    month_val = Dates.month(date_val)
    day_val = Dates.day(date_val)
    
    # Split second into integer and fractional parts
    sec_int = floor(Int, second)
    sec_frac = second - sec_int
    nanosec = sec_frac * 1e9
    
    return (year, month_val, day_val, hour, minute, Float64(sec_int), nanosec)
end
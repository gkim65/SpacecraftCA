using PyCall
using GaussianFilters
using Dates, Printf
using LinearAlgebra
using Distributions
using Random
using Statistics

# Import brahe Python library - lazy initialization
const _bh_prop_cache = Ref{Union{PyObject, Nothing}}(nothing)

function get_brahe_prop()
    if _bh_prop_cache[] === nothing
        _bh_prop_cache[] = pyimport("brahe")
        # Initialize EOP and space weather data
        _bh_prop_cache[].initialize_eop()
        _bh_prop_cache[].initialize_sw()
    end
    return _bh_prop_cache[]
end


####### Monte Carlo Based Covariance Matrix
# Quite slow, possibly not worth using

function mc_propagate_mean_cov(mean_x, cov_mat, u, epc0, T, n_samples=200)
    L = cholesky(Symmetric(cov_mat)).L
    samples = [ mean_x .+ L * randn(6) for _ in 1:n_samples ]
    propagated = [ step(s, u, epc0, T) for s in samples ]     # step returns km-state
    X = reduce(hcat, propagated)
    mean_prop = mean(X, dims=2)[:,1]
    cov_prop = cov(Matrix(X)', dims=1) # sample cov on rows
    return mean_prop, cov_prop
end


####### UKF based covariance propagation

function spaceXEpoch_brahe(epc_str="2025310104542.000")
    """
    Parse SpaceX epoch string and return brahe Epoch object
    """
    bh_prop = get_brahe_prop()
    
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
    
    # Create brahe epoch
    return bh_prop.Epoch.from_datetime(year, month_val, day_val, hour, minute, second, 0.0, bh_prop.TimeSystem.UTC)
end


# TODO: Test out different thrust_magnitudes
function apply_thrust(eci, thrust_direction, thrust_magnitude = 10)
    x = eci[1:3]
    v = eci[4:6]
    thrust = thrust_magnitude * v/norm(v) * thrust_direction # this is our u > control, it can be 1, -1, or 0
    return vcat(x, v + thrust)
end

# Assuming x in ECI frame (km and km/s)
# T is in seconds
# epc0 is a brahe Epoch object
function step(x, u, epc0, T)
    """
    Propagate state x forward by T seconds using brahe
    x: state in km and km/s
    u: control vector [thrust_direction]
    epc0: brahe Epoch object
    T: propagation time in seconds
    Returns: state in km and km/s
    """
    bh_prop = get_brahe_prop()
    
    epcf = epc0 + T
    x_state = 1000.0 .* Float64.(x)  # Convert to meters

    # Apply thrust
    x_state = apply_thrust(x_state, u[1])
    
    # Create propagation config
    prop_config = bh_prop.NumericalPropagationConfig.default()
    
    # Full force model
    force_config = bh_prop.ForceModelConfig.default()
    
    # Spacecraft parameters: [mass, drag_area, Cd, srp_area, Cr]
    np = pyimport("numpy")
    params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])
    
    # Create propagator
    prop = bh_prop.NumericalOrbitPropagator(
        epc0,
        x_state,
        prop_config,
        force_config,
        params=params
    )
    
    # Propagate to final epoch
    prop.propagate_to(epcf)
    
    # Get final state
    final_state = prop.state()
    
    # Convert back to km
    return collect(final_state) ./ 1000.0
end

function symmetric_from_lower(v)
    # figure out n from number of lower-triangular elements
    n = floor(Int, sqrt(2*length(v) + 0.25) - 0.5)
    A = zeros(eltype(v), n, n)
    k = 1
    for i in 1:n, j in 1:i
        A[i, j] = v[k]
        A[j, i] = v[k]   # mirror across diagonal
        k += 1
    end
    return A
end

# nonlinear observation function. must be a function of both states (x) and actions (u) even if either are not used.
function observe(x,u)
    return x
end

function rRTNtoECI(x)
    """
    Compute rotation matrix from RTN (Radial-Tangential-Normal) to ECI frame
    x: state vector [x, y, z, vx, vy, vz] in ECI frame
    Returns: 3x3 rotation matrix R such that v_ECI = R * v_RTN
    """
    r = x[1:3]  # position vector
    v = x[4:6]  # velocity vector
    
    # Radial direction (normalized position vector)
    R_hat = r / norm(r)
    
    # Normal direction (normalized angular momentum)
    h = cross(r, v)
    N_hat = h / norm(h)
    
    # Tangential direction (completes right-handed system)
    T_hat = cross(N_hat, R_hat)
    
    # Build rotation matrix: columns are RTN basis vectors in ECI frame
    return hcat(R_hat, T_hat, N_hat)
end

function covRTNtoECI(x, covariance)
    R = rRTNtoECI(x)  # 3x3
    return [R zeros(3,3); zeros(3,3) R] * covariance * [R' zeros(3,3); zeros(3,3) R']
end

function predictStep(m::NonlinearDynamicsModel, x::AbstractVector{<:Number}, 
                 u::AbstractVector{<:Number}, epc0, T)
    return m.f(x, u, epc0, T)
end


# Unscented Kalman Filter functions

"""
    predict(filter::UnscentedKalmanFilter, b0::GaussianBelief, u::AbstractVector)

Uses Unscented Kalman filter to run prediction step on gaussian belief b0,
given control vector u.
"""
function predictEpc(filter::UnscentedKalmanFilter, b0::GaussianBelief, u::AbstractVector{<:Number}, epc0, T)

    # Motion update

    n = length(b0.μ)

    # approximate Gaussian belief with sigma points
    points, w_μ, w_Σ = unscented_transform(b0, filter.λ, filter.α, filter.β)

    # iterate over each sigma point and propagate it through motion function
    pointsp = [predictStep(filter.d, point, u, epc0, T) for point in points]

    # apply inverse unscented transform to approximate new Gaussian
    bp = unscented_transform_inverse(pointsp, w_μ, w_Σ)

    # add process noise
    Σp = bp.Σ + filter.d.W

    return GaussianBelief(bp.μ, Σp)
end







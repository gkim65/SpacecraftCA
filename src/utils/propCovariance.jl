using SatelliteDynamics
using GaussianFilters
using Dates, Printf
using LinearAlgebra
using Distributions
using Random
using Statistics


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

function spaceXEpoch(epc_str="2025310104542.000")
    # Parse components (Julia uses 1-based indexing!)
    year = parse(Int, epc_str[1:4])
    julian_day = parse(Int, epc_str[5:7])
    hour = parse(Int, epc_str[8:9])
    minute = parse(Int, epc_str[10:11])
    second = parse(Float64, epc_str[12:end])  # includes fractional part

    # Convert Julian day to month/day
    date_val = Date(year) + Day(julian_day - 1)
    month_val = Dates.month(date_val)
    day_val = Dates.day(date_val)
    return SatelliteDynamics.Epoch(year, month_val, day_val, hour, minute, second)
end


# TODO: Test out different thrust_magnitudes
function apply_thrust(eci, thrust_direction, thrust_magnitude = 10)
    x = eci[1:3]
    v = eci[4:6]
    thrust = thrust_magnitude * v/norm(v) * thrust_direction # this is our u > control, it can be 1, -1, or 0
    return vcat(x, v + thrust)
end

# Assuming x in ECI frame
# T is in seconds
function  step(x, u, epc0, T) #epc_str="2025320194042.000", T=60)

    epcf = epc0 + T
    x_state = 1000.0 .* Float64.(x)

    x_state = apply_thrust(x_state, u[1])
    
    # Initialize State Vector
    orb  = EarthInertialState(epc0, x_state, dt=60.0,
               mass=1.0, n_grav=0, m_grav=0,
               drag=true, srp=true,
               moon=true, sun=true,
               relativity=true
    )

    # Simulate orbit
    t, epc, eci = sim!(orb, epcf)

    return vec(eci[:, end]) ./ 1000.0

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







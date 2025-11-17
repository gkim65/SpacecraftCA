"""
Orbital propagation functions using SatelliteDynamics.jl.

Provides realistic orbital propagation with SRP, moon, sun, and relativity effects.
"""

using SatelliteDynamics
using Dates
using Printf
using LinearAlgebra
using Distributions

"""
Propagate orbital state forward in time using SatelliteDynamics.jl.

Uses realistic dynamics including:
- Solar radiation pressure (SRP)
- Moon perturbations
- Sun perturbations  
- Relativity effects

# Arguments
- `x`: State vector [x, y, z, vx, vy, vz] in km, km/s
- `u`: Action (not used, can be nothing)
- `epc_str`: Epoch string in format "YYYYJJJHHMMSS.fff" (e.g., "2025296104542.000")
- `T`: Time step in seconds (default: 60)

# Returns
- Propagated state vector [x, y, z, vx, vy, vz] in km, km/s
"""
function step(x, u, epc_str="2025296104542.000", T=60)
    x_c, y_c, z_c, vx, vy, vz = x
    
    # Parse epoch string components (Julia uses 1-based indexing!)
    year = parse(Int, epc_str[1:4])
    julian_day = parse(Int, epc_str[5:7])
    hour = parse(Int, epc_str[8:9])
    minute = parse(Int, epc_str[10:11])
    second = parse(Int, epc_str[12:13])
    nanosecond = parse(Float64, epc_str[14:end])
    
    # Convert Julian day to month/day
    date_val = Date(year) + Day(julian_day - 1)
    month_val = Dates.month(date_val)
    day_val = Dates.day(date_val)
    epc0 = Epoch(year, month_val, day_val, hour, minute, second, nanosecond)
    epcf = epc0 + T
    
    # Convert state from km to meters (SatelliteDynamics uses meters)
    x_state = 1000.0 .* Float64.(x)
    
    # Initialize State Vector with realistic dynamics
    orb = EarthInertialState(epc0, x_state, dt=1.0,
               mass=1.0, n_grav=0, m_grav=0,
               drag=false, srp=true,
               moon=true, sun=true,
               relativity=true
    )
    
    # Propagate the orbit
    t, epc, eci = sim!(orb, epcf)
    
    # Return final propagated Cartesian state in km and km/s
    return vec(eci[:, end]) ./ 1000.0
end

"""
Propagate both state and covariance forward in time.

TEMPORARY: Simple covariance propagation using additive process noise.
TODO: Replace with UKF-based propagation (team member working on this).

Current implementation:
- Propagates mean state using step()
- Adds process noise to covariance (simple linear approximation)

Future UKF implementation will:
- Propagate sigma points through nonlinear dynamics
- Compute new covariance from propagated sigma points
- More accurate for nonlinear orbital dynamics

# Arguments
- `x`: State vector [x, y, z, vx, vy, vz] in km, km/s
- `Σ`: Covariance matrix (6x6)
- `u`: Action (not used, can be nothing)
- `epc_str`: Epoch string in format "YYYYJJJHHMMSS.fff"
- `T`: Time step in seconds (default: 60)
- `W`: Process noise covariance matrix (6x6), default: T*0.05*I

# Returns
- Tuple `(x_prop, Σ_prop)`: Propagated state vector and covariance matrix
"""
function step_with_covariance(x, Σ, u, epc_str="2025296104542.000", T=60; W=nothing)
    # TEMPORARY: Simple propagation - will be replaced with UKF
    # Propagate mean state using step()
    x_prop = step(x, u, epc_str, T)
    
    # Default process noise: W = dt * 0.05 * I (from your original code)
    if W === nothing
        W = T * 0.05 * Matrix{Float64}(I, 6, 6)
    end
    
    # Simple covariance propagation: add process noise
    # This is a linear approximation - UKF will handle nonlinear dynamics properly
    Σ_prop = Σ + W
    
    return x_prop, Σ_prop
end

"""
UKF-based state and covariance propagation.

TODO: Implement UKF propagation using GaussianFilters.jl
This will replace step_with_covariance() for accurate nonlinear covariance propagation.

# Arguments
- `x`: State vector [x, y, z, vx, vy, vz] in km, km/s
- `Σ`: Covariance matrix (6x6)
- `u`: Action (not used, can be nothing)
- `epc_str`: Epoch string in format "YYYYJJJHHMMSS.fff"
- `T`: Time step in seconds
- `W`: Process noise covariance matrix (6x6)

# Returns
- Tuple `(x_prop, Σ_prop)`: Propagated state vector and covariance matrix
"""
function step_with_covariance_ukf(x, Σ, u, epc_str="2025296104542.000", T=60; W=nothing)
    # TODO: Implement UKF propagation
    # This will use GaussianFilters.jl to:
    # 1. Generate sigma points from current state and covariance
    # 2. Propagate each sigma point through step() function
    # 3. Compute new mean and covariance from propagated sigma points
    # 4. Add process noise
    
    error("step_with_covariance_ukf() not yet implemented - use step_with_covariance() for now")
end

"""
Advance epoch string by a time step.

# Arguments
- `epc_str`: Epoch string in format "YYYYJJJHHMMSS.fff"
- `T`: Time step in seconds

# Returns
- New epoch string advanced by T seconds
"""
function advance_epoch_str(epc_str::AbstractString, T::Real)
    yyyy = parse(Int, epc_str[1:4])
    jjj  = parse(Int, epc_str[5:7])
    hh   = parse(Int, epc_str[8:9])
    mm   = parse(Int, epc_str[10:11])
    ss   = parse(Int, epc_str[12:13])
    frac = parse(Float64, epc_str[14:end])
    
    base_date = Date(yyyy) + Day(jjj - 1)
    dt = DateTime(base_date) + Hour(hh) + Minute(mm) + Second(ss) + Millisecond(round(Int, frac))
    dt2 = dt + Millisecond(round(Int, T*1000))
    
    @sprintf("%04d%03d%02d%02d%02d.%03d",
             Dates.year(dt2), Dates.dayofyear(Date(dt2)), Dates.hour(dt2), 
             Dates.minute(dt2), Dates.second(dt2), Dates.millisecond(dt2))
end


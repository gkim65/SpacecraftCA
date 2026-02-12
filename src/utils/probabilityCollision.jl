using HCubature
using Distributions


function integrate_circle(gaussian, radius)
    f(polarcoords) = polarcoords[1] * pdf(gaussian, [polarcoords[1] * cos(polarcoords[2]), polarcoords[1] * sin(polarcoords[2])])  # Polar coordinates integrand
    result, _ = hcubature(f, (0, 0), (radius, 2*π))
    return result
end

function fosterPcAnalytical(object1_x, object1_Σ, object2_x, object2_Σ; object1_radius = 10, object2_radius = 10)
    """
    Calculate the probability of collision between an agent and debris using the Foster analytical method.
    Ephemeris Datatype:
    - `epoch`: Epoch representing the time of the ephemeris data.
    - `x`: Vector of the state vector `[x, y, z, u, v, w]`. This is in the ECI Reference Frame, x,y,z, are the position coordinates, and u,v,w are the velocity coordinates. https://en.wikipedia.org/wiki/Earth-centered_inertial
    - `cov`: 6x6 covariance matrix representing the uncertainty in the state vector. This covariance is in the RTN frame - radial, tangential, normal. https://www.researchgate.net/figure/Geocentric-Equatorial-and-RTN-co-ordinate-systems_fig1_267324871
    - `r_hardbody`: Float64 representing the hardbody radius of the satellite.
    - `id`: Identifier for the object, either `:agent` or `:debris`.

    # Arguments
    - `eph_agent`: Ephemeris data for the agent.
    - `eph_debris`: Ephemeris data for the debris.
    
    # Outputs
    - `Pc`: The probability of collision between the agent and debris.

    Notes:

    - Foster Monte Carlo and Foster Analytical should result in the same Pc.
    """
    
    # Safety check for NaN/Inf in inputs
    if any(isnan.(object1_x)) || any(isinf.(object1_x)) || 
       any(isnan.(object2_x)) || any(isinf.(object2_x)) ||
       any(isnan.(object1_Σ)) || any(isinf.(object1_Σ)) ||
       any(isnan.(object2_Σ)) || any(isinf.(object2_Σ))
        return 0.0  # Return zero collision probability if inputs are invalid
    end

    u_var = object1_Σ[1, 1] + object2_Σ[1, 1]

    # U axis is orthogonal to the velocity plane
    u_axis = cross(object1_x[4:6], object2_x[4:6])
    u_norm = norm(u_axis)
    if u_norm < 1e-10  # Parallel velocities
        return 0.0  # Can't compute collision probability
    end
    u_axis /= u_norm
    
    # V axis is in the direction of relative velocity
    v_axis = object2_x[4:6] - object1_x[4:6]
    v_norm = norm(v_axis)
    if v_norm < 1e-10  # Same velocity
        return 0.0
    end
    v_axis /= v_norm
    
    w_axis = cross(u_axis, v_axis)
    # This matrix rotates from UVW to XYZ
    R_inv = hcat(u_axis, v_axis, w_axis)
    
    # Check for NaN/Inf before inversion
    if any(isnan.(R_inv)) || any(isinf.(R_inv))
        return 0.0
    end
    
    # This matrix rotates from XYZ to UVW
    R = inv(R_inv)

    # Recenter the coordinate system to spacecraft
    debris_pos = object2_x[1:3] - object1_x[1:3]
    # Rotate debris position to UVW frame
    debris_pos = R * debris_pos
    U0 = debris_pos[1]
    W0 = debris_pos[3]

    theta_r_spacecraft = acos(dot(v_axis, object1_x[4:6]) / (norm(object1_x[4:6])))
    theta_r_debris = acos(dot(v_axis, object2_x[4:6]) / (norm(object2_x[4:6])))

    w_var = object1_Σ[2, 2] * (sin(theta_r_spacecraft) ^ 2) + object1_Σ[3, 3] * (cos(theta_r_spacecraft) ^ 2) + object2_Σ[2, 2] * (sin(theta_r_debris) ^ 2) + object2_Σ[3, 3] * (cos(theta_r_debris) ^ 2)

    uw_mean = [U0, W0]
    uw_cov = [u_var 0;
              0 w_var]
    rhb1 = object1_radius
    rhb2 = object2_radius
    d_hb = rhb1+rhb2


    gaussian = MvNormal(uw_mean, uw_cov)
    Pc = integrate_circle(gaussian, d_hb)

    return Pc
end

function fosterPcState(pomdp::SpacecraftCAPOMDP, state::SpacecraftCAState)
    if state.TCA <= 0
        xs_tca = state.xs
        xd_tca = state.xd
        Σs_tca = state.Σs
        Σd_tca = state.Σd
    else
        T_total = state.TCA * pomdp.dt_seconds
        
        bp_s = unscented_kalman_filter(pomdp, state.xs, state.Σs, [0.0], T_total)
        bp_d = unscented_kalman_filter(pomdp, state.xd, state.Σd, [0.0], T_total)
        
        xs_tca = bp_s.μ
        xd_tca = bp_d.μ
        Σs_tca = Matrix(bp_s.Σ)
        Σd_tca = Matrix(bp_d.Σ)
    end
    
    Σs_rtn = getRTNCovariance(xs_tca, Σs_tca)
    Σd_rtn = getRTNCovariance(xd_tca, Σd_tca)
    Σs_6x6 = zeros(6, 6)
    Σd_6x6 = zeros(6, 6)
    Σs_6x6[1:3, 1:3] = Σs_rtn
    Σd_6x6[1:3, 1:3] = Σd_rtn
    return fosterPcAnalytical(
        xs_tca, Σs_6x6, 
        xd_tca, Σd_6x6,
        object1_radius=state.rs,
        object2_radius=state.rd
    )
end

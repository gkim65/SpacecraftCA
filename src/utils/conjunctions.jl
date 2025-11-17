"""
Functions for generating satellite-debris conjunction scenarios.

Port of conjunction generation functions from ManeuverTimeMDP, adapted to use
satellite/debris (s/d) naming and SpacecraftCAState.
"""

using LinearAlgebra
using SatelliteDynamics
using Distributions
using Random

"""
Sample from a distribution in RTN (Radial-Tangential-Normal) coordinates.

# Arguments
- `xyz`: 6-element state vector in ECI coordinates [x, y, z, vx, vy, vz]
- `Σ`: 3x3 covariance matrix in RTN coordinates

# Returns
- 6-element state vector in ECI coordinates with sampled offset
"""
function sample_from_rtn(xyz, Σ)
    @assert size(Σ) == (3, 3)
    @assert size(xyz) == (6,)
    rvec = xyz[1:3] / norm(xyz[1:3])
    tvec = xyz[4:6] / norm(xyz[4:6])
    nvec = cross(rvec, tvec)
    nvec = nvec / norm(nvec)
    R = hcat(rvec, tvec, nvec)

    rtn_offset = rand(MvNormal([0, 0, 0], Σ))
    xyz_offset = R * rtn_offset
    return vcat(xyz[1:3] + xyz_offset, xyz[4:6])
end

"""
Sample covariance matrices for satellite and debris.

# Arguments
- `seed`: Random seed (optional)
- `Σs_μ`: Mean values for satellite covariance diagonal [R, T, N] (default: [10^1, 10^3, 10^1])
- `Σd_μ`: Mean values for debris covariance diagonal [R, T, N] (default: [10^2, 10^6, 10^2])
- `Σs_σ`: Std dev for satellite covariance diagonal (default: [2, 20, 2])
- `Σd_σ`: Std dev for debris covariance diagonal (default: [20, 200, 20])

# Returns
- `(Σs, Σd)`: Tuple of 6x6 diagonal covariance matrices
"""
function sample_covariance(;
    seed=false, 
    Σs_μ=[10^1, 10^3, 10^1], 
    Σd_μ=[10^2, 10^6, 10^2], 
    Σs_σ=[2, 20, 2], 
    Σd_σ=[20, 200, 20]
)
    if seed !== false
        Random.seed!(seed)
    end
    # Generate Covariance distributions based on examples from spacetrack.org
    # which happen at TCA = 3 - 1 day in advance.
    Σs_distribution = MvNormal(Σs_μ, Diagonal(Σs_σ))
    Σd_distribution = MvNormal(Σd_μ, Diagonal(Σd_σ))

    # Generate Covariances (expand 3x3 RTN to 6x6 full state)
    Σs_3x3 = Diagonal(rand(Σs_distribution))
    Σd_3x3 = Diagonal(rand(Σd_distribution))
    
    # Create 6x6 matrices (position covariance only, velocity assumed known)
    Σs = zeros(6, 6)
    Σd = zeros(6, 6)
    Σs[1:3, 1:3] = Σs_3x3
    Σd[1:3, 1:3] = Σd_3x3
    
    return Σs, Σd
end

"""
Generate a random satellite orbit in ECI coordinates.

# Arguments
- `seed`: Random seed (optional)

# Returns
- 6-element state vector [x, y, z, vx, vy, vz] in ECI frame (km, km/s)
"""
function random_eci(; seed=false)
    if seed !== false
        Random.seed!(seed)
    end
    epc = Epoch(2024, 1, 1, 12, 0, 0, 0.0)
    R = R_EARTH + 400e3 + 200e3 * rand()  # LEO: 400-600 km altitude
    e = 0.01 + 0.1 * rand()                # Eccentricity
    i = 75.0 + 15.0 * rand()                # Inclination (degrees)
    Ω = 45.0 + 45.0 * rand()                # RAAN (degrees)
    ω = 30.0 + 30.0 * rand()                # Argument of perigee (degrees)
    M = 360.0 * rand()                      # Mean anomaly (degrees)
    new_eci = sOSCtoCART([R, e, i, Ω, ω, M])
    return new_eci
end

"""
Generate a random debris orbit relative to a satellite.

# Arguments
- `eci_satellite`: Satellite state vector in ECI frame
- `rtn_cov_diag`: Diagonal of RTN covariance for debris offset (default: [1e1, 1e4, 1e4])
- `seed`: Random seed (optional)

# Returns
- 6-element debris state vector [x, y, z, vx, vy, vz] in ECI frame (km, km/s)
"""
function random_debris(eci_satellite; rtn_cov_diag=[1e1, 1e4, 1e4], seed=false)
    if seed !== false
        Random.seed!(seed)
    end
   
    # Use the vector and the arbitrary covariance to generate a random debris position
    rtn_offset_distribution = MvNormal([0, 0, 0], Diagonal(rtn_cov_diag))
    rtn_offset = rand(rtn_offset_distribution)
    rvec = eci_satellite[1:3] / norm(eci_satellite[1:3])  # radial direction
    tvec = eci_satellite[4:6] / norm(eci_satellite[4:6])  # along-track direction
    nvec = cross(rvec, tvec)  # cross-track direction
    nvec = nvec / norm(nvec)
    R = hcat(rvec, tvec, nvec)
    debris_x = eci_satellite[1:3] + R * rtn_offset

    # For the debris to be at TCA, it must meet the following conditions:
    # 1. The debris velocity must be perpendicular to the vector from debris to satellite
    # 2. The debris must be in a valid orbit
    # 3. These conditions constrain 2 dimensions of the 3D velocity vector
    
    relative_vector = debris_x - eci_satellite[1:3]
    A = transpose(hcat(debris_x, relative_vector, [0, 0, 1]))
    b = [0, 0, rand(-1:1e-18:1)]
    debris_v_dir = A \ b
    debris_v = debris_v_dir / norm(debris_v_dir) * norm(eci_satellite[4:6])

    return vcat(debris_x, debris_v)
end

"""
Generate a satellite-debris conjunction pair.

# Arguments
- `seed`: Random seed (optional)
- `orbit`: If true, return orbit objects instead of state vectors (default: false)
- `debris_cov_diag`: Diagonal of RTN covariance for debris generation (default: [1e1, 1e5, 1e6])

# Returns
- `(eci_satellite, eci_debris)`: Tuple of state vectors or orbit objects
"""
function generate_conjunction(;
    seed=false, 
    orbit=false, 
    debris_cov_diag=[1e1, 1e5, 1e6]
)
    if seed !== false
        Random.seed!(seed)
    end

    eci_satellite = random_eci()
    eci_debris = random_debris(eci_satellite, rtn_cov_diag=debris_cov_diag)

    if orbit
        # Convert to orbit objects (if needed)
        epc = Epoch(2024, 1, 1, 12, 0, 0, 0.0)
        orb_satellite = EarthInertialState(epc, eci_satellite, dt=1.0,
                                          mass=1.0, n_grav=0, m_grav=0,
                                          drag=false, srp=false,
                                          moon=false, sun=false,
                                          relativity=false)
        orb_debris = EarthInertialState(epc, eci_debris, dt=1.0,
                                       mass=1.0, n_grav=0, m_grav=0,
                                       drag=false, srp=false,
                                       moon=false, sun=false,
                                       relativity=false)
        return orb_satellite, orb_debris
    end

    return eci_satellite, eci_debris
end

"""
Generate a single SpacecraftCAState for initializing the POMDP.

# Arguments
- `seed`: Random seed (optional)
- `debris_cov_diag`: Diagonal of RTN covariance for debris generation (default: [1e1, 1e4, 1e4])
- `syntheticTCA`: Initial TCA value (default: 9)
- `rs_range`: Range for satellite radius in meters (default: (5.0, 10.0))
- `rd_range`: Range for debris radius in meters (default: (5.0, 10.0))
- `Σs_μ`: Mean values for satellite covariance (optional)
- `Σd_μ`: Mean values for debris covariance (optional)

# Returns
- `SpacecraftCAState`: Initial state for the POMDP
"""
function generate_one_state(;
    seed=false, 
    debris_cov_diag=[1e1, 1e4, 1e4], 
    syntheticTCA=9, 
    rs_range=(5.0, 10.0), 
    rd_range=(5.0, 10.0),
    Σs_μ=[10^1, 10^3, 10^1],
    Σd_μ=[10^2, 10^6, 10^2]
)
    if seed !== false
        Random.seed!(seed)
    end
    eci_satellite, eci_debris = generate_conjunction(seed=seed, debris_cov_diag=debris_cov_diag)
    Σs, Σd = sample_covariance(seed=seed, Σs_μ=Σs_μ, Σd_μ=Σd_μ)
    rs = rand(Distributions.Uniform(rs_range...)) 
    rd = rand(Distributions.Uniform(rd_range...))
    return SpacecraftCAState(syntheticTCA, eci_satellite, eci_debris, Σs, Σd, rs, rd)
end


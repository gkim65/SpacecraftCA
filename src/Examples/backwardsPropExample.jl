using Pkg
Pkg.activate("SpacecraftCA")
Pkg.instantiate()

include("../utils/genConjunctions.jl")

using SatelliteDynamics
# Declare simulation initial Epoch
epc0 = Epoch(2019, 1, 1, 12, 0, 0, 0.0) 

# Declare initial state in terms of osculating orbital elements
oe0  = [R_EARTH + 500e3, 0.01, 75.0, 45.0, 30.0, 0.0]

# Convert osculating elements to Cartesean state
eci0 = sOSCtoCART(oe0, use_degrees=true)

# Set the propagation end time to one orbit period after the start
T    = orbit_period(oe0[1])
epcf = epc0 - T

# Initialize State Vector
orb  = EarthInertialState(epc0, eci0, dt=1.0,
            mass=1.0, n_grav=0, m_grav=0,
            drag=false, srp=false,
            moon=false, sun=false,
            relativity=false
)

# Propagate the orbit
t, epc, eci = sim_backwards!(orb, epcf)
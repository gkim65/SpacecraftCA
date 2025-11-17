using Pkg
Pkg.activate("SpacecraftCA")
Pkg.instantiate()

include("../utils/propCovariance.jl")


# epc_str is year, julian day, hour, min, sec, nanosecond
epc_str="2025320194042.000"
epc0 = spaceXEpoch(epc_str)
T = 3600 # seconds
u = zeros(60) # dt is every minute right now

# Example state vector X from SpaceX Starlink (pos_x, pos_y, pos_z, vel_x, vel_y, vel_z)
x = [1330.2616056555, 5488.3189463136, 4020.4428044921, -7.0711182832, -0.2910718924, 2.7303984809]

# Example covariance given from SpaceX Starlink (Lower triangular matrix)
covariance = [4.6168919761e-07 -3.5086167747e-07 7.0813978079e-07 -4.3282984159e-11 -1.8658558907e-10 1.1186913214e-06 7.8384624467e-10 -8.1707665340e-10 1.0005306574e-12 1.8219354109e-12 -4.3894746158e-10 3.6432917302e-10 -1.3868235724e-12 -7.5949340937e-13 4.6485913441e-13 -2.7403008039e-13 3.0716036457e-13 1.5174087582e-09 1.1798576905e-16 -1.6071743466e-15 5.5777455775e-12]

# Convert from RTN to ECI frame
C_eci = covRTNtoECI(x, symmetric_from_lower(covariance))

# Set initial belief of state and covariance
b0 = GaussianBelief(Float64.(x), Symmetric(C_eci))

# Dynamics and Observation Models
W = diagm([1e-6, 1e-6, 1e-6, 1e-12, 1e-12, 1e-12])
dmodel = NonlinearDynamicsModel(step, W);
V = diagm([1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9])
omodel = NonlinearObservationModel(observe,V)

# Full Filter
ukf = UnscentedKalmanFilter(dmodel, omodel);

# Number of prediction steps > only propagate to next hour
N = 2

# Container for predicted beliefs
predictions = Vector{GaussianBelief}(undef, N)
predictions[1] = b0

u = [0] # this is our u > control, it can be 1, -1, or 0
for k in 2:N
    # propagate belief without measurement (prediction step)
    predictions[k] = predictEpc(ukf, predictions[k-1], u, epc0, T) 
end



# To check after, for an hour later measurement
x60 = [3680.6749229388,-3668.0041030438,-4599.8983214543,5.9944271440,4.4649970941,1.2369107836]
covariance60 = [2.7222484482e-06 -2.4894424513e-06 4.9342923494e-05 -6.7912778851e-09 6.8358081174e-08 4.9648737110e-06 3.1589630400e-09 -4.5883822222e-08 -7.2091304693e-11 4.4415123559e-11 -2.7531319919e-09 1.6318991863e-09 1.2065158199e-11 -2.2135592939e-12 2.9335599779e-12 -7.0640408384e-12 1.1526935406e-10 2.0169314089e-09 -1.0908467763e-13 7.7273196772e-15 2.7348179556e-12]
C_eci60 = covRTNtoECI(x60, symmetric_from_lower(covariance60))

# Compare and see if similar
C_eci60
predictions[2].Σ



# # Monte carlo based covariance propagation
# # Takes a while to run this with large samples (right now set to 100)
mean_prop, cov_prop = mc_propagate_mean_cov(Float64.(x), C_eci, u, epc0, T, 100)
println("Monte-Carlo propagated covariance diag: ", diag(cov_prop))
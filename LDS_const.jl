"""
----------------------------------------------
Types
----------------------------------------------
"""
mutable struct Preference
	x₁::Array
	x₂::Array
	ψ::Vector
	pref::Int64
end

"""
----------------------------------------------
General Size Parameters
----------------------------------------------
"""
M = 1000 # Number of weights to sample to estimate the objective function
num_features = 4
num_steps = 5
step_time = 5
ctrl_size = 2

"""
----------------------------------------------
Variables Updated During Iterations
----------------------------------------------
"""
W = zeros(num_features, M)
prefs = Vector{Preference}()
w_hist = Vector{Array{Float64,2}}()

"""
----------------------------------------------
MCMC Parameters
----------------------------------------------
"""
burn = 5000
T = 100

"""
----------------------------------------------
Particle Filter Parameters
----------------------------------------------
"""
num_particles = 1000
particle_noise = Normal(0, 0.001)

"""
----------------------------------------------
Discrete Filter Parameters
----------------------------------------------
"""
probs = (1/M)*ones(M)
@load "points1000.jld2"
weighted_points = vcat([probs' for i = 1:4]...).*points
dist_hist = []
#push!(dist_hist, copy(probs))

"""
----------------------------------------------
Optimization/Dynamics Parameters
----------------------------------------------
"""
lb = repeat([-0.1, -0.2], num_steps*2)
ub = repeat([0.1, 0.2], num_steps*2)
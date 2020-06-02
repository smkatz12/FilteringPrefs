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
M = 150 # Number of weights to sample to estimate the objective function
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
Optimization/Dynamics Parameters
----------------------------------------------
"""
lb = repeat([-0.1, -0.2], num_steps*2)
ub = repeat([0.1, 0.2], num_steps*2)


"""
----------------------------------------------
Nearest Neighbor Parameters
----------------------------------------------
"""
# n_bins = 20

# @load "neighbor_info.jld2" X u_binned
# kdtree = KDTree(X)

"""
----------------------------------------------
Neural Net Parameters
----------------------------------------------
"""
ns = 4
nhidden = 500
k = num_steps*step_time

nn_ϕ = Chain(Dense(ns, nhidden, relu), Dense(nhidden, num_features, relu))
@load "nn_weights.bson" weights
Flux.loadparams!(nn_ϕ, weights)
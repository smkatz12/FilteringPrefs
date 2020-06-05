using Mamba
using Optim
using LinearAlgebra
using ForwardDiff
using LineSearches
using Distributions
using JLD2

include("LDS_const.jl")
include("LDS_functions.jl")

function auto_reward_iteration(w_true; query_type=:info_gain)
	converged = false
	ind=0
	while !converged
		ind += 1
		sample_w(M)
		u_tot = get_inputs(query_type)
		x₁, x₂, ψ = get_ψ(u_tot)

		p = sigmoid(w_true'*ψ)
		if p ≥ rand()
			pref = 1
		else
			pref = -1
		end

		push!(prefs, Preference(x₁, x₂, ψ, pref))
		if ind ≥ 30
			converged = true
		end
	end
end

function auto_reward_iteration_df(w_true, num_iter; query_type=:info_gain)
	update_times = zeros(num_iter)
	opt_times = zeros(num_iter)
	converged = false
	for i = 1:num_iter
		println("Iter: $i")
		start = time()
		u_tot = get_inputs(query_type, discrete=true)
		opt_times[i] = time() - start
		x₁, x₂, ψ = get_ψ(u_tot)

		p = sigmoid(w_true'*ψ)
		if p ≥ rand()
			pref = 1
		else
			pref = -1
		end

		push!(prefs, Preference(x₁, x₂, ψ, pref))
		start = time()
		discrete_update(prefs[end])
		update_times[i] = time() - start
		weighted_points = vcat([probs' for i = 1:num_features]...).*points
	end
	return update_times, opt_times
end

function auto_reward_iteration_pf(w_true; query_type=:info_gain)
	converged = false
	ind=0
	particles = get_initial_particles()
	while !converged
		ind += 1
		println("Iter: $ind")
		u_tot = get_inputs(query_type)
		x₁, x₂, ψ = get_ψ(u_tot)

		p = sigmoid(w_true'*ψ)
		if p ≥ rand()
			pref = 1
		else
			pref = -1
		end

		push!(prefs, Preference(x₁, x₂, ψ, pref))
		particles = pf_update(particles, prefs[end])
		if ind ≥ 80
			converged = true
		end
	end
end
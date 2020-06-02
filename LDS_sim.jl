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
		start = time()
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

function auto_reward_iteration_nn_julia(w_true; query_type=:info_gain, num_queries=30)
	X = []
	for i = 1:num_queries
		start = time()
		sample_w(M)

		u_tot = get_inputs(query_type)

		x₁, x₂, ψ = get_ψ(u_tot)
		x₁ = reshape(x₁, (1, length(x₁)))
		x₂ = reshape(x₂, (1, length(x₂)))
		X = i == 1 ? hcat(x₁, x₂) : [X; hcat(x₁, x₂)]

		p = sigmoid(w_true'*ψ)
		println(p)
		# p = round(p) # This line is if want perfect decision making (comment to put back human decision model)
		if p ≥ rand()
			pref = 1
		else
			pref = -1
		end

		push!(prefs, Preference(x₁, x₂, ψ, pref))
		println("Iter: $i, Elapsed time: $(time()-start)")
	end

	# Save everything necessary for training
	Y = [prefs[i].pref for i in 1:length(prefs)]
	for i = 1:length(Y)
		Y[i] == -1 ? Y[i] = 0 : nothing
	end

	@save "nn_inputs.jld2" X Y w_true
end
using Mamba
using Optim
using LinearAlgebra
using ForwardDiff
using LineSearches
using Distributions
using JLD2
using NearestNeighbors
using DelimitedFiles
using Flux

#include("find_nn.jl")
include("LDS_const.jl")
include("LDS_functions.jl")
include("learn_features.jl")

function auto_reward_iteration(w_true; query_type=:info_gain)
	converged = false
	ind=0
	while !converged
		ind += 1
		start = time()
		sample_w(M)
		u_tot = get_inputs(query_type)
		#println("u_tot: $u_tot")
		x₁, x₂, ψ = get_ψ(u_tot)

		# ϕ₁ = get_ϕ(u_tot[1:num_steps*ctrl_size])
		# println("ϕ₁: $ϕ₁")
		# ϕ₂ = get_ϕ(u_tot[num_steps*ctrl_size+1:end])
		# println("ϕ₂: $ϕ₂")
		# println("ψ: $ψ")

		p = sigmoid(w_true'*ψ)
		if p ≥ rand()
			pref = 1
		else
			pref = -1
		end

		push!(prefs, Preference(x₁, x₂, ψ, pref))
#		println("Iter: $ind, Elapsed time: $(time()-start)")
		# TODO: STOPPING CONDITION!!!!!!!!!!!
		# This one is temporary for debugging
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

# function auto_reward_iteration_nn_features(w_true; query_type=:info_gain)
# 	X = []
# 	for i = 1:30
# 		start = time()
# 		sample_w(M)

# 		u_tot = get_inputs_nn(query_type)

# 		ψ_nn = get_ψ_nn(u_tot)
# 		ψ_true = get_ψ(u_tot)

# 		p = sigmoid(w_true'*ψ_nn)
# 		p = round(p) # This line is if want perfect decision making (comment to put back human decision model)
# 		if p ≥ rand()
# 			pref = 1
# 		else
# 			pref = -1
# 		end

# 		push!(prefs, Preference(ψ_nn, pref))
# 		println("Iter: $i, Elapsed time: $(time()-start)")
# 	end
# end

# function auto_reward_iteration_write(w_true, binFile; query_type=:info_gain)
# 	converged = false
# 	ind=0
# 	while !converged
# 		ind += 1
# 		start = time()
# 		sample_w(M)

# 		# Write W to the file
# 		if !isfile(binFile)
# 			s = open(binFile, "w")
# 		else
# 			s = open(binFile, "a")
# 		end
# 		write_W(s)
# 		close(s)

# 		u_tot = get_inputs(query_type)

# 		# Write u to the file
# 		s = open(binFile, "a")
# 		write_u(s, u_tot)
# 		close(s)

# 		# println("u_tot: $u_tot")
# 		ψ = get_ψ(u_tot)

# 		ϕ₁ = get_ϕ(u_tot[1:num_steps*ctrl_size])
# 		# println("ϕ₁: $ϕ₁")
# 		ϕ₂ = get_ϕ(u_tot[num_steps*ctrl_size+1:end])
# 		# println("ϕ₂: $ϕ₂")
# 		# println("ψ: $ψ")

# 		p = sigmoid(w_true'*ψ)
# 		if p ≥ rand()
# 			pref = 1
# 		else
# 			pref = -1
# 		end

# 		push!(prefs, Preference(ψ, pref))
# #		println("Iter: $ind, Elapsed time: $(time()-start)")
# 		# TODO: STOPPING CONDITION!!!!!!!!!!!
# 		# This one is temporary for debugging
# 		if ind ≥ 30
# 			converged = true
# 		end
# 	end
# end

# function auto_reward_iteration_nn(w_true; query_type=:info_gain)
# 	xa = []
# 	xb = []
# 	for i = 1:150
# 		start = time()
# 		sample_w(M)

# 		u_tot = get_inputs(query_type)

# 		ψ = get_ψ(u_tot)
# 		if i == 1
# 			xa = get_x_mat(u_tot[1:num_steps*ctrl_size])
# 			xb = get_x_mat(u_tot[num_steps*ctrl_size+1:end])
# 		else
# 			xa = [xa; get_x_mat(u_tot[1:num_steps*ctrl_size])]
# 			xb = [xb; get_x_mat(u_tot[1:num_steps*ctrl_size])]
# 		end

# 		p = sigmoid(w_true'*ψ)
# 		if p ≥ rand()
# 			pref = 1
# 		else
# 			pref = -1
# 		end

# 		push!(prefs, Preference(ψ, pref))
# 		println("Iter: $i, Elapsed time: $(time()-start)")
# 	end

# 	# Write all the stuff I need to files
# 	# Inputs to neural network
# 	open("feature_learning/NetworkFiles/inputs_delim_file.txt", "w") do io
# 		writedlm(io, [xa; xb])
# 	end
# 	# Reward weights (eventually will not use this)
# 	w_mean = [mean(W[i,:]) for i in 1:num_features]
# 	open("feature_learning/NetworkFiles/linear_reward_delim_file.txt", "w") do io
# 		writedlm(io, w_mean')
# 	end
# 	# User responses
# 	responses = [prefs[i].pref for i in 1:length(prefs)]
# 	open("feature_learning/NetworkFiles/user_responses_delim_file.txt", "w") do io
# 		writedlm(io, responses)
# 	end
# end

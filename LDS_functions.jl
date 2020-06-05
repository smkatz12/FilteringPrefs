
"""
----------------------------------------------
MCMC Sampling
----------------------------------------------
"""

function sample_w(M)
	w₀ = zeros(num_features)
	Σ = Matrix{Float64}(I, num_features, num_features) ./ 10000 # From batch active code
	w = AMMVariate(w₀, Σ, logf)
	for i = 1:(M*T+burn)
		sample!(w, adapt = (i <= burn))
		if i > burn && mod(i,T) == 0
			ind = convert(Int64, (i-burn)/T)
			W[:,ind] = w[:]
		end
	end
	# for i = 1:M
	# 	W[:,i] = W[:,i]./norm(W[:,i])
	# end
	push!(w_hist, copy(W))
end

function logf(w::DenseVector)
	if norm(w) > 1
	 	return -Inf
	elseif isempty(prefs)
		return 0.0
	else
		sum_over_prefs = 0
		for i = 1:length(prefs)
			sum_over_prefs += min.(prefs[i].pref*w'*prefs[i].ψ, 0)
		end
		return sum_over_prefs
	end
end

"""
----------------------------------------------
Particle Filter
----------------------------------------------
"""

function pf_update(particles, pref)
	# Simulate dynamics (just adding small amount of noise)
	for i = 1:num_particles
		for j = 1:num_features
			particles[i][j] += rand(particle_noise)
		end
	end
	# Update weights
	weights = zeros(num_particles)
	for i = 1:num_particles
		weights[i] = p_w_given_pref(particles[i], pref)
	end
	weights ./= sum(weights)
	# Resample
	dist = Categorical(weights)
	inds = rand(dist, num_particles)
	println(length(unique(inds)))
	old_particles = copy(particles)
	for i = 1:num_particles
		particles[i] = old_particles[inds[i]]
	end
	# Update W for rest of simulation
	for i = 1:num_particles
		W[:,i] = particles[i]
	end
	push!(w_hist, copy(W))
	return particles
end

function p_w_given_pref(w, pref)
	return 1/(1 + exp(-pref.pref*w'*pref.ψ))
end

# Sample uniformly in unit square and then reject until have 1000
function get_initial_particles()
	particles = []
	counter = 0
	dist = Uniform(-1, 1)
	while counter < num_particles
		sample = [rand(dist) for i = 1:num_features]
		if norm(sample) ≤ 1
			push!(particles, sample)
			counter += 1
		end
	end
	return particles
end

"""
----------------------------------------------
Discrete Filter
----------------------------------------------
"""

function discrete_update(pref)
	for i = 1:M
		probs[i] = p_w_given_pref(points[:,i], pref)*probs[i]
	end
	probs ./= sum(probs)
	push!(dist_hist, copy(probs))
end

function f_weighted(ψ::Vector)
	return (1 ./ (1 .+ exp.(-points'*ψ)))
end

function sample_hypersphere(num_points)
	dist = Normal(0,1)
	points = zeros(num_features, num_points)
	for i = 1:num_points
		point = [rand(dist) for i = 1:num_features]
		point ./= norm(point)
		points[:,i] = point
	end
	return points
end

"""
----------------------------------------------
Active Query Selection
----------------------------------------------
"""

function get_inputs(query_type; discrete=false)
	initvals = rand.(Distributions.Uniform.(lb,ub))
	if query_type == :random
		u = initvals
	elseif query_type == :knn
		point = bin_samples(W, n_bins)
		idx, _ = knn(kdtree, point, 1, true)
		u = u_binned[idx[1], :]
	else
		inner_optimizer = LBFGS(linesearch = LineSearches.BackTracking())
		objec = query_type == :info_gain ? objec1 : objec2
		if discrete
			objec = objec3
		end
		res = Optim.optimize(objec, lb, ub, initvals, Fminbox(inner_optimizer), Optim.Options(show_trace=false), autodiff=:forward)
		u = Optim.minimizer(res)
	end
	return u
end

function objec1(u::Vector)
	_, _, ψ = get_ψ(u)
	pq1 = f(ψ)
	pq2 = f(-ψ)
	return -(sum(pq1 .* log2.(M .* pq1 ./ sum(pq1))) + sum(pq2 .* log2.(M .* pq2 ./ sum(pq2))))
end

function objec2(u::Vector)
	_, _, ψ = get_ψ(u)
	return -min(sum(1 .- f(ψ)), sum(1 .- f(-ψ)))
end

function objec3(u::Vector)
	_, _, ψ = get_ψ(u)
	pq1 = f_weighted(ψ)
	pq2 = f_weighted(-ψ)
	ws_pq1 = sum([probs[i]*pq1[i] for i = 1:M])
	ws_pq2 = sum([probs[i]*pq2[i] for i = 1:M])
	sum1 = sum([probs[i]*(pq1[i] .* log2.(M .* pq1[i] ./ ws_pq1)) for i = 1:M])
	sum2 = sum([probs[i]*(pq2[i] .* log2.(M .* pq2[i] ./ ws_pq2)) for i = 1:M])
	return -(sum1 + sum2)
end

# function objec3(u::Vector)
# 	_, _, ψ = get_ψ(u)
# 	f1 = f_weighted(ψ)
# 	f2 = f_weighted(-ψ)
# 	sum1 = sum([probs[i]*f1[i] for i = 1:M])
# 	sum2 = sum([probs[i]*f2[i] for i = 1:M])
# 	return -min(sum1, sum2)
# end

function f(ψ::Vector)
	return 1 ./ (1 .+ exp.(-W'*ψ))
end

function get_ψ(u_tot::Vector)
	x₁, ϕ₁ = get_ϕ(u_tot[1:num_steps*ctrl_size])
	x₂, ϕ₂ = get_ϕ(u_tot[num_steps*ctrl_size+1:end])
	ψ = ϕ₁ - ϕ₂
	return x₁, x₂, ψ
end

function get_ϕ(u::Vector)
	x = get_x_mat(u)
	return x, ϕ(x)
end

function ϕ(x::Array)
	return [3mean(x[2,:]), 3mean(x[4,:]), 3mean(abs.(x[1,:] .- 1)), 3mean(abs.(x[3,:] .- 1))]
end

function get_x_mat(u::Vector)
	init_x = zeros(4,1)

	A = [1.0 1.0 0.0 0.0;
		 0.0 1.0 0.0 0.0;
		 0.0 0.0 1.0 0.5;
		 0.0 0.0 0.0 1.0]

	B = [0.0 0.0;
		 1.0 0.0;
		 0.0 0.0;
		 0.0 0.3]

	x = []
	ind = 0
	for i = 1:num_steps
		for j = 1:step_time
			ind += 1
			if ind == 1
				x = A*init_x + B*u[ctrl_size*(i-1)+1:ctrl_size*i]
			else
				x = hcat(x, A*x[:,end] + B*u[ctrl_size*(i-1)+1:ctrl_size*i])
			end
		end
	end

	return x
end

"""
----------------------------------------------
Other Functions
----------------------------------------------
"""

function sigmoid(z)
	return 1 / (1 + exp(-z))
end

function post_process_w_hist(w_hist::Vector, w_true::Vector)
	num_iter = length(w_hist)
	num_features = length(w_true)
	w_mean_hist = zeros(num_iter, num_features)
	m = zeros(num_iter)

	for i = 1:num_iter
		W = w_hist[i]
		w_mean = [mean(W[i,:]) for i in 1:num_features]
		w_mean /= norm(w_mean) # Normalize w_mean
		w_mean_hist[i,:] = w_mean
		m[i] = w_mean'*w_true/(norm(w_mean)*norm(w_true))
	end

	return m, w_mean_hist
end

function post_process_dist_hist(dist_hist::Vector, w_true::Vector)
	num_iter = length(dist_hist)
	num_features = length(w_true)
	m = zeros(num_iter)

	for i = 1:num_iter
		w_mean = points[:,argmax(dist_hist[i])]
		m[i] = w_mean'*w_true/(norm(w_mean)*norm(w_true))
	end

	return m
end
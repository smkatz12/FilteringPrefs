
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
	for i = 1:M
		W[:,i] = W[:,i]./norm(W[:,i])
	end
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
Active Query Selection
----------------------------------------------
"""

function get_inputs(query_type)
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
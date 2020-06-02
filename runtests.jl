using Random

num_tests = 1

w_mat = rand(Uniform(-1,1), (num_tests,num_features))
# Normalize
for i = 1:num_tests
	w_mat[i,:] ./= norm(w_mat[i,:])
end

##########################################################
# Random
##########################################################
println("Running Random...")
Random.seed!(129)

# Create matrix to store results
m_mat = zeros(30, num_tests)

start = time()
for i = 1:num_tests
	@eval begin
		w_hist = Vector{Array{Float64,2}}()
		W = zeros(num_features, M)
		prefs = Vector{Preference}()
	end
	println("Iter: $i;   w_true = $(w_mat[i,:])")
	auto_reward_iteration(w_mat[i,:], query_type = :random)
	m, _ = post_process_w_hist(w_hist, w_mat[i,:])
	m_mat[:,i] = m
end
println("Elapsed time: $(time() - start)")

m_mean_random = [mean(m_mat[i,:]) for i=1:size(m_mat,1)]

##########################################################
# Volume Removal
##########################################################
println("Running Volume Removal...")
Random.seed!(129)

# Create matrix to store results
m_mat = zeros(30, num_tests)

start = time()
for i = 1:num_tests
	@eval begin
		w_hist = Vector{Array{Float64,2}}()
		W = zeros(num_features, M)
		prefs = Vector{Preference}()
	end
	println("Iter: $i;   w_true = $(w_mat[i,:])")
	auto_reward_iteration(w_mat[i,:], query_type = :vol_rem)
	m, _ = post_process_w_hist(w_hist, w_mat[i,:])
	m_mat[:,i] = m
end
println("Elapsed time: $(time() - start)")

m_mean_vol_rem = [mean(m_mat[i,:]) for i=1:size(m_mat,1)]

##########################################################
# Information Gain
##########################################################
println("Running Information Gain...")
Random.seed!(129)

# Create matrix to store results
m_mat = zeros(30, num_tests)

start = time()
for i = 1:num_tests
	@eval begin
		w_hist = Vector{Array{Float64,2}}()
		W = zeros(num_features, M)
		prefs = Vector{Preference}()
	end
	println("Iter: $i;   w_true = $(w_mat[i,:])")
	auto_reward_iteration(w_mat[i,:], query_type = :info_gain)
	m, _ = post_process_w_hist(w_hist, w_mat[i,:])
	m_mat[:,i] = m
end
println("Elapsed time: $(time() - start)")

m_mean_info_gain = [mean(m_mat[i,:]) for i=1:size(m_mat,1)]
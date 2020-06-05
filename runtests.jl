using Random

num_tests = 5

w_mat = rand(Uniform(-1,1), (num_tests,num_features))
# Normalize
for i = 1:num_tests
	w_mat[i,:] ./= norm(w_mat[i,:])
end

##########################################################
# Information Gain
##########################################################
# println("Running Information Gain...")
# Random.seed!(129)

# # Create matrix to store results
# m_mat = zeros(30, num_tests)

# start = time()
# for i = 1:num_tests
# 	@eval begin
# 		w_hist = Vector{Array{Float64,2}}()
# 		W = zeros(num_features, M)
# 		prefs = Vector{Preference}()
# 	end
# 	println("Iter: $i;   w_true = $(w_mat[i,:])")
# 	auto_reward_iteration_df(w_mat[i,:], query_type = :info_gain)
# 	m, _ = post_process_w_hist(w_hist, w_mat[i,:])
# 	m_mat[:,i] = m
# end
# println("Elapsed time: $(time() - start)")

# m_mean_info_gain = [mean(m_mat[i,:]) for i=1:size(m_mat,1)]

println("Running Discrete Information Gain...")
Random.seed!(13)

# Create matrix to store results
num_iter = 30
m_mat = zeros(num_iter, num_tests)

start = time()
for i = 1:num_tests
	@eval begin
		probs = (1/M)*ones(M)
		weighted_points = vcat([probs' for i = 1:4]...).*points
		dist_hist = []
		W = zeros(num_features, M)
		prefs = Vector{Preference}()
	end
	println("Iter: $i;   w_true = $(w_mat[i,:])")
	auto_reward_iteration_df(w_mat[i,:], num_iter, query_type = :info_gain)
	m = post_process_dist_hist(dist_hist, w_mat[i,:])
	m_mat[:,i] = m
end
println("Elapsed time: $(time() - start)")

m_mean_info_gain = [mean(m_mat[i,:]) for i=1:size(m_mat,1)]
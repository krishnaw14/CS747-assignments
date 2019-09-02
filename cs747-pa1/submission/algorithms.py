import numpy as np 

def round_robin(num_arms, emperical_means, num_pulls, t, epsilon):
	arm = t%num_arms
	return arm

def epsilon_greedy(num_arms, emperical_means, num_pulls, t, epsilon):
	choice = np.random.choice([0,1], p=[epsilon, 1-epsilon])
	if choice == 0:
		arm = np.random.randint(0, num_arms)
	else:
		arm = np.argmax(emperical_means)
	return arm

def ucb(num_arms, emperical_means, num_pulls, t, epsilon):
	if t < num_arms:
		arm = t
	else:
		ucb_values = emperical_means + np.sqrt(2*np.log(t)/num_pulls)
		arm = np.argmax(ucb_values)
	return arm

def kl_ucb(num_arms, emperical_means, num_pulls, t, epsilon):
	if t < num_arms:
		arm = t
	else:
		def get_kld(p, q):
			if p == 1 or p == 0 or q == 0 or q == 1:
				return 0
			kld = p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
			return kld

		def solve_for_q(p, rhs_value):
			# Solved via Bisection method
			q_l = p
			q_r = 1
			ERROR_PRECISION = 1e-5
			q = 0.5*(q_l+q_r)
			kld = get_kld(p,q)
			while((abs(rhs_value-kld) > ERROR_PRECISION) and q_r - q_l > ERROR_PRECISION):
				
				if kld > rhs_value:
					q_r = q
				else:
					q_l = q
				q = 0.5*(q_l+q_r)
				kld = get_kld(p,q)
			return q

		q = np.zeros(emperical_means.shape)
		for i in range(num_arms):
			rhs_value = (np.log(t) + 3*np.log(np.log(t)))/num_pulls[i]
			q[i] = solve_for_q(emperical_means[i], rhs_value)
		arm = np.argmax(q)
	return arm

def thompson_sampling(num_arms, emperical_means, num_pulls, t, epsilon):
	num_successes = np.int32(emperical_means*num_pulls)
	num_failures = num_pulls - num_successes
	x_values = np.zeros(emperical_means.shape)
	for i in range(num_arms):
		x_values[i] = np.random.beta(num_successes[i]+1, num_failures[i]+1)
	arm = np.argmax(x_values)
	return arm

ALGORITHM_DICT = {
	'round-robin': round_robin,
	'epsilon-greedy': epsilon_greedy,
	'ucb': ucb,
	'kl-ucb': kl_ucb,
	'thompson-sampling': thompson_sampling
	}

def get_algorithm_by_name(algorithm_name):
	assert algorithm_name in ALGORITHM_DICT.keys(), "Algorithm choices: {}".format(list(ALGORITHM_DICT.keys()))
	return ALGORITHM_DICT[algorithm_name]

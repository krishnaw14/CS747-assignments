import numpy as np 
from algorithms import get_algorithm_by_name

class Agent():

	def __init__(self, instance_path, algorithm_name, random_seed, 
		epsilon, horizon):

		instances = np.loadtxt(instance_path)
		self.num_arms = len(instances)
		self.bernoulli_arm_means = instances
		self.algorithm = get_algorithm_by_name(algorithm_name)
		self.random_seed = random_seed
		self.epsilon = epsilon
		self.horizon = horizon

		np.random.seed(self.random_seed)

	def pull_arm(self, arm_index):
		reward = np.random.binomial(n=1, p=self.bernoulli_arm_means[arm_index])
		return reward

	def calculate_regret(self):
		expected_cummulated_reward = self.solve_expected_cumulative_reward()
		optimal_arm_bernoulli_mean = np.max(self.bernoulli_arm_means)
		regret = optimal_arm_bernoulli_mean*self.horizon - expected_cummulated_reward
		return regret

	def solve_expected_cumulative_reward(self):
		num_pulls = np.zeros_like(self.bernoulli_arm_means)
		num_successfull_pulls = np.zeros_like(self.bernoulli_arm_means)
		emperical_means = np.zeros_like(self.bernoulli_arm_means)
		cumulative_reward = 0

		for t in range(self.horizon):
			arm_index = self.algorithm(self.num_arms, emperical_means, num_pulls, t, 
				self.epsilon)
			reward = self.pull_arm(arm_index)
			num_pulls[arm_index] += 1
			num_successfull_pulls[arm_index] += reward
			emperical_means[arm_index] = num_successfull_pulls[arm_index]/num_pulls[arm_index]
			cumulative_reward += reward

		# import pdb; pdb.set_trace();
			

		return cumulative_reward


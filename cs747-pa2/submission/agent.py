from __future__ import print_function
import numpy as np 
from pulp import *

from mdp import MDP
from algorithms import linear_programming, howard_policy_iteration

class Agent():

	def __init__(self, txt_file_path, algorithm_name):
		self.mdp = MDP(txt_file_path)
		self.algorithm = self.get_algorithm_by_name(algorithm_name)

	def get_algorithm_by_name(self, algorithm_name):
		assert algorithm_name in ['lp', 'hpi'], "Algorith Choices: [lp, hpi]"
		if algorithm_name == 'lp':
			return linear_programming
		else:
			return howard_policy_iteration

	def construct_policy(self):
		value_function, policy = self.algorithm(self.mdp.num_states, self.mdp.num_actions, self.mdp.gamma, 
			self.mdp.transition_function, self.mdp.reward_function)

		for state in range(self.mdp.num_states):
			print(value_function[state], policy[state])


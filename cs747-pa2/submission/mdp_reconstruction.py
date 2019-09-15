import numpy as np 

from mdp import MDP
from algorithms import howard_policy_iteration, linear_programming, evaluate_policy, evaluate_action_value_function

class MDPreconstruction():

	def __init__(self, num_states, num_actions=2):
		self.num_states = num_states
		self.num_actions = num_actions
		# self.gamma = gamma
		self.transition_function = self.get_transition_function()
		self.reward_function = self.get_reward_function()

		self.value_functions_0 = []
		self.value_functions_1 = []

		self.p1, self.p2, self.p3 = True, True, True

		# alpha = np.zeros((self.num_states,))
		# for i in range(self.num_states):
		# 	alpha[i] = np.sum(self.reward_function[i][0][(i+1)%self.num_states]*self.transition_function[i][0][(i+1)%self.num_states])
		# import pdb; pdb.set_trace()


	def get_transition_function(self):
		transition_function = np.zeros((self.num_states, self.num_actions, self.num_states))
		for state in range(self.num_states):
			for action in range(self.num_actions):
				if action == 0:
					target_state = (state+1)%self.num_states
					transition_function[state][action][target_state] = 1
				elif action == 1:
					transition_function[state][action][state] = 0.3
					target_state = (state-1)%self.num_states
					transition_function[state][action][target_state] = 0.7
				
		return transition_function

	def get_reward_function(self):
		reward_function = np.zeros((self.num_states, self.num_actions, self.num_states))
		for state in range(self.num_states):
			for action in range(self.num_actions):
				if action == 0:
					reward_function[state][action][(state+1)%self.num_states] = 8
					# reward_function[state][action][(state-1)%self.num_states] = -5
				elif action == 1:
					reward_function[state][action][state] = -8
					reward_function[state][action][(state-1)%self.num_states] = 7/0.7
					# reward_function[state][action][state] = 5
		reward_function[-1][0][0] = -1.4
		reward_function[-1][1][-1] = -1.4
		reward_function[-1][1][-2] = -1.4

		reward_function[1][0][2] = -3.9
		reward_function[1][1][1] = -3.9
		reward_function[1][1][0] = -3.9
		return reward_function

	def construct_policy(self, gamma):
		value_function, policy = linear_programming(self.num_states, self.num_actions, gamma, 
			self.transition_function, self.reward_function)
		action_value_function = evaluate_action_value_function(value_function, 4, 2, gamma, 
			self.transition_function, self.reward_function)

		# print("PIVOT GAMMA: ", gamma)

		for i in range(self.num_states):
			print(value_function[i], policy[i])
		# import pdb; pdb.set_trace()
		# if np.count_nonzero(policy) == 0:
		# 	if self.p1:
		# 		print("PIVOT GAMMA: ", gamma, np.count_nonzero(policy))
		# 		self.p1 = False
		# elif np.count_nonzero(policy) == 1:
		# 	if self.p2:
		# 		print("PIVOT GAMMA: ", gamma, np.count_nonzero(policy))
		# 		self.p2 = False
		# elif np.count_nonzero(policy) == 2:
		# 	if self.p3:
		# 		print("PIVOT GAMMA: ", gamma, np.count_nonzero(policy))
		# 		self.p3 = False
		# import pdb; pdb.set_trace()

	def plot_value_function(self, gamma):
		value_function_0 = evaluate_policy(np.zeros((4,), dtype=np.int), self.num_states, self.num_actions, gamma, 
			self.transition_function, self.reward_function)
		self.value_functions_0.append(value_function_0[0])
		value_function_1 = evaluate_policy(np.ones((4,), dtype=np.int), self.num_states, self.num_actions, gamma, 
			self.transition_function, self.reward_function)
		self.value_functions_1.append(value_function_1[0])

		# import pdb; pdb.set_trace()

	def write_mdp(self):

		file = open('mdp-family.txt', "w")
		file.write(str(self.num_states) + '\n')
		file.write(str(self.num_states) + '\n')
		for state in range(self.num_states):
			for action in range(self.num_actions):
				for target_state in range(self.num_states):
					file.write(str(self.reward_function[state][action][target_state]) + "\t")
				file.write('\n')
		for state in range(self.num_states):
			for action in range(self.num_actions):
				for target_state in range(self.num_states):
					file.write(str(self.transition_function[state][action][target_state]) + "\t")
				file.write('\n')
		file.write(str(0.5)+'\n')
		file.write('continuing')
		file.close()

if __name__ == '__main__':
	gamma_values = np.linspace(0.01,0.99,100)
	mdp_recon = MDPreconstruction(6)
	# for gamma in gamma_values:
	# 	print("\n-------------------------gamma = {}---------------------------\n".format(gamma))
	# 	# mdp_recon.plot_value_function(gamma)
	# 	mdp_recon.construct_policy(gamma)
	# 	print("\n----------------------------------------------------\n")

	mdp_recon.write_mdp()



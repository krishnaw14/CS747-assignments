from __future__ import print_function
import numpy as np 
from algorithms import linear_programming

class Solver:

	def __init__(self, data_path, target_path=None):

		with open(data_path) as f:
			data = f.readlines()

		self.num_states = int(data[0])
		self.num_actions = int(data[1])
		self.gamma = float(data[2])

		self.trajectory = np.array([np.float32(d.split()) for d in data[3:-1]])
		self.last_state = int(data[-1])

		self.gamma_list = np.array([self.gamma**i for i in range(len(self.trajectory))])

		if target_path is not None and target_path != data_path:
			self.target = np.loadtxt(target_path)

	def model_based_learning(self):
		# DOES NOT WORK AS ALL STATE-ACTION PAIRS ARE NOT ENCOUNTERED IN THE GIVEN TRAJECTORY.

		total_transitions = np.zeros((self.num_states, self.num_actions, self.num_states))
		total_rewards = np.zeros((self.num_states, self.num_actions, self.num_states))
		total_visits = np.zeros((self.num_states, self.num_actions))

		for i in range(len(self.trajectory[:-1])):
			state, action, reward = self.trajectory[i]
			state, action = int(state), int(action)
			target_state = int(self.trajectory[i+1][0])

			total_visits[state][action] += 1
			total_transitions[state][action][target_state] += 1
			total_rewards[state][action][target_state] += 1

		transition_function = np.divide(total_transitions, np.expand_dims(total_visits,-1), 
			out=np.zeros_like(total_transitions), where=np.expand_dims(total_visits,-1)!=0)
		reward_function = np.divide(total_rewards, total_transitions, 
			out=np.zeros_like(total_transitions), where=total_transitions!=0)

		optimal_value_function, optimal_policy = linear_programming(self.num_states, self.num_actions, self.gamma, 
			transition_function, reward_function)

		print(optimal_value_function)

	def monte_carlo_learning(self):
		# USED FOR PREDICTION
		value_functions = np.zeros((self.num_states,1))

		# first_occurance = []
		for state in range(self.num_states):
			indices = np.where(self.trajectory[:,0] == state)[0]

			reward = 0.0
			for start_idx in indices:
				reward += np.sum([self.gamma_list[0:len(self.trajectory)-start_idx]*self.trajectory[start_idx:,-1] ] )
				# break 

			value_functions[state] = reward/len(indices)

		for value_function in value_functions.squeeze():
			print(value_function)

		# print('Answer:', value_functions.squeeze())
		# print('Expected:', self.target)

		# error = np.sum( (value_functions.squeeze()-self.target)**2 )
		# print("Error: ", error)

	def td_learning(self, alpha=0.5):

		value_functions = np.zeros((self.num_states,1))
		for i in range(len(self.trajectory)):
			state = int(self.trajectory[i][0])
			target_state = int(self.last_state) if i == len(self.trajectory)-1 else int(self.trajectory[i+1][0])
			reward = self.trajectory[i][-1]

			value_functions[state] += alpha*(reward+self.gamma*value_functions[target_state] - value_functions[state])

		print('Answer:', value_functions)
		print('Expected:', self.target)

		error = np.sum( (value_functions.squeeze()-self.target)**2 )
		print("Error: ", error)
		return error

	def td_lambda(self, lambda_=0.95, alpha=0.1):
		# TD-lambda learning with eligibility trace

		value_functions = np.zeros((self.num_states,1))
		eligibility = np.zeros((self.num_states,1))

		total_time = len(self.trajectory)

		for t in range(total_time):
			state = int(self.trajectory[t][0])
			target_state = int(self.last_state) if t == len(self.trajectory)-1 else int(self.trajectory[t+1][0])
			reward = self.trajectory[t][-1]
			eligibility[state] += 1.0
			value_functions += alpha*eligibility*(reward+self.gamma*value_functions[target_state] - value_functions[state])
			eligibility *= lambda_ * self.gamma 

		# error = np.sum( (value_functions.squeeze()-self.target)**2 )

		# return error






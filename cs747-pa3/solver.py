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

		if target_path is not None:
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
		value_functions = np.zeros((self.num_states,1))
		total_visits = np.zeros((self.num_states,1))

		

		# first_occurance = []
		for state in range(self.num_states):
			indices = np.where(self.trajectory[:,0] == state)[0]

			reward = 0.0
			for start_idx in indices:
				reward += np.sum([self.gamma_list[0:len(self.trajectory)-start_idx]*self.trajectory[start_idx:,-1] ] ) 

			value_functions[state] = reward/len(indices)

		print('Answer:', value_functions)
		print('Expected:', self.target)

		# import pdb; pdb.set_trace()

		error = np.sum( (value_functions.squeeze()-self.target)**2 )
		print("Error: ", error)






import numpy as np 
from algorithms import linear_programming

class Solver:

	def __init__(self, data_path):

		with open(data_path) as f:
			data = f.readlines()

		self.num_states = int(data[0])
		self.num_actions = int(data[1])
		self.gamma = float(data[2])

		self.trajectory = np.array([np.float32(d.split()) for d in data[3:]])

		# import pdb; pdb.set_trace()

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

		for sample in self.trajectory:
			if len(sample) == 1:
				total_visits[int(sample[0])]+=1
			else:
				total_visits[int(sample[0])]+=1
				value_functions[int(sample[0])] += float(sample[-1])

		import pdb; pdb.set_trace()

		value_functions /= total_visits
		
		print(value_functions)





import numpy as np 

class MDP():

	def __init__(self, txt_file_path):
		self.load_mdp_from_txt_file(txt_file_path)

	def load_mdp_from_txt_file(self, txt_file_path):
		# Load File
		with open(txt_file_path) as f:
			mdp_content = f.readlines()
		self.num_states = int(mdp_content[0])
		self.num_actions = int(mdp_content[1])
		self.type = mdp_content[-1].split()[0]
		self.gamma = float(mdp_content[-2])  #if self.type == 'continuing' else 1
		
		self.done = False # To denote termination
		self.transition_function = np.zeros((self.num_states, self.num_actions, self.num_states))
		self.reward_function = np.zeros((self.num_states, self.num_actions, self.num_states))

		line_counter = 0
		for state in range(self.num_states):
			for action in range(self.num_actions):
				reward_line = mdp_content[2+line_counter]
				self.reward_function[state][action] = np.float32(reward_line.split())
				transition_line = mdp_content[2+self.num_states*self.num_actions+line_counter]
				self.transition_function[state][action] = np.float32(transition_line.split())
				line_counter += 1

	def get_reward(self, state, action, target_state):
		reward =  self.reward_function[state][action][target_state]
		return reward

	def step(self, state, action):
		target_state = np.random.choice(self.num_states, p = self.transition_function[state][action])
		reward =  self.reward_function[state][action][target_state]
		if self.type == 'episodic' and target_state == self.S-1:
			self.done = True
		return target_state, reward






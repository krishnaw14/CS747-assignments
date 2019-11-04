import numpy as np 
import os 

class WindyGridWorld(object):

	def __init__(self):
		self.num_rows = 7
		self.num_columns = 10
		self.num_actions = 4 # 0: left, 1: right, 2: up, 3: down

		self.wind_strength = [0,0,0,1,1,1,2,2,1,0] 

		self.initial_state = [3, 0]
		self.terminal_state = [3, 7]

		self.reset()

	def get_next_state(self, action):
		# Apply wind
		self.state[0] = min(self.state[0] + self.wind_strength[self.state[1]], self.num_rows-1)

		if action == 0: # left
			self.state[1] = max(0, self.state[1]-1)
		elif action == 1: # right
			self.state[1] = min(self.state[1] + 1, self.num_columns-1)
		elif action == 2: # up
			self.state[0] = min(self.state[0] + 1, self.num_rows-1)
		elif action == 3: # down
			self.state[0] = max(0, self.state[0]-1)

	def get_reward(self):
		if self.state == self.terminal_state:
			self.done = True
			self.reward = 0

	def step(self, action):
		assert action >=0 and action <= 3, "Invalid Action - only 4 actions (0-3) are allowed"

		self.get_next_state(action)
		self.get_reward()

		return self.state, self.reward, self.done

	def reset(self):
		self.state = self.initial_state
		self.reward = -1
		self.done = False



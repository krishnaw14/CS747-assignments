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

	def get_next_state(self, state, action):

		if action == 0: # left
			new_state = [max(state[0] - self.wind_strength[state[1]], 0), max(0, state[1]-1)]
		elif action == 1: # right
			new_state = [max(state[0] - self.wind_strength[state[1]], 0), min(state[1] + 1, self.num_columns-1)]
		elif action == 2: # up
			new_state = [max(0, state[0]-1-self.wind_strength[state[1]]), state[1]]
		elif action == 3: # down
			new_state = [max(min(state[0] + 1 - self.wind_strength[state[1]], self.num_rows-1),0), state[1]]

		return new_state

	def get_reward(self, state):
		if state == self.terminal_state:
			return 0
		else: 
			return -1

	def step(self, state, action):
		assert action >=0 and action <= 3, "Invalid Action - only 4 actions (0-3) are allowed"
		return self.get_next_state(state, action), self.get_reward(state), (state == self.terminal_state)

	def reset(self):
		return self.initial_state.copy()


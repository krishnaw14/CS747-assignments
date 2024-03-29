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
		assert action >=0 and action < self.num_actions, "Invalid Action - only 4 actions (0-3) are allowed"
		return self.get_next_state(state, action), self.get_reward(state), (state == self.terminal_state)

	def reset(self):
		return self.initial_state.copy()

class WindyGridWorldwithKingMoves(WindyGridWorld):
	def __init__(self):
		super(WindyGridWorldwithKingMoves, self).__init__()
		self.num_actions = 8 

	def get_next_state(self, state, action):
		if action == 0: # left
			new_state = [max(state[0] - self.wind_strength[state[1]], 0), max(0, state[1]-1)]
		elif action == 1: # right
			new_state = [max(state[0] - self.wind_strength[state[1]], 0), min(state[1] + 1, self.num_columns-1)]
		elif action == 2: # up
			new_state = [max(0, state[0]-1-self.wind_strength[state[1]]), state[1]]
		elif action == 3: # down
			new_state = [max(min(state[0] + 1 - self.wind_strength[state[1]], self.num_rows-1),0), state[1]]
		elif action == 4: # left-up
			new_state = [max(state[0]-1-self.wind_strength[state[1]], 0), max(0, state[1]-1)]
		elif action == 5: # left-down
			new_state = [max(min(state[0]+1-self.wind_strength[state[1]],self.num_rows-1), 0), max(0, state[1]-1)]
		elif action == 6: # right-up
			new_state = [max(state[0]-1-self.wind_strength[state[1]], 0), min(state[1] + 1, self.num_columns-1)]
		elif action == 7: # right-down
			new_state = [max(min(state[0]+1-self.wind_strength[state[1]],self.num_rows-1), 0), min(state[1] + 1, self.num_columns-1)]

		return new_state

class StochasticWindyGridWorld(WindyGridWorldwithKingMoves):
	def __init__(self):
		super(StochasticWindyGridWorld, self).__init__()

	def get_next_state(self, state, action):

		stochastic_param = np.random.choice(3)
		if stochastic_param == 0:
			var = 0
		elif stochastic_param == 1:
			var = 1
		elif stochastic_param == 2:
			var = -1

		if action == 0: # left
			new_state = [min(max(state[0] - self.wind_strength[state[1]]+var, 0), self.num_rows-1), max(0, state[1]-1)]
		elif action == 1: # right
			new_state = [min(max(state[0] - self.wind_strength[state[1]]+var, 0), self.num_rows-1), min(state[1] + 1, self.num_columns-1)]
		elif action == 2: # up
			new_state = [min(max(0, state[0]-1-self.wind_strength[state[1]]+var), self.num_rows-1), state[1]]
		elif action == 3: # down
			new_state = [max(min(state[0] + 1 - self.wind_strength[state[1]]+var, self.num_rows-1),0), state[1]]
		elif action == 4: # left-up
			new_state = [min(max(state[0]-1-self.wind_strength[state[1]]+var, 0), self.num_rows-1), max(0, state[1]-1)]
		elif action == 5: # left-down
			new_state = [max(min(state[0]+1-self.wind_strength[state[1]]+var,self.num_rows-1), 0), max(0, state[1]-1)]
		elif action == 6: # right-up
			new_state = [min(max(state[0]-1-self.wind_strength[state[1]]+var, 0), self.num_rows-1), min(state[1] + 1, self.num_columns-1)]
		elif action == 7: # right-down
			new_state = [max(min(state[0]+1-self.wind_strength[state[1]]+var,self.num_rows-1), 0), min(state[1] + 1, self.num_columns-1)]

		return new_state


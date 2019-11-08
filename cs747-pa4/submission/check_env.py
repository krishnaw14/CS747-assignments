from __future__ import print_function
import numpy as np 

from environment import WindyGridWorld

env = WindyGridWorld()

print('Initial State: {}, Terminal State: {}'.format(env.state, env.terminal_state))

while not env.done:
	action = int(input())
	next_state, reward, done = env.step(action)
	print('Next State: {}, Reward: {}'.format(next_state, reward))

print('Game Over!')

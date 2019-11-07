import numpy as np 
import matplotlib.pyplot as plt
from environment import WindyGridWorld

class SarsaAgent():

	def __init__(self, alpha, epsilon, save_plot_path):
		self.env = WindyGridWorld()

		self.alpha = alpha
		self.epsilon = epsilon

		self.Q_values = np.zeros((self.env.num_rows, self.env.num_columns, self.env.num_actions))

		self.save_plot_path = save_plot_path

	def get_action(self, state):
		choice = np.random.choice([0,1], p=[self.epsilon, 1-self.epsilon])
		if choice == 0:
			action = np.random.randint(0, self.env.num_actions)
		else:
			q_value = self.Q_values[state[0],state[1]]
			action = np.random.choice(np.argwhere(q_value == np.amax(q_value)).flatten())
		return action

	def play_episode(self):

		time_steps = 0
		state = self.env.reset()
		action = self.get_action(state)

		while state != self.env.terminal_state:
			
			next_state, reward, done = self.env.step(state, action)
			next_action = self.get_action(next_state)

			self.Q_values[state[0]][state[1]][action] += self.alpha*(reward + 
				self.Q_values[next_state[0]][next_state[1]][next_action] - self.Q_values[state[0]][state[1]][action])

			state = next_state
			action = next_action

			time_steps += 1

		return time_steps

	def learn(self):	

		time_step_values, episode_values = [], []
		num_episodes = 180
		time_steps = 0

		for i in range(num_episodes):
			time_step = self.play_episode()

			time_steps += time_step
			time_step_values.append(time_steps)
			episode_values.append(i)

		plt.plot(time_step_values, episode_values)
		plt.xlabel('Time Steps')
		plt.ylabel('Episodes')
		plt.title('Episodes vs Time Steps')
		plt.savefig(self.save_plot_path)
		plt.clf()

		
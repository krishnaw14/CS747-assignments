import numpy as np 
import matplotlib.pyplot as plt
from environment import WindyGridWorld, WindyGridWorldwithKingMoves, StochasticWindyGridWorld

class SarsaAgent():

	def __init__(self, env, alpha, epsilon, save_plot_path):
		self.env = env

		self.alpha = alpha
		self.epsilon_initial = epsilon
		self.epsilon  = epsilon

		self.Q_values = np.zeros((self.env.num_rows, self.env.num_columns, self.env.num_actions))

		self.save_plot_path = save_plot_path
		# import pdb; pdb.set_trace()

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

	def learn(self, num_seed_runs):	

		num_episodes = 180
		final_time_step_values, final_episode_values = np.zeros((num_episodes,)), np.zeros((num_episodes,))
		minimum_step_values = []

		for seed in range(num_seed_runs):
			np.random.seed(seed)
			time_step_values, episode_values,  = [], []
			time_steps = 0
			self.Q_values = np.zeros((self.env.num_rows, self.env.num_columns, self.env.num_actions))
			self.epsilon = self.epsilon_initial

			for i in range(num_episodes):
				time_step = self.play_episode()

				time_steps += time_step
				time_step_values.append(time_steps)
				episode_values.append(i)
				self.epsilon = self.epsilon_initial/(i+1)

			min_step = np.min(np.array(time_step_values[1:]) - np.array(time_step_values[0:-1]))
			minimum_step_values.append(min_step)
			final_time_step_values += np.array(time_step_values)
			final_episode_values += np.array(episode_values)

		min_step = np.mean(minimum_step_values)
		cumulative_step = final_time_step_values[-1]/num_seed_runs
		print('Average Minimum steps to reach the target over {} seed runs of {} episodes: {}'.format(num_seed_runs, 
			num_episodes, min_step))
		print('Average Cumulative steps to reach the target over {} seed runs of {} episodes: {}'.format(num_seed_runs, 
			num_episodes, cumulative_step))
		plt.plot(final_time_step_values/num_seed_runs, np.int32(final_episode_values/num_seed_runs), label=self.save_plot_path)

		plt.xlabel('Time Steps')
		plt.ylabel('Episodes')
		plt.title('Episodes vs Time Steps')
		# plt.savefig(self.save_plot_path)
		# plt.clf()

		
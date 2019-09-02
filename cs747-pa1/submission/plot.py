from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt 

HORIZON_VALUES = np.array([50, 200, 800, 3200, 12800, 51200, 204800])
ALGO_NAMES = ['round-robin', 'epsilon-greedy_0.002', 'epsilon-greedy_0.02', 'epsilon-greedy_0.2', 
'ucb', 'kl-ucb', 'thompson-sampling']

def initialize_dict():
	dict_ = {}
	for algo_name in ALGO_NAMES:
		dict_[algo_name] = {}
		for horizon in HORIZON_VALUES:
			dict_[algo_name][horizon] = []
	return dict_

def append_dict(dict_, line_list):
	algo_name = line_list[1].split()[0]
	if algo_name == 'epsilon-greedy':
		algo_name += ('_'+line_list[3].split()[0])
	horizon = int(line_list[-2].split()[0])
	regret = float(line_list[-1].split()[0])
	# import pdb; pdb.set_trace()
	dict_[algo_name][horizon].append(regret)

def plot_dict(dict_, title):
	plt.clf()
	for algo_name in dict_:
		x_values = []
		y_values = []
		for horizon in dict_[algo_name]:
			avg_regret = np.mean(dict_[algo_name][horizon])
			x_values.append(horizon)
			y_values.append(avg_regret)
		plt.plot(np.log(x_values), y_values, label=algo_name)
	plt.title(title)
	plt.xlabel('log(T)')
	plt.ylabel('regret')
	plt.legend()
	plt.savefig(title+'.png')


if __name__ == '__main__':

	filename = 'outputData.txt'
	with open(filename) as f:
		content = f.readlines()

	instance_1_dict = initialize_dict()
	instance_2_dict = initialize_dict()
	instance_3_dict = initialize_dict()

	for line in content:
		line_list = line.split(',')
		if line_list[0] == '../instances/i-1.txt':
			append_dict(instance_1_dict, line_list)
		elif line_list[0] == '../instances/i-2.txt':
			append_dict(instance_2_dict, line_list)
		elif line_list[0] == '../instances/i-3.txt':
			append_dict(instance_3_dict, line_list)
		else:
			print('Error in outputData.txt')
			exit(1)

	plot_dict(instance_1_dict, 'Instance_1_plot')
	plot_dict(instance_2_dict, 'Instance_2_plot')
	plot_dict(instance_3_dict, 'Instance_3_plot')





			
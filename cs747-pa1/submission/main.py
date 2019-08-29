import numpy as np
import argparse
from agent import Agent

def get_parser():
	parser = argparse.ArgumentParser(description='Multi-Arm Bandit Problem')
	parser.add_argument('--instance', type=str, required=True,
		help='Path to emperical means of different arms')
	parser.add_argument('--algorithm', type=str, required=True,
		help='Arm Sampling Algorithm')
	parser.add_argument('--randomSeed', type=int, required=True,
		help='Seed for benchmarking and reproduction')
	parser.add_argument('--horizon', type=int, required=True,
		help='Total Time Period of sampling')
	parser.add_argument('--epsilon', type=float, required=True,
		help='Epsilon Value for epsilon-greedy algorithm') 
	return parser

if __name__ == '__main__':
	parser = get_parser()
	args = parser.parse_args()
	print('Hello World')
	instance_file = parser.instance
	algorithm = parser.algorithm
	random_seed = parser.randomSeed
	epsilon = parser.epsilon
	horizon = parser.horizon

	np.random.seed(random_seed)
	agent = Agent(instance_file, algorithm, random_seed, epsilon, horizon)

	print(agent.calculate_regret())
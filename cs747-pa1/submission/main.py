from __future__ import print_function
import numpy as np
import argparse
import sys
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
	instance_file = args.instance
	algorithm = args.algorithm
	random_seed = args.randomSeed
	epsilon = args.epsilon
	horizon = args.horizon

	np.random.seed(random_seed)
	agent = Agent(instance_file, algorithm, random_seed, epsilon, horizon)

	expected_regret = agent.calculate_regret()

	print("{}, {}, {}, {}, {}, {}".format(instance_file, algorithm, random_seed, epsilon, horizon, expected_regret))
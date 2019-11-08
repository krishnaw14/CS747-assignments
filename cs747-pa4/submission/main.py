from __future__ import print_function
from agent import SarsaAgent
from environment import WindyGridWorld, WindyGridWorldwithKingMoves, StochasticWindyGridWorld

import numpy as np 
import argparse
import matplotlib.pyplot as plt
import os


def get_args():
	parser = argparse.ArgumentParser(description='Markov Decision Problem')
	parser.add_argument('--baseline', action='store_true', default=False,
		help='Solving for BaseLine WindyGridWorld Environment')
	parser.add_argument('--kingmoves', action='store_true', default=False,
		help='Solving for WindyGridWorld Environment with King Moves')
	parser.add_argument('--stochastic', action='store_true', default=False,
		help='Solving for Stochastic WindyGridWorld Environment with King Moves')
	parser.add_argument('--num_seed_runs', type=int, default=10, 
		help='Number of seeds to average the learnt values')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	os.makedirs('plots', exist_ok=True)

	args = get_args()

	if not args.baseline and not args.kingmoves and not args.stochastic:
		print('Please pass the correct argument flag')

	if args.baseline:
		print('----------------- BaseLine -----------------')
		env = WindyGridWorld()
		agent = SarsaAgent(env, alpha=0.5, epsilon=0.1, save_plot_path='plots/baseline.png')
		agent.learn(num_seed_runs = args.num_seed_runs)

	if args.kingmoves:
		print('----------------- KingMoves -----------------')
		env = WindyGridWorldwithKingMoves()
		agent = SarsaAgent(env, alpha=0.5, epsilon=0.1, save_plot_path='plots/kingmoves.png')
		agent.learn(num_seed_runs = args.num_seed_runs)

	if args.stochastic:
		print('----------------- Stochastic with KingMoves -----------------')
		env = StochasticWindyGridWorld()
		agent = SarsaAgent(env, alpha=0.5, epsilon=0.1, save_plot_path='plots/stochastic.png')
		agent.learn(num_seed_runs = args.num_seed_runs)


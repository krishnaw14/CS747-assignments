import argparse
import numpy as np
from agent import Agent
from mdp import MDP

def get_parser():
	parser = argparse.ArgumentParser(description='Markov Decision Problem')
	parser.add_argument('--mdp', type=str, required=True,
		help='Path to mdp specifcations file in txt format')
	parser.add_argument('--algorithm', type=str, required=True,
		help='Algorithm')
	return parser

if __name__ == '__main__':
	parser = get_parser()
	args = parser.parse_args()

	agent = Agent(args.mdp, args.algorithm)
	agent.construct_policy()

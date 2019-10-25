import sys
from solver import Solver

if __name__ == '__main__':
	data_path = sys.argv[1]

	solver = Solver(data_path)
	solver.model_based_learning()
	solver.monte_carlo_learning()
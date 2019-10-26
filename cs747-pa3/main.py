import sys
from solver import Solver

if __name__ == '__main__':

	solver = Solver(sys.argv[1], sys.argv[-1])
	# solver.model_based_learning()
	solver.monte_carlo_learning()
from __future__ import print_function
import sys
import numpy as np
from solver import Solver

error_values = []

def grid_search():
	for lambda_ in np.linspace(0.70,0.99,30):
		for alpha in np.linspace(0.01, 0.055,30):
			error = solver.td_lambda(lambda_=lambda_, alpha=alpha)
			if error < 0.0005:
				print('----------------lambda = {}, alpha= {}---------------------'.format(lambda_ ,alpha))
				print(error)

solver = Solver(sys.argv[1], sys.argv[-1])

solver.monte_carlo_learning()

# grid_search()
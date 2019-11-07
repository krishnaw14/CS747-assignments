from agent import SarsaAgent
import numpy as np 
import matplotlib.pyplot as plt
import os
from environment import WindyGridWorld, WindyGridWorldwithKingMoves, StochasticWindyGridWorld

os.makedirs('plots', exist_ok=True)

env = WindyGridWorld()
agent = SarsaAgent(env, alpha=0.5, epsilon=0.1, save_plot_path='baseline')
agent.learn(num_seed_runs = 10)

env = WindyGridWorldwithKingMoves()
agent = SarsaAgent(env, alpha=0.5, epsilon=0.2, save_plot_path='kingmoves')
agent.learn(num_seed_runs = 10)

# env = StochasticWindyGridWorld()
# agent = SarsaAgent(env, alpha=0.5, epsilon=0.2, save_plot_path='stochastic')
# agent.learn(num_seed_runs = 10)

plt.legend()
plt.show()
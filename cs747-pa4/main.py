from agent import SarsaAgent
import numpy as np 
import os

os.makedirs('plots', exist_ok=True)

agent = SarsaAgent(alpha=0.5, epsilon=0.1, save_plot_path='plots/baseline.png')
agent.learn()

from pulp import *
import numpy as np

def linear_programming(num_states, num_actions, gamma, transition_function, reward_function):

	prob = pulp.LpProblem('MDP Linear Solver', pulp.LpMinimize)

	# Define Decision Variable
	decision_variables = []
	for i in range(num_states):
		var = pulp.LpVariable('X' + str(i)) #make variables binary
		decision_variables.append(var)

	# Define Objective
	prob += sum(decision_variables)

	# Define Constraints
	for state in range(num_states):
		lhs = decision_variables[i]
		for action in range(num_actions):
			rhs = 0
			for target_state in range(num_states):
				rhs += transition_function[state][action][target_state]*(
					reward_function[state][action][target_state]+gamma*decision_variables[target_state])
			prob += (lhs>=rhs), "constraint_{}_{}".format(state,action)

	while True:
		optimisation_result = prob.solve()
		flag = True
		for v in prob.variables():
			if v.varValue != 0:
				flag = False
				break
		if flag == False:
			break

	for constraint in prob.constraints:
		# import pdb; pdb.set_trace() 
		print("constraint:", prob.constraints[constraint].value() - prob.constraints[constraint].constant)
		# print(prob.constraints[constraint])
	for v in prob.variables():
		print(v.name, "=", v.varValue)
	import pdb; pdb.set_trace()


def howard_policy_iteration(num_states, num_actions, gamma, transition_function, reward_function):

	def evaluate_policy(policy):
		# See Report for derivation: Solving for V = [I-gammaT]^-1Alpha
		alpha = np.zeros((num_states,1))
		T = np.zeros((num_states, num_states))
		for state in range(num_states):
			action = policy[state]
			alpha[state] = np.sum(transition_function[state][action]*reward_function[state][action])
			T[state][:] = transition_function[state][action]
		# import pdb; pdb.set_trace()
		value_function = np.matmul(np.linalg.inv(1-gamma*T), alpha)
		return value_function

	def evaluate_action_value_function(value_function):
		action_value_function = np.zeros((num_states, num_actions))
		for state in range(num_states):
			for action in range(num_actions):
				action_value_function[state][action] = np.sum(
					transition_function[state][action]*(reward_function[state][action] + gamma*value_function)
					)
		return action_value_function

	policy = np.zeros((num_states, 1), dtype=np.int8)
	improvable_state_exists = True
	num_iterations = 0
	while improvable_state_exists:
		value_function = evaluate_policy(policy)
		action_value_function = evaluate_action_value_function(value_function)

		flag = True
		for state in range(num_states):
			if policy[state] != np.argmax(action_value_function[state]):
				policy[state] = np.argmax(action_value_function[state])
				check_for_improvable_state = False
		num_iterations += 1

		if flag == True:
			improvable_state_exists = False

	return value_function, policy

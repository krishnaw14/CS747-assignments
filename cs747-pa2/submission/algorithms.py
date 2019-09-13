from pulp import *
import numpy as np

def evaluate_action_value_function(value_function, num_states, num_actions, gamma, transition_function, 
	reward_function):
	action_value_function = np.zeros((num_states, num_actions))
	for state in range(num_states):
		for action in range(num_actions):
			action_value_function[state][action] = np.sum(
				transition_function[state][action]*(reward_function[state][action] + gamma*value_function)
				)
	return action_value_function

def evaluate_policy(policy, num_states, num_actions, gamma, transition_function, reward_function):
		# See notes.txt for derivation: Solving for V = [I-gammaT]^-1Alpha
		alpha = np.zeros((num_states,1))
		T = np.zeros((num_states, num_states))
		for state in range(num_states):
			action = policy[state]
			alpha[state] = np.sum(transition_function[state][action]*reward_function[state][action])
			T[state][:] = transition_function[state][action]
		try:
			value_function = np.matmul(np.linalg.inv(np.eye(num_states)-gamma*T), alpha)
		except np.linalg.LinAlgError as e:
			print(e, "Cannot Perform Policy Evaluation!")
			exit(1)
		return value_function

def linear_programming(num_states, num_actions, gamma, transition_function, reward_function):

	prob = pulp.LpProblem('MDP Linear Solver', LpMinimize)

	# Define Decision Variable
	decision_variables = []
	for i in range(num_states):
		var = LpVariable('V' + str(i)) 
		decision_variables.append(var)

	# Define Objective
	objective = sum(decision_variables)
	prob += objective

	# Define Constraints
	for state in range(num_states):
		lhs = decision_variables[state]
		for action in range(num_actions):
			rhs = 0
			for target_state in range(num_states):
				rhs += transition_function[state][action][target_state]*(
					reward_function[state][action][target_state]+gamma*decision_variables[target_state])
			prob += (lhs>=rhs)

	# Solve the Linear Programming Formulation
	optimization_result = prob.solve()

	# Collect the solution of the linear programming
	optimal_value_function = np.array([v.varValue for v in prob.variables()])

	# Get Action Value function from value function
	optimal_action_value_function = evaluate_action_value_function(optimal_value_function, num_states, 
		num_actions, gamma, transition_function, reward_function)

	# Action maximising the action value function for a given state gives us the required optimal policy
	optimal_policy = np.argmax(optimal_action_value_function, axis=1)

	return optimal_value_function, optimal_policy


def howard_policy_iteration(num_states, num_actions, gamma, transition_function, reward_function):

	# Randomly initialize a policy. This will be changed during policy improvement.
	policy = np.random.randint(low=0, high=num_actions, size=(num_states,))
	improvable_state_exists = True

	# Iterate until the existence of an improvable state
	while improvable_state_exists:
		value_function = evaluate_policy(policy, num_states, num_actions, gamma, transition_function, reward_function)
		action_value_function = evaluate_action_value_function(value_function, num_states, num_actions, gamma,
			transition_function, reward_function)

		flag = True # To check if an improvable state was detected for the give policy
		for state in range(num_states):
			if policy[state] != np.argmax(action_value_function[state]): # Check if there is an improvable state
				policy[state] = np.argmax(action_value_function[state]) # Flip the state if there is an improvable state
				flag = False

		if flag == True: # If policy is not improved, it implies that there are no improvable state. 
			improvable_state_exists = False

	return value_function.squeeze(), policy

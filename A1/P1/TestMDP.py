from MDP import *

''' Construct simple MDP as described in Lecture 1b Slides 17-18'''
# Transition function: |A| x |S| x |S'| array
# First set of arrays is for action "Advertise", the next is for action "Save"
# Array entry 0: P&U, 1: P&F, 2: R&U, 3: R&F
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
# Reward function: |A| x |S| array
R = np.array([[0,0,10,10],[0,0,10,10]]) # This is different from the formalization in Sut_Bar since the reward is determined by the current state rather than jointly determined with the next state.
# Discount factor: scalar in [0,1)
discount = 0.9        
# MDP object
mdp = MDP(T,R,discount)

'''Test each procedure'''

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

[V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))
policy = mdp.extractPolicy(V) # Extract policy from value function
print(f"\nV for value iteration: {V}\nnIterations: {nIterations}\nepsilon: {epsilon:.5f}\npolicy: {policy}\n")

V = mdp.evaluatePolicy(np.array([1,0,1,0])) # Evaluate policy
print(f"V for policy evaluation: {V}\n")

[policy,V,iterId] = mdp.policyIteration(np.array([0,0,0,0]))
print(f"V for policy iteration: {V}\nnIterations: {iterId}\npolicy: {policy}\n")

[V,iterId,epsilon] = mdp.evaluatePolicyPartially(np.array([1,0,1,0]),np.array([0,10,0,13]))
print(f"V for partial policy evaluation: {V}\nnIterations: {iterId}\ntolerance: {epsilon:.3f}\n")

[policy,V,iterId,tolerance] = mdp.modifiedPolicyIteration(np.array([1,0,1,0]),np.array([0,10,0,13]))
print(f"V for modified policy iteration: {V}\nnIterations: {iterId}\ntolerance: {tolerance:.3f}\npolicy: {policy}\n")
import numpy as np
import MDP
import matplotlib.pyplot as plt

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:]) # This line is a cumulative sum along the array of probabilities from s to s'. The cumulative sum only changes over indices where the state transition probabilities are non-zero.
        nextState = np.where(cumProb >= np.random.rand(1))[0][0] # This generates a random value in the interval [0,1). 50% of the time this is > 0.5, and 50% of the time it will be less than 0.5. Therefore this in combination with the cumulative sum above will select the accessible states at random according to their probabilities.
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  
        When epsilon > 0: perform epsilon exploration (i.e., with probability epsilon, select action at random)
        When epsilon == 0 and temperature > 0: perform Boltzmann exploration with temperature parameter
        When epsilon == 0 and temperature == 0: no exploration (i.e., selection action with best Q-value)

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''
        # Constants
        nTrials = 10

        # initialize variables before trials: (avg_cum_r = zeros)
        # var List () <- (avg_cum_r)
        # initialize (avg_cum_r)
        avg_cum_r = np.zeros(nEpisodes)

        for trial in range(nTrials):
            # start trial
            # initializate variables before episodes: (Q <- zeros, state_action_count <- zeros)
            # var List loop (trail) (avg_cum_r) <- (Q, step_s_a)
            # carry through (avg_cum_r) initialize (Q, step_s_a)
            Q = np.copy(initialQ)
            step_s_a = np.zeros([self.mdp.nActions, self.mdp.nStates])

            for episode in range(nEpisodes):
                # start episode
                # initialize variables before steps: (cum_r <- 0, s <- s0)
                # var List loop (trial, episode) (avg_cum_r, Q, step_s_a) <- (cum_r, s)
                # carry through (avg_cum_r, Q, step_s_a) initialize (cum_r, s)
                cum_r = 0
                s = s0

                for step in range(nSteps):
                    # start step
                    # select Action: (a <- selected according to epsilon greedy, boltzman, or max Q value)
                    # var List loop (trial, episode, step) (avg_cum_r, Q, step_s_a, cum_r, s) <- (a)
                    # carry through (avg_cum_r, Q, step_s_a, cum_r, s) initialize (a) 
                    if epsilon > 0: # epsilon greedy
                        if np.random.rand(1)[0] < epsilon:
                            a = np.random.choice(range(self.mdp.nActions))  # epsilon greedy exploration
                        else:
                            a = Q[:,s].argmax() # epsilon greedy exploitation
                    elif epsilon == 0 and temperature > 0:  # boltzman
                        Prob_a = np.exp(Q[:,s]/temperature)/np.sum(np.exp(Q[:,s]/temperature))
                        cumProb_a = np.cumsum(Prob_a)
                        a = np.where(cumProb_a >= np.random.rand(1))[0][0]  # boltzman exploration & exploitation according to temperature
                    elif epsilon == 0 and temperature == 0:
                        a = Q[:,s].argmax() # exploitation

                    # generate next state and reward using stochastic model and current state and selected action (r <- model(s,a), sn <- model(s,a))
                    # var List loop (trial, episode, step) (avg_cum_r, Q, step_s_a, cum_r, s, a) <- (r, s_n)
                    # carry through (avg_cum_r, Q, step_s_a, cum_r, s, a) initialize (r, s_n)
                    [r, s_n] = self.sampleRewardAndNextState(s,a)

                    # update variables: (state_action_count += 1, Q with TD, cum_r += gamma^t*r, s <- sn)
                    # var List loop (trial, episode, step) (avg_cum_r, Q, step_s_a, cum_r, s, a, r, s_n)
                    # carry through (avg_cum_r, a, r, s_n) set (Q, step_s_a, cum_r, s)
                    step_s_a[a,s] += 1  # update state_action_count
                    Q[a,s] = Q[a,s] + (1./step_s_a[a,s])*(r + self.mdp.discount*Q[:,s_n].max() - Q[a,s])    # Temporal Difference Control
                    cum_r += (self.mdp.discount**step)*r
                    s = s_n

                    # end of step
                    # Remove variables (a, r, s_n)
                    # var List loop (trial, episode, step) (avg_cum_r, Q, step_s_a, cum_r, s)
                    del a, r, s_n

                # remove variables (s)
                # var List loop (trial, episode) (avg_cum_r, Q, step_s_a, cum_r)
                del s

                # update variables (avg_cum_r <- cum_r/nTrials)
                # var List loop (trial, episode) (avg_cum_r, Q, step_s_a, cum_r)
                # carry through (Q, step_s_a, cum_r) set(avg_cum_r)
                avg_cum_r[episode] += cum_r/nTrials

                # end of episode
                # remove variables (cum_r)
                # var List loop(trial, episode) (avg_cum_r, Q, step_s_a)
                del cum_r

            # end of trial
            # Remove variables (step_s_a)
            # var List loop (trial) (avg_cum_r, Q)
            del step_s_a

        policy = Q.argmax(axis=0)

        return [Q, policy, avg_cum_r]    

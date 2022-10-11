import numpy as np
import MDP
import RL2
import matplotlib.pyplot as plt


def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0


# Multi-arm bandit problems (3 arms with probabilities 0.25, 0.5 and 0.75)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.25],[0.5],[0.75]])
discount = 0.999
mdp = MDP.MDP(T,R,discount)
banditProblem = RL2.RL2(mdp,sampleBernoulli)

# Test epsilon greedy strategy
empiricalMeans, _ = banditProblem.epsilonGreedyBandit(nIterations=200)
print("\nepsilonGreedyBandit results")
print(empiricalMeans)

# Test Thompson sampling strategy
empiricalMeans, _ = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=200)
print("\nthompsonSamplingBandit results")
print(empiricalMeans)

# Test UCB strategy
empiricalMeans, _ = banditProblem.UCBbandit(nIterations=200)
print("\nUCBbandit results")
print(empiricalMeans)

epGreedy_curves = []
Thompson_curves = []
UCB_curves = []
for trial in range(1000):
    eM_epGreedy, r_seq_epGreedy =  banditProblem.epsilonGreedyBandit(nIterations=200)
    eM_Thompson, r_seq_Thompson =  banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=200)
    eM_UCB, r_seq_UCB =  banditProblem.UCBbandit(nIterations=200)
    epGreedy_curves += [r_seq_epGreedy]
    Thompson_curves += [r_seq_Thompson]
    UCB_curves += [r_seq_UCB]

def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis = 0)
    std = np.std(vars, axis = 0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    # plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std, 200), color=color, alpha=0.3)

plot_arrays(epGreedy_curves, color = 'b', label = 'epsilon Greedy')
plot_arrays(Thompson_curves, color = 'r', label = 'Thompson Sampling')
plot_arrays(UCB_curves, color = 'g', label = 'UCB')

plt.legend(loc='best')
plt.xlabel('Iteration')
plt.ylabel('Reward Earned at each Iteration')
plt.ylim([0.2, 0.8])
plt.show()
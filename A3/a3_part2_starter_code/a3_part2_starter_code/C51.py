from turtle import update
import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# C51
# Based on Slide 11
# cs.uwaterloo.ca/~ppoupart/teaching/cs885-winter22/slides/cs885-module5.pdf

# Constants
SEEDS = [1, 2, 3, 4, 5]
t = utils.torch.TorchHelper()
DEVICE = t.device
OBS_N = 4               # State space size
ACT_N = 2               # Action space size
STARTING_EPSILON = 1.0  # Starting epsilon
STEPS_MAX = 10000       # Gradually reduce epsilon over these many steps
EPSILON_END = 0.1       # At the end, keep epsilon at this value
MINIBATCH_SIZE = 64     # How many examples to sample per train step
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATE = 5e-4    # Learning rate for Adam optimizer
TRAIN_AFTER_EPISODES = 10   # Just collect episodes for these many episodes
TRAIN_EPOCHS = 25       # Train for these many epochs every time
BUFSIZE = 10000         # Replay buffer size
EPISODES = 500          # Total number of episodes to learn over
TEST_EPISODES = 10      # Test episodes
HIDDEN = 512            # Hidden nodes
TARGET_NETWORK_UPDATE_FREQ = 10 # Target network update frequency

# Suggested constants
ATOMS = 51              # Number of atoms for distributional network
ZRANGE = [0, 100]       # Range for Z projection
DELTA_Z = (ZRANGE[1] - ZRANGE[0])/(ATOMS-1.0) 
Z_SUP = torch.range(ZRANGE[0], ZRANGE[1], DELTA_Z).detach()

# Global variables
EPSILON = STARTING_EPSILON
Z = None

# Create environment
# Create replay buffer
# Create distributional networks
# Create optimizer
def create_everything(seed):
    utils.seed.seed(seed)
    env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)
    env.reset(seed=seed)
    test_env = utils.envs.TimeLimit(utils.envs.NoisyCartPole(), 500)
    test_env.reset(seed=seed)
    buf = utils.buffers.ReplayBuffer(BUFSIZE)
    Z = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
    ).to(DEVICE)
    Zt = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N*ATOMS)
    ).to(DEVICE)
    OPT = torch.optim.Adam(Z.parameters(), lr = LEARNING_RATE)
    return env, test_env, buf, Z, Zt, OPT

# Create epsilon-greedy policy
def policy(env, obs):

    global EPSILON, EPSILON_END, STEPS_MAX, Z
    obs = t.f(obs).view(-1, OBS_N)  # Convert to torch tensor
    
    # With probability EPSILON, choose a random action
    # Rest of the time, choose argmax_a Q(s, a) 
    if np.random.rand() < EPSILON:
        action = np.random.randint(ACT_N)
    else:
        ## TODO: use Z to compute greedy action
        with torch.no_grad():
            probs = torch.nn.Softmax(dim=1)(Z(obs).view(ACT_N, ATOMS))
            expected_return = (probs*Z_SUP).sum(dim=1)
            action = expected_return.argmax().item()
    
    # Epsilon update rule: Keep reducing a small amount over
    # STEPS_MAX number of steps, and at the end, fix to EPSILON_END
    EPSILON = max(EPSILON_END, EPSILON - (1.0 / STEPS_MAX))
    
    return action

# Update networks
def update_networks(epi, buf, Z, Zt, OPT):
    
    ## TODO: Implement this function
    S, A, R, S_prime, done = buf.sample(MINIBATCH_SIZE, t)

    p = torch.zeros((MINIBATCH_SIZE, ATOMS))
    with torch.no_grad():
        probs_prime = torch.nn.Softmax(dim=2)(Zt(S_prime).view(MINIBATCH_SIZE, ACT_N, ATOMS))
        expected_return_prime = (probs_prime*Z_SUP).sum(dim=2)
        a_greedy = expected_return_prime.argmax(dim=1)
        a_greedy = a_greedy.unsqueeze(1).unsqueeze(2).expand(MINIBATCH_SIZE, 1, ATOMS)

        Tau_z = torch.clip(R.view(MINIBATCH_SIZE, 1) + GAMMA * Z_SUP, ZRANGE[0], ZRANGE[1])
        Tau_z = torch.where(done.unsqueeze(1).expand(MINIBATCH_SIZE, ATOMS) == 1, R.unsqueeze(1).expand(MINIBATCH_SIZE, ATOMS), Tau_z)

        index = (Tau_z - ZRANGE[0])/DELTA_Z
        l_index = torch.clip(t.l(torch.floor(index)), 0, ATOMS - 1)
        u_index = torch.clip(t.l(torch.ceil(index)), 0, ATOMS - 1)

        probs_greedy = torch.gather(input=probs_prime, index=a_greedy, dim=1)  
        p.scatter_add_(dim=1, index = l_index, src = probs_greedy.squeeze()*(u_index - index)) # For repeated indices, torch.scatter_add accumulates the sum, rather than simply overwriting at the index non-deterministically
        p.scatter_add_(dim=1, index = u_index, src = probs_greedy.squeeze()*(index - l_index)) # For repeated indices, torch.scatter_add accumulates the sum, rather than simply overwriting at the index non-deterministically

    OPT.zero_grad()
    zp_out = torch.nn.Softmax(dim=2)(Z(S).view(MINIBATCH_SIZE, ACT_N, ATOMS))
    a_experience = A.unsqueeze(1).unsqueeze(2).expand(MINIBATCH_SIZE, 1, ATOMS)
    zp_a = torch.gather(input=zp_out, index=a_experience, dim=1)
    loss = -(p * torch.log(zp_a.squeeze())).sum(-1).mean()
    loss.backward()
    OPT.step()

    # Update target network
    if epi%TARGET_NETWORK_UPDATE_FREQ==0:
        Zt.load_state_dict(Z.state_dict())

    return loss


# Play episodes
# Training function
def train(seed):

    global EPSILON, Z
    print("Seed=%d" % seed)

    # Create environment, buffer, Z, Z target, optimizer
    env, test_env, buf, Z, Zt, OPT = create_everything(seed)

    # epsilon greedy exploration
    EPSILON = STARTING_EPSILON

    testRs = [] 
    last25testRs = []
    print("Training:")
    pbar = tqdm.trange(EPISODES)
    for epi in pbar:

        # Play an episode and log episodic reward
        S, A, R = utils.envs.play_episode_rb(env, policy, buf)  # This updates the buffer with experiences
        
        # Train after collecting sufficient experience
        if epi >= TRAIN_AFTER_EPISODES:

            # Train for TRAIN_EPOCHS
            for tri in range(TRAIN_EPOCHS): 
                update_networks(epi, buf, Z, Zt, OPT)

        # Evaluate for TEST_EPISODES number of episodes
        Rews = []
        for epj in range(TEST_EPISODES):
            S, A, R = utils.envs.play_episode(test_env, policy, render = False)
            Rews += [sum(R)]
        testRs += [sum(Rews)/TEST_EPISODES]

        # Update progress bar
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        pbar.set_description("R25(%g)" % (last25testRs[-1]))

    pbar.close()
    print("Training finished!")
    env.close()

    return last25testRs

# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 200
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,500), color=color, alpha=0.3)

if __name__ == "__main__":

    # Train for different seeds
    curves = []
    for seed in SEEDS:
        curves += [train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves, 'b', 'c51')
    plt.legend(loc='best')
    plt.show()
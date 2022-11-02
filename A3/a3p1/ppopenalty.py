import gym
import numpy as np
import utils.envs, utils.seed, utils.buffers, utils.torch
import torch
import tqdm
import matplotlib.pyplot as plt
import copy
import warnings
warnings.filterwarnings("ignore")

# Constants
SEEDS = [1,2,3,4,5]
t = utils.torch.TorchHelper()
DEVICE = t.device
GAMMA = 0.99            # Discount factor in episodic reward objective
LEARNING_RATES = {      # Learning rate for optimizers
    "Pi": 1e-4,
    "V": 1e-3,
}
EPOCHS = 300            # Total number of episodes to learn over
EPISODES_PER_EPOCH = 20 # Epsides per epoch
TEST_EPISODES = 10      # Test episodes
HIDDEN = 64             # Hidden size
BUFSIZE = 10000         # Buffer size
CLIP_PARAM = 0.1        # Clip parameter
MINIBATCH_SIZE = 64     # Minibatch size
TRAIN_EPOCHS = 50       # Training epochs
OBS_N = None
ACT_N = None
Pi = None

# Create environment
# Create replay buffer
# Create networks
# Create optimizers
def create_everything(seed):

    global OBS_N, ACT_N, penalty_param, Pi
    utils.seed.seed(seed)
    env = utils.envs.TimeLimit(utils.envs.ConstrainedCartPole(), 200)
    env.reset(seed=seed)
    test_env = utils.envs.TimeLimit(utils.envs.ConstrainedCartPole(), 200)
    test_env.reset(seed=10+seed)
    OBS_N = env.observation_space.shape[0]  # State space size
    ACT_N = env.action_space.n              # Action space size
    buf = utils.buffers.ReplayBuffer(BUFSIZE, OBS_N, t)
    Pi = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, ACT_N)
    ).to(DEVICE)
    V = torch.nn.Sequential(
        torch.nn.Linear(OBS_N, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 1)
    ).to(DEVICE)
    OPTPi = torch.optim.Adam(Pi.parameters(), lr = 1e-4)
    OPTPic = torch.optim.Adam(Pi.parameters(), lr = 1e-5)
    OPTV = torch.optim.Adam(V.parameters(), lr = 1e-3)
    return env, test_env, buf, Pi, V, OPTPi, OPTV, OPTPic

# Policy
def policy(env, obs):

    probs = torch.nn.Softmax(dim=-1)(Pi(t.f(obs)))
    return np.random.choice(ACT_N, p = probs.cpu().detach().numpy())

# Training function
def update_networks(epoch_data, buf, Pi, V, OPTPi, OPTV, OPTPic):
    
    # Sample from buffer
    S, A, returns, old_log_probs = buf.sample(MINIBATCH_SIZE)
    log_probs = torch.nn.LogSoftmax(dim=-1)(Pi(S)).gather(1, A.view(-1, 1)).view(-1)

    # Critic update
    OPTV.zero_grad()
    objective1 = (returns - V(S)).pow(2).mean()
    objective1.backward()
    OPTV.step()

    # Actor update
    OPTPi.zero_grad()
    advantages = returns - V(S)
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1-CLIP_PARAM, 1+CLIP_PARAM)
    ppo_obj = torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    objective2 = -ppo_obj
    objective2.backward()
    OPTPi.step()

    if penalty_param != 0:
        epoch_data = copy.deepcopy(epoch_data)

        # Error checking
        # assert len(epoch_data['S']) == EPISODES_PER_EPOCH, "states array data not for 20 episodes"
        # assert len(epoch_data['A']) == EPISODES_PER_EPOCH, "actions array data not for 20 episodes"
        # assert len(epoch_data['ret_c']) == EPISODES_PER_EPOCH, "costs return array data not for 20 episodes"
        # assert len(epoch_data['I_Gc0']) == EPISODES_PER_EPOCH, "I_Gc0 array data not for 20 episodes"

        for episode in range(EPISODES_PER_EPOCH):
            if epoch_data['I_Gc0']:
                OPTPic.zero_grad()
                log_probs_c = torch.nn.LogSoftmax(dim=-1)(Pi(epoch_data['S'][episode])).gather(1, epoch_data['A'][episode].detach().view(-1, 1)).view(-1)
                objective3 = epoch_data['I_Gc0'][episode].detach() * (epoch_data['ret_c'][episode].detach() * log_probs_c).sum()
                objective3.backward()
                OPTPic.step()


# Play episodes
# Training function
def train(seed):

    global penalty_param
    print("Seed=%d" % seed)

    # Create environments, buffer, networks, optimizers
    env, test_env, buf, Pi, V, OPTPi, OPTV, OPTPic = create_everything(seed)

    # Train PPO for EPOCH times
    testRs = []
    testRcs = []
    last25testRs = []
    last25testRcs = []
    print("Training:")
    pbar = tqdm.trange(EPOCHS)
    for epi in pbar:

        # Collect experience
        all_S, all_A = [], []
        all_returns = []
        epoch_S = []
        epoch_A = []
        epoch_c = []
        epoch_I_Gc0 = []
        for epj in range(EPISODES_PER_EPOCH):
            
            # Play an episode and log episodic reward
            S, A, R, Rc = utils.envs.play_episode(env, policy, constraint=True)
            all_S += S[:-1] # ignore last state
            all_A += A
            epoch_S += [t.f(np.array(S[:-1]))]
            epoch_A += [t.l(np.array(A))]
            
            # Create returns 
            discounted_rewards = copy.deepcopy(R)
            for i in range(len(R)-1)[::-1]:
                discounted_rewards[i] += GAMMA * discounted_rewards[i+1]
            discounted_rewards = t.f(discounted_rewards)
            all_returns += [discounted_rewards]
                       
            # Create cost returns
            discounted_costs = copy.deepcopy(Rc)
            for i in range(len(Rc)-1)[::-1]:
                discounted_costs[i] += GAMMA * discounted_costs[i+1]
            discounted_costs = t.f(discounted_costs)
            epoch_c += [discounted_costs]
            if discounted_costs[0] > penalty_param:
                epoch_I_Gc0 += [t.f(1.)]
            else:
                epoch_I_Gc0 += [t.f(0.)]

        S, A = t.f(np.array(all_S)), t.l(np.array(all_A))
        returns = torch.cat(all_returns, dim=0).flatten()

        # add to replay buffer
        log_probs = torch.nn.LogSoftmax(dim=-1)(Pi(S)).gather(1, A.view(-1, 1)).view(-1)
        buf.add(S, A, returns, log_probs.detach())

        epoch_data = {'S':epoch_S, 'A':epoch_A, 'ret_c':epoch_c, 'I_Gc0':epoch_I_Gc0}

        # update networks
        for i in range(TRAIN_EPOCHS):
            update_networks(epoch_data, buf, Pi, V, OPTPi, OPTV, OPTPic)

        # evaluate
        Rews = []
        Rewcs = []
        for epj in range(TEST_EPISODES):
            S, A, R, Rc = utils.envs.play_episode(test_env, policy, constraint=True)
            Rews += [sum(R)]
            Rewcs += [sum(Rc)]
        testRs += [sum(Rews)/TEST_EPISODES]
        testRcs += [sum(Rewcs)/TEST_EPISODES]

        # Show mean episodic test reward over last 25 episodes
        last25testRs += [sum(testRs[-25:])/len(testRs[-25:])]
        last25testRcs += [sum(testRcs[-25:])/len(testRcs[-25:])]
        pbar.set_description("R25(%g), Rc25(%g)" % (last25testRs[-1], last25testRcs[-1]))

    pbar.close()
    print("Beta: ", penalty_param, " Seed: ", seed, ", Training finished!")
    env.close()
    
    return last25testRs, last25testRcs

# Plot mean curve and (mean-std, mean+std) curve with some transparency
# Clip the curves to be between 0, 200
def plot_arrays(ax, vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    ax.plot(range(len(mean)), mean, color=color, label=label)
    ax.fill_between(range(len(mean)), np.maximum(mean-std, 0), np.minimum(mean+std,200), color=color, alpha=0.3)
    ax.set_xlabel("Episode")

if __name__ == "__main__":

    fig, ax = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(10)

    # Train for different seeds
    plt_colours = {0:"red", 1:"blue", 5:"green", 10:"purple", 300:"black"}
    for penalty_param in [0, 1, 5, 10]:
        curves = []
        curvesc = []
        for seed in SEEDS:
            R, Rc = train(seed)
            curves += [R]
            curvesc += [Rc]

        # Plot the curve for the given seeds
        if penalty_param == 0:
            plt_label = "ppo"
        else:
            plt_label = f"beta = {penalty_param}"
        plot_arrays(ax[0], curves, plt_colours[penalty_param], plt_label)
        plot_arrays(ax[1], curvesc, plt_colours[penalty_param], plt_label)
        ax[0].set_ylabel("Reward")
        ax[1].set_ylabel("Cost")
        plt.legend(loc='best')

    plt.show()
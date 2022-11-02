import matplotlib.pyplot as plt 
import DQN
import C51


curves = []
for seed in DQN.SEEDS:
    curves += [DQN.train(seed)]

# Plot the curve for the given seeds
DQN.plot_arrays(curves, 'b', 'DQN')
plt.legend(loc='best')


for seed in C51.SEEDS:
    curves += [C51.train(seed)]

# Plot the curve for the given seeds
C51.plot_arrays(curves, 'r', 'C51')
plt.legend(loc='best')

plt.show()
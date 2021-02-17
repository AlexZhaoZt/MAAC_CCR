import pickle
import numpy as np
import matplotlib.pyplot as plt

def running_average(data, subset_size=10):
    new_data = np.zeros(data.shape[0] - subset_size + 1)
    for i in range(new_data.shape[0]):
        new_data[i] = np.mean(data[i : i + subset_size])
    return new_data

path_name = "models/wall/maac_ccr1/run5/"
reward_file_name = "rewards.pkl"

with open(path_name + reward_file_name, 'rb') as f:
    reward_data = np.array(pickle.load(f))
    # hack to multiply with agent number and episode length
    reward_data *= 6
    reward_data *= 35

reward_data = running_average(reward_data)

plt.plot(reward_data)

plt.savefig(path_name + 'new_rewards.png')
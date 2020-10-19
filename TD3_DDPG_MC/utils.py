import numpy as np
from collections import OrderedDict
import math
import random
import gym
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
	def __init__(self, max_size=1e6):
		self.storage = []
		self.max_size = max_size
		self.ptr = 0

	def add(self, data):
		if len(self.storage) == self.max_size:
			self.storage[int(self.ptr)] = data
			self.ptr = (self.ptr + 1) % self.max_size
		else:
			self.storage.append(data)

	def sample(self, batch_size):
		ind = np.random.randint(0, len(self.storage), size=batch_size)
		x, y, u, r, d = [], [], [], [], []

		for i in ind: 
			X, Y, U, R, D = self.storage[i]
			x.append(np.array(X, copy=False))
			y.append(np.array(Y, copy=False))
			u.append(np.array(U, copy=False))
			r.append(np.array(R, copy=False))
			d.append(np.array(D, copy=False))

		return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)

def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()

class Hot_Plug(object):
    def __init__(self, model):
        self.model = model
        self.params = OrderedDict(self.model.named_parameters())
    def update(self, lr=0.1):
        for param_name in self.params.keys():
            path = param_name.split('.')
            cursor = self.model
            for module_name in path[:-1]:
                cursor = cursor._modules[module_name]
            if lr > 0:
                cursor._parameters[path[-1]] = self.params[param_name] - lr*self.params[param_name].grad
            else:
                cursor._parameters[path[-1]] = self.params[param_name]
    def restore(self):
        self.update(lr=0)

class Critic_Network(nn.Module):
    def __init__(self, hidden_dim):
        super(Critic_Network, self).__init__()
        self.fc1 = nn.Linear(hidden_dim,100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = nn.functional.softplus(self.fc3(x))
        return torch.mean(x)

def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

def plot_results(path):
    model_path_1 = '%s/evaluation_reward.npy' % (path)
    plot_path = '%s/results.jpg' % (path)

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def sliding_mean(data_array, window=5):
        data_array = np.array(data_array)
        new_list = []
        for i in range(len(data_array)):
            indices = list(range(max(i - window + 1, 0),
                                 min(i + window + 1, len(data_array))))
            avg = 0
            for j in indices:
                avg += data_array[j]
            avg /= float(len(indices))
            new_list.append(avg)

        return np.array(new_list)

    smooth_curve = True

    def get_data(model_path):
        average_reward_train = np.load(model_path)

        if smooth_curve:
            clean_statistics_train = sliding_mean(average_reward_train, window=30)
        else:
            clean_statistics_train = average_reward_train

        return clean_statistics_train

    model_path = []
    model_path.append(model_path_1)
    results_all = []
    for file in model_path:
        results_all.append(get_data(file))

    for rel in results_all:
        if smooth_curve:
            results_tmp = rel.tolist()
        else:
            results_tmp = rel
        frame_id = results_tmp.index(max(results_tmp))
        print(frame_id)
        print(max(results_tmp))
        print('******************************')

    plt.title('Learning curve')
    plt.xlabel('Episode'), plt.ylabel('Average Reward'), plt.legend(loc='best')

    plt.plot(results_all[0], color='#FFA500', label="exploration")

    plt.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig(plot_path)
    plt.close('all')


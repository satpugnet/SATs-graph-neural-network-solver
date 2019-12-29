import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torch
import os
import time
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, NNConv, SplineConv
from torch_geometric.data import InMemoryDataset
import matplotlib.pyplot as plt
import pickle
import scipy.io
import matplotlib.pyplot as plt

from torchdiffeq import odeint_adjoint as odeint

np.random.seed(0)
torch.manual_seed(0)

pi = torch.tensor(np.pi)

#################################################
#
# generate ODE data
#
#################################################


# simple example, y = exp(y^2), dy/dt = 2t exp(y^2) = 2*y*y

# derivative
# def func_true(y, y):
#     return y * y

time_step = 100
t = torch.linspace(0., 1, time_step)
y_init = torch.tensor([1.])
true_y = torch.exp(t**2)

beta = 0
class Lambda(nn.Module):
    def forward(self, y, t):
        return beta * y * t


with torch.no_grad():
    beta = 1
    y1 = odeint(Lambda(), y_init, t, method='adams')
    beta = 2
    y2 = odeint(Lambda(), y_init, t, method='adams')

#################################################
#
# Construct Graph
#
#################################################



#################################################
#
# Network
#
#################################################

class Net_FC(torch.nn.Module):
    def __init__(self):
        super(Net_FC, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, y, t):
        x = torch.stack((y.view(-1), t.view(-1))).view(2)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc3(x)
        return x * y * t


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.conv1 = GCNConv(32, 32)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.fc3(x)
        return x

#################################################
#
# Train
#
#################################################


n = time_step
num_train_node = n//2
shuffle_idx = np.random.permutation(n)
train_mask = shuffle_idx[:num_train_node]
test_mask = shuffle_idx[num_train_node:]

iter = 0

func = Net_FC()
optimizer = optim.RMSprop(func.parameters(), lr=1e-2)
end = time.time()


for itr in range(100):
    optimizer.zero_grad()

    pred_y = odeint(func, y_init, t).view(-1)
    loss = F.mse_loss(pred_y[train_mask], true_y[train_mask])
    loss.backward()
    optimizer.step()

    test_error = F.mse_loss(pred_y[test_mask], true_y[test_mask])
    runtime = time.time() - end
    print('Iter {:04d} | Train Loss {:.6f} | Test Error {:.6f} | Time {:.4f}'.format(itr, loss.item(), test_error.item(), runtime))


    # if itr % 10 == 0:
    #     with torch.no_grad():
    #         pred_y = odeint(func, y_init, y, method='adams').view(-1)
    #         loss = F.mse_loss(pred_y, true_y)
    #         print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item(), runtime))
    #         iter = iter + 1

    end = time.time()



#################################################
#
# plot
#
#################################################


pred_y = pred_y.detach().numpy()

fig, ax = plt.subplots()
ax.set_xlabel('y')
ax.set_ylabel('y')
ax.plot(t.numpy(), true_y.numpy(), 'g-')
ax.plot(t.numpy(), pred_y, 'b--')
# ax.plot(y.numpy(), y2.numpy(), 'r--')
ax.set_xlim(t.min(), t.max())
ax.set_ylim(1, 4)
plt.show()


path = "/Users/lizongyi/Downloads/GNN-PDE/neural_ode/"
if not os.path.exists(path):
    os.makedirs(path)
np.savetxt(path + "y_t.txt", pred_y)

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

np.random.seed(0)
torch.manual_seed(0)

# Load Data
data = scipy.io.loadmat('pinn_data/burgers/burgers_sine001.mat')

t_star = data['y'].flatten()[:, None]  # np.linspace(0.0, 1.0, 100) or (0.0, 10.0, 101)
x_star = data['x'].flatten()[:, None]  # np.linspace(-1.0, 1.0, 256) or (-8.0, 8.0, 256)
Exact = np.real(data['usol'])[:,:]


### take a subset
t_star = t_star[::10]
x_star = x_star[::4]
Exact = Exact[::4,::10]


T = t_star.shape[0] # 100/10 = 10
N = x_star.shape[0] # 256/4 = 64

print("T", T, "N", N)
print(t_star.shape, x_star.shape, Exact.shape)
#print(t_star, x_star, Exact)


#################################################
#
# construct the graph
#
#################################################


edge_index = []
edge_attr = []
edge_index = []
edge_attr = []

h = 3
for x in range(N):
    for i in range(1, h+1):
        if (x + i < N):
            edge_index.append((x, x + i))
            edge_attr.append(1/N)
            edge_index.append((x + i, x))
            edge_attr.append(-1/N)


x = torch.tensor(x_star, dtype=torch.float)
t = torch.tensor(t_star, dtype=torch.float)
y = torch.tensor(Exact, dtype=torch.float).transpose(0,1)
edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
edge_attr = torch.tensor(edge_attr, dtype=torch.float)
print(x.shape)
print(y.shape)
print(edge_index.shape)
print(edge_attr.shape)
data = Data(x=x, t=t, y=y, edge_index=edge_index, edge_attr=edge_attr)



path = "fenics/burger"

if not os.path.exists(path):
    os.mkdir(path)
    print("Directory ", path, " Created ")
else:
    print("Directory ", path, " already exists")


#################################################
#
# Network
#
#################################################

def weighted_mse_loss(output, target, weight):
    return torch.sum(weight * (output - target) ** 2)


class Net_RNN(torch.nn.Module):
    def __init__(self):
        super(Net_RNN, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 26)
        self.conv1 = GCNConv(32, 32)
        self.conv2 = GCNConv(32, 32)
        # self.conv4 = GCNConv(32, 32)
        # self.conv5 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 26)
        self.fc3 = nn.Linear(26, 32)
        self.fc4 = nn.Linear(32, 1)


    def forward(self, data):
        T, N = data.y.shape
        #print(N, T)
        x, t, edge_index = data.x, data.t, data.edge_index

        boundary = data.y[0,:].view(-1,1)
        res = torch.zeros((T,N)).to(device)

        #print(x.shape, boundary.shape, edge_index.shape, res.shape)
        res[0, :] = boundary.view(1, -1)

        # initialization
        h0 = F.relu(self.fc1(boundary))
        h = F.relu(self.fc2(h0))

        for i in range(1, T):
            t_i = t[i].repeat(N,3)
            x_i = x.repeat((1,3))
            h = torch.cat((h, x_i, t_i), dim=1)
            h = F.relu(self.conv1(h, edge_index))
            h = F.relu(self.conv2(h, edge_index))
            # h = F.relu(self.conv4(h, edge_index))
            # h = F.relu(self.conv5(h, edge_index))
            h = F.relu(self.conv3(h, edge_index))

            y = F.relu(self.fc3(h))
            y = self.fc4(y)
            res[i, :] = y.view(1, -1)

        return res.view(-1)

class Net_MP_RNN(nn.Module):
    def __init__(self):
        super(Net_MP_RNN, self).__init__()

        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 26)

        nn1 = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1024))
        self.conv1 = NNConv(32, 32, nn1, aggr='mean')

        # nn2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1024))
        # self.conv2 = NNConv(32, 32, nn2, aggr='mean')

        nn3 = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 32*26))
        self.conv3 = NNConv(32, 26, nn3, aggr='mean')

        self.fc3 = nn.Linear(26, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, data):
        x, t, edge_index, edge_attr = data.x, data.t, data.edge_index, data.edge_attr
        boundary = data.y[0,:].view(-1,1)
        res = torch.zeros((T,N)).to(device)

        #print(x.shape, boundary.shape, edge_index.shape, res.shape)
        res[0, :] = boundary.view(1, -1)

        # initialization
        h0 = F.relu(self.fc1(boundary))
        h = F.relu(self.fc2(h0))

        for i in range(1, T):
            t_i = t[i].repeat(N,3)
            x_i = x.repeat((1,3))
            h = torch.cat((h, x_i, t_i), dim=1)
            h = F.relu(self.conv1(h, edge_index, edge_attr))
            # h = F.relu(self.conv2(h, edge_index, edge_attr))
            h = F.relu(self.conv3(h, edge_index, edge_attr))

            y = F.relu(self.fc3(h))
            y = self.fc4(y)
            res[i, :] = y.view(1, -1)

        return res.view(-1)


#################################################
#
# train
#
#################################################

n = N * T
num_train_node = n//2

shuffle_idx = np.random.permutation(n)
train_mask = shuffle_idx[:num_train_node]
test_mask = shuffle_idx[num_train_node:]

trivial_error = F.mse_loss(torch.zeros(num_train_node), data.y.reshape(-1)[test_mask])
max_error = torch.max(data.y**2)
print('trivial error:', trivial_error)
print('max error:', max_error)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
target = data.y.reshape(-1).to(device)
model = Net_MP_RNN().to(device)
#model = Net().to(device)
# model = Net_RNN().to(device)
#model = Net_full().to(device)
#model = Net_skip().to(device)

test_loss = []
train_loss = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


model.train()
for epoch in range(10000):
    tic = time.time()

    if epoch == 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    if epoch == 2000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    # if epoch == 4000:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.0001

    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out[train_mask], target[train_mask])
    loss.backward()
    optimizer.step()

    loss2 = torch.sqrt(loss)/torch.sqrt(trivial_error)
    train_loss.append(loss2)

    tac = time.time()
    with torch.no_grad():
        test_error = F.mse_loss(out[test_mask], target[test_mask])
        test_error2 = torch.sqrt(test_error)/torch.sqrt(trivial_error)
        test_loss.append(test_error2)
        print("{0} train: {1:.4f}, {2:.4f}, test: {3:.4f}, {4:.4f}, time: {5:2f}".format(epoch, (loss/trivial_error).item(), (loss2).item(), (test_error/trivial_error).item(), (test_error2).item(), tac-tic))


torch.save(model, path + "/model")

#################################################
#
# plot
#
#################################################

train_loss = np.log10(np.array(train_loss))
test_loss = np.log10(np.array(test_loss))

plt.plot(train_loss)
plt.plot(test_loss)
plt.show()

np.savetxt(path + "/train_loss_rnn_net_full.txt", train_loss)
np.savetxt(path + "/test_loss_rnn_net_full.txt", test_loss)

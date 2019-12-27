import torch
import time
import os
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, NNConv, SplineConv
from torch_geometric.data import InMemoryDataset
import matplotlib.pyplot as plt
import pickle


np.random.seed(0)
torch.manual_seed(0)


def u(x):
    res = 1
    dim = len(x)
    for d in range(dim):
        res = res * np.sin(np.pi * x[d])
    return res

dim = 3
h = 16
n = h**dim
depth = np.int(np.log2(h))
print(dim, h, depth)



pre_index = 0

x = []
y = np.zeros(h**dim)
edge_index= []
edge_attr = []


hierarchy_n = []
hierarchy_l = []
hierarchy_x = []
hierarchy_y = []
hierarchy_edge_index = []
hierarchy_edge_attr = []
for l in range(depth, depth-3, -1):
    h_l = 2**l
    n_l = h_l ** dim
    print("l", l, "h", h_l, "n", n_l)

    hierarchy_n.append(n_l)
    hierarchy_l.append(l)

    xs = np.linspace(0.0, 1.0, h_l)
    xs_grid = []
    edge_index_inner = []
    edge_attr_inner = []
    for i in range(h_l):
        for j in range(h_l):
            for k in range(h_l):
                xs_grid.append((xs[i],xs[j],xs[k]))
                index = i * h_l**2 + j * h_l + k + pre_index

                if (k +1 <= h_l - 1):
                    edge_index_inner.append((index, index + 1))
                    edge_attr_inner.append((1, 0, 0, 0))
                    edge_index_inner.append((index + 1, index))
                    edge_attr_inner.append((-1, 0, 0, 0))
                if (j + 1 <= h_l - 1):
                    edge_index_inner.append((index, index + h_l))
                    edge_attr_inner.append((0, 1, 0, 0))
                    edge_index_inner.append((index + h_l, index))
                    edge_attr_inner.append((0, -1, 0, 0))
                if (i + 1 <= h_l - 1):
                    edge_index_inner.append((index, index + h_l**2))
                    edge_attr_inner.append((0, 0, 1, 0))
                    edge_index_inner.append((index + h_l**2, index))
                    edge_attr_inner.append((0, 0, -1, 0))

    x = x + xs_grid
    print(len(edge_index_inner), len(edge_attr_inner))
    edge_index = edge_index + edge_index_inner
    edge_attr = edge_attr + edge_attr_inner


    ### y, only for the first grid


    if l == depth:
        for i in range(n_l):
            y[i] = u(xs_grid[i])

    ### inter grid edge

    edge_index_inter = []
    edge_attr_inter = []

    if l != depth-2:
        for i in range(h_l):
            i_upgrid = i//2
            for j in range(h_l):
                j_upgrid = j//2
                for k in range(h_l):
                    k_upgrid = k//2

                    index = i * h_l ** 2 + j * h_l + k + pre_index
                    index_upgrid = i_upgrid * (h_l/2)**2 + j_upgrid * (h_l/2) + k_upgrid + n_l + pre_index

                    if (index_upgrid > 4672): print("error",l,i,j,k)

                    edge_index_inter.append((index, index_upgrid))
                    edge_attr_inter.append((0, 0, 0, 1))
                    edge_index_inter.append((index_upgrid, index))
                    edge_attr_inter.append((0, 0, 0, -1))

        print(len(edge_index_inter), len(edge_attr_inter))
        edge_index = edge_index + edge_index_inter
        edge_attr  = edge_attr + edge_attr_inter

    pre_index = pre_index + n_l


print(len(edge_index), len(edge_attr))
x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)
edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0,1)
edge_attr = torch.tensor(edge_attr, dtype=torch.float)

print(x.shape)
print(y.shape)
print(edge_index.shape)
print(edge_attr.shape)

data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)


#################################################
#
# Graph neural network structure
#
#################################################


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dim, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        return x

class Net_MP(nn.Module):
    def __init__(self):
        super(Net_MP, self).__init__()
        nn1 = nn.Sequential(nn.Linear(dim+1, 16), nn.ReLU(), nn.Linear(16, dim * 32))
        self.conv1 = NNConv(dim, 32, nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(dim+1, 16), nn.ReLU(), nn.Linear(16, 1024))
        self.conv2 = NNConv(32, 32, nn2, aggr='mean')

        self.fc1 = torch.nn.Linear(32, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#################################################
#
# train
#
#################################################


num_train_node = n//2

shuffle_idx = np.random.permutation(n)
train_mask = shuffle_idx[:num_train_node]
test_mask = shuffle_idx[num_train_node:]

trivial_error = F.mse_loss(torch.zeros((num_train_node,1)), data.y.view(-1,1)[test_mask])
max_error = torch.max(data.y**2)
print('trivial error:', trivial_error)
print('max error:', max_error)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

model = Net_MP().to(device)
#model = Net_separate().to(device)
# model = Net().to(device)
#model = Net_full().to(device)
#model = Net_skip().to(device)

test_loss = []
train_loss = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


model.train()
for epoch in range(2000):
    tic = time.time()

    if epoch == 500:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005

    if epoch == 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.002

    if epoch == 2000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out[train_mask], data.y.view(-1,1)[train_mask])
    loss.backward()
    optimizer.step()

    loss2 = torch.sqrt(loss)/torch.sqrt(trivial_error)
    train_loss.append(loss2)

    tac = time.time()
    with torch.no_grad():
        test_error = F.mse_loss(out[test_mask], data.y.view(-1,1)[test_mask])
        test_error2 = torch.sqrt(test_error)/torch.sqrt(trivial_error)
        test_loss.append(test_error2)
        print("{0} train: {1:.4f}, {2:.4f}, test: {3:.4f}, {4:.4f}, time: {5:2f}".format(epoch, (loss/trivial_error).item(), (loss2).item(), (test_error/trivial_error).item(), (test_error2).item(), tac-tic))

path = "fenics/multi_res_connected/"

if not os.path.exists(path):
    os.mkdir(path)
    print("Directory ", path, " Created ")
else:
    print("Directory ", path, " already exists")
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

np.savetxt(path + "/train_loss_net.txt", train_loss)
np.savetxt(path + "/test_loss_net.txt", test_loss)

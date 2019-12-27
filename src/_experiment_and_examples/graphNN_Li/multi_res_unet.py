
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


np.random.seed(0)
torch.manual_seed(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

### generate y ###
xs = np.linspace(0.0, 1.0, h)
xs_list = [xs] * dim
xs_grid = np.vstack([xx.ravel() for xx in np.meshgrid(*xs_list)]).T

y = np.zeros(n_l)
for i in range(n_l):
    y[i] = u(xs_grid[i])

inner_graphs = []
inner_graphs_edge_index = []
# inner_graphs_edge_attr = []
for l in range(1, depth+1):
    h_l = 2**l
    n_l = h_l ** dim


    xs = np.linspace(0.0, 1.0, h_l)
    xs_list = [xs] * dim
    xs_grid = np.vstack([xx.ravel() for xx in np.meshgrid(*xs_list)]).T
    print(l, h_l, xs_grid.shape)

    y = np.zeros(n_l)
    for i in range(n_l):
        y[i] = u(xs_grid[i])


    edge_index = []
    edge_attr = []
    for i in range(n_l):
        for j in range(i):
            if (np.linalg.norm(xs_grid[i] - xs_grid[j])) < 1.01 /(h_l-1):
                edge_index.append((i,j))
                edge_index.append((j,i))

    edge_index = np.array(edge_index, dtype=np.int).transpose()
    edge_attr = np.array(edge_attr)

    inner_graphs.append(torch.tensor(xs_grid, dtype=torch.float))

    hierarchy_edge_index.append(torch.tensor(edge_index, dtype=torch.long))
    hierarchy_edge_attr.append(torch.tensor(edge_attr, dtype=torch.float))
    print(edge_index.shape, edge_attr.shape)

#os.mkdir("/Users/lizongyi/Downloads/GNN-PDE/fenics/data_multi_res/")
path = "fenics/data_multi_res/" + str(dim)+str(h)

if not os.path.exists(path):
    os.mkdir(path)
    print("Directory ", path, " Created ")
else:
    print("Directory ", path, " already exists")

pickle.dump(hierarchy_n, open(path + "/hierarchy_n", "wb"))
pickle.dump(hierarchy_l, open(path + "/hierarchy_l", "wb"))
pickle.dump(hierarchy_x, open(path + "/hierarchy_x", "wb"))
pickle.dump(hierarchy_y, open(path + "/hierarchy_y", "wb"))
pickle.dump(hierarchy_edge_index, open(path + "/hierarchy_edge_index", "wb"))
pickle.dump(hierarchy_edge_attr, open(path + "/hierarchy_edge_attr", "wb"))


dim = 3
h = 16
depth = np.int(np.log2(h))
print(dim, h, depth)

#################################################
#
# load data
#
#################################################

path = "fenics/data_multi_res/" + str(dim)+str(h)

hierarchy_n = pickle.load(open(path + "/hierarchy_n", "rb"))
hierarchy_l = pickle.load(open(path + "/hierarchy_l", "rb"))
hierarchy_x = pickle.load(open(path + "/hierarchy_x", "rb"))
hierarchy_y = pickle.load(open(path + "/hierarchy_y", "rb"))
hierarchy_edge_index = pickle.load(open(path + "/hierarchy_edge_index", "rb"))
hierarchy_edge_attr = pickle.load(open(path + "/hierarchy_edge_attr", "rb"))

print(hierarchy_n)
print(hierarchy_l)
print(hierarchy_x[-1].shape)
print(hierarchy_y[-1].shape)
print(hierarchy_edge_index[-1].shape)
print(hierarchy_edge_attr[-1].shape)

depth = len(hierarchy_n)
dataset = []
for l in range(depth):
    data = Data(x=hierarchy_x[l], y=hierarchy_y[l], edge_index=hierarchy_edge_index[l], edge_attr=hierarchy_edge_attr[l]).to(device)

    dataset.append(data)
#################################################
#
# network
#
#################################################

def Upsample(x, dim, channels, scale):
    x = x.transpose(0, 1)
    for i in range(dim):
        x = F.upsample(x.view(1, channels, -1), scale_factor=scale, mode='nearest').view(32, -1)
        channels = channels*2
    x = x.transpose(0, 1)
    return x

class Net_graph(torch.nn.Module):
    def __init__(self):
        super(Net_graph, self).__init__()
        self.conv11 = GCNConv(dim, 32)
        self.conv12 = GCNConv(32, 32)
        self.conv13 = GCNConv(32, 32)

        self.conv21 = GCNConv(dim, 32)
        self.conv22 = GCNConv(32, 32)
        self.conv23 = GCNConv(32, 32)

        self.conv31 = GCNConv(dim, 32)
        self.conv32 = GCNConv(32, 32)
        self.conv33 = GCNConv(32, 32)

        self.conv41 = GCNConv(dim, 32)
        self.conv42 = GCNConv(32, 32)
        self.conv43 = GCNConv(32, 32)

        self.fc = nn.Linear(32,1)

    def forward(self, dataset):
        x1, edge_index1 = dataset[-1].x, dataset[-1].edge_index

        x1 = self.conv11(x1, edge_index1)
        x1 = F.relu(x1)
        x1 = self.conv12(x1, edge_index1)
        x1 = F.relu(x1)
        x1 = self.conv13(x1, edge_index1)

        x2, edge_index2 = dataset[-2].x, dataset[-2].edge_index

        x2 = self.conv21(x2, edge_index2)
        x2 = F.relu(x2)
        x2 = self.conv22(x2, edge_index2)
        x2 = F.relu(x2)
        x2 = self.conv23(x2, edge_index2)

        x3, edge_index3 = dataset[-3].x, dataset[-3].edge_index

        x3 = self.conv31(x3, edge_index3)
        x3 = F.relu(x3)
        x3 = self.conv32(x3, edge_index3)
        x3 = F.relu(x3)
        x3 = self.conv33(x3, edge_index3)

        # x4, edge_index4 = dataset[-4].x, dataset[-4].edge_index
        #
        # x4 = self.conv41(x4, edge_index4)
        # x4 = F.relu(x4)
        # x4 = self.conv42(x4, edge_index4)
        # x4 = F.relu(x4)
        # x4 = self.conv43(x4, edge_index4)


        x2 = Upsample(x2, channels=32, dim=dim, scale=2)
        x3 = Upsample(x3, channels=32, dim=dim, scale=4)
        # x4 = Upsample(x4, channels=32, dim=dim, scale=8)


        #x1 = x1 + x2.view(-1, 32) + x3.view(-1, 32) + x4.view(-1, 32)
        #x1 = x1 + x2.view(-1, 32) + x3.view(-1, 32)
        x = F.relu(x1)
        x = self.fc(x)
        return x


class Net_MP(nn.Module):
    def __init__(self):
        super(Net_MP, self).__init__()
        nn1 = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, dim * 32))
        nn1_short = nn.Linear(dim, dim * 32)
        self.conv1 = NNConv(dim, 32, nn1_short, aggr='add')

        nn2 = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, 1024))
        nn2_short = nn.Linear(dim, 1024)
        self.conv2 = NNConv(32, 32, nn2_short, aggr='add')

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

n = hierarchy_n[-1]
y = hierarchy_y[-1].to(device)

num_train_node = n//2

shuffle_idx = np.random.permutation(n)
train_mask = shuffle_idx[:num_train_node]
test_mask = shuffle_idx[num_train_node:]

trivial_error = F.mse_loss(torch.zeros((num_train_node,1)).to(device), y.view(-1,1)[test_mask])
max_error = torch.max(y**2)
print('trivial error:', trivial_error)
print('max error:', max_error)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = Net_MP().to(device)
model = Net_graph().to(device)
#model = Net_full().to(device)
#model = Net_skip().to(device)

test_loss = []
train_loss = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

#dataset = dataset.to(device)
model.train()
for epoch in range(500):
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
    out = model(dataset)
    loss = F.mse_loss(out[train_mask], y.view(-1,1)[train_mask])
    loss2 = torch.sqrt(loss)/torch.sqrt(trivial_error)
    train_loss.append(loss2)

    loss.backward()
    optimizer.step()

    tac = time.time()
    with torch.no_grad():
        test_error = F.mse_loss(out[test_mask], y.view(-1,1)[test_mask])
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

np.savetxt(path + "/train_loss_net.txt", train_loss)
np.savetxt(path + "/test_loss_net.txt", test_loss)

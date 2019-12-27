import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.data import InMemoryDataset
import matplotlib.pyplot as plt
import random

#################################################
#
# Networks
#
#################################################


def u(x):
    res = 1
    dim = len(x)
    for d in range(dim):
        res = res * np.pi * np.sin(np.pi * x[d])
    return res

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(4, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

        #return F.log_softmax(x, dim=1)

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
# load model
#
#################################################

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)



dim = 5
features = 0
n_b = 1000
n_i = 4000
n = n_b + n_i
m = n_b//dim
h = n**(1/dim)

path = "/Users/lizongyi/Downloads/GNN-PDE/fenics/data_sin_high/" + str(dim)
x = np.loadtxt(path + "/x.txt")
y = np.loadtxt(path + "/y.txt")
#z = np.loadtxt(path + "/z.txt")
#x = np.concatenate((x,z), axis = 1)
edge_index = np.loadtxt(path + "/edge_index.txt")
edge_attr = np.loadtxt(path + "/edge_attr.txt")

print(x.shape, y.shape, edge_index.shape, edge_attr.shape)



#################################################
#
# update graph
#
#################################################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net_MP().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# model.load_state_dict(torch.load(path + "/model"))

model = torch.load(path + "/model")
model.eval()


m = 100

test_x = np.random.rand(m,dim)
test_y = np.zeros(m)
for i in range(m):
    test_y[i] = u(test_x[i])

test_edge_index = []
test_edge_attr = []
counter = 0
for j in range(m):
    for i in range(n):
        if (np.linalg.norm(x[i,:dim] - test_x[j,:dim]) <= 1.21/h) :
            test_edge_index.append((i, n+j))
            test_edge_attr.append(x[i] - test_x[j])
            test_edge_index.append((n+j, i))
            test_edge_attr.append(test_x[j] - x[i])

            counter = counter + 1

print(counter)
test_edge_index = np.array(test_edge_index, dtype=np.int16).transpose()
test_edge_attr = np.array(test_edge_attr)
print(edge_index.shape, edge_attr.shape)
print(test_edge_index.shape, test_edge_attr.shape)
edge_index = np.concatenate((edge_index,test_edge_index), axis=1)
edge_attr = np.concatenate((edge_attr,test_edge_attr), axis=0)


test_x = torch.tensor(test_x, dtype=torch.float)
test_y = torch.tensor(test_y, dtype=torch.float)
print(test_x.shape, test_y.shape)
x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)
print(x.shape, test_y.shape)
x = torch.cat((x, test_x), dim=0)
y = torch.cat((y, test_y), dim=0)
print(x.shape, y.shape)

edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_attr = torch.tensor(edge_attr, dtype=torch.float)
print(edge_index.shape, edge_attr.shape)

new_graph = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

#################################################
#
# eval
#
#################################################


model.eval()
pred = model(new_graph)
ori_loss = F.mse_loss(pred[:n], y.view(-1, 1)[:n])
loss = F.mse_loss(pred[n:], y.view(-1, 1)[n:])
max_error = torch.max(y**2)
print(ori_loss/max_error)
print(loss/max_error)


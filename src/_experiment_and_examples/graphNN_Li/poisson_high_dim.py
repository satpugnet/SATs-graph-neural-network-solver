import torch
import time
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, NNConv, SplineConv
from torch_geometric.data import InMemoryDataset
import matplotlib.pyplot as plt


np.random.seed(0)
torch.manual_seed(0)

#################################################
#
# generate analytic PDE data
#
#################################################
def weighted_mse_loss(output, target, weight):
    print(output/weight)
    print(target/weight)
    return torch.mean((output/weight - target/weight) ** 2)

def u(x):
    res = 1
    dim = len(x)
    for d in range(dim):
        res = res * np.pi * np.sin(np.pi * x[d])
    return res

# def f(x, y):
#     return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


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

x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)
edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_attr = torch.tensor(edge_attr, dtype=torch.float)

dataset = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

print(x.shape, y.shape, edge_index.shape)


#################################################
#
# GNN structure
#
#################################################
dim = dim + features
class Net_graph(torch.nn.Module):
    def __init__(self):
        super(Net_graph, self).__init__()
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

class Net_skip(torch.nn.Module):
    def __init__(self):
        super(Net_skip, self).__init__()
        self.conv1 = GCNConv(dim, 32-dim)
        self.conv2 = GCNConv(32, 32-dim)
        self.conv3 = GCNConv(32, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = torch.cat((x,data.x),dim=1)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.cat((x, data.x),dim=1)
        x = self.conv3(x, edge_index)
        return x

class Net_full(torch.nn.Module):
    def __init__(self):
        super(Net_full, self).__init__()
        self.conv1 = GCNConv(dim, 32)
        self.conv2 = GCNConv(32, 32)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
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


num_train_node = n//2

shuffle_idx = np.random.permutation(n)
train_mask = shuffle_idx[:num_train_node]
test_mask = shuffle_idx[num_train_node:]

trivial_error = F.mse_loss(torch.zeros((num_train_node,1)), y.view(-1,1)[test_mask])
max_error = torch.max(y**2)
print('trivial error:', trivial_error)
print('max error:', max_error)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net_MP().to(device)
#model = Net_graph().to(device)
#model = Net_full().to(device)
#model = Net_skip().to(device)

test_loss = []
train_loss = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

data = dataset.to(device)
model.train()
for epoch in range(1000):
    tic = time.time()
    #
    if epoch == 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005

    if epoch == 500:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.002

    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out[train_mask], data.y.view(-1,1)[train_mask])
    #loss = weighted_mse_loss(target=data.y.view(-1,1)[train_mask], output=out[train_mask], weight=data.y.view(-1,1)[train_mask] )
    # for name, W in model.named_parameters():
    #     print(name, W)

    loss.backward()
    optimizer.step()
    train_loss.append(loss)

    tac = time.time()
    with torch.no_grad():
        test_error = F.mse_loss(out[test_mask], data.y.view(-1,1)[test_mask])
        #test_error = weighted_mse_loss(target=data.y.view(-1,1)[test_mask], output=out[test_mask], weight=data.y.view(-1,1)[test_mask] )
        test_loss.append(test_error)
        print(epoch, "train:", loss/trivial_error, loss/max_error, "test:", test_error/trivial_error, test_error/max_error, "time:", round(tac-tic, 4))

model.eval()
pred = model(data)
error = F.mse_loss(pred[test_mask], data.y.view(-1,1)[test_mask])/max_error
print('test L2 error: {:.4f}'.format(error))



torch.save(model, path + "/model")

#################################################
#
# plot
#
#################################################

train_loss = np.log(np.array(train_loss) / max_error.numpy())
test_loss = np.log(np.array(test_loss) / max_error.numpy())

plt.plot(train_loss)
plt.plot(test_loss)
plt.show()

np.savetxt(path + "/train_loss_mp1_net.txt", train_loss)
np.savetxt(path + "/test_loss_mp1_net.txt", test_loss)


#################################################
#
# functionality to add node
#
#################################################

# m = 100
# dim = 2
#
#
# n = data.x.shape[0]
# h = int(np.sqrt(n)) - 1
# print(n,h)
#
# test_x = np.random.rand(m,2)
# test_y = np.zeros((m,1))
# for i in range(m):
#     test_y[i,0] = u(test_x[i,0], test_x[i,1])
#
# # construct new graph
# test_x = torch.tensor(test_x, dtype=torch.float)
# test_y = torch.tensor(test_y, dtype=torch.float)
#
# x = torch.cat((data.x, test_x), dim =0)
# #y = torch.cat((data.y, torch.tensor([0.])), dim =0)
# edge_index = data.edge_index
#
# counter = 0
# for j in range(m):
#     for i in range(n):
#         #print(x[i, :dim] - node_x[:dim], torch.norm(x[i,:dim] - node_x[:dim]))
#         if (torch.norm(x[i,:dim] - test_x[j,:dim]) <= 1.001/h) :
#
#             edge1 = torch.tensor([i,n+j], dtype=torch.long).reshape(2,1)
#             edge2 = torch.tensor([n+j,i], dtype=torch.long).reshape(2,1)
#             edge_index = torch.cat((edge_index, edge1, edge2), dim=1)
#             counter = counter + 1
#
# print(counter)
# new_graph = Data(x=x, y=y, edge_index=edge_index)
#
# model.train()
# for epoch in range(10):
#     optimizer.zero_grad()
#     out = model(new_graph)
#     loss = F.mse_loss(out[train_mask], new_graph.y.view(-1, 1)[train_mask])
#     print(epoch, loss, F.mse_loss(out[n : ], test_y.view(-1, 1)))
#     loss.backward()
#     optimizer.step()
#
# model.eval()
# pred = model(new_graph)
#



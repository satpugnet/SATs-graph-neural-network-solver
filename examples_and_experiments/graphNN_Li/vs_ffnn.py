import torch
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

def u (x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f (x, y):
    return 2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

n = 1000
x_int = np.random.rand(n,2)

x_out1 = np.random.rand(n//2,1)
x_out2 = np.random.rand(n//2,1)
zeros = np.zeros((n//2, 1))
x_out1 = np.concatenate((x_out1, zeros), axis=1)
x_out2 = np.concatenate((zeros, x_out2), axis=1)
x_out = np.concatenate((x_out1,x_out2), axis=0)


print(x_out.shape)
y_int = np.zeros((n,1))
y_out = np.zeros((n,1))
for i in range(n):
    y_int[i] = u(x_int[i,0], x_int[i,1]) #f(x_int[i,0], x_int[i,1])
    y_out[i] = u(x_out[i,0], x_out[i,1])

x = np.concatenate((x_int,x_out), axis=0)
y = np.concatenate((y_int,y_out), axis=0)


#################################################
#
# generate edges
#
#################################################


h = 0.05
counter = 0
edges = []
edge_attr = []
for i in range(n):
    for j in range(i):
        if (np.linalg.norm(x[i] - x[j]) <= h):
            counter = counter + 1
            # print(counter, i,j, x[i], x[j])
            edges.append((i, j))
            edge_attr.append(x[i] - x[j])
            edges.append((j, i))
            edge_attr.append(x[j] - x[i])

print(counter)
edges = np.array(edges).transpose()
edge_index = np.array(edges, dtype=np.int16)
edge_attr = np.array(edge_attr)


print(x.shape, y.shape, edge_index.shape, edge_attr.shape)



x = torch.tensor(x, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)
edge_index = torch.tensor(edges, dtype=torch.long)
edge_attr = torch.tensor(edge_attr, dtype=torch.float)
dataset = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)



#################################################
#
# GNN structure
#
#################################################

class Net_graph(torch.nn.Module):
    def __init__(self):
        super(Net_graph, self).__init__()
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 16)
        # self.conv21 = GCNConv(16, 16)
        # self.conv22 = GCNConv(16, 16)
        # self.conv23 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 1)


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
        self.conv1 = GCNConv(2, 30)
        self.conv2 = GCNConv(32, 30)
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
        self.conv1 = GCNConv(2, 32)
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
        nn1 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 64))
        nn1_short = nn.Linear(2, 64)
        self.conv1 = NNConv(2, 32, nn1_short, aggr='add')

        nn2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1024))
        nn2_short = nn.Linear(2, 1024)
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



n = 2*n
num_train_node = int(n/2)
shuffle_idx = np.random.permutation(n)
train_mask = shuffle_idx[:num_train_node]
test_mask = shuffle_idx[num_train_node:]

print('trivial error:', F.mse_loss(torch.zeros((num_train_node ,1)), y.view(-1,1)[test_mask]))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Model = Net_MP().to(device)
model = Net_graph().to(device)
#model = Net_full().to(device)
#model = Net_skip().to(device)

test_loss = []
train_loss = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

data = dataset.to(device)
model.train()
for epoch in range(5000):
    #
    # if epoch == 200:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.001

    if epoch == 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out[train_mask], data.y.view(-1,1)[train_mask])
    # for name, W in model.named_parameters():
    #     print(name, W)

    loss.backward()
    optimizer.step()
    train_loss.append(loss)

    with torch.no_grad():
        test_error = F.mse_loss(out[test_mask], data.y.view(-1,1)[test_mask])
        test_loss.append(test_error)
        print(epoch, loss, test_error)

model.eval()
pred = model(data)
error = F.mse_loss(pred[test_mask], data.y.view(-1,1)[test_mask])
print('test L2 error: {:.4f}'.format(error))



#torch.save(model, "/Users/lizongyi/Downloads/GNN-PDE/fenics/model")

#################################################
#
# plot
#
#################################################

train_loss = np.log(np.array(train_loss))
test_loss = np.log(np.array(test_loss))

plt.plot(train_loss)
plt.plot(test_loss)
plt.show()


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


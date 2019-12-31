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
data = scipy.io.loadmat('pinn_data/burgers_shock.mat')

t_star = data['y'].flatten()[:, None]  # np.linspace(0.0, 1.0, 100)
x_star = data['x'].flatten()[:, None]  # np.linspace(-1.0, 1.0, 256)
Exact = np.real(data['usol'])


### take a subset
# t_star = t_star[::10]
# x_star = x_star[::4]
# Exact = Exact[::4, ::10]
# t_star = t_star[:96]
# Exact = Exact[:, :96]

T = t_star.shape[0]
N = x_star.shape[0]
h_t = 1/T
h_x = 1/N
print(t_star.shape, x_star.shape, Exact.shape)
#print(t_star, x_star, Exact)


#################################################
#
# construct the graph
#
#################################################

grid = np.vstack([xx.ravel() for xx in np.meshgrid(x_star,t_star)]).T

edge_index_space = []
edge_index_time_forward = []
edge_index_time_backward = []
edge_attr_space = []
edge_attr_time_forward = []
edge_attr_time_backward = []
boundary_index = []
interior_index = []
counter = 0
for t in range(T):
    for x in range(N):
        i = t*N + x
        if(x != N-1):
            edge_index_space.append((i, i + 1))
            edge_attr_space.append((h_x, 0))
            edge_index_space.append((i + 1, i))
            edge_attr_space.append((-h_x, 0))

        if(t != T-1):
            edge_index_time_forward.append((i, i+N))
            edge_attr_time_forward.append((0, h_t))
            edge_index_time_backward.append((i+N, i))
            edge_attr_time_backward.append((0, -h_t))

        if(t==0 or x==0 or x==N-1):
            boundary_index.append(i)
            counter = counter + 1
        else:
            interior_index.append(i)
print(counter)

edge_index = edge_index_space + edge_index_time_forward
edge_attr = edge_attr_space + edge_attr_time_forward
edge_index_full = edge_index + edge_index_time_backward
edge_attr_full = edge_attr + edge_attr_time_backward


path = "fenics/burger/"

if not os.path.exists(path):
    os.mkdir(path)
    print("Directory ", path, " Created ")
else:
    print("Directory ", path, " already exists")

grid = torch.tensor(grid, dtype=torch.float)
Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0,1)
edge_attr = torch.tensor(edge_attr, dtype=torch.float)
edge_index_full = torch.tensor(edge_index_full, dtype=torch.long).transpose(0,1)
edge_attr_full = torch.tensor(edge_attr_full, dtype=torch.float)
boundary_index = torch.tensor(boundary_index, dtype=torch.long)
interior_index = torch.tensor(interior_index, dtype=torch.long)


data = Data(x=grid, y=Exact, edge_index=edge_index, edge_attr=edge_attr, boundary_index=boundary_index, interior_index=interior_index )
data_full = Data(x=grid, y=Exact, edge_index=edge_index_full, edge_attr=edge_attr_full, boundary_index=boundary_index, interior_index=interior_index)

pickle.dump(grid, open(path + "/grid", "wb"))
pickle.dump(Exact, open(path + "/Exact", "wb"))
pickle.dump(edge_index, open(path + "/edge_index", "wb"))
pickle.dump(edge_attr, open(path + "/edge_attr", "wb"))
pickle.dump(edge_index_full, open(path + "/edge_index_full", "wb"))
pickle.dump(edge_attr_full, open(path + "/edge_attr_full", "wb"))

print(grid.shape)
print(Exact.shape)
print(edge_index.shape)
print(edge_attr.shape)
print(edge_index_full.shape)
print(edge_attr_full.shape)


#################################################
#
# Network
#
#################################################

def weighted_mse_loss(output, target, weight):
    return torch.sum(weight * (output - target) ** 2)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.conv1 = GCNConv(32, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        # self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        # x = self.fc2(x)
        # x = F.relu(x)
        x = self.fc3(x)
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

        nn3 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1024))
        nn3_short = nn.Linear(2, 1024)
        self.conv3 = NNConv(32, 32, nn3_short, aggr='add')

        self.fc1 = torch.nn.Linear(32, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net_separate(torch.nn.Module):
    def __init__(self):
        super(Net_separate, self).__init__()
        self.fc_boundary1 = nn.Linear(3, 20)
        self.fc_boundary2 = nn.Linear(20, 20)
        self.fc_interior1 = nn.Linear(2, 20)
        self.fc_interior2 = nn.Linear(20, 20)
        self.conv1 = GCNConv(20, 20)
        self.conv2 = GCNConv(20, 20)
        self.conv3 = GCNConv(20, 20)
        self.conv4 = GCNConv(20, 20)
        self.conv5 = GCNConv(20, 20)
        self.conv6 = GCNConv(20, 20)
        self.fc1 = nn.Linear(20, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, data):
        x, y, edge_index, boundary_index, interior_index = data.x, data.y, data.edge_index, data.boundary_index, data.interior_index

        #print(x, boundary_index, edge_index)
        x_boundary = torch.cat((x[boundary_index], y[boundary_index].view(-1,1)), dim=1)
        x_boundary = self.fc_boundary1(x_boundary)
        x_boundary = F.relu(x_boundary)
        x_boundary = self.fc_boundary2(x_boundary)

        x_interior = x[interior_index]
        x_interior = self.fc_interior1(x_interior)
        x_interior = F.relu(x_interior)
        x_interior = self.fc_interior2(x_interior)

        x = torch.zeros((x.shape[0],x_boundary.shape[1])).to(device)
        x[boundary_index] = x_boundary
        x[interior_index] = x_interior

        x = F.relu(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = F.relu(self.conv5(x, edge_index))
        x = F.relu(self.conv6(x, edge_index))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net_RNN(torch.nn.Module):
    def __init__(self):
        super(Net_RNN, self).__init__()
        self.fc1 = nn.Linear(2, 20)
        self.conv1 = GCNConv(20, 20)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

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

trivial_error = F.mse_loss(torch.zeros((num_train_node,1)), data.y.view(-1,1)[test_mask])
max_error = torch.max(data.y**2)
print('trivial error:', trivial_error)
print('max error:', max_error)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

#model = Net_MP().to(device)
#model = Net_separate().to(device)
model = Net().to(device)
#model = Net_full().to(device)
#model = Net_skip().to(device)

test_loss = []
train_loss = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

weight = torch.ones(n).to(device)
weight[boundary_index] = weight[boundary_index]/counter
weight[interior_index] = weight[interior_index]/(n-counter)

print(weight)

model.train()
for epoch in range(2000):
    tic = time.time()

    if epoch == 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005

    if epoch == 500:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.002

    if epoch == 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out[train_mask], data.y.view(-1,1)[train_mask])
    loss_weighted = weighted_mse_loss(output=out[train_mask].view(-1), target=data.y[train_mask], weight=weight[train_mask])

    loss.backward()
    #loss_weighted.backward()
    loss2 =  torch.sqrt(loss)/torch.sqrt(trivial_error)
    optimizer.step()
    train_loss.append(loss2)

    tac = time.time()
    with torch.no_grad():
        test_error = F.mse_loss(out[test_mask], data.y.view(-1,1)[test_mask])
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

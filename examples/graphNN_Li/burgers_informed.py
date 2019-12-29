import torch
import os
import time
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable, grad
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


## take a subset
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
        self.fc3 = nn.Linear(32, 1)


    def forward(self, data):
        x, t, edge_index = data.x, data.t, data.edge_index
        X = torch.cat((x.view(-1, 1), t.view(-1, 1)), dim=1)

        X = self.fc1(X)
        X = F.relu(X)
        X = F.relu(self.conv1(X, edge_index))
        X = F.relu(self.conv2(X, edge_index))
        X = F.relu(self.conv3(X, edge_index))
        X = self.fc3(X)
        return X.view(-1)



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
u_target = data.y.reshape(-1).to(device)
model = Net().to(device)
# model = Net_MP().to(device)
#model = Net_full().to(device)
#model = Net_skip().to(device)

test_loss = []
train_loss = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)



dataset = []

x = Variable(data.x[:, 0], requires_grad=True)
t = Variable(data.x[:, 1], requires_grad=True)
data = Data(x=x, t=t, edge_index=edge_index).to(device)
size = n
# x_repeat = x.repeat(size, 1)
# x_repeat.requires_grad_(True)
# t_repeat = y.repeat(size, 1)
# t_repeat.requires_grad_(True)
#
# for i in range(size):
#     data = Data(x=x_repeat[i], y=t_repeat[i], edge_index=edge_index).to(device)
#     dataset.append(data)
#
# loader = DataLoader(dataset, batch_size=size)

model.train()
for epoch in range(500):
    tic = time.time()

    # if epoch == 200:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.001
    #
    # if epoch == 1000:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.0001

    # if epoch == 4000:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.0001


    # for batch in loader:
    #     u_repeat = model(batch)
    # u = u_repeat[:size]

    model.zero_grad()
    # u_t_jocabian = grad(u_repeat, t_repeat, grad_outputs=torch.eye(size).view(-1).to(device),retain_graph=True)[0]
    # u_t = torch.diag(u_t_jocabian)
    # u_x_jocabian = grad(u_repeat, x_repeat, grad_outputs=torch.eye(size).view(-1).to(device),retain_graph=True)[0]
    # u_x = torch.diag(u_x_jocabian)
    # # u_xx_jocabian = grad(torch.diag(u_x).view(-1), x_repeat, grad_outputs=torch.eye(size).view(-1).to(device),  create_graph=True, retain_graph=True)[0]
    # # u_xx = torch.diag(u_x_jocabian)


    # size = u.shape[0]
    u = model(data)
    u_t = torch.zeros(size).to(device)
    u_x = torch.zeros(size).to(device)
    u_xx = torch.zeros(size).to(device)
    for i in range(size):
        u_t[i] = grad(u[i], t,  create_graph=True, retain_graph=True)[0][i]
        u_x[i] = grad(u[i], x,  create_graph=True, retain_graph=True)[0][i]
        # u_xx[i] = grad(u_x[i], x,  create_graph=True, retain_graph=True)[0][i]
        # print(u_t[i], u_x[i], u_xx[i])


    f = u_t + u * u_x  #- (0.01 / np.pi) * u_xx

    optimizer.zero_grad()
    loss0 = F.mse_loss(u[train_mask], u_target[train_mask])
    loss1 = torch.mean(f[train_mask]**2)
    loss = loss0 + loss1
    loss.backward()
    optimizer.step()

    loss2 = torch.sqrt(F.mse_loss(u[train_mask], u_target[train_mask]))/torch.sqrt(trivial_error)
    train_loss.append(loss2)

    tac = time.time()
    with torch.no_grad():
        test_error = F.mse_loss(u[test_mask], u_target[test_mask]) + torch.sum(f[test_mask]**2)
        test_error2 = torch.sqrt(F.mse_loss(u[test_mask], u_target[test_mask]))/torch.sqrt(trivial_error)
        test_loss.append(test_error2)
        print("{0} train: {1:.4f} + {2:.6f} = {3:.4f}, {4:.4f}, test: {5:.4f}, {6:.4f}, time: {7:2f}".format(epoch, loss0.item(), loss1.item(), (loss).item(), (loss2).item(), (test_error).item(), (test_error2).item(), tac-tic))


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

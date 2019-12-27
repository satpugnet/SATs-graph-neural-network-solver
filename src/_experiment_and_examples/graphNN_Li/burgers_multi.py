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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data
data = scipy.io.loadmat('pinn_data/burgers_shock.mat')

t_star = data['y'].flatten()[:, None]  # np.linspace(0.0, 1.0, 100)
x_star = data['x'].flatten()[:, None]  # np.linspace(-1.0, 1.0, 256)
Exact = np.real(data['usol'])


### take a subset
# t_star = t_star[::10]
# x_star = x_star[::4]
# Exact = Exact[::4, ::10]
t_star = t_star[:96]
Exact = Exact[:, :96]

T = t_star.shape[0]
N = x_star.shape[0]
h_t = 1/T
h_x = 1/N
print(t_star.shape, x_star.shape, Exact.shape)
#print(t_star, x_star, Exact)


#################################################
#
# construct multi graphs
#
#################################################

dim = 2
depth = 5
print(dim, T, N, depth)

dataset = []
for l in range(depth):
    h_x_l = 16 * 2**l
    h_t_l = 6 * 2**l
    n_l = h_x_l * h_t_l

    xs = np.linspace(0.0, 1.0, h_x_l)
    ts = np.linspace(0.0, 1.0, h_t_l)
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ts)]).T
    print(l, h_x_l, h_t_l, grid.shape)

    y = Exact[::2**(depth-l-1), ::2**(depth-l-1)]

    edge_index = []
    edge_attr = []
    edge_index = []
    edge_attr = []
    boundary_index = []
    interior_index = []
    counter = 0
    for t in range(h_t_l):
        for x in range(h_x_l):
            i = t * h_x_l + x
            if (x != h_x_l - 1):
                edge_index.append((i, i + 1))
                edge_attr.append((h_x, 0))
                edge_index.append((i + 1, i))
                edge_attr.append((-h_x, 0))

            if (t != h_t_l - 1):
                edge_index.append((i, i + h_x_l))
                edge_attr.append((0, h_t))

            if (t == 0 or x == 0 or x == h_x_l - 1):
                boundary_index.append(i)
                counter = counter + 1
            else:
                interior_index.append(i)
    print(counter)

    x = torch.tensor(grid, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0,1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    print(x.shape)
    print(y.shape)
    print(edge_index.shape)
    print(edge_attr.shape)
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, boundary_index=boundary_index, interior_index=interior_index).to(device)
    dataset.append(data)


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
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.fc1(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
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
        self.fc_boundary1 = nn.Linear(3, 32)
        self.fc_boundary2 = nn.Linear(32, 32)
        self.fc_interior1 = nn.Linear(2, 32)
        self.fc_interior2 = nn.Linear(32, 32)
        self.conv1 = GCNConv(32, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.fc = nn.Linear(32, 1)

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
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x


def Upsample(x, dim, channels, scale):
    x = x.transpose(0, 1)
    for i in range(dim):
        x = F.upsample(x.view(1, channels, -1), scale_factor=scale, mode='nearest').view(32, -1)
        channels = channels*2
    x = x.transpose(0, 1)
    return x

class Net_mul(torch.nn.Module):
    def __init__(self):
        super(Net_mul, self).__init__()
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

        self.conv51 = GCNConv(dim, 32)
        self.conv52 = GCNConv(32, 32)
        self.conv53 = GCNConv(32, 32)

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

        x4, edge_index4 = dataset[-4].x, dataset[-4].edge_index

        x4 = self.conv41(x4, edge_index4)
        x4 = F.relu(x4)
        x4 = self.conv42(x4, edge_index4)
        x4 = F.relu(x4)
        x4 = self.conv43(x4, edge_index4)

        x5, edge_index5 = dataset[-5].x, dataset[-5].edge_index

        x5 = self.conv51(x5, edge_index5)
        x5 = F.relu(x5)
        x5 = self.conv52(x5, edge_index5)
        x5 = F.relu(x5)
        x5 = self.conv53(x5, edge_index5)


        x2 = Upsample(x2, channels=32, dim=dim, scale=2)
        x3 = Upsample(x3, channels=32, dim=dim, scale=4)
        x4 = Upsample(x4, channels=32, dim=dim, scale=8)
        x5 = Upsample(x5, channels=32, dim=dim, scale=16)


        x1 = x1 + x2.view(-1, 32) + x3.view(-1, 32) + x4.view(-1, 32) + x5.view(-1,32)
        #x1 = x1 + x2.view(-1, 32) + x3.view(-1, 32)
        x = F.relu(x1)
        x = self.fc(x)
        return x

#################################################
#
# train
#
#################################################

n = N * T
num_train_node = n//2
y = torch.tensor(Exact, dtype=torch.float).view(-1)

shuffle_idx = np.random.permutation(n)
train_mask = shuffle_idx[:num_train_node]
test_mask = shuffle_idx[num_train_node:]



trivial_error = F.mse_loss(torch.zeros((num_train_node,1)), y.view(-1,1)[test_mask])
max_error = torch.max(y**2)
print('trivial error:', trivial_error)
print('max error:', max_error)


# data = dataset.to(device)

#model = Net_MP().to(device)
#model = Net_separate().to(device)
#model = Net().to(device)
model = Net_mul().to(device)
#model = Net_skip().to(device)

test_loss = []
train_loss = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
y = y.to(device)
weight = torch.ones(n).to(device)
weight[boundary_index] = weight[boundary_index]/counter
weight[interior_index] = weight[interior_index]/(n-counter)

print(weight)

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
    out = model(dataset)
    loss = F.mse_loss(out[train_mask], y.view(-1,1)[train_mask])
    loss_weighted = weighted_mse_loss(output=out[train_mask].view(-1), target=y[train_mask], weight=weight[train_mask])

    loss.backward()
    #loss_weighted.backward()
    optimizer.step()
    train_loss.append(loss)

    tac = time.time()
    with torch.no_grad():
        test_error = F.mse_loss(out[test_mask], y.view(-1,1)[test_mask])
        test_loss.append(test_error)
        print("{0} train: {1:.4f}, {2:.4f}, test: {3:.4f}, time: {4:2f}".format(epoch, (loss/trivial_error).item(), (loss_weighted/trivial_error).item(), (test_error/trivial_error).item(), tac-tic))


path = "fenics/burger/"

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

train_loss = np.log10(np.array(train_loss) / max_error.numpy())
test_loss = np.log10(np.array(test_loss) / max_error.numpy())

plt.plot(train_loss)
plt.plot(test_loss)
plt.show()

np.savetxt(path + "/train_loss_m_net.txt", train_loss)
np.savetxt(path + "/test_loss_m_net.txt", test_loss)

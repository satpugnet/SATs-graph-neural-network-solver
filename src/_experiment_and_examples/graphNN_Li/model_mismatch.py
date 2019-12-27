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



def u(x, coeff):
    return np.sin(np.pi * coeff[0] * x[0]) * np.sin(np.pi * coeff[1] * x[1])

h = 50
dim = 2
n = h**dim

xs = np.linspace(0.0, 1.0, h)
xs_list = [xs] * dim
x = np.vstack([xx.ravel() for xx in np.meshgrid(*xs_list)]).T


print(h, n, x.shape)
y1 = np.zeros(n)
y2 = np.zeros(n)
coeff1 = np.array([2.1,2.1])
coeff2 = np.array([2,2])
for i in range(n):
    y1[i] = u(x[i], coeff1)
    y2[i] = u(x[i], coeff2)

edge_index = []
for i in range(h):
    for j in range(h):
        index = i*h + j
        if j+1 < h:
            edge_index.append((index,index+1))
            edge_index.append((index+1,index))
        if i+1 < h:
            edge_index.append((index,index+h))
            edge_index.append((index+h,index))

print(edge_index)
x = torch.tensor(x, dtype=torch.float)
y1 = torch.tensor(y1, dtype=torch.float)
y2 = torch.tensor(y2, dtype=torch.float)
edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0,1)
print(edge_index)
data = Data(x=x,y1=y1,y2=y2,edge_index=edge_index)

#################################################
#
# Graph neural network structure
#
#################################################


class Net_base(torch.nn.Module):
    def __init__(self):
        super(Net_base, self).__init__()
        self.conv1 = GCNConv(2, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x.view(-1)



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(3, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 1)

    def forward(self, data):
        x, y1, edge_index = data.x, data.y1, data.edge_index

        x = torch.cat((x,y1.view(-1,1)), dim=1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x.view(-1)


class Net_skip(torch.nn.Module):
    def __init__(self):
        super(Net_skip, self).__init__()
        self.conv1 = GCNConv(3, 29)
        self.conv2 = GCNConv(32, 29)
        self.conv3 = GCNConv(32, 1)


    def forward(self, data):
        x, y1, edge_index = data.x, data.y1, data.edge_index

        input = torch.cat((x,y1.view(-1,1)), dim=1)

        x = self.conv1(input, edge_index)
        x = F.relu(x)
        x = torch.cat((x, input),dim=1)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.cat((x, input),dim=1)
        x = self.conv3(x, edge_index)
        return x.view(-1)


#################################################
#
# train
#
#################################################

residual = data.y2 - data.y1
target = data.y2


num_train_node = n//2

shuffle_idx = np.random.permutation(n)
train_mask = shuffle_idx[:num_train_node]
test_mask = shuffle_idx[num_train_node:]

# trivial_error = F.mse_loss(torch.zeros(num_train_node), residual[test_mask])
# max_error = torch.max(residual**2)
# print('trivial error:', trivial_error)
# print('max error:', max_error)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

residual = residual.to(device)
target = target.to(device)

model = Net().to(device)
model_base = Net_base().to(device)
# model = Net_RNN().to(device)
#model = Net_full().to(device)
# model = Net_skip().to(device)

test_loss = []
train_loss = []
test_loss_base = []
train_loss_base = []

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
optimizer_base = torch.optim.Adam(model_base.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(1000):
    tic = time.time()

    if epoch == 400:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
        for param_group in optimizer_base.param_groups:
            param_group['lr'] = 0.001

    if epoch == 2000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
        for param_group in optimizer_base.param_groups:
            param_group['lr'] = 0.0001

    optimizer.zero_grad()
    optimizer_base.zero_grad()
    out = model(data)
    out_base = model_base(data)

    loss = F.mse_loss(out[train_mask], residual[train_mask])
    loss.backward()
    optimizer.step()
    loss_base = F.mse_loss(out_base[train_mask], target[train_mask])
    loss_base.backward()
    optimizer_base.step()

    train_loss.append(loss)
    train_loss_base.append(loss_base)


    tac = time.time()
    with torch.no_grad():
        test_error = F.mse_loss(out[test_mask], residual[test_mask])
        test_error_base = F.mse_loss(out_base[test_mask], target[test_mask])
        test_loss.append(test_error)
        test_loss_base.append(test_error_base)
        print("{0} train: {1:.4f}, {2:.4f}, test: {3:.4f}, {4:.4f}, time: {5:2f}".format(epoch, (loss).item(), (loss_base).item(), (test_error).item(), (test_error_base).item(), tac-tic))


path = "fenics/model_mismatch"
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
train_loss_base = np.log10(np.array(train_loss_base))
test_loss_base = np.log10(np.array(test_loss_base))

plt.plot(train_loss)
plt.plot(test_loss)
plt.plot(train_loss_base)
plt.plot(test_loss_base)
plt.show()

np.savetxt(path + "/train_loss_residual.txt", train_loss)
np.savetxt(path + "/test_loss_residual.txt", test_loss)
np.savetxt(path + "/train_loss_residual_base.txt", train_loss_base)
np.savetxt(path + "/test_loss_residual_base.txt", test_loss_base)

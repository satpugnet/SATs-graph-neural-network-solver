import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.utils as utils
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, NNConv

#################################################
#
# support functions
#
#################################################


def subgraph(subset,
             edge_index,
             edge_attr=None,
             relabel_nodes=False,
             num_nodes=None):
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.

    Args:
        subset (LongTensor, ByteTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    n_mask = subset

    if relabel_nodes:
        n_idx = torch.zeros(n_mask.size(0), dtype=torch.long)
        n_idx[subset] = torch.arange(subset.sum().item())

    mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index = n_idx[edge_index]

    return edge_index, edge_attr

def get_cluster(n_x, n_y):
    n_l = n_x * n_y
    index2 = np.array(range(n_l//4)).reshape(n_x//2, n_y//2) # torch.repeat is different from numpy
    index2 = index2.repeat(2, axis = 0).repeat(2, axis = 1)
    index2 = torch.tensor(index2, dtype=torch.long).reshape(-1)
    return index2

def upsample(x, dim, channels, scale):
    x = x.transpose(0, 1)
    for i in range(dim):
        x = F.upsample(x.view(1, channels, -1), scale_factor=scale, mode='nearest').view(channels, -1)
        channels = channels*2
    x = x.transpose(0, 1)
    return x

def index_to_mask(index, n):
    mask = torch.zeros(n, dtype=torch.uint8)
    mask[index] = 1
    return mask


#################################################
#
# construct graph data
#
#################################################

num_epoch = 1000

path = "cfd_data/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ni = 392
nj = 120

n = 26430
N = ni * nj

mask = np.loadtxt(path + "mask.txt", dtype=np.uint8)
index = np.loadtxt(path + "index.txt", dtype=np.int32)

mesh = np.loadtxt(path + "mesh.txt", dtype=np.float).reshape(-1,2)
mesh_full = np.loadtxt(path + "mesh_full.txt", dtype=np.float).reshape(-1,2)
sdf = np.loadtxt(path + "sdf.txt", dtype=np.float)
sdf_full = np.loadtxt(path + "sdf_full.txt", dtype=np.float)

mask = torch.tensor(mask, dtype=torch.uint8)
index = torch.tensor(index, dtype=torch.long)
mesh = torch.tensor(mesh, dtype=torch.float).reshape(-1,2)
sdf = torch.tensor(sdf, dtype=torch.float).reshape(-1,1)
sdf_full = torch.tensor(sdf_full, dtype=torch.float).reshape(-1,1)
print(mask.shape, index.shape, mesh.shape, sdf.shape, mesh_full.shape, sdf_full.shape)

def graph(n_x, n_y, mesh):
    edge_index = []
    edge_attr = []
    for x in range(n_x):
        for y in range(n_y):
            i = x * n_y + y
            if(y != n_y-1):
                edge_index.append((i, i + 1))
                edge_attr.append(mesh[i+1,:] - mesh[i,:])
                edge_index.append((i + 1, i))
                edge_attr.append(mesh[i,:] - mesh[i+1,:])

            if(x != n_x-1):
                edge_index.append((i, i+n_y))
                edge_attr.append(mesh[i+n_y,:] - mesh[i,:])
                edge_index.append((i+n_y, i))
                edge_attr.append(mesh[i,:] - mesh[i+n_y,:])

    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0,1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    return edge_index, edge_attr


edge_index, edge_attr = graph(ni, nj, mesh_full)
print(edge_index.shape, edge_attr.shape)
edge_index_sub, edge_attr_sub = subgraph(mask, edge_index, edge_attr=edge_attr)
print(edge_index_sub.shape, edge_attr_sub.shape)


depth = 3
edge_index_list = []
edge_attr_list = []
index_list = []
mesh_full = mesh_full.reshape(ni, nj, 2)
for l in range(0, depth):
    ni_l = ni // (2**l)
    nj_l = nj // (2**l)
    n_l = ni_l * nj_l
    index_l = index // (2**(2*l))
    mask_l = index_to_mask(index_l, n_l)

    mesh_full_l = mesh_full[::(2**l), ::(2**l), :].reshape(-1, 2)
    edge_index_l, edge_attr_l = graph(ni_l, nj_l, mesh_full_l)
    edge_index_sub_l, edge_attr_sub_l = subgraph(mask_l, edge_index_l, edge_attr=edge_attr_l, relabel_nodes=True)
    edge_index_list.append(edge_index_sub_l.to(device))
    edge_attr_list.append(edge_attr_sub_l.to(device))
    index_list.append(index_l)


dataset = []
mesh_full = torch.tensor(mesh_full, dtype=torch.float).reshape(-1,2)
for i in range(0, 21):
    solution = np.loadtxt(path + str(i) + "/solution.txt", dtype=np.float).reshape(-1,4)[:,1:4]
    solution = torch.tensor(solution, dtype=torch.float)
    parameter = torch.tensor(i, dtype=torch.float).repeat(N,1)
    x = torch.cat([mesh_full, sdf_full, parameter], dim=1)
    data = Data(x=x, y=solution, edge_index=edge_index, edge_attr=edge_attr).to(device)
    dataset.append(data)

features = 4
# number of train data
num_train = 14
train_loader = DataLoader(dataset[:num_train], batch_size=1, shuffle=True)
test_loader = DataLoader(dataset[num_train:], batch_size=1, shuffle=False)

#################################################
#
# architecture
#
#################################################

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(features, 32)
        self.conv1 = GCNConv(32, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.fc2 = nn.Linear(32, 3)


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

        return x



class Net_MP(nn.Module):
    def __init__(self):
        super(Net_MP, self).__init__()
        nn1 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, features*32))
        self.conv1 = NNConv(features, 32, nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1024))
        self.conv2 = NNConv(32, 32, nn2, aggr='mean')

        nn3 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1024))
        self.conv3 = NNConv(32, 32, nn3, aggr='mean')

        self.fc1 = torch.nn.Linear(32, 32)
        self.fc2 = torch.nn.Linear(32, 3)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

n_x = ni
n_y = nj
class Net_multi(torch.nn.Module):
    def __init__(self):
        super(Net_multi, self).__init__()
        self.fc1 = nn.Linear(features, 32)
        self.conv1 = GCNConv(32, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 32)
        self.conv5 = GCNConv(32, 32)
        self.fc2 = nn.Linear(32, 3)


    def forward(self, data, edge_index_list, index_list):

        ### step 1,  depth0
        x = data.x

        x = self.fc1(x)
        x = F.relu(x)
        x[index_list[0]] = self.conv1(x[index_list[0]], edge_index_list[0])
        x = F.relu(x)

        x1 = x.reshape(n_x, n_y, 32)
        n_x1 = n_x//2
        n_y1 = n_y//2

        ### step 2, depth 1
        x1 = x1[::2,::2,:].reshape(-1,32)
        x1[index_list[1]] = self.conv2(x1[index_list[1]], edge_index_list[1])
        x1 = F.relu(x1)

        x2 = x1.reshape(n_x1, n_y1, 32)

        ### step 3, depth 2
        x2 = x2[::2, ::2, :].reshape(-1, 32)
        x2[index_list[2]] = self.conv3(x2[index_list[2]], edge_index_list[2])
        x2 = F.relu(x2)

            ### unpooling
        x2_up = upsample(x2, dim=2, channels=32, scale=2).reshape(-1, 32)
        x1_up = x1 + x2_up

        ### step4, depth 1
        x1_up[index_list[1]] = self.conv4(x1_up[index_list[1]], edge_index_list[1])
        x1_up = F.relu(x1_up)

        x_up = upsample(x1_up, dim=2, channels=32, scale=2).reshape(-1, 32)
        x = x + x_up

        ### step5, depth 0
        x[index_list[0]] = self.conv5(x[index_list[0]], edge_index_list[0])
        x = F.relu(x)

        x = self.fc2(x)

        return x[index_list[0]]

class Net_multi_MP(torch.nn.Module):
    def __init__(self):
        super(Net_multi_MP, self).__init__()

        self.fc1 = torch.nn.Linear(features, 32)

        nn1 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1024))
        self.conv1 = NNConv(32, 32, nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1024))
        self.conv2 = NNConv(32, 32, nn2, aggr='mean')

        nn3 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1024))
        self.conv3 = NNConv(32, 32, nn3, aggr='mean')

        nn4 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1024))
        self.conv4 = NNConv(32, 32, nn4, aggr='mean')

        nn5 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1024))
        self.conv4 = NNConv(32, 32, nn5, aggr='mean')

        self.fc2 = torch.nn.Linear(32, 3)


    def forward(self, data, edge_index_list, index_list):

        ### step 1,  depth0
        x = data.x

        x = self.fc1(x)
        x = F.relu(x)
        x[index_list[0]] = self.conv1(x[index_list[0]], edge_index_list[0])
        x = F.relu(x)

        x1 = x.reshape(n_x, n_y, 32)
        n_x1 = n_x//2
        n_y1 = n_y//2

        ### step 2, depth 1
        x1 = x1[::2,::2,:].reshape(-1,32)
        x1[index_list[1]] = self.conv2(x1[index_list[1]], edge_index_list[1])
        x1 = F.relu(x1)

        x2 = x1.reshape(n_x1, n_y1, 32)

        ### step 3, depth 2
        x2 = x2[::2, ::2, :].reshape(-1, 32)
        x2[index_list[2]] = self.conv3(x2[index_list[2]], edge_index_list[2])
        x2 = F.relu(x2)

            ### unpooling
        x2_up = upsample(x2, dim=2, channels=32, scale=2).reshape(-1, 32)
        x1_up = x1 + x2_up

        ### step4, depth 1
        x1_up[index_list[1]] = self.conv4(x1_up[index_list[1]], edge_index_list[1])
        x1_up = F.relu(x1_up)

        x_up = upsample(x1_up, dim=2, channels=32, scale=2).reshape(-1, 32)
        x = x + x_up

        ### step5, depth 0
        x[index_list[0]] = self.conv5(x[index_list[0]], edge_index_list[0])
        x = F.relu(x)

        x = self.fc2(x)

        return x[index_list[0]]


#################################################
#
# training
#
#################################################



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
# model = Net_multi().to(device)
# model = Net_MP().to(device)
model = Net_multi().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
trivial_error = 1 # F.mse_loss(torch.zeros((N,1)), dataset[0].y.view(-1,1))

test_loss = []
train_loss = []
model.train()
for epoch in range(num_epoch):

    if epoch == 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005

    if epoch == 400:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001


    train_error = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch, edge_index_list, index_list)

        loss = F.mse_loss(out, batch.y.view(-1,3))
        train_error += loss

        loss.backward()
        optimizer.step()
    train_loss.append(train_error / len(train_loader))

    test_error = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch, edge_index_list, index_list)
            test_error += F.mse_loss(pred, batch.y.view(-1, 3))
    test_loss.append(test_error / len(test_loader))

    print(epoch, 'train loss: {:.4f}'.format(torch.log10(train_error/ len(train_loader))),
                 'test error: {:.4f}'.format(torch.log10(test_error/ len(test_loader))))



#################################################
#
# save
#
#################################################

path = "cfd_data/out_multi_mp"

if not os.path.exists(path):
    os.mkdir(path)
    print("Directory ", path, " Created ")
else:
    print("Directory ", path, " already exists")
# torch.save(model, path + "/model")

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

np.savetxt(path + "/train_loss.txt", train_loss)
np.savetxt(path + "/test_loss.txt", test_loss)

for i in range(14,21):
    out = model(dataset[i].to(device), edge_index_list, index_list).detach().cpu().numpy()
    # out = out.reshape(128//level, 256//level,2)
    # plot(out[:,:,0])
    # plot(out[:, :, 1])
    print(out.shape)
    np.savetxt(path + "/figure" + str(i)+ ".txt", out.reshape(-1))

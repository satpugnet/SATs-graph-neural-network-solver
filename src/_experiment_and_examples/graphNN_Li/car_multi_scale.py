import os
import h5py
import torch
import numpy as np
import time
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

def load_flow(filename, shape):
  stream_flow = h5py.File(filename, 'r')
  flow_state_vel = np.array(stream_flow['Velocity_0'][:])
  flow_state_vel = flow_state_vel.reshape([shape[0], shape[1]+128, 3])[0:shape[0],0:shape[1],0:2]
  stream_flow.close()
  return flow_state_vel

def load_boundary(filename, shape):
  stream_boundary = h5py.File(filename, 'r')
  boundary_cond = np.array(stream_boundary['Gamma'][:])
  boundary_cond = boundary_cond.reshape([shape[0], shape[1]+128, 1])[0:shape[0],0:shape[1],:]
  stream_boundary.close()
  return boundary_cond

def plot(image):
    plt.imshow(image)
    plt.colorbar()
    plt.show()

def subgraph(index,subset,
             edge_index,
             edge_attr=None,
             relabel_nodes=False):
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

    if isinstance(subset, list) or isinstance(subset, tuple):
        subset = torch.tensor(subset, dtype=torch.long)

    n_mask = subset

    n_idx = torch.zeros(n_mask.size(0), dtype=torch.long)
    n_idx[index] = torch.arange(subset.sum().item()).reshape(-1,1)

    mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
    mask = torch.nonzero(mask)
    edge_index = edge_index[:, mask].reshape(2,-1)
    edge_attr = edge_attr[mask, :].reshape(-1,3) if edge_attr is not None else None

    if relabel_nodes:
        edge_index = n_idx[edge_index]

    return edge_index, edge_attr


#################################################
#
# construct graph data
#
#################################################

sample = 3
level = 2**sample
num_epoch = 100
features = 3 + 2
shape = [128, 256]
flow_name = "car_data/computed_car_flow/sample_1/fluid_flow_0002.h5"

boundary_np = load_boundary(flow_name, shape) # (128, 256, 1)
sflow_true = load_flow(flow_name, shape) # (128, 256, 2)
sflow_plot = np.sqrt(np.square(sflow_true[:,:,0]) + np.square(sflow_true[:,:,1]))  - .05 *boundary_np[:,:,0]
# plot(sflow_plot)


n_x = 128 //level
n_y = 256 //level
N = n_x * n_y

def grid(n_x, n_y):

    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []
    for y in range(n_y):
        for x in range(n_x):
            i = y * n_x + x
            if(x != n_x-1):
                edge_index.append((i, i + 1))
                edge_attr.append((1, 0, 0))
                edge_index.append((i + 1, i))
                edge_attr.append((-1, 0, 0))

            if(y != n_y-1):
                edge_index.append((i, i+n_x))
                edge_attr.append((0, 1, 0))
                edge_index.append((i+n_x, i))
                edge_attr.append((0, -1, 0))

    X = torch.tensor(grid, dtype=torch.float)
    #Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0,1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


X, edge_index, edge_attr = grid(n_x, n_y)
print(edge_index.shape, edge_attr.shape)

depth = 8 - sample

edge_index_global = []
edge_attr_global = []
X_global = []
num_nodes = 0

for l in range(depth):
    h_x_l = n_x // (2 ** l)
    h_y_l = n_y // (2 ** l)
    n_l = h_x_l * h_y_l


    X, edge_index_inner, edge_attr_inner = grid(h_y_l, h_x_l)

    # update index
    edge_index_inner = edge_index_inner + num_nodes
    edge_index_global.append(edge_index_inner)
    edge_attr_global.append(edge_attr_inner)
    # construct X
    X_l = torch.tensor(l, dtype=torch.float).repeat(n_l, 1)
    X = torch.cat([X, X_l], dim=1)
    X_global.append(X)

    index1 = torch.tensor(range(n_l), dtype=torch.long)
    index1 = index1 + num_nodes
    num_nodes += n_l

    # #construct inter-graph edge
    if l != depth-1:
        index2 = np.array(range(n_l//4)).reshape(h_x_l//2, h_y_l//2)  # torch.repeat is different from numpy
        index2 = index2.repeat(2, axis = 0).repeat(2, axis = 1)
        index2 = torch.tensor(index2).reshape(-1)
        index2 = index2 + num_nodes

        edge_index_inter1 = torch.cat([index1,index2], dim=-1).reshape(2,-1)
        edge_index_inter2 = torch.cat([index2,index1], dim=-1).reshape(2,-1)
        edge_index_inter = torch.cat([edge_index_inter1, edge_index_inter2], dim=1)

        edge_attr_inter1 = torch.tensor((0, 0, 1), dtype=torch.float).repeat(n_l, 1)
        edge_attr_inter2 = torch.tensor((0, 0,-1), dtype=torch.float).repeat(n_l, 1)
        edge_attr_inter = torch.cat([edge_attr_inter1, edge_attr_inter2], dim=0)

        edge_index_global.append(edge_index_inter)
        edge_attr_global.append(edge_attr_inter)

print(N, num_nodes)
X = torch.cat(X_global, dim=0)
edge_index = torch.cat(edge_index_global, dim=1)
edge_attr = torch.cat(edge_attr_global, dim=0)
print(X.shape,  edge_index.shape, edge_attr.shape)

# for each car, modify the first layer of graph

path = "car_data/computed_car_flow/sample_"
dataset = []
mask_full = np.ones(num_nodes)
shape = [128, 256]
n = 128//level * 256//level
for i in range(1, 29):
    filename = path + str(i) + "/fluid_flow_0002.h5"
    boundary_np = load_boundary(filename, shape)[::level,::level].reshape(-1)
    sflow_true = load_flow(filename, shape)[::level,::level,:].reshape(-1, 2)
    signed_distance_function = np.loadtxt(path + str(i) + "/sdf.txt").reshape(shape)[::level,::level]
    #sflow_plot = np.sqrt(np.square(sflow_true[:, :, 0]) + np.square(sflow_true[:, :, 1])) - .05 * boundary_np[:, :, 0]

    boundary = torch.zeros(num_nodes,1)
    boundary_np = torch.tensor(boundary_np, dtype=torch.float).reshape(-1,1)
    boundary[:n] = boundary_np
    sflow_true = torch.tensor(sflow_true, dtype=torch.float)
    sdf = torch.zeros(num_nodes,1)
    sdf_np = torch.tensor(signed_distance_function, dtype=torch.float).reshape(-1,1)
    sdf[:n] = sdf_np

    x = torch.cat([X, boundary, sdf], dim=1)

    valid = (boundary_np==0)
    valid_index = np.where(valid)[0]
    mask = mask_full
    mask[:n] = valid.reshape(-1)
    mask = torch.tensor(mask, dtype=torch.long)
    index = torch.nonzero(mask)

    n_valid = torch.sum(valid)
    n_total = torch.sum(mask)
    mask_index = torch.tensor(range(n_valid), dtype=torch.long)

    x_sub = x[index, :].reshape(n_total, features)
    y_sub = sflow_true[valid_index, :]
    edge_index_sub, edge_attr_sub = subgraph(index, mask, edge_index, edge_attr, relabel_nodes=True)
    print(edge_index_sub.shape, edge_attr_sub.shape)
    print(n_valid, x_sub.shape, edge_index_sub.shape, boundary_np.shape)
    data = Data(x=x_sub, y=y_sub, edge_index=edge_index_sub, edge_attr=edge_attr_sub, mask_index=mask_index)
    dataset.append(data)


# number of train data
num_train = 20
train_loader = DataLoader(dataset[:num_train], batch_size=4, shuffle=True)
test_loader = DataLoader(dataset[num_train:], batch_size=4, shuffle=False)

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
        self.fc2 = nn.Linear(32, 2)


    def forward(self, data):
        x, edge_index, mask_index = data.x, data.edge_index, data.mask_index

        x = self.fc1(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = x[mask_index]
        x = F.relu(x)
        x = self.fc2(x)

        return x



class Net_MP(nn.Module):
    def __init__(self):
        super(Net_MP, self).__init__()
        nn1 = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, features * 32))
        self.conv1 = NNConv(features, 32, nn1, aggr='mean')

        # nn2 = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1024))
        # self.conv2 = NNConv(32, 32, nn2, aggr='mean')

        nn3 = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1024))
        self.conv3 = NNConv(32, 32, nn3, aggr='mean')

        self.fc1 = torch.nn.Linear(32, 32)
        self.fc2 = torch.nn.Linear(32, 2)

    def forward(self, data):
        x, edge_index, edge_attr, mask_index = data.x, data.edge_index, data.edge_attr, data.mask_index
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        # x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = x[mask_index]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#################################################
#
# training
#
#################################################



N = n_x * n_y
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
model = Net_MP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
trivial_error = 1 # F.mse_loss(torch.zeros((N,1)), dataset[0].y.view(-1,1))

test_loss = []
train_loss = []
model.train()
for epoch in range(num_epoch):

    if epoch == 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.005

    if epoch == 500:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001


    train_error = 0
    tic = time.time()

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)

        loss = F.mse_loss(out, batch.y.view(-1,2))
        train_error += loss

        loss.backward()
        optimizer.step()
    train_loss.append(train_error / len(train_loader))

    test_error = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            test_error += F.mse_loss(pred, batch.y.view(-1, 2))
    test_loss.append(test_error / len(test_loader))

    tac = time.time()
    print(epoch, 'train loss: {:.4f}'.format(torch.log10(train_error/ len(train_loader))),
                 'test error: {:.4f}'.format(torch.log10(test_error/ len(test_loader))),
                'time: {:.2f}.'.format(tac-tic))



#################################################
#
# save
#
#################################################

path = "out_connect_net"

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

# for i in range(20,28):
#     out = model(dataset[i].to(device)).detach().cpu().numpy()
#     out = out.reshape(128//level, 256//level,2)
#     plot(out[:,:,0])
#     # plot(out[:, :, 1])
#     print(out.shape)
#     np.savetxt(path + "/figure" + str(i)+ ".txt", out.reshape(-1))

import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.utils as utils
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, NNConv, global_add_pool


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


#################################################
#
# construct graph data
#
#################################################

level = 3
num_epoch = 100
shape = [128, 256]
flow_name = "car_data/computed_car_flow/sample_1/fluid_flow_0002.h5"

boundary_np = load_boundary(flow_name, shape) # (128, 256, 1)
sflow_true = load_flow(flow_name, shape) # (128, 256, 2)
sflow_plot = np.sqrt(np.square(sflow_true[:,:,0]) + np.square(sflow_true[:,:,1]))  - .05 *boundary_np[:,:,0]
# plot(sflow_plot)


n_x = 256 // level
n_y = 128 // level
N = n_x * n_y
# grid of 128 * 256
# xs = [i for i in range(128)]
# ys = [i for i in range(256)]
xs = np.linspace(0.0, 1.0, n_x)
ys = np.linspace(0.0, 1.0, n_y)
grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T
print(grid)

edge_index = []
edge_attr = []
for y in range(n_y):
    for x in range(n_x):
        i = y * n_x + x
        if(x != n_x-1):
            edge_index.append((i, i + 1))
            edge_attr.append((1, 0))
            edge_index.append((i + 1, i))
            edge_attr.append((-1, 0))

        if(y != n_y-1):
            edge_index.append((i, i+n_x))
            edge_attr.append((0, 1))
            edge_index.append((i+n_x, i))
            edge_attr.append((0, -1))

# or
# edge_index, pos = utils.grid(h_x, h_y)
print(len(edge_index))
print(len(edge_attr))

X = torch.tensor(grid, dtype=torch.float)
#Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0,1)
edge_attr = torch.tensor(edge_attr, dtype=torch.float)


path = "car_data/computed_car_flow/sample_"
dataset = []
shape = [128, 256]
# for i in range(1, 29):
#     filename = path + str(i) + "/fluid_flow_0002.h5"
#     boundary_np = load_boundary(filename, shape).reshape(-1)
#     sflow_true = load_flow(filename, shape).reshape(-1, 2)
#     #sflow_plot = np.sqrt(np.square(sflow_true[:, :, 0]) + np.square(sflow_true[:, :, 1])) - .05 * boundary_np[:, :, 0]
#
#     mask = np.where(boundary_np == 0)
#     mask = torch.tensor(mask, dtype=torch.long).reshape(-1)
#     subset = torch.tensor(boundary_np, dtype=torch.uint8)
#     X_sub = X[mask]
#     Exact_sub = torch.tensor(sflow_true[mask, :], dtype=torch.float)
#     edge_index_sub, edge_attr_sub = subgraph(subset, edge_index, edge_attr=edge_attr, relabel_nodes=True)
#
#     data = Data(x=X_sub, y=Exact_sub, edge_index=edge_index_sub, edge_attr=edge_attr_sub)
#     print(X_sub.shape, Exact_sub.shape, edge_index_sub.shape, edge_attr_sub.shape)
#     dataset.append(data)

for i in range(1, 29):
    filename = path + str(i) + "/fluid_flow_0002.h5"
    boundary_np = load_boundary(filename, shape)[::level,::level].reshape(-1)
    sflow_true = load_flow(filename, shape)[::level,::level].reshape(-1, 2)
    signed_distance_function = np.loadtxt(path + str(i) + "/sdf.txt").reshape(shape)[::level,::level]

    #sflow_plot = np.sqrt(np.square(sflow_true[:, :, 0]) + np.square(sflow_true[:, :, 1])) - .05 * boundary_np[:, :, 0]

    boundary_np = torch.tensor(boundary_np, dtype=torch.float).reshape(-1,1)
    sflow_true = torch.tensor(sflow_true, dtype=torch.float)
    sdf = torch.tensor(signed_distance_function, dtype=torch.float).reshape(-1,1)
    print(X.shape, boundary_np.shape, sdf.shape)
    x = torch.cat([X, boundary_np, sdf], dim=1)
    data = Data(x=x, y=sflow_true, edge_index=edge_index, edge_attr=edge_attr)
    dataset.append(data)


features = 4
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
        nn1 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 64))
        self.conv1 = NNConv(features, 32, nn1, aggr='mean')

        nn2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1024))
        self.conv2 = NNConv(32, 32, nn2, aggr='mean')

        self.fc1 = torch.nn.Linear(32, 32)
        self.fc2 = torch.nn.Linear(32, 2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#################################################
#
# training
#
#################################################


n_x = 128//level
n_y = 256//level
N = n_x * n_y
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
# model = Net_MP().to(device)
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

    print(epoch, 'train loss: {:.4f}'.format(torch.log10(train_error/ len(train_loader))),
                 'test error: {:.4f}'.format(torch.log10(test_error/ len(test_loader))))



#################################################
#
# save
#
#################################################

path = "car_data/out_local_connected"

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
#     # out = out.reshape(128//level, 256//level,2)
#     # plot(out[:,:,0])
#     # plot(out[:, :, 1])
#     print(out.shape)
#     np.savetxt(path + "/figure" + str(i)+ ".txt", out.reshape(-1))

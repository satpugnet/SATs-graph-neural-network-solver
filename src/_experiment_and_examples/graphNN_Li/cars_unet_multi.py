import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.utils as utils
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, NNConv, max_pool

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

#################################################
#
# construct graph data
#
#################################################

level = 2**0
depth = 3
num_epoch = 1000
shape = [128, 256]
flow_name = "car_data/computed_car_flow/sample_1/fluid_flow_0002.h5"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


boundary_np = load_boundary(flow_name, shape) # (128, 256, 1)
sflow_true = load_flow(flow_name, shape) # (128, 256, 2)
sflow_plot = np.sqrt(np.square(sflow_true[:,:,0]) + np.square(sflow_true[:,:,1]))  - .05 *boundary_np[:,:,0]
# plot(sflow_plot)


n_x = 128 // level
n_y = 256 // level
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
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0,1).to(device)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).to(device)

    return X, edge_index, edge_attr

X, edge_index, edge_attr = grid(n_y, n_x)

path = "car_data/computed_car_flow/sample_"
dataset = []
shape = [128, 256]
edge_index_list = []
edge_attr_list = []

for l in range(0, depth):
    n_y_l = n_y // (2**l)
    n_x_l = n_x // (2**l)
    _, edge_index_l, edge_attr_l = grid(n_y_l, n_x_l)
    edge_index_list.append(edge_index_l)
    edge_attr_list.append(edge_attr_l)

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
    data = Data(x=x, y=sflow_true, edge_index=edge_index, edge_attr=edge_attr).to(device)
    dataset.append(data)


features = 4
# number of train data
num_train = 20

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

class Net_multi(torch.nn.Module):
    def __init__(self):
        super(Net_multi, self).__init__()
        self.fc1 = nn.Linear(features, 32)
        self.conv1 = GCNConv(32, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 32)
        self.conv5 = GCNConv(32, 32)
        # self.conv12 = GCNConv(32, 32)
        # self.conv22 = GCNConv(32, 32)
        # self.conv32 = GCNConv(32, 32)
        # self.conv42 = GCNConv(32, 32)
        # self.conv52 = GCNConv(32, 32)
        self.fc2 = nn.Linear(32, 2)


    def forward(self, data, edge_index_list):

        ### step 1,  depth0
        x, edge_index = data.x, data.edge_index

        x = self.fc1(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index_list[0])
        # x = self.conv12(x, edge_index_list[0])
        x = F.relu(x)

        x1 = x.reshape(n_x, n_y, 32)
        n_x1 = n_x//2
        n_y1 = n_y//2

        ### step 2, depth 1
        x1 = x1[::2,::2,:].reshape(-1,32)
        x1 = self.conv2(x1, edge_index_list[1])
        # x1 = self.conv22(x1, edge_index_list[1])
        x1 = F.relu(x1)

        x2 = x1.reshape(n_x1, n_y1, 32)

        ### step 3, depth 2
        x2 = x2[::2, ::2, :].reshape(-1, 32)
        x2 = self.conv3(x2, edge_index_list[2])
        # x2 = self.conv32(x2, edge_index_list[2])
        x2 = F.relu(x2)

            ### unpooling
        x2_up = upsample(x2, dim=2, channels=32, scale=2).reshape(-1, 32)
        x1_up = x1 + x2_up

        ### step4, depth 1
        x1_up = self.conv4(x1_up, edge_index_list[1])
        # x1_up = self.conv42(x1_up, edge_index_list[1])
        x1_up = F.relu(x1_up)

        x_up = upsample(x1_up, dim=2, channels=32, scale=2).reshape(-1, 32)
        x = x + x_up

        ### step5, depth 0
        x = self.conv5(x, edge_index_list[0])
        # x = self.conv52(x, edge_index_list[0])
        x = F.relu(x)

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
# model = Net().to(device)
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
        out = model(batch, edge_index_list)

        loss = F.mse_loss(out, batch.y.view(-1,2))
        train_error += loss

        loss.backward()
        optimizer.step()
    train_loss.append(train_error / len(train_loader))

    test_error = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch, edge_index_list)
            test_error += F.mse_loss(pred, batch.y.view(-1, 2))
    test_loss.append(test_error / len(test_loader))

    print(epoch, 'train loss: {:.4f}'.format(torch.log10(train_error/ len(train_loader))),
                 'test error: {:.4f}'.format(torch.log10(test_error/ len(test_loader))))



#################################################
#
# save
#
#################################################

path = "car_data/out_multi10"

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
#     print(i)
#     out = model(dataset[i].to(device)).detach().cpu().numpy()
#     # out = out.reshape(128//level, 256//level,2)
#     # plot(out[:,:,0])
#     # plot(out[:, :, 1])
#     print(out.shape)
#     np.savetxt(path + "/figure" + str(i)+ ".txt", out.reshape(-1))

i = 20
for batch in test_loader:
    out = model(batch, edge_index_list).detach().cpu().numpy()
    out = out.reshape(128//level, 256//level,2)
    plot(out[:,:,0])
    # plot(out[:, :, 1])
    # print(out.shape)
    np.savetxt(path + "/figure" + str(i)+ ".txt", out.reshape(-1))
    i = i + 1

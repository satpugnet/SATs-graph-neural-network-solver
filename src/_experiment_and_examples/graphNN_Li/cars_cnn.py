import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.utils as utils
from torch.utils.data import DataLoader
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


#################################################
#
# construct graph data
#
#################################################

level = 1
num_epoch = 1000
shape = [128, 256]
flow_name = "car_data/computed_car_flow/sample_1/fluid_flow_0002.h5"

boundary_np = load_boundary(flow_name, shape) # (128, 256, 1)
sflow_true = load_flow(flow_name, shape) # (128, 256, 2)
sflow_plot = np.sqrt(np.square(sflow_true[:,:,0]) + np.square(sflow_true[:,:,1]))  - .05 *boundary_np[:,:,0]
# plot(sflow_plot)


n_x = 256 // level
n_y = 128 // level
N = n_x * n_y


path = "car_data/computed_car_flow/sample_"

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
# architecture@
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
            param_group['lr'] = 0.001

    if epoch == 4000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001


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

path = "car_data/out"

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

for i in range(20,28):
    out = model(dataset[i].to(device)).detach().cpu().numpy()
    # out = out.reshape(128//level, 256//level,2)
    # plot(out[:,:,0])
    # plot(out[:, :, 1])
    print(out.shape)
    np.savetxt(path + "/figure" + str(i)+ ".txt", out.reshape(-1))

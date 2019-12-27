import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.data import InMemoryDataset
import matplotlib.pyplot as plt
import random


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

Plot = True

#################################################
#
# generate PDE data
#
#################################################


# # 2*2 mesh on [0,1] * [0,1] domain
#
# # location
# x = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float)
# # value
# y = torch.tensor([0, 0, 0, 0], dtype=torch.float)
#
# # edge (4 edges)
# edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
#                            [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
#
# dataset = Data(x=x, y=y, edge_index=edge_index)
#
# print(dataset)



# h*h mesh on [0,1] * [0,1] domain
#raw_data = np.loadtxt("/Users/lizongyi/Downloads/GNN-PDE/fenics/mysol00.txt")
#raw_data = np.loadtxt("/Users/lizongyi/Downloads/GNN-PDE/test.txt", delimiter=',').reshape(-1,3)


#np.random.shuffle(raw_data)
# print(raw_data.shape)
# n = raw_data.shape[0]
# h = int(np.sqrt(n))-1
# print('h', h)
# x = torch.tensor(raw_data[:,:2], dtype=torch.float)
# y = torch.tensor(raw_data[:,2], dtype=torch.float)

# counter = 0
# edges = list()
# # generate edge
# for i in range(n):
#     for j in range(i+1,n):
#         if (np.linalg.norm(x[i]-x[j]) <= 1/h):
#             counter = counter + 1
#             print(counter,x[i], x[j])
#             edges.append((i, j))
#             edges.append((j, i))
# edges = np.array(edges).transpose()
# edge_index = torch.tensor(edges, dtype=torch.long)

h = 10
n = (h+1)**2
features = 4

num_train_node = int(n/2)
train_mask = np.array(range(num_train_node))
test_mask = np.array(range(num_train_node,n))

dataset = []

for a in range(10):
    for b in range(10):
        path = "/Users/lizongyi/Downloads/GNN-PDE/fenics/data_sin_25/mysol" + str(a) + str(b)
        raw_x = np.loadtxt(path + "/x.txt")
        raw_y = np.loadtxt(path + "/y.txt")
        raw_edge = np.loadtxt(path + "/edge.txt")
        x = torch.tensor(raw_x, dtype=torch.float)
        y = torch.tensor(raw_y, dtype=torch.float)
        edge_index = torch.tensor(raw_edge,dtype=torch.long)
        dataset.append(Data(x=x, y=y, edge_index=edge_index))

# number of train data
num_train = 50
num_data_per_batch = 10
random.shuffle(dataset)

train_loader = DataLoader(dataset[:num_train], batch_size=num_train // num_data_per_batch, shuffle=True)
test_loader = DataLoader(dataset[num_train:], batch_size=(len(dataset) - num_train) // num_data_per_batch, shuffle=False)

# class PoissionDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None):
#         super(PoissionDataset, self).__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])
#
#     @property
#     def raw_file_names(self):
#         return ['some_file_1']
#
#     @property
#     def processed_file_names(self):
#         return ['data.pt']
#
#     def _download(self):
#         pass
#
#     def process(self):
#         # Read data into huge `Data` list.
#         data_list = [...]
#
#         if self.pre_filter is not None:
#             data_list [data for data in data_list if self.pre_filter(data)]
#
#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]
#
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])


#################################################
#
# Graph neural network structure
#
#################################################


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv21 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv21(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x

        #return F.log_softmax(x, dim=1)

class Net_skip(torch.nn.Module):
    def __init__(self):
        super(Net_skip, self).__init__()
        self.conv1 = GCNConv(features, 32-features)
        self.conv2 = GCNConv(32, 32-features)
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

#################################################
#
# train
#
#################################################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
#model = Net_skip().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# data = dataset.to(device)
# model.train()
# for epoch in range(100):
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.mse_loss(out[train_mask], data.y.view(-1,1)[train_mask])
#     print(epoch, loss)
#     loss.backward()
#     optimizer.step()
#
# model.eval()
# pred = model(data)
# error = ((pred[test_mask] - data.y[test_mask])**2).mean()
# print('test L2 error: {:.4f}'.format(error))

test_loss = []
train_loss = []
model.train()
for epoch in range(1000):
    train_error = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        #y = torch.cat([data.y for data in batch])
        loss = F.mse_loss(out, batch.y.view(-1,1))
        train_error = train_error + loss

        loss.backward()
        optimizer.step()
    train_loss.append(train_error / len(train_loader))

    test_error = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            test_error += F.mse_loss(pred, batch.y.view(-1, 1))
    test_loss.append(test_error / len(test_loader))

    print(epoch, 'train loss: {:.4f}'.format(train_error/len(train_loader)),
                 'test L2 error: {:.4f}'.format(test_error/len(test_loader)))

model.eval()
test_error = 0
for batch in test_loader:
    batch = batch.to(device)
    pred = model(batch)
    test_error += F.mse_loss(pred, batch.y.view(-1,1))
print('test L2 error: {:.4f}'.format(test_error/len(test_loader)))


#################################################
#
# save
#
#################################################

#torch.save(model, "/Users/lizongyi/Downloads/GNN-PDE/fenics/model")
torch.save(model.state_dict(), "/Users/lizongyi/Downloads/GNN-PDE/fenics/model")

#################################################
#
# plot
#
#################################################

if(Plot):
    plt.plot(train_loss, label='train loss')
    plt.plot(test_loss, label='test loss')
    plt.legend(loc='upper right')
    plt.show()


#################################################
#
# functionality to add node
#
#################################################

def add_node(data, node_x, dim):
    n = data.x.shape[0]
    h = int(np.sqrt(n)) - 1
    print(n,h)
    coefficient = data.x[0,dim:]
    node_x = torch.cat((node_x,coefficient), dim = 0)


    # construct new graph
    x = torch.cat((data.x, node_x.view(1,-1)), dim =0)
    edge_index = data.edge_index
    for i in range(n):
        if (torch.norm(x[i,:dim] - node_x[:dim]) <= 1/h) :
            edge1 = torch.tensor([i,n], dtype=torch.long).reshape(2,1)
            edge2 = torch.tensor([n,i], dtype=torch.long).reshape(2,1)
            edge_index = torch.cat((edge_index, edge1, edge2), dim = 1)

    new_graph = Data(x=x, y=data.y, edge_index=edge_index)
    model.eval()
    pred = model(new_graph)

    return pred[-1]

# out = add_node(dataset[0], torch.tensor([0.95, 0.95], dtype=torch.float), dim=2)
# print(out)

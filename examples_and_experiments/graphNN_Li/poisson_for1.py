import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.data import InMemoryDataset
import matplotlib.pyplot as plt


np.random.seed(0)
torch.manual_seed(0)

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



#  h*h mesh on [0,1] * [0,1] domain
raw_data = np.loadtxt("/Users/lizongyi/Downloads/GNN-PDE/fenics/data_sin_1616/mysol00.txt")
#raw_data = np.loadtxt("/Users/lizongyi/Downloads/GNN-PDE/test.txt", delimiter=',').reshape(-1,3)


np.random.shuffle(raw_data)
print(raw_data.shape)
n = raw_data.shape[0]
h = int(np.sqrt(n))-1
print('h', h)
x = torch.tensor(raw_data[:,:2], dtype=torch.float)
y = torch.tensor(raw_data[:,2], dtype=torch.float)

counter = 0
edges = list()
# generate edge
for i in range(n):
    for j in range(i+1,n):
        if (np.linalg.norm(x[i]-x[j]) <= 1.001/h):
            counter = counter + 1
            print(counter,x[i], x[j])
            edges.append((i, j))
            edges.append((j, i))
edges = np.array(edges).transpose()
edge_index = torch.tensor(edges, dtype=torch.long)

dataset = Data(x=x, y=y, edge_index=edge_index)

num_train_node = int(n/2)
train_mask = np.array(range(num_train_node))
test_mask = np.array(range(num_train_node,n))

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
        self.conv1 = GCNConv(2, 16)
        self.conv3 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        return x

        #return F.log_softmax(x, dim=1)

#################################################
#
# train
#
#################################################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

test_loss = []
train_loss = []

data = dataset.to(device)
model.train()
for epoch in range(500):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out[train_mask], data.y.view(-1,1)[train_mask])

    loss.backward()
    optimizer.step()

    with torch.no_grad():
        test_error = F.mse_loss(out[test_mask], data.y.view(-1,1)[test_mask])
        print(epoch, loss, test_error)

model.eval()
pred = model(data)
error = F.mse_loss(pred[test_mask], data.y.view(-1,1)[test_mask])
print('test L2 error: {:.4f}'.format(error))



#torch.save(model, "/Users/lizongyi/Downloads/GNN-PDE/fenics/model")

#################################################
#
# plot
#
#################################################


# plt.plot(train_loss)
# plt.plot(test_loss)
# plt.show()


#################################################
#
# functionality to add node
#
#################################################


dim = 2
node_x = torch.tensor([0., 0.])
n = data.x.shape[0]
h = int(np.sqrt(n)) - 1
print(n,h)
coefficient = data.x[0,dim:]
node_x = torch.cat((node_x,coefficient), dim = 0)


# construct new graph
x = torch.cat((data.x, node_x.view(1,-1)), dim =0)
y = torch.cat((data.y, torch.tensor([0.])), dim =0)
edge_index = data.edge_index
for i in range(n):
    #print(x[i, :dim] - node_x[:dim], torch.norm(x[i,:dim] - node_x[:dim]))
    if (torch.norm(x[i,:dim] - node_x[:dim]) <= 1.001/h) :

        edge1 = torch.tensor([i,n], dtype=torch.long).reshape(2,1)
        edge2 = torch.tensor([n,i], dtype=torch.long).reshape(2,1)
        edge_index = torch.cat((edge_index, edge1, edge2), dim = 1)
        print(i, x[i,:dim], edge1)

new_graph = Data(x=x, y=y, edge_index=edge_index)

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(new_graph)
    loss = F.mse_loss(out[train_mask], new_graph.y.view(-1, 1)[train_mask])
    print(epoch, loss, out[-1])
    loss.backward()
    optimizer.step()

model.eval()
pred = model(new_graph)
print(pred.shape)


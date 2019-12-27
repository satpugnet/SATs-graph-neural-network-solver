import torch
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


def u(x):
    res = 1
    dim = len(x)
    for d in range(dim):
        res = res * np.sin(np.pi * x[d])
    return res

dim = 3
h = 16
depth = np.int(np.log2(h))
print(dim, h, depth)

hierarchy_n = []
hierarchy_l = []
hierarchy_x = []
hierarchy_y = []
hierarchy_edge_index = []
hierarchy_edge_attr = []
for l in range(1, depth+1):
    h_l = 2**l
    n_l = h_l ** dim

    hierarchy_n.append(n_l)
    hierarchy_l.append(l)

    xs = np.linspace(0.0, 1.0, h_l)
    xs_list = [xs] * dim
    xs_grid = np.vstack([xx.ravel() for xx in np.meshgrid(*xs_list)]).T
    print(l, h_l, xs_grid.shape)

    y = np.zeros(n_l)
    for i in range(n_l):
        y[i] = u(xs_grid[i])

    # adjacent_v = []
    # for i in range(h):
    #     adjacent_v.append((i, (i+1) % h))
    #     adjacent_v.append(((i+1) % h), i)
    # adjacent_v = np.array(adjacent_v, dtype=np.int16)
    edge_index = []
    edge_attr = []
    for i in range(n_l):
        for j in range(i):
            if (np.linalg.norm(xs_grid[i] - xs_grid[j])) < 1.01 /(h_l-1):
                edge_index.append((i,j))
                edge_attr.append(xs_grid[i] - xs_grid[j])
                edge_index.append((j,i))
                edge_attr.append(xs_grid[j] - xs_grid[i])

    edge_index = np.array(edge_index, dtype=np.int).transpose()
    edge_attr = np.array(edge_attr)

    hierarchy_x.append(torch.tensor(xs_grid, dtype=torch.float))
    hierarchy_y.append(torch.tensor(y, dtype=torch.float))
    hierarchy_edge_index.append(torch.tensor(edge_index, dtype=torch.long))
    hierarchy_edge_attr.append(torch.tensor(edge_attr, dtype=torch.float))
    print(edge_index.shape, edge_attr.shape)

#os.mkdir("/Users/lizongyi/Downloads/GNN-PDE/fenics/data_multi_res/")
path = "fenics/data_multi_res/" + str(dim)+str(h)

if not os.path.exists(path):
    os.mkdir(path)
    print("Directory ", path, " Created ")
else:
    print("Directory ", path, " already exists")

pickle.dump(hierarchy_n, open(path + "/hierarchy_n", "wb"))
pickle.dump(hierarchy_l, open(path + "/hierarchy_l", "wb"))
pickle.dump(hierarchy_x, open(path + "/hierarchy_x", "wb"))
pickle.dump(hierarchy_y, open(path + "/hierarchy_y", "wb"))
pickle.dump(hierarchy_edge_index, open(path + "/hierarchy_edge_index", "wb"))
pickle.dump(hierarchy_edge_attr, open(path + "/hierarchy_edge_attr", "wb"))

#################################################
#
# network
#
#################################################



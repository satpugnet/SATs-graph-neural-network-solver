import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch import nn
from C_GNNs.abstract_gnn import AbstractGNN


class GCN2LayerLinear1LayerGNN(AbstractGNN):

    def __init__(self, num_node_features, num_y_features):
        super(GCN2LayerLinear1LayerGNN, self).__init__(num_node_features, num_y_features)
        self._conv1 = GCNConv(num_node_features, 16)
        self._conv2 = GCNConv(16, num_y_features)
        self._fc2 = nn.Linear(num_y_features, 1)

    def __str__(self):
        return "{}(conv1({}), conv2({}), fc2({}))".format(self.__class__.__name__, self._conv1, self._conv2, self._fc2)

    # TODO: check how to take into account edge attribute
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self._conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) # TODO: uncomment or remove
        x = self._conv2(x, edge_index)
        x = F.relu(x)
        x = global_add_pool(x, data.batch)
        x = self._fc2(x)

        return torch.sigmoid(x)

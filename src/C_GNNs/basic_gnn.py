import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch import nn

from C_GNNs.abstract_gnn import AbstractGNN


class BasicGNN(AbstractGNN):

    def __init__(self, num_node_features, num_y_features):
        super(BasicGNN, self).__init__(num_node_features, num_y_features)
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_y_features)
        self.fc2 = nn.Linear(num_y_features, 1)

    # TODO: check how to take into account edge attribute
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) # TODO: uncomment or remove
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_add_pool(x, data.batch)
        x = self.fc2(x)

        return torch.sigmoid(x)

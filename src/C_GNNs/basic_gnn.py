import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch import nn


class BasicGNN(torch.nn.Module):
    def __init__(self, num_node_features, y_length):
        super(BasicGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, y_length)
        self.fc2 = nn.Linear(y_length, 1)

    # TODO: check how to take into account edge attribute
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_add_pool(x, data.batch)
        x = self.fc2(x)

        return torch.sigmoid(x)
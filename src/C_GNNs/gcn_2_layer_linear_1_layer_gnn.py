import time

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch import nn
from C_GNNs.abstract_gnn import AbstractGNN


class GCN2LayerLinear1LayerGNN(AbstractGNN):

    def __init__(self, sigmoid_output=True):
        super(GCN2LayerLinear1LayerGNN, self).__init__(sigmoid_output)

    def initialise_channels(self, in_channels, out_channels, num_edge_features=None):
        super().initialise_channels(in_channels, out_channels, num_edge_features)
        self._conv1 = GCNConv(in_channels, 16)
        self._conv2 = GCNConv(16, out_channels)

        self._fc2 = nn.Linear(out_channels, 1)

    def __repr__(self):
        return "{}(conv1({}), conv2({}), fc2({}))".format(self.__class__.__name__, self._conv1, self._conv2, self._fc2)

    def _perform_pre_pooling(self, x, edge_index, edge_attr):
        x = F.relu(self._conv1(x, edge_index))
        x = F.dropout(x, training=self.training)  # TODO: uncomment or remove
        x = F.relu(self._conv2(x, edge_index))

        return x

    def _perform_pooling(self, x, batch):
        x = global_add_pool(x, batch)

        return x

    def _perform_post_pooling(self, x, edge_index, edge_attr):
        x = self._fc2(x)

        return x


from abc import ABC

import torch
from torch import nn
from torch_geometric.nn import NNConv, global_add_pool

from C_GNNs.abstract_gnn import AbstractGNN
import torch.nn.functional as F


class NNConvGNN(AbstractGNN, ABC):

    def __init__(self, sigmoid_output=True):
        super().__init__(sigmoid_output=True)

    def initialise_channels(self, in_channels, out_channels, num_edge_features=None):
        super().initialise_channels(in_channels, out_channels, num_edge_features)

        # nn1 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 64))
        self._nn1 = nn.Linear(num_edge_features, in_channels * 32)
        self._conv1 = NNConv(in_channels, 32, self._nn1, aggr='add')

        # nn2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1024))
        self._nn2 = nn.Linear(num_edge_features, 1024)
        self._conv2 = NNConv(32, 32, self._nn2, aggr='add')

        # nn3 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1024))
        self._nn3 = nn.Linear(num_edge_features, 1024)
        self._conv3 = NNConv(32, 32, self._nn3, aggr='add')

        self._pooling = global_add_pool

        self._fc1 = torch.nn.Linear(32, 32)
        self._fc2 = torch.nn.Linear(32, 1)

    def __repr__(self):
        return "{}(nn1({}), conv1({}), nn2({}), conv2({}), nn3({}), conv3({}), pooling ({}), fc1({}), fc2({}), sigmoid_output({}))"\
            .format(
                self.__class__.__name__,
                self._nn1,
                self._conv1,
                self._nn2,
                self._conv2,
                self._nn3,
                self._conv3,
                self._pooling.__name__,
                self._fc1,
                self._fc2,
                self._sigmoid_output
            )

    def _perform_pre_pooling(self, x, edge_index, edge_attr):
        x = F.relu(self._conv1(x, edge_index, edge_attr))
        x = F.relu(self._conv2(x, edge_index, edge_attr))
        x = F.relu(self._conv3(x, edge_index, edge_attr))
        x = F.relu(self._fc1(x))

        return x

    def _perform_pooling(self, x, batch):
        x = self._pooling(x, batch)

        return x

    def _perform_post_pooling(self, x, edge_index, edge_attr):
        x = self._fc2(x)

        return x

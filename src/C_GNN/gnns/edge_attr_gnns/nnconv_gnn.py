import torch
from torch import nn
from torch_geometric.nn import NNConv

import torch.nn.functional as F

from C_GNN.gnns.abstract_edge_attr_gnn import AbstractEdgeAttrGNN
from C_GNN.gnns.edge_atr_gnns_enums.aggr_enum import Aggr
from C_GNN.poolings.add_pooling import AddPooling


class NNConvGNN(AbstractEdgeAttrGNN):

    def __init__(self, sigmoid_output, dropout_prob, pooling, num_hidden_neurons, deep_nn, aggr):
        '''
        Defines a GNN architecture which uses NNConv.
        :param sigmoid_output: Whether to output a sigmoid.
        :param dropout_prob: The probability of dropout.
        :param deep_nn: Whether to use a deep neural net of shallow one.
        :param num_hidden_neurons: The number of hidden neurons in the hidden layers.
        :param aggr: The aggregation to use for the feed forward neural network.
        '''
        super().__init__(sigmoid_output, dropout_prob, pooling, num_hidden_neurons, deep_nn, aggr)

    def initialise_channels(self, in_channels, out_channels, num_edge_features=None):
        super().initialise_channels(in_channels, out_channels, num_edge_features)

        if self._deep_nn:
            self._nn1 = nn.Sequential(nn.Linear(num_edge_features, int(self._num_hidden_neurons / 4)), nn.LeakyReLU(), nn.Linear(int(self._num_hidden_neurons / 4), in_channels * self._num_hidden_neurons))
        else:
            self._nn1 = nn.Linear(num_edge_features, in_channels * self._num_hidden_neurons)
        self._conv1 = NNConv(in_channels, self._num_hidden_neurons, self._nn1, aggr=self._aggr.value)

        if self._deep_nn:
            self._nn2 = nn.Sequential(nn.Linear(num_edge_features, int(self._num_hidden_neurons / 4)), nn.LeakyReLU(), nn.Linear(int(self._num_hidden_neurons / 4), self._num_hidden_neurons * self._num_hidden_neurons))
        else:
            self._nn2 = nn.Linear(num_edge_features, self._num_hidden_neurons * self._num_hidden_neurons)
        self._conv2 = NNConv(self._num_hidden_neurons, self._num_hidden_neurons, self._nn2, aggr=self._aggr.value)

        if self._deep_nn:
            self._nn3 = nn.Sequential(nn.Linear(num_edge_features, int(self._num_hidden_neurons / 4)), nn.LeakyReLU(), nn.Linear(int(self._num_hidden_neurons / 4), self._num_hidden_neurons * self._num_hidden_neurons))
        else:
            self._nn3 = nn.Linear(num_edge_features, self._num_hidden_neurons * self._num_hidden_neurons)
        self._conv3 = NNConv(self._num_hidden_neurons, self._num_hidden_neurons, self._nn3, aggr=self._aggr.value)

        self._fc1 = torch.nn.Linear(self._post_pulling_num_neurons, self._num_hidden_neurons)
        self._fc2 = torch.nn.Linear(self._num_hidden_neurons, out_channels)

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(),
                **{
                   "nn1": self._nn1,
                   "conv1": self._conv1,
                    "nn2": self._nn2,
                    "conv2": self._conv2,
                    "nn3": self._nn3,
                    "conv3": self._conv3,
                    "fc1": self._fc1,
                    "fc2": self._fc2,
                }}

    def _perform_pre_pooling(self, x, edge_index, edge_attr):
        x = F.leaky_relu(self._conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self._dropout_prob, training=self.training)
        x = F.leaky_relu(self._conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self._dropout_prob, training=self.training)
        x = F.leaky_relu(self._conv3(x, edge_index, edge_attr))

        return x

    def _perform_post_pooling(self, x, edge_index, edge_attr):
        x = F.leaky_relu(self._fc1(x))
        x = self._fc2(x)

        return x

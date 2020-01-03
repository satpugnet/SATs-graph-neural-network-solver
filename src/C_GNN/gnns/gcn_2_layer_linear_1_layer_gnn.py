import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from C_GNN.abstract_gnn import AbstractGNN


class GCN2LayerLinear1LayerGNN(AbstractGNN):

    def __init__(self, sigmoid_output, dropout_prob, pooling, num_hidden_neurons):
        '''
        Defines a GNN architecture with 2 convolution layers and one linear layer.
        :param sigmoid_output: Whether to output a sigmoid.
        :param dropout_prob: The probability of dropout.
        '''
        super(GCN2LayerLinear1LayerGNN, self).__init__(sigmoid_output, dropout_prob, pooling, num_hidden_neurons)

    def initialise_channels(self, in_channels, out_channels, num_edge_features=None):
        super().initialise_channels(in_channels, out_channels, num_edge_features)

        self._conv1 = GCNConv(in_channels, self._num_hidden_neurons)
        self._conv2 = GCNConv(self._num_hidden_neurons, self._num_hidden_neurons)

        self._fc2 = nn.Linear(self._post_pulling_num_neurons, out_channels)

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(),
                **{
                   "conv1": self._conv1,
                   "conv2": self._conv2,
                    "fc2": self._fc2
                }}

    def _perform_pre_pooling(self, x, edge_index, edge_attr):
        x = F.leaky_relu(self._conv1(x, edge_index))
        x = F.dropout(x, p=self._dropout_prob, training=self.training)
        x = F.leaky_relu(self._conv2(x, edge_index))

        return x

    def _perform_post_pooling(self, x, edge_index, edge_attr):
        x = self._fc2(x)

        return x


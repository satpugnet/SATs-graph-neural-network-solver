import torch
from torch import nn
from torch_geometric.nn import global_add_pool, NNConv
import torch.nn.functional as F

from C_GNNs.abstract_gnn import AbstractGNN


class RepeatingNNConvGNN(AbstractGNN):
    def __init__(self, sigmoid_output=True, dropout_prob=0.5, deep_nn=False, num_hidden_neurons=8, conv_repetition=20,
                 ratio_test_train_rep=4):
        super().__init__(sigmoid_output, dropout_prob)
        self._deep_nn = deep_nn
        self._num_hidden_neurons = num_hidden_neurons
        self._conv_repetition = conv_repetition
        self._ratio_test_train_rep = ratio_test_train_rep

    def initialise_channels(self, in_channels, out_channels, num_edge_features=None):
        if self._deep_nn:
            self._nn1 = nn.Sequential(nn.Linear(num_edge_features, 16), nn.LeakyReLU(), nn.Linear(16, in_channels * self._num_hidden_neurons))
        else:
            self._nn1 = nn.Linear(num_edge_features, in_channels * self._num_hidden_neurons)
        self._conv1 = NNConv(in_channels, self._num_hidden_neurons, self._nn1, aggr='add')

        if self._deep_nn:
            self._nn2 = nn.Sequential(nn.Linear(num_edge_features, 16), nn.LeakyReLU(), nn.Linear(16, self._num_hidden_neurons * self._num_hidden_neurons))
        else:
            self._nn2 = nn.Linear(num_edge_features, self._num_hidden_neurons * self._num_hidden_neurons)
        self._conv2 = NNConv(self._num_hidden_neurons, self._num_hidden_neurons, self._nn2, aggr='add')

        self._pooling = global_add_pool

        self._fc1 = torch.nn.Linear(self._num_hidden_neurons, self._num_hidden_neurons)
        self._fc2 = torch.nn.Linear(self._num_hidden_neurons, 1)

    def __repr__(self):
        return "{}(nn1({}), conv1({}), dropout({}), conv_repetition({}), ratio_test_train_rep({}), nn2({}), conv2({}), " \
               "pooling ({}), fc1({}), fc2({}), sigmoid_output({}))"\
            .format(
                self.__class__.__name__,
                self._nn1,
                self._conv1,
                self._conv_repetition,
                self._ratio_test_train_rep,
                self._dropout_prob,
                self._nn2,
                self._conv2,
                self._pooling.__name__,
                self._fc1,
                self._fc2,
                self._sigmoid_output
            )

    def _perform_pre_pooling(self, x, edge_index, edge_attr):
        x = F.leaky_relu(self._conv1(x, edge_index, edge_attr))

        self._iterate_nnconv(x, edge_index, edge_attr)

        return x

    def _iterate_nnconv(self, x, edge_index, edge_attr):
        iteration_number = self._conv_repetition if self.training else self._conv_repetition * self._ratio_test_train_rep

        for i in range(iteration_number):
            x = F.dropout(x, p=self._dropout_prob, training=self.training)
            x = F.leaky_relu(self._conv2(x, edge_index, edge_attr))

    def _perform_pooling(self, x, batch):
        x = self._pooling(x, batch)

        return x

    def _perform_post_pooling(self, x, edge_index, edge_attr):
        x = F.leaky_relu(self._fc1(x))
        x = self._fc2(x)

        return x

import torch
from torch import nn
from torch_geometric.nn import NNConv, GCNConv
import torch.nn.functional as F

from C_GNN.gnns.abstract_edge_attr_gnn import AbstractEdgeAttrGNN

from C_GNN.gnns.enums.nn_types import NNTypes


class RepeatingNNConvGNN(AbstractEdgeAttrGNN):
    def __init__(self, sigmoid_output, dropout_prob, pooling, num_hidden_neurons, deep_nn, conv_repetition,
                 ratio_test_train_rep, aggr, num_layers_per_rep, nn_type):
        '''
        Defines a GNN architecture which uses NNConv and repeat a fixed number of time in the feedforward phase for training.
        :param sigmoid_output: Whether to output a sigmoid.
        :param dropout_prob: The probability of dropout.
        :param deep_nn: Whether to use a deep neural net of shallow one.
        :param num_hidden_neurons: The number of hidden neurons in the hidden layers.
        :param conv_repetition: The range in which to uniformly pick for the number of repetition of the ConvGNN.
        :param ratio_test_train_rep: The ratio of the number of repetition of the ConvGNN for the testing and training.
        '''
        super().__init__(sigmoid_output, dropout_prob, pooling, num_hidden_neurons, deep_nn, aggr)
        self._ratio_test_train_rep = ratio_test_train_rep
        self._conv_repetition = conv_repetition
        self._num_layers_per_rep = num_layers_per_rep
        self._nn_type = nn_type

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(),
                **{
                   "conv1": self._conv1,
                    "conv_repetition": self._conv_repetition,
                    "ratio_test_train_rep": self._ratio_test_train_rep,
                    "convs": self._convs,
                    "conv3": self._conv3,
                    "conv4": self._conv4,
                    "fc1": self._fc1,
                    "fc2": self._fc2,
                    "nn_type": self._nn_type
                }}

    def initialise_channels(self, in_channels, out_channels, num_edge_features=None):
        super().initialise_channels(in_channels, out_channels, num_edge_features)

        if self._nn_type == NNTypes.GNN:
            self.__initialise_gnn(in_channels, num_edge_features)

        elif self._nn_type == NNTypes.GCN:
            self.__initialise_gcn(in_channels)

        self._fc1 = torch.nn.Linear(self._post_pulling_num_neurons, self._num_hidden_neurons)
        self._fc2 = torch.nn.Linear(self._num_hidden_neurons, out_channels)

    def __initialise_gnn(self, in_channels, num_edge_features=None):
        self.__uses_edge_attr = True

        if self._deep_nn:
            self._nn1 = nn.Sequential(nn.Linear(num_edge_features, int(self._num_hidden_neurons / 4)), nn.LeakyReLU(), nn.Linear(int(self._num_hidden_neurons / 4), in_channels * self._num_hidden_neurons))
        else:
            self._nn1 = nn.Linear(num_edge_features, in_channels * self._num_hidden_neurons)
        self._conv1 = NNConv(in_channels, self._num_hidden_neurons, self._nn1, aggr=self._aggr.value)

        convs = []
        for _ in range(self._num_layers_per_rep):
            if self._deep_nn:
                self._nn2 = nn.Sequential(nn.Linear(num_edge_features, int(self._num_hidden_neurons / 4)), nn.LeakyReLU(), nn.Linear(int(self._num_hidden_neurons / 4), self._num_hidden_neurons * self._num_hidden_neurons))
            else:
                self._nn2 = nn.Linear(num_edge_features, self._num_hidden_neurons * self._num_hidden_neurons)
            convs.append(NNConv(self._num_hidden_neurons, self._num_hidden_neurons, self._nn2, aggr=self._aggr.value))

        if self._deep_nn:
            self._nn1 = nn.Sequential(nn.Linear(num_edge_features, int(self._num_hidden_neurons / 4)), nn.LeakyReLU(), nn.Linear(int(self._num_hidden_neurons / 4), in_channels * self._num_hidden_neurons))
        else:
            self._nn1 = nn.Linear(num_edge_features, in_channels * self._num_hidden_neurons)
        self._convs = nn.ModuleList(convs)

        self._conv3 = NNConv(self._num_hidden_neurons, in_channels, self._nn1, aggr=self._aggr.value)

        if self._deep_nn:
            self._nn1 = nn.Sequential(nn.Linear(num_edge_features, int(self._num_hidden_neurons / 4)), nn.LeakyReLU(), nn.Linear(int(self._num_hidden_neurons / 4), in_channels * self._num_hidden_neurons))
        else:
            self._nn1 = nn.Linear(num_edge_features, in_channels * self._num_hidden_neurons)
        self._conv4 = NNConv(in_channels, self._num_hidden_neurons, self._nn1, aggr=self._aggr.value)

    def __initialise_gcn(self, in_channels):
        self.__uses_edge_attr = False

        self._conv1 = GCNConv(in_channels, self._num_hidden_neurons, improved=True)

        convs = []
        for _ in range(self._num_layers_per_rep):
            convs.append(GCNConv(self._num_hidden_neurons, self._num_hidden_neurons, improved=True))

        self._convs = nn.ModuleList(convs)

        self._conv3 = GCNConv(self._num_hidden_neurons, in_channels, improved=True)

        self._conv4 = GCNConv(in_channels, self._num_hidden_neurons, improved=True)


    def _perform_pre_pooling(self, x, edge_index, edge_attr):
        x = self._iterate_nnconv(x, edge_index, edge_attr)

        x = F.leaky_relu(self._conv4(x, edge_index, edge_attr) if self.__uses_edge_attr else self._conv4(x, edge_index))

        return x

    def _iterate_nnconv(self, x, edge_index, edge_attr):
        num_iterations = self._compute_num_iterations()

        for i in range(num_iterations):
            x = F.dropout(x, p=self._dropout_prob, training=self.training)
            x = F.leaky_relu(self._conv1(x, edge_index, edge_attr) if self.__uses_edge_attr else self._conv1(x, edge_index))

            for conv in self._convs:
                if self.__uses_edge_attr:
                    x = F.leaky_relu(conv(x, edge_index, edge_attr) if self.__uses_edge_attr else conv(x, edge_index))

            x = F.leaky_relu(self._conv3(x, edge_index, edge_attr) if self.__uses_edge_attr else self._conv3(x, edge_index))

        return x

    def _compute_num_iterations(self):
        return self._conv_repetition if self.training else self._conv_repetition * self._ratio_test_train_rep

    def _perform_post_pooling(self, x, edge_index, edge_attr):
        x = F.leaky_relu(self._fc1(x))
        x = self._fc2(x)

        return x

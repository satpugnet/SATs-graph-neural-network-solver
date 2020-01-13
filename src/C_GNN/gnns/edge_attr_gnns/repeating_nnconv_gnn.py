import torch
from torch import nn
from torch_geometric.nn import NNConv, GCNConv, GraphConv
import torch.nn.functional as F

from C_GNN.gnns.abstract_edge_attr_gnn import AbstractEdgeAttrGNN

from C_GNN.gnns.enums.nn_types import NNTypes


class RepeatingNNConvGNN(AbstractEdgeAttrGNN):
    def __init__(self, sigmoid_output, dropout_prob, pooling, num_hidden_neurons, deep_nn, conv_repetition,
                 ratio_test_train_rep, aggr, num_layers_per_rep, nn_type, rep_layer_in_out_num_neurons=None):
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
        self._rep_layer_in_out_num_neurons = rep_layer_in_out_num_neurons

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
        
        if self._rep_layer_in_out_num_neurons is None:
            self._rep_layer_in_out_num_neurons = in_channels

        if self._nn_type == NNTypes.GNN:
            self.__uses_edge_attr = True
            self.__initialise_nn(self.__create_NNConv, in_channels, num_edge_features)

        elif self._nn_type == NNTypes.GCN:
            self.__uses_edge_attr = False
            self.__initialise_nn(self.__create_GCNConv, in_channels, num_edge_features)

        elif self._nn_type == NNTypes.GraphConv:
            self.__uses_edge_attr = False
            self.__initialise_nn(self.__create_GraphConv, in_channels, num_edge_features)

        self._fc1 = torch.nn.Linear(self._post_pulling_num_neurons, self._num_hidden_neurons)
        self._fc2 = torch.nn.Linear(self._num_hidden_neurons, out_channels)

    def __initialise_nn(self, conv_init_func, in_channels, num_edge_features=None):
        self._conv0 = conv_init_func(in_channels, self._rep_layer_in_out_num_neurons, num_edge_features)

        self._conv1 = conv_init_func(self._rep_layer_in_out_num_neurons, self._num_hidden_neurons, num_edge_features)
        convs = []
        for _ in range(self._num_layers_per_rep):
            convs.append(conv_init_func(self._num_hidden_neurons, self._num_hidden_neurons, num_edge_features))
        self._convs = nn.ModuleList(convs)
        self._conv3 = conv_init_func(self._num_hidden_neurons, self._rep_layer_in_out_num_neurons, num_edge_features)

        self._conv4 = conv_init_func(self._rep_layer_in_out_num_neurons, self._num_hidden_neurons, num_edge_features)
    
    def __create_NNConv(self, in_channels, out_channels, num_edge_features=None):
        if self._deep_nn:
            nn_value = nn.Sequential(nn.Linear(num_edge_features, int(out_channels / 4)), nn.LeakyReLU(), nn.Linear(int(out_channels / 4), in_channels * out_channels))
        else:
            nn_value = nn.Linear(num_edge_features, in_channels * out_channels)
        return NNConv(in_channels, out_channels, nn_value, aggr=self._aggr.value)

    def __create_GCNConv(self, in_channels, out_channels, num_edge_features=None):
        return GCNConv(in_channels, out_channels, improved=True)

    def __create_GraphConv(self, in_channels, out_channels, num_edge_features=None):
        return GraphConv(in_channels, out_channels, aggr=self._aggr.value)

    def _perform_pre_pooling(self, x, edge_index, edge_attr):
        x = F.leaky_relu(self._conv0(x, edge_index, edge_attr) if self.__uses_edge_attr else self._conv0(x, edge_index))
        
        x = self._iterate_nnconv(x, edge_index, edge_attr)
        
        x = F.leaky_relu(self._conv4(x, edge_index, edge_attr) if self.__uses_edge_attr else self._conv4(x, edge_index))
        
        return x

    def _iterate_nnconv(self, x, edge_index, edge_attr):
        num_iterations = self._compute_num_iterations()
            
        for i in range(num_iterations):
            x = F.dropout(x, p=self._dropout_prob, training=self.training)
            x = F.leaky_relu(self._conv1(x, edge_index, edge_attr) if self.__uses_edge_attr else self._conv1(x, edge_index))

            for conv in self._convs:
                x = F.leaky_relu(conv(x, edge_index, edge_attr) if self.__uses_edge_attr else conv(x, edge_index))

            x = F.leaky_relu(self._conv3(x, edge_index, edge_attr) if self.__uses_edge_attr else self._conv3(x, edge_index))
        
        return x

    def _compute_num_iterations(self):
        return self._conv_repetition if self.training else self._conv_repetition * self._ratio_test_train_rep

    def _perform_post_pooling(self, x, edge_index, edge_attr):
        x = F.leaky_relu(self._fc1(x))
        x = self._fc2(x)
        
        return x

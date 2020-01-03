import random

import torch.nn.functional as F

from C_GNN.gnns.edge_atr_gnns_enums.aggr_enum import Aggr
from C_GNN.gnns.edge_attr_gnns.repeating_nnconv_gnn import RepeatingNNConvGNN
from C_GNN.poolings.add_pooling import AddPooling


class VariableRepeatingNNConvGNN(RepeatingNNConvGNN):

    def __init__(self, sigmoid_output=True, dropout_prob=0.5, pooling=AddPooling(), num_hidden_neurons=8, deep_nn=False,
                 conv_min_max_rep=(10, 20), ratio_test_train_rep=4, aggr=Aggr.ADD):
        '''
        Defines a GNN architecture which uses NNConv and repeat a random number of time in the feedforward phase for training.
        :param sigmoid_output: Whether to output a sigmoid.
        :param dropout_prob: The probability of dropout.
        :param deep_nn: Whether to use a deep neural net of shallow one.
        :param num_hidden_neurons: The number of hidden neurons in the hidden layers.
        :param conv_min_max_rep: The range in which to uniformly pick for the number of repetition of the ConvGNN.
        :param ratio_test_train_rep: The ratio of the number of repetition of the ConvGNN for the testing and training.
        '''
        super().__init__(sigmoid_output, dropout_prob, pooling, num_hidden_neurons, deep_nn, conv_min_max_rep[1],
                         ratio_test_train_rep, aggr)
        self._conv_min_max_rep = conv_min_max_rep

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(),
                **{
                    "conv_min_max_rep": self._conv_min_max_rep
                }}

    def _iterate_nnconv(self, x, edge_index, edge_attr):
        if self.training:
            iteration_number = random.randint(self._conv_min_max_rep[0], self._conv_min_max_rep[1])
        else:
            iteration_number = int(self._conv_repetition * self._ratio_test_train_rep)

        for i in range(iteration_number):
            x = F.dropout(x, p=self._dropout_prob, training=self.training)
            x = F.leaky_relu(self._conv2(x, edge_index, edge_attr))


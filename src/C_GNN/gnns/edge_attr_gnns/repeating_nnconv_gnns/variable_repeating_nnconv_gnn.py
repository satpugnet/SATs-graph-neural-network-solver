import random

import torch.nn.functional as F

from C_GNN.gnns.edge_attr_gnns.repeating_nnconv_gnn import RepeatingNNConvGNN


class VariableRepeatingNNConvGNN(RepeatingNNConvGNN):

    def __init__(self, sigmoid_output, dropout_prob, pooling, num_hidden_neurons, deep_nn, conv_min_max_rep,
                 ratio_test_train_rep, aggr, num_repeating_layer):
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
                         ratio_test_train_rep, aggr, num_repeating_layer)
        self._conv_min_max_rep = conv_min_max_rep

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(),
                **{
                    "conv_min_max_rep": self._conv_min_max_rep
                }}

    def _compute_num_iterations(self):
        if self.training:
            num_iterations = random.randint(self._conv_min_max_rep[0], self._conv_min_max_rep[1])
        else:
            num_iterations = int(self._conv_repetition * self._ratio_test_train_rep)

        return num_iterations


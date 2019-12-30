from abc import ABC, abstractmethod

import torch


class AbstractGNN(torch.nn.Module, ABC):

    def __init__(self, sigmoid_output=True, dropout_prob=0.5):
        super(AbstractGNN, self).__init__()
        self._sigmoid_output = sigmoid_output
        self._dropout_prob = dropout_prob

    # Initialise the dynamic part of the data later in a method so that we can build the object in the config, this
    # could be replaced by a constructor pattern
    @abstractmethod
    def initialise_channels(self, in_channels, out_channels, num_edge_features=None):
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._num_edge_features = num_edge_features

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self._perform_pre_pooling(x, edge_index, edge_attr)
        x = self._perform_pooling(x, batch)
        x = self._perform_post_pooling(x, edge_index, edge_attr)

        return torch.sigmoid(x) if self._sigmoid_output else x

    @abstractmethod
    def _perform_pre_pooling(self, x, edge_index, edge_attr):
        pass

    @abstractmethod
    def _perform_pooling(self, x, batch):
        pass

    @abstractmethod
    def _perform_post_pooling(self, x, edge_index, edge_attr):
        pass
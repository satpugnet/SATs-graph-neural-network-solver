from abc import ABC, abstractmethod

import torch


class AbstractGNN(torch.nn.Module, ABC):

    def __init__(self, num_node_features, num_y_features):
        super(AbstractGNN, self).__init__()
        self._num_node_features = num_node_features
        self._num_y_features = num_y_features

    @abstractmethod
    def forward(self, data):
        pass



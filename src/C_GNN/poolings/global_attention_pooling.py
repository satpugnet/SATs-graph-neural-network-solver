from torch import nn
from torch_geometric.nn import GlobalAttention

from C_GNN.abstract_pooling import AbstractPooling


class GlobalAttentionPooling(AbstractPooling):
    def __init__(self, num_hidden_neurons, deep_nn):
        super().__init__()
        self._num_hidden_neurons = num_hidden_neurons
        self._deep_nn = deep_nn

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(), **{
            "gate_nn": self._gate_nn,
            "nn": self._nn,
        }}

    def _initialise(self, in_channels, out_channels):
        if self._deep_nn:
            self._gate_nn = nn.Sequential(nn.Linear(in_channels, int(self._num_hidden_neurons)), nn.LeakyReLU(), nn.Linear(int(self._num_hidden_neurons), 1))
        else:
            self._gate_nn = nn.Linear(in_channels, 1)

        if self._deep_nn:
            self._nn = nn.Sequential(nn.Linear(in_channels, int(self._num_hidden_neurons)), nn.LeakyReLU(), nn.Linear(int(self._num_hidden_neurons), out_channels))
        else:
            self._nn = nn.Linear(in_channels, out_channels)
            
        self._pooling = GlobalAttention(self._gate_nn, self._nn)

        return out_channels

    def pool(self, x, batch):
        return self._pooling(x, batch)

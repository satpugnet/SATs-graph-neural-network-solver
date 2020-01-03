from torch import nn
from torch_geometric.nn import GlobalAttention, Set2Set

from C_GNN.abstract_pooling import AbstractPooling


class SetToSetPooling(AbstractPooling):
    def __init__(self, processing_steps, num_layers):
        super().__init__()
        self._processing_steps = processing_steps
        self._num_layers = num_layers

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(), **{
            "in_channels": self._in_channels,
            "processing_steps": self._processing_steps,
            "num_layers": self._num_layers
        }}

    def _initialise(self, in_channels, out_channels):
        self._in_channels = in_channels
        self._pooling = Set2Set(in_channels, self._processing_steps, self._num_layers)

        return in_channels * 2

    def pool(self, x, batch):
        return self._pooling(x, batch)

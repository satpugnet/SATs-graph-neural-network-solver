from torch_geometric.nn import global_mean_pool, global_max_pool, global_sort_pool

from C_GNN.abstract_pooling import AbstractPooling


class SortPooling(AbstractPooling):
    def __init__(self, k):
        super().__init__()
        self._k = k

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(), **{
            "k": self._k
        }}

    def _initialise(self, in_channels, out_channels):
        return in_channels * self._k

    def pool(self, x, batch):
        return global_sort_pool(x, batch, self._k)

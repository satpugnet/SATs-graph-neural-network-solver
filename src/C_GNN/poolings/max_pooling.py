from torch_geometric.nn import global_mean_pool, global_max_pool

from C_GNN.abstract_pooling import AbstractPooling


class MaxPooling(AbstractPooling):
    def __init__(self):
        super().__init__()

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(), **{}}

    def pool(self, x, batch):
        return global_max_pool(x, batch)

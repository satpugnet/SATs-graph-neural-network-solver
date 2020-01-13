from torch_geometric.nn import DataParallel

from utils.abstract_repr import AbstractRepr


class DataParallelWrapper(DataParallel, AbstractRepr):

    def __init__(self, module):
        super().__init__(module)

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(), **{}}
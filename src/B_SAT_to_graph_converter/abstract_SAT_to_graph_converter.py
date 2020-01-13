import time
from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Data

from utils import logger
from utils.abstract_repr import AbstractRepr


class AbstractSATToGraphConverter(ABC, AbstractRepr):

    def __init__(self):
        '''
        The algorithm used to convert from SAT problems to graphs.
        '''
        pass

    def _get_fields_for_repr(self):
        return {}

    def convert_all(self, SAT_problems):
        graphs_data = []

        for i in range(len(SAT_problems)):
            logger.get().debug("{:.1f}% SAT problem converted\r".format(i / len(SAT_problems) * 100))
            graphs_data.append(self.__convert(SAT_problems[i]))

        return graphs_data

    def __convert(self, SAT_problem):
        x = torch.tensor(self._compute_x(SAT_problem), dtype=torch.float)
        y = torch.tensor([SAT_problem.is_sat], dtype=torch.float)

        edge_index_raw, edge_attr_raw = self.__convert_edges_from_dict(self._compute_edges(SAT_problem.clauses, SAT_problem.n_vars))
        edge_index = torch.tensor(edge_index_raw, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_raw, dtype=torch.float)

        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

    def __convert_edges_from_dict(self, edges):
        edge_start = []
        edge_end = []
        edge_attr = []

        for key in edges:
            edge_start.append(key[0])
            edge_end.append(key[1])
            edge_attr.append(edges[key])

        return [edge_start, edge_end], edge_attr

    @abstractmethod
    def _compute_x(self, SAT_problem):
        # Example x: [[2, 3, 4], [4, 7, 5], [4, 7, 5]]
        pass

    @abstractmethod
    def _compute_edges(self, clauses, n_vars):
        # Example edge_index: [[0, 1, 1], [0, 1, 0]]
        #         edge_attr:  [[4, 2], [3, 2], [1, 2]]
        pass

    # TODO: make that this is done at the end if a flag is_undirected is activate, every edge is doubled instead of
    # putting both in every time
    def _add_undirected_edge_to_dict(self, edge_dict, edge_end1, edge_end2, value):
        edge_dict[(edge_end1, edge_end2)] = value
        edge_dict[(edge_end2, edge_end1)] = value

        return edge_dict
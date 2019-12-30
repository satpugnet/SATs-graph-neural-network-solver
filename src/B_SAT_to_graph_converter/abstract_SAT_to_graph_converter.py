from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Data


class AbstractSATToGraphConverter(ABC):

    def __init__(self):
        pass

    def __repr__(self):
        return "{}(max_clause_length({}))".format(self.__class__.__name__, self._max_clause_length)

    def convert_all(self, SAT_problems):
        print("Converting SAT problems to graphs")
        graphs_data = []

        for SAT_problem in SAT_problems:
            graphs_data.append(self.__convert(SAT_problem))

        print("Finished to convert SAT problems to graphs")
        return graphs_data

    def __convert(self, SAT_problem):
        x = torch.tensor(self._compute_x(SAT_problem), dtype=torch.float)
        y = torch.tensor([SAT_problem.is_sat], dtype=torch.float)

        edge_index_raw, edge_attr_raw = self._compute_edges(SAT_problem)
        edge_index = torch.tensor(edge_index_raw, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_raw, dtype=torch.float)

        # x = torch.tensor([[2, 3, 4], [4, 7, 5], [4, 7, 5]], dtype=torch.float)
        # y = torch.tensor([1], dtype=torch.long)
        # edge_index = torch.tensor([[0, 1, 1], [0, 1, 0]], dtype=torch.long)
        # edge_attr = torch.tensor([[4, 2], [3, 2], [1, 2]], dtype=torch.long)

        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

    @abstractmethod
    def _compute_x(self, SAT_problem):
        pass

    @abstractmethod
    def _compute_edges(self, SAT_problem):
        pass

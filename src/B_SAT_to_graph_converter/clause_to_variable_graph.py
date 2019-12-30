import torch

from B_SAT_to_graph_converter.abstract_SAT_to_graph_converter import AbstractSATToGraphConverter


class ClausToVariableGraph(AbstractSATToGraphConverter):

    def __init__(self):
        super().__init__()

    def _compute_x(self, SAT_problem):
        pass

    def _compute_edges(self, SAT_problem):
        pass

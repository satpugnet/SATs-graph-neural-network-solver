from abc import ABC, abstractmethod

from B_SAT_to_graph_converter.abstract_SAT_to_graph_converter import AbstractSATToGraphConverter


class AbstractClauseVarGraphConverter(AbstractSATToGraphConverter, ABC):

    def __init__(self):
        super().__init__()
        self.__opposite_edge_attr = [1]

    def _compute_x(self, SAT_problem):
        # Example x: [[2, 3, 4], [4, 7, 5], [4, 7, 5]]
        return self._compute_clauses_node(SAT_problem.n_clauses) + self._compute_lits_node(SAT_problem.n_vars, SAT_problem.lits_present)

    @abstractmethod
    def _compute_clauses_node(self, n_clauses):
        pass

    @abstractmethod
    def _get_clause_node_index(self, clause_index, n_clauses, n_vars):
        pass

    @abstractmethod
    def _compute_lits_node(self, n_vars, lits_present):
        pass

    @abstractmethod
    def _get_lit_node_index(self, lit_index, n_clauses, n_vars):
        pass

    def _compute_edges(self, SAT_problem):
        neg_edges = self.__compute_opposite_edges(SAT_problem.n_clauses, SAT_problem.n_vars)
        var_clause_edges = self._compute_other_edges(SAT_problem.clauses, SAT_problem.n_vars)

        return self._sum_edges([neg_edges, var_clause_edges])

    def __compute_opposite_edges(self, n_clauses, n_vars):
        opposite_edges = {}
        for i in range(n_vars):
            current_var = i + 1
            pos_lit_index = self._get_lit_node_index(current_var, n_clauses, n_vars)
            neg_lit_index = self._get_lit_node_index(-current_var, n_clauses, n_vars)
            opposite_edges = self._add_undirected_edge_to_dict(opposite_edges, pos_lit_index, neg_lit_index, self.__opposite_edge_attr)

        return opposite_edges

    @abstractmethod
    def _compute_other_edges(self, clauses, n_vars):
        pass

    def _sum_edges(self, edges_list):
        final_edges = edges_list[0]

        for edge in edges_list[1:]:
            final_edges_index, final_edges_attr = zip(*final_edges.items())
            edge_index, edge_attr = zip(*edge.items())
            final_edges = dict(zip(final_edges_index + edge_index, self.__sum_edges_attr_with_padding(final_edges_attr, edge_attr)))

        return final_edges

    def __sum_edges_attr_with_padding(self, edges_attr1, edges_attr2):
        new_edges_attr1 = [attr + [0] * len(edges_attr2[0]) for attr in edges_attr1]
        new_edges_attr2 = [[0] * len(edges_attr1[0]) + attr for attr in edges_attr2]

        return new_edges_attr1 + new_edges_attr2



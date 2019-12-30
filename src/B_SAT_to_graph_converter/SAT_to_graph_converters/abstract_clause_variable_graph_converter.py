from abc import ABC, abstractmethod

from B_SAT_to_graph_converter.abstract_SAT_to_graph_converter import AbstractSATToGraphConverter


class AbstractClauseVariableGraphConverter(AbstractSATToGraphConverter, ABC):

    def __init__(self):
        super().__init__()
        self.__opposite_edge_attr = [1]

    @property
    @abstractmethod
    def _lit_node_feature(self):
        pass

    def _compute_x(self, SAT_problem):
        # Example x: [[2, 3, 4], [4, 7, 5], [4, 7, 5]]
        return self._compute_clauses_node(SAT_problem.n_clauses) + self._compute_lits_node(SAT_problem.n_vars, SAT_problem.lits_present)

    @abstractmethod
    def _compute_clauses_node(self, n_clauses):
        pass

    @abstractmethod
    def _get_clause_node_index(self, clause_index, n_clauses, n_vars):
        pass

    def _compute_lits_node(self, n_vars, lits_present):
        lits_nodes = [[0] * len(self._lit_node_feature)] * n_vars * 2

        for lit in lits_present:
            if lit > 0:
                lit_index = lit - 1
                lits_nodes[lit_index] = self._lit_node_feature
            else:
                lit_index = n_vars + abs(lit) - 1
                lits_nodes[lit_index] = [-elem for elem in self._lit_node_feature]

        return lits_nodes

    @abstractmethod
    def _get_lit_node_index(self, lit, n_clauses, n_vars):
        pass

    def _compute_edges(self, SAT_problem):
        neg_edges = self.__compute_opposite_edges(SAT_problem.n_clauses, SAT_problem.n_vars)
        other_edges = self._compute_other_edges(SAT_problem.clauses, SAT_problem.n_vars)

        return self._sum_edges([neg_edges, other_edges])

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
        '''Takes a list of dict or a list of int (size of padding to add to all attributes) as an input'''
        final_edges = {}

        for edge in edges_list:

            final_edges_index, final_edges_attr = zip(*final_edges.items()) if len(final_edges) != 0 else [[], []]

            if isinstance(edge, int):
                edge_index, edge_attr = [], [[0] * edge]

            elif isinstance(edge, dict):
                if len(edge) == 0:
                    raise Exception("The edge should either be an integer or contain some edges.")
                edge_index, edge_attr = zip(*edge.items())

            new_edge_index = list(final_edges_index) + list(edge_index)
            new_edge_attr = self.__sum_edges_attr_with_padding(final_edges_attr, edge_attr)
            final_edges = dict(zip(new_edge_index, new_edge_attr))

        return final_edges

    def __sum_edges_attr_with_padding(self, edges_attr1, edges_attr2):
        len_edges_attr2 = len(edges_attr2[0]) if len(edges_attr2) != 0 else 0
        new_edges_attr1 = [attr + [0] * len_edges_attr2 for attr in edges_attr1]

        len_edges_attr1 = len(edges_attr1[0]) if len(edges_attr1) != 0 else 0
        new_edges_attr2 = [[0] * len_edges_attr1 + attr for attr in edges_attr2]

        return new_edges_attr1 + new_edges_attr2



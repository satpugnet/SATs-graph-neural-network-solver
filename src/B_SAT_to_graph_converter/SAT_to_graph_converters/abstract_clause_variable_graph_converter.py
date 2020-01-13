from abc import ABC, abstractmethod

from B_SAT_to_graph_converter.abstract_SAT_to_graph_converter import AbstractSATToGraphConverter


class AbstractClauseVariableGraphConverter(AbstractSATToGraphConverter, ABC):

    def __init__(self):
        '''
        Converting SAT problems to graphs using the clauses and variables as nodes.
        '''
        super().__init__()
        self.__opposite_edge_attr = [1]

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(), **{}}

    @property
    @abstractmethod
    def _include_opposite_lit_edges(self):
        pass

    def _compute_x(self, SAT_problem):
        # Example x: [[2, 3, 4], [4, 7, 5], [4, 7, 5]]
        self._clauses_nodes = self._compute_clauses_nodes(SAT_problem.clauses, SAT_problem.n_vars)
        return self._clauses_nodes + self._compute_lits_node(SAT_problem.n_vars, SAT_problem.lits_present)

    @abstractmethod
    def _compute_clauses_nodes(self, clauses, n_vars):
        pass

    def _get_clause_node_index(self, clause_index):
        # clause_index starts at 0
        return clause_index

    @abstractmethod
    def _compute_lits_node(self, n_vars, lits_present):
        pass

    # TODO: Deal better with the case when one of the litteral is not present (for example in [[-1], [-3, -1]], there is no 2),
    # currently it simply put an attribute of an unconnected node but this should be completly removed from the graph
    def _compute_default_lits_node(self, n_vars, lits_present, lit_node_feature, only_positive_nodes):
        lits_nodes = [[0] * len(lit_node_feature)] * n_vars * (1 if only_positive_nodes else 2)

        if only_positive_nodes:
            lits_present = list(set([abs(lit) for lit in lits_present]))

        for lit in lits_present:
            if lit > 0:
                lit_index = lit - 1
                lits_nodes[lit_index] = lit_node_feature
            else:
                lit_index = n_vars + abs(lit) - 1
                lits_nodes[lit_index] = [-elem for elem in lit_node_feature]

        return lits_nodes

    def _get_lit_node_index(self, lit, n_clauses, n_vars, only_positive_nodes):
        # lit index starts at 1 and can be positive and negative (for positive and negative variables)
        if lit > 0 or only_positive_nodes:
            position_index = abs(lit) - 1
        else:
            position_index = n_vars + abs(lit) - 1
        return len(self._clauses_nodes) + position_index

    def _compute_edges(self, clauses, n_vars):
        neg_edges = {}
        if self._include_opposite_lit_edges:
            neg_edges = self._compute_opposite_edges(len(clauses), n_vars)
        other_edges = self._compute_extra_edges(clauses, n_vars)

        return self._sum_edges([neg_edges, other_edges])

    def _compute_opposite_edges(self, n_clauses, n_vars):
        opposite_edges = {}
        for i in range(n_vars):
            current_var = i + 1
            pos_lit_index = self._get_lit_node_index(current_var, n_clauses, n_vars, False)
            neg_lit_index = self._get_lit_node_index(-current_var, n_clauses, n_vars, False)
            opposite_edges = self._add_undirected_edge_to_dict(opposite_edges, pos_lit_index, neg_lit_index, self.__opposite_edge_attr)

        return opposite_edges

    @abstractmethod
    def _compute_extra_edges(self, clauses, n_vars):
        pass

    def _sum_edges(self, edges_list):
        '''Takes a list of dict or a list of int (size of padding to add to all attributes) as an input'''
        final_edges = {}

        for edge in edges_list:

            final_edges_index, final_edges_attr = zip(*final_edges.items()) if len(final_edges) != 0 else [[], []]

            edge_index = []
            edge_attr = []

            if isinstance(edge, int):
                edge_index, edge_attr = [], [[0] * edge]

            elif isinstance(edge, dict):
                if len(edge) != 0:
                    edge_index, edge_attr = zip(*edge.items())

            else:
                raise Exception("The edge should either be an integer or a dict.")

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



from B_SAT_to_graph_converter.SAT_to_graph_converters.abstract_clause_variable_graph_converter import \
    AbstractClauseVariableGraphConverter


class ClauseToVariableGraph(AbstractClauseVariableGraphConverter):

    def __init__(self, represent_opposite_lits_in_edge):
        '''
        Converting SAT problems to graphs using both the clauses and variables as nodes.
        '''
        super().__init__()
        self.__represent_opposite_lits_in_edge = represent_opposite_lits_in_edge
        self.__clause_var_edge_attr = [1]
        self.__clause_node_feature = [1, 0]
        self.__var_node_feature = [0, 1]

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(),
                **{
                   "represent_opposite_lits_in_edge": self.__represent_opposite_lits_in_edge
               }}

    @property
    def _include_opposite_lit_edges(self):
        return not self.__represent_opposite_lits_in_edge

    def _compute_clauses_nodes(self, clauses, n_vars):
        return [self.__clause_node_feature] * len(clauses)

    def _compute_lits_node(self, n_vars, lits_present):
        return self._compute_default_lits_node(n_vars, lits_present, self.__var_node_feature, only_positive_nodes=self.__represent_opposite_lits_in_edge)

    def _compute_extra_edges(self, clauses, n_vars):
        return self._sum_edges([self.__compute_var_to_clause_edges(clauses, n_vars)])

    def __compute_var_to_clause_edges(self, clauses, n_vars):
        edges = {}
        for i in range(len(clauses)):
            for lit in clauses[i]:
                clause_nodes_index = self._get_clause_node_index(i)
                lit_nodes_index = self._get_lit_node_index(lit, len(clauses), n_vars, self.__represent_opposite_lits_in_edge)

                if self.__represent_opposite_lits_in_edge:
                    clause_var_edge_attr = self.__clause_var_edge_attr if lit > 0 else [-attr for attr in self.__clause_var_edge_attr]
                else:
                    clause_var_edge_attr = self.__clause_var_edge_attr

                edges = self._add_undirected_edge_to_dict(edges, clause_nodes_index, lit_nodes_index, clause_var_edge_attr)

        return edges

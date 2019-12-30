from B_SAT_to_graph_converter.SAT_to_graph_converters.abstract_clause_var_graph_converter import AbstractClauseVarGraphConverter


class ClauseToVariableGraph(AbstractClauseVarGraphConverter):

    def __init__(self):
        super().__init__()
        self.__clause_var_edge_attr = [1]
        self.__clause_node_feature = [1, 0]

    @property
    def _lit_node_feature(self):
        return [0, 1]

    def _compute_clauses_node(self, n_clauses):
        return [self.__clause_node_feature] * n_clauses

    def _get_clause_node_index(self, clause_index, n_clauses, n_vars):
        # clause_index starts at 0
        return clause_index

    def _get_lit_node_index(self, lit, n_clauses, n_vars):
        # lit index starts at 1 and can be positive and negative (for positive and negative variables)

        position_index = lit if lit > 0 else n_vars + abs(lit)
        return n_clauses - 1 + position_index

    def _compute_other_edges(self, clauses, n_vars):
        return self._sum_edges([self.__compute_var_to_clause_edges(clauses, n_vars)])

    def __compute_var_to_clause_edges(self, clauses, n_vars):
        edges = {}
        for i in range(len(clauses)):
            for lit in clauses[i]:
                clause_nodes_index = self._get_clause_node_index(i, len(clauses), n_vars)
                lit_nodes_index = self._get_lit_node_index(lit, len(clauses), n_vars)
                edges = self._add_undirected_edge_to_dict(edges, clause_nodes_index, lit_nodes_index, self.__clause_var_edge_attr)

        return edges

from B_SAT_to_graph_converter.SAT_to_graph_converters.abstract_clause_variable_graph_converter import \
    AbstractClauseVariableGraphConverter


class ClauseToClauseGraph(AbstractClauseVariableGraphConverter):

    def __init__(self, max_num_vars):
        super().__init__()
        self.__clause_to_clause_edge_attr = [1]
        self.__lit_positive_node_attr = 1
        self.__lit_negative_node_attr = -1
        self.__max_num_vars = max_num_vars

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(),
                **{
                   "max_num_vars": self.__max_num_vars
               }}

    @property
    def _include_opposite_lit_edges(self):
        return False

    def _compute_clauses_nodes(self, clauses, n_vars):
        if n_vars > self.__max_num_vars:
            raise Exception("There are more variable in the SAT problem than the max_num_vars specifies as the max")

        clauses_nodes = []
        for clause in clauses:
            node_attr = [0] * self.__max_num_vars

            for lit in clause:
                node_attr[abs(lit) - 1] = self.__lit_positive_node_attr if lit > 0 else self.__lit_negative_node_attr

            clauses_nodes.append(node_attr)

        return clauses_nodes

    def _compute_lits_node(self, n_vars, lits_present):
        return []

    def _compute_extra_edges(self, clauses, n_vars):
        edges = {}
        for i in range(len(clauses)):
            for j in range(i, len(clauses)):
                if i == j:
                    continue
                clause_nodes_index_1 = self._get_clause_node_index(i)
                clause_nodes_index_2 = self._get_clause_node_index(j)
                edges = self._add_undirected_edge_to_dict(edges, clause_nodes_index_1, clause_nodes_index_2, self.__clause_to_clause_edge_attr)

        return edges

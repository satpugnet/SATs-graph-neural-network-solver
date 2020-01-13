from B_SAT_to_graph_converter.SAT_to_graph_converters.abstract_clause_variable_graph_converter import \
    AbstractClauseVariableGraphConverter


class VariableToVariableGraph(AbstractClauseVariableGraphConverter):

    def __init__(self, max_num_clauses):
        '''
        Converting SAT problems to graphs using only the variables as nodes.
        '''
        super().__init__()
        self._max_num_clauses = max_num_clauses

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(),
                **{
                   "max_num_clauses": self._max_num_clauses
               }}

    @property
    def _include_opposite_lit_edges(self):
        return True

    def _compute_clauses_nodes(self, clauses, n_vars):
        return []

    def _compute_lits_node(self, n_vars, lits_present):
        return self._compute_default_lits_node(n_vars, lits_present, [1], only_positive_nodes=False)

    def _compute_extra_edges(self, clauses, n_vars):
        edges = {}

        for i in range(len(clauses)):
            current_clause = clauses[i]

            for lit1 in current_clause:
                for lit2 in current_clause:

                    lit_index1 = self._get_lit_node_index(lit1, len(clauses), n_vars, False)
                    lit_index2 = self._get_lit_node_index(lit2, len(clauses), n_vars, False)

                    if lit_index1 != lit_index2:
                        if (lit_index1, lit_index2) in edges:
                            edge_attr = edges[(lit_index1, lit_index2)]
                        else:
                            edge_attr = [0] * self._max_num_clauses

                        if i >= self._max_num_clauses:
                            raise Exception("The max_num_clauses value ({}) is too low for the given clause of size {}".format(self._max_num_clauses, str(i)))
                        edge_attr[i] = 1
                        edges[(lit_index1, lit_index2)] = edge_attr

        return edges if len(edges) != 0 else self._max_num_clauses
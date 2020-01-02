from B_SAT_to_graph_converter.SAT_to_graph_converters.abstract_clause_variable_graph_converter import \
    AbstractClauseVariableGraphConverter


class VariableToVariableGraph(AbstractClauseVariableGraphConverter):

    def __init__(self, max_clause_length):
        '''
        Converting SAT problems to graphs using only the variables as nodes.
        '''
        super().__init__()
        self._max_clause_length = max_clause_length

    def __repr__(self):
        return "{}(max_clause_length({}))".format(self.__class__.__name__, self._max_clause_length)

    @property
    def _lit_node_feature(self):
        return [1]

    def _compute_clauses_node(self, n_clauses):
        return []

    def _get_clause_node_index(self, clause_index, n_clauses, n_vars):
        return clause_index

    def _get_lit_node_index(self, lit, n_clauses, n_vars):
        return n_vars + abs(lit) - 1 if lit < 0 else lit - 1

    def _compute_edges(self, clauses, n_vars):
        edges = {}

        for i in range(len(clauses)):
            current_clause = clauses[i]

            for lit1 in current_clause:
                for lit2 in current_clause:

                    lit_index1 = self._get_lit_node_index(lit1, len(clauses), n_vars)
                    lit_index2 = self._get_lit_node_index(lit2, len(clauses), n_vars)

                    if lit_index1 != lit_index2:
                        if (lit_index1, lit_index2) in edges:
                            edge_attr = edges[(lit_index1, lit_index2)]
                        else:
                            edge_attr = [0] * self._max_clause_length

                        if i >= self._max_clause_length:
                            raise Exception("The max_clause_length value is too low for the given clause of size " + str(i))
                        edge_attr[i] = 1
                        edges[(lit_index1, lit_index2)] = edge_attr

        return edges if len(edges) != 0 else self._max_clause_length
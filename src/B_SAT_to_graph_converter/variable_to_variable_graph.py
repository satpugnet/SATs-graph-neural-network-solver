from B_SAT_to_graph_converter.abstract_SAT_to_graph_converter import AbstractSATToGraphConverter


class VariableToVariableGraph(AbstractSATToGraphConverter):

    def __init__(self, max_clause_length):
        super().__init__()
        self._max_clause_length = max_clause_length

    def _compute_x(self, SAT_problem):
        return [[1]] * SAT_problem.n_vars + [[-1]] * SAT_problem.n_vars

    def _compute_edges(self, SAT_problem):
        edges = {}

        for i in range(len(SAT_problem.clauses)):
            current_clause = SAT_problem.clauses[i]

            for j in current_clause:
                for k in current_clause:
                    j = self.__lit_to_node_value(j, SAT_problem.n_vars)
                    k = self.__lit_to_node_value(k, SAT_problem.n_vars)
                    if j != k:
                        if (j, k) in edges:
                            edge_attr = edges[(j, k)]
                        else:
                            edge_attr = [0] + [0] * self._max_clause_length

                        edge_attr[i + 1] = 1
                        edges[(j, k)] = edge_attr

        edges = self.__connect_opposite_with_special_edge(edges, SAT_problem.n_vars)

        edge_start = []
        edge_end = []
        edge_attr = []

        for key in edges:
            edge_start.append(key[0])
            edge_end.append(key[1])
            edge_attr.append(edges[key])

        return [edge_start, edge_end], edge_attr

    def __connect_opposite_with_special_edge(self, edges, n_vars):
        for i in range(0, n_vars):

            if (i, i + n_vars) in edges:
                edge_attr = edges[(i, i + n_vars)]

            else:
                edge_attr = [0] + [0] * self._max_clause_length

            edge_attr[0] = 1
            edges[(i, i + n_vars)] = edge_attr
            edges[(i + n_vars, i)] = edge_attr

        return edges

    def __lit_to_node_value(self, lit, n_vars):
        if lit < 0:
            node_value = n_vars + abs(lit) - 1
        else:
            node_value = lit

        return node_value


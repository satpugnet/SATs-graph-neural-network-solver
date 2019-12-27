import torch
from torch_geometric.data import Data


class SATToGraphConverter:

    def __init__(self, max_clause_length):
        self.max_clause_length = max_clause_length

    def convert_all(self, SAT_problems):
        print("Converting SAT problems to graphs")
        graphs_data = []

        for SAT_problem in SAT_problems:
            graphs_data.append(self.__convert(SAT_problem))

        print("Finished to convert SAT problems to graphs")
        return graphs_data

    def __convert(self, SAT_problem):
        x = torch.tensor([[1]] * SAT_problem.n_lits, dtype=torch.float)
        y = torch.tensor([SAT_problem.is_sat], dtype=torch.float)

        edge_index_raw, edge_attr_raw = self.__compute_edges(SAT_problem.clauses, SAT_problem.n_vars)
        edge_index = torch.tensor(edge_index_raw, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_raw, dtype=torch.long)

        return Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

    def __compute_edges(self, clauses, n_vars):
        edges = {}

        for i in range(len(clauses)):
            current_clause = clauses[i]
            for j in current_clause:
                for k in current_clause:
                    j = self.__lit_to_node_value(j, n_vars)
                    k = self.__lit_to_node_value(k, n_vars)
                    if j != k:
                        if (j, k) in edges:
                            edge_attr = edges[(j, k)]
                        else:
                            edge_attr = [0] * self.max_clause_length

                        edge_attr[i] = 1
                        edges[(j, k)] = edge_attr

        edge_start = []
        edge_end = []
        edge_attr = []

        for key in edges:
            edge_start.append(key[0])
            edge_end.append(key[1])
            edge_attr = edges[key]

        return [edge_start, edge_end], edge_attr

    def __lit_to_node_value(self, lit, n_vars):
        if lit < 0:
            node_value = n_vars + abs(lit) - 1
        else:
            node_value = lit

        return node_value

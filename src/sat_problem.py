import torch
class SATProblem:
    def __init__(self, clauses, is_sat):
        self.n_vars = max(abs(min(clauses)), max(clauses))
        self.n_lits = 2 * self.n_vars
        self.n_clauses = len(clauses)

        self.is_sat = is_sat


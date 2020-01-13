

class SATProblem:
    def __init__(self, clauses, is_sat):
        lits_used = set([int(lit) for clause in clauses for lit in clause])
        self.n_vars = max(abs(min(lits_used)), max(lits_used))

        self.n_lits = 2 * self.n_vars
        self.n_clauses = len(clauses)
        self.clauses = clauses

        self.is_sat = is_sat

        self.lits_present = self.__compute_present_lit(self.n_vars)

    def __compute_present_lit(self, n_vars):
        return list(range(1, n_vars + 1)) + list(range(-n_vars, 0))


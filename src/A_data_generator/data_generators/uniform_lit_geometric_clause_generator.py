import numpy as np
import random
from random import shuffle

from A_data_generator.abstract_data_generator import AbstractDataGenerator


class UniformLitGeometricClauseGenerator(AbstractDataGenerator):

    def __init__(self, max_n_vars, max_n_clause, out_dir="../data_generated", percentage_sat=0.50, seed=None, min_n_vars=1,
                 min_n_clause=1, lit_distr_p=0.4, include_trivial_clause=False):
        super().__init__(out_dir, percentage_sat, seed, min_n_vars, max_n_vars, min_n_clause, max_n_clause)
        self._lit_distr_p = lit_distr_p
        self._include_trivial_clause = include_trivial_clause

    def __repr__(self):
        return "{}(percentage_sat({:.2f}), seed({}), min_n_vars({}), max_n_vars({}), min_n_clause({}), " \
               "max_n_clause({}), lit_distr_p({:.2f}), self._include_trivial_clause({}))"\
            .format(self.__class__.__name__, self._percentage_sat, self._seed, self._min_n_vars, self._max_n_vars,
                    self._min_n_clause, self._max_n_clause, self._lit_distr_p, self._include_trivial_clause)

    def _generate_CNF(self):
        n_vars = random.randint(self._min_n_vars, self._max_n_vars)
        n_clause = random.randint(self._min_n_clause, self._max_n_clause)

        clauses = []
        for i in range(n_clause):
            lit_to_draw_from_geom = np.random.geometric(self._lit_distr_p) + 1
            max_n_lit = max(n_vars, 1)
            current_clause_n_lit = min(lit_to_draw_from_geom, max_n_lit)
            current_clause = self.__generate_clause(n_vars, int(current_clause_n_lit), self._include_trivial_clause)
            clauses.append(current_clause)

        return n_vars, clauses

    def __generate_clause(self, n_vars, n_lit_drawn, include_trivial_clause):
        min_lit = -n_vars if include_trivial_clause else 1
        lits_to_pick_from = list(range(min_lit, n_vars + 1))
        if 0 in lits_to_pick_from:
            lits_to_pick_from.remove(0)
        shuffle(lits_to_pick_from)

        lits_picked = lits_to_pick_from[:n_lit_drawn]
        # Randomly negate the lits in the clause
        if not include_trivial_clause:
            lits_picked = [lit if random.random() < 0.5 else -lit for lit in lits_picked]

        return lits_picked

    def _make_filename(self, n_vars, n_clause, is_sat, iter_num):
        return "sat=%i_n_vars=%.3d_n_clause=%.3d_lit_distr_p=%.2f_seed=%d-%i.sat" % \
               (is_sat, n_vars, n_clause, self._lit_distr_p, self._seed, iter_num)
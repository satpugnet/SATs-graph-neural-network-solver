import random
import time

import numpy as np

from A_data_generator.abstract_data_generator import AbstractDataGenerator
from random import randint

class PairedProblemGenerator(AbstractDataGenerator):
    '''
    This is an implementation of the data generation used in "Learning a SAT solver from a single-bit supervision".
    '''

    def __init__(self, seed=None, min_max_n_vars=(10, 40)):
        assert(min_max_n_vars[0] > 1)
        super().__init__(None, seed, min_max_n_vars, (None, None))


    def _generate_CNF(self):

        is_sat = True
        num_vars = randint(self._min_max_n_vars[0], self._min_max_n_vars[1])
        clauses = []

        while is_sat:
            current_num_lit = 1 + np.random.binomial(1, 0.7) + np.random.geometric(0.4)
            if current_num_lit <= num_vars:
                current_lits = random.sample(range(1, num_vars + 1), current_num_lit)
                current_lits = [lit if random.random() < 0.5 else -lit for lit in current_lits]
                clauses.append(current_lits)
                is_sat = self._is_satisfiable(clauses)

        sat_clause = clauses[0:-1] + [[-lit for lit in clauses[-1]]]

        return [clauses, sat_clause]


import errno
import os
import random
import shutil
import time
from abc import ABC, abstractmethod

import numpy as np

from A_data_generator.deterministic_solvers.PyMiniSolvers import minisolvers
from utils import logger


class AbstractDataGenerator(ABC):
    def __init__(self, out_dir="../data_generated", percentage_sat=0.50, seed=None, min_n_vars=None, max_n_vars=None,
                 min_n_clause=None, max_n_clause=None):
        self._seed = seed if seed is not None else time.time_ns() % 100000
        random.seed(self._seed)
        np.random.seed(self._seed)

        self._out_dir = out_dir
        self._percentage_sat = percentage_sat
        self._min_n_vars = min_n_vars
        self._max_n_vars = max_n_vars
        self._min_n_clause = min_n_clause
        self._max_n_clause = max_n_clause

    def generate(self, number_dimacs):
        number_sat_required = int(number_dimacs * self._percentage_sat)
        number_unsat_required = int(number_dimacs - number_sat_required)

        i = 0
        while i < number_dimacs:
            logger.get().debug("Generation of SATs problem at: " + str(int(i / number_dimacs * 100)) + "% (" + str(number_sat_required)
                + " SAT left and " + str(number_unsat_required) + " UNSAT left)")
            n_vars, clauses = self._generate_CNF()

            if not self._has_correct_num_var_and_clauses(n_vars, clauses):
                logger.get().warning("Warning: the last clause has incorrect number of variable or clauses")
                continue

            is_sat = self._is_satisfiable(n_vars, clauses)
            if not self._has_correct_satisfiability(is_sat, number_sat_required, number_unsat_required):
                logger.get().warning("Warning: the last clause has incorrect satisfiability according to the requested ratio of SAT to UNSAT")
                continue

            if is_sat:
                number_sat_required -= 1
            else:
                number_unsat_required -= 1
            i += 1

            out_filename = "{}/{}_{}".format(str(self._out_dir), self.__class__.__name__,
                           self._make_filename(n_vars, len(clauses), is_sat, i))
            self._save_sat_problem_to(out_filename, n_vars, clauses)

    def _is_satisfiable(self, n_vars, clauses):
        solver = minisolvers.MinisatSolver()
        for i in range(n_vars):
            solver.new_var(dvar=True)

        for clause in clauses:
            solver.add_clause(clause)

        return solver.solve()

    def _has_correct_satisfiability(self, is_sat, number_sat_required, number_unsat_required):
        correct_satisfiability = (is_sat and number_sat_required != 0) or (not is_sat and number_unsat_required != 0)

        return correct_satisfiability

    def _has_correct_num_var_and_clauses(self, n_vars, clauses):
        correct_n_vars = n_vars <= self._max_n_vars if self._max_n_vars is not None else True
        correct_n_vars &= n_vars >= self._min_n_vars if self._min_n_vars is not None else True

        correct_n_clauses = len(clauses) <= self._max_n_clause if self._max_n_clause is not None else True
        correct_n_clauses &= len(clauses) >= self._min_n_clause if self._min_n_clause is not None else True

        return correct_n_vars and correct_n_clauses

    @abstractmethod
    def _generate_CNF(self):
        pass

    def delete_all(self):
        try:
            shutil.rmtree(self._out_dir)
        except FileNotFoundError as e:
            pass

    @abstractmethod
    def _make_filename(self, n_vars, n_clause, is_sat, iter_num):
        pass

    def _save_sat_problem_to(self, out_filename, n_vars, clauses):
        if not os.path.exists(os.path.dirname(out_filename)):
            try:
                os.makedirs(os.path.dirname(out_filename))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(out_filename, 'w') as file:
            file.write("p cnf %d %d\n" % (n_vars, len(clauses)))

            for clause in clauses:
                for lit in clause:
                    file.write("%d " % int(lit))
                file.write("\n")

            file.close()




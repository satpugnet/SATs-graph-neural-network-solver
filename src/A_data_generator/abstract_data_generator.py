import errno
import os
import random
import shutil
import time
from abc import ABC, abstractmethod

import numpy as np

from pysat.solvers import Glucose3

from utils import logger
from utils.abstract_repr import AbstractRepr


class AbstractDataGenerator(ABC, AbstractRepr):
    def __init__(self, percentage_sat=0.50, seed=None, min_max_n_vars=(None, None),
                 min_max_n_clauses=(None, None)):
        '''
        Generates SATs data in the form of Dimacs that can be used later on for training.
        :param percentage_sat: The percentage of SAT to UNSAT problems.
        :param seed: The seed used if any.
        :param min_max_n_vars: The min and max number of variable in the problems.
        :param min_max_n_clauses: The min and max number of clauses in the problems.
        '''
        self._seed = seed if seed is not None else int(round(time.time() * 10**6)) % 100000
        random.seed(self._seed)
        np.random.seed(self._seed)

        self._percentage_sat = percentage_sat

        self._min_max_n_vars = min_max_n_vars
        self._min_max_n_clauses = min_max_n_clauses

    def _get_fields_for_repr(self):
        return {
            "percentage_sat": self._percentage_sat,
            "seed": self._seed,
            "min_max_n_vars": self._min_max_n_vars,
            "min_max_n_clauses": self._min_max_n_clauses
        }

    def generate(self, number_dimacs, out_dir):
        number_sat_required = int(number_dimacs * self._percentage_sat) if self._percentage_sat is not None else None
        number_unsat_required = int(number_dimacs - number_sat_required) if self._percentage_sat is not None else None
        start_time = time.time()

        total_sat = 0
        total_unsat = 0
        while total_sat + total_unsat < number_dimacs:
            logger.get().debug("Generation of SATs problem at: " + str(int((total_sat + total_unsat) / number_dimacs * 100)) +
                               "% " + ("(" + str(total_sat) + " SAT and " + str(total_unsat) + " UNSAT)\r"))

            n_vars, clauses = self._generate_CNF()

            if not self._has_correct_num_var_and_clauses(n_vars, clauses):
                logger.get().warning("Warning: the last generated SAT has incorrect number of variable or clauses, trying again")
                continue

            is_sat = self._is_satisfiable(clauses)
            if self._percentage_sat is not None and not self._has_correct_satisfiability(is_sat, total_sat, total_unsat, number_sat_required, number_unsat_required):
                logger.get().warning("Warning: the last generated SAT has incorrect satisfiability according to the requested ratio of SAT to UNSAT, trying again")
                continue

            if is_sat:
                total_sat += 1
            else:
                total_unsat += 1

            out_filename = "{}/{}_{}".format(str(out_dir), self.__class__.__name__,
                           self._make_filename(n_vars, len(clauses), is_sat, total_sat + total_unsat))
            self._save_sat_problem_to(out_filename, n_vars, clauses)


    def _is_satisfiable(self, clauses):
        solver = Glucose3()

        for clause in clauses:
            solver.add_clause(clause)

        return solver.solve()

    def _has_correct_satisfiability(self, is_sat, total_sat, total_unsat, number_sat_required, number_unsat_required):
        if number_sat_required is None or number_unsat_required is None:
            return True

        correct_satisfiability = (is_sat and total_sat != number_sat_required) or (not is_sat and total_unsat != number_unsat_required)

        return correct_satisfiability

    def _has_correct_num_var_and_clauses(self, n_vars, clauses):
        correct_n_vars = n_vars <= self._min_max_n_vars[1] if self._min_max_n_vars[1] is not None else True
        correct_n_vars &= n_vars >= self._min_max_n_vars[0] if self._min_max_n_vars[0] is not None else True

        correct_n_clauses = len(clauses) <= self._min_max_n_clauses[1] if self._min_max_n_clauses[1] is not None else True
        correct_n_clauses &= len(clauses) >= self._min_max_n_clauses[0] if self._min_max_n_clauses[0] is not None else True

        return correct_n_vars and correct_n_clauses

    @abstractmethod
    def _generate_CNF(self):
        pass

    def delete_all(self, out_dir):
        try:
            shutil.rmtree(out_dir)
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




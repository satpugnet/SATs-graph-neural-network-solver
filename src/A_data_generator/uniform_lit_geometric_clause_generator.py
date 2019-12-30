import errno
import inspect
import os
import shutil
import time

import numpy as np
import random

from A_data_generator.abstract_data_generator import AbstractDataGenerator
from A_data_generator.deterministic_solvers.PyMiniSolvers import minisolvers


class UniformLitGeometricClauseGenerator(AbstractDataGenerator):

    def __init__(self, out_dir="../data_generated", percentage_sat=0.50, seed=None, min_n_vars=1, max_n_vars=5,
                 min_n_clause=2, max_n_clause=5, lit_distr_p=0.4):
        super().__init__()
        self._seed = seed if seed is not None else time.time_ns() % 100000
        random.seed(self._seed)
        np.random.seed(self._seed)
        self._percentage_sat = percentage_sat
        self._out_dir = out_dir
        self._min_n_vars = min_n_vars
        self._max_n_vars = max_n_vars
        self._min_n_clause = min_n_clause
        self._max_n_clause = max_n_clause
        self._lit_distr_p = lit_distr_p

    def __repr__(self):
        return "{}(out_dir({}), percentage_sat({:.2f}), seed({}), min_n_vars({}), max_n_vars({}), min_n_clause({}), " \
               "max_n_clause({}), lit_distr_p({:.2f}))".format(self.__class__.__name__, self._out_dir,
                                                               self._percentage_sat, self._seed,
                                                               self._min_n_vars, self._max_n_vars, self._min_n_clause,
                                                               self._max_n_clause, self._lit_distr_p)

    def generate(self, number_dimacs):
        number_sat_required = number_dimacs * self._percentage_sat
        number_unsat_required = number_dimacs - number_sat_required

        i = 0
        while i < number_dimacs:
            print("Generation of SATs problem at: " + str(int(i / number_dimacs * 100)) + "% (" + str(number_sat_required)
                  + " SAT left and " + str(number_unsat_required) + " UNSAT left)")
            n_vars, n_clause, clauses, is_sat = self.__gen_clause()

            if((is_sat and number_sat_required == 0) or (not is_sat and number_unsat_required == 0)):
                continue
            if is_sat:
                number_sat_required -= 1
            else:
                number_unsat_required -= 1
            i += 1

            out_filename = self.__make_filename(self._out_dir, n_vars, n_clause, is_sat, i)
            self.__save_sat_problem_to(out_filename, n_vars, clauses)

    def delete_all(self):
        try:
            shutil.rmtree(self._out_dir)
        except FileNotFoundError as e:
            pass

    def __gen_clause(self):
        n_vars = random.randint(self._min_n_vars, self._max_n_vars)
        n_clause = random.randint(self._min_n_clause, self._max_n_clause)
        solver = minisolvers.MinisatSolver()
        for i in range(n_vars):
            solver.new_var(dvar=True)

        clauses = []
        for i in range(n_clause):
            lit_to_draw_from_geom = np.random.geometric(self._lit_distr_p) + 1
            max_n_lit = max(n_vars, 1)
            current_clause_n_lit = min(lit_to_draw_from_geom, max_n_lit)
            current_clause = self.__generate_clause(n_vars, current_clause_n_lit)
            clauses.append(current_clause)
            solver.add_clause(current_clause)

        is_sat = solver.solve()

        return n_vars, n_clause, clauses, is_sat

    def __generate_clause(self, n_lits, n_lit_drawn):
        ''' This does not generate trivial unsatisfiable clauses such with a var and its negation'''
        # TODO: make it so it can generate (x and not x)

        lits = [i + 1 for i in range(n_lits)]
        lits_drawn = np.random.choice(lits, size=int(n_lit_drawn), replace=False)

        return [lit if random.random() < 0.5 else -lit for lit in lits_drawn]

    def __make_filename(self, out_dir, n_vars, n_clause, is_sat, iter_num):
        return "%s/sat=%i_n_vars=%.3d_n_clause=%.3d_lit_distr_p=%.2f_seed=%d-%i.sat" % \
               (out_dir, is_sat, n_vars, n_clause, self._lit_distr_p, self._seed, iter_num)

    def __save_sat_problem_to(self, out_filename, n_vars, clauses):
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
                    file.write("%d " % lit)
                file.write("\n")

            file.close()

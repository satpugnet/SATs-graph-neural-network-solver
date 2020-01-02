import random
from random import shuffle

import numpy as np

from A_data_generator.abstract_data_generator import AbstractDataGenerator
from A_data_generator.data_generators.distr_based_generators.distr_based_generator_enum import Distribution
from utils import logger


class DistrBasedGenerator(AbstractDataGenerator):

    def __init__(self, percentage_sat=0.50, seed=None, min_max_n_vars=(1, 30), min_max_n_clauses=(20, 30),
                 var_num_distr=Distribution.UNIFORM, var_num_distr_params=[], clause_num_distr=Distribution.UNIFORM,
                 clause_num_distr_params=[], lit_in_clause_distr=Distribution.GEOMETRIC, lit_in_clause_distr_params=[0.4],
                 include_trivial_clause=False):
        '''
        Generate SATs based on classical distributions.
        :param percentage_sat: The percentage of SAT to UNSAT problems.
        :param seed: The seed used if any.
        :param min_max_n_vars: The min and max number of variable in the problems.
        :param min_max_n_clauses: The min and max number of clauses in the problems.
        :param var_num_distr: The distribution used to generate the number of variable in a problem.
        :param var_num_distr_params: The distribution parameters.
        :param clause_num_distr: The distribution used to generate the number of clauses in a problem.
        :param clause_num_distr_params: The distribution parameters.
        :param lit_in_clause_distr: The distribution used to generate the number of clauses in a problem.
        :param lit_in_clause_distr_params: The distribution parameters.
        :param include_trivial_clause: Whether to include clause containing a variable and its opposite such as (x and not x).
        '''
        super().__init__(percentage_sat, seed, min_max_n_vars, min_max_n_clauses)
        self._var_num_distr = var_num_distr
        self._var_num_distr_params = var_num_distr_params
        self._clause_num_distr = clause_num_distr
        self._clause_num_distr_params = clause_num_distr_params
        self._lit_in_clause_distr = lit_in_clause_distr
        self._lit_in_clause_distr_params = lit_in_clause_distr_params

        self._include_trivial_clause = include_trivial_clause

    def __repr__(self):
        return "{}(percentage_sat({:.2f}), seed({}), min_max_n_vars({}), min_max_n_clauses({}), " \
               "var_num_distr({}), var_num_distr_params({}), clause_num_distr({}), clause_num_distr_params({}), " \
               "lit_in_clause_distr({}), lit_in_clause_distr_params({}), self._include_trivial_clause({}))"\
            .format(self.__class__.__name__, self._percentage_sat, self._seed, self._min_max_n_vars, self._min_max_n_clauses,
                    self._var_num_distr, self._var_num_distr_params, self._clause_num_distr, self._clause_num_distr_params,
                    self._lit_in_clause_distr, self._lit_in_clause_distr_params, self._include_trivial_clause)

    def _generate_CNF(self):
        n_vars = self._compute_num_vars()
        n_clause = self._compute_num_clauses()

        clauses = []
        for i in range(n_clause):
            current_clause = self.__generate_clause(
                n_vars,
                self._compute_num_lits_in_clause(n_vars),
                self._include_trivial_clause
            )
            clauses.append(current_clause)

        return n_vars, clauses

    def _compute_num_vars(self):
        return self.__generate_value_from_distr_bounded(
            self._var_num_distr,
            self._min_max_n_vars[0],
            self._min_max_n_vars[1],
            self._var_num_distr_params
        )

    def _compute_num_clauses(self):
        return self.__generate_value_from_distr_bounded(
            self._clause_num_distr,
            self._min_max_n_clauses[0],
            self._min_max_n_clauses[1],
            self._clause_num_distr_params
        )

    def _compute_num_lits_in_clause(self, n_vars):
        return self.__generate_value_from_distr_bounded(
            self._lit_in_clause_distr,
            1,
            n_vars, # TODO: potentially you could have more than n_vars, you could have 2 * n_vars (total number of litteral) although it makes the problem trivial
            self._lit_in_clause_distr_params
        )

    def __generate_value_from_distr_bounded(self, distr, lower_bound, upper_bound, params):
        while True:
            value = self.__generate_value_from_distr(distr, lower_bound, upper_bound, params)
            if lower_bound <= value <= upper_bound:
                return value
            else:
                logger.get().warning("Generated a value (" + str(value) + ") outside of the range (" + str(lower_bound) + ", " +
                                     str(upper_bound) + ") of value, trying again")

    def __generate_value_from_distr(self, distr, lower_bound, upper_bound, params):
        if distr == Distribution.UNIFORM:
            return random.randint(lower_bound, upper_bound)
        elif distr == Distribution.GEOMETRIC:
            return np.random.geometric(params[0])
        elif distr == Distribution.POISSON:
            return np.random.poisson(params[0]) + 1
        elif distr == Distribution.BINOMIAL:
            return np.random.binomial(params[0], params[1])
        elif distr == Distribution.HYPERGEOMETRIC:
            return np.random.hypergeometric(params[0], params[1], params[2]) + 1
        elif distr == Distribution.NORMAL:
            return int(np.random.normal(params[0], params[1]))

    def __generate_clause(self, n_vars, n_lit_drawn, include_trivial_clause):
        lits_to_pick_from = list(range(1, n_vars + 1))
        if include_trivial_clause:
            lits_to_pick_from += list(range(-n_vars, 0))
        shuffle(lits_to_pick_from)

        lits_picked = lits_to_pick_from[:n_lit_drawn]
        # Randomly negate the lits in the clause
        if not include_trivial_clause:
            lits_picked = [lit if random.random() < 0.5 else -lit for lit in lits_picked]

        return lits_picked

    def _make_filename(self, n_vars, n_clause, is_sat, iter_num):
        return "sat=%i_n_vars=%.3d_n_clause=%.3d_seed=%d-%i.sat" % \
               (is_sat, n_vars, n_clause, self._seed, iter_num)

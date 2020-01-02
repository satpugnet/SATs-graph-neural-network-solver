import random
from collections import OrderedDict

from cnfformula.families.pigeonhole import PigeonholePrinciple

from A_data_generator.abstract_data_generator import AbstractDataGenerator


class PigeonHolePrincipleGenerator(AbstractDataGenerator):

    def __init__(self, percentage_sat=0.50, seed=None, min_max_n_vars=None, min_max_n_clauses=None,
                 min_max_n_pigeons=(1, 10), min_max_n_holes=(1, 10)):
        '''
        Generate SATs based on the pigeon hole principle.
        :param percentage_sat: The percentage of SAT to UNSAT problems.
        :param seed: The seed used if any.
        :param min_max_n_vars: The min and max number of variable in the problems.
        :param min_max_n_clauses: The min and max number of clauses in the problems.
        :param min_max_n_pigeons: The min and max number of pigeons in the problems.
        :param min_max_n_holes: The min and max number of holes in the problems.
        '''
        super().__init__(percentage_sat, seed, min_max_n_vars, min_max_n_clauses)
        self._min_max_n_pigeons = min_max_n_pigeons
        self._min_max_n_holes = min_max_n_holes

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(),
                **{
                   "min_max_n_pigeons": self._min_max_n_pigeons,
                   "min_max_n_holes": self._min_max_n_holes
               }}

    def _generate_CNF(self):
        n_pigeons = random.randint(self._min_max_n_pigeons[0], self._min_max_n_pigeons[1])
        n_holes = random.randint(self._min_max_n_holes[0], self._min_max_n_holes[1])

        cnf = PigeonholePrinciple(n_pigeons, n_holes)
        clauses = self._convert_from_dimac_to_list(cnf.dimacs())
        n_vars = len(list(cnf.variables()))

        # is_sat = cnf.is_satisfiable(cmd="cryptominisat5", sameas="cryptominisat")[0]

        return n_vars, clauses

    def _convert_from_dimac_to_list(self, dimacs):
        lines = dimacs[1:].splitlines()
        for line in lines:
            lines = lines[1:]
            if line[0:6] == "p cnf ":
                break

        result = []
        for line in lines:
            lits = [int(lit) for lit in line.split()]
            lits = lits[:-1]
            result.append(lits)

        return result

    def _make_filename(self, n_vars, n_clause, is_sat, iter_num):
        return "sat=%i_n_vars=%.3d_n_clause=%.3d_seed=%d-%i.sat" % \
               (is_sat, n_vars, n_clause, self._seed, iter_num)


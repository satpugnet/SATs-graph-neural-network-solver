import random
import time

from cnfformula.families.pigeonhole import PigeonholePrinciple

from A_data_generator.abstract_data_generator import AbstractDataGenerator
from cnfformula.families.cliquecoloring import CliqueColoring


class PigeonHolePrincipleGenerator(AbstractDataGenerator):

    def __init__(self, out_dir="../data_generated", percentage_sat=0.50, min_n_vars=None, max_n_vars=None,
                 min_n_clause=None, max_n_clause=None, seed=None, min_n_pigeons=1, max_n_pigeons=10,
                 min_n_holes=1, max_n_holes=10):
        super().__init__(out_dir, percentage_sat, seed, min_n_vars, max_n_vars, min_n_clause, max_n_clause)
        self._min_n_pigeons = min_n_pigeons
        self._max_n_pigeons = max_n_pigeons
        self._min_n_holes = min_n_holes
        self._max_n_holes = max_n_holes

    def __repr__(self):
        return "{}(percentage_sat({:.2f}), seed({}), min_n_pigeon({}), max_n_pigeon({}), min_n_holes({}), max_n_holes({}))"\
            .format(self.__class__.__name__,  self._percentage_sat, self._seed,
                    self._min_n_pigeons, self._max_n_pigeons, self._min_n_holes, self._max_n_holes)

    def _generate_CNF(self):
        n_pigeons = random.randint(self._min_n_pigeons, self._max_n_pigeons)
        n_holes = random.randint(self._min_n_holes, self._max_n_holes)

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


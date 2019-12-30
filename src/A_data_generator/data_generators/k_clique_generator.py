import time

from A_data_generator.abstract_data_generator import AbstractDataGenerator
from cnfformula.families.cliquecoloring import CliqueColoring


class KCliqueGenerator(AbstractDataGenerator):

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "{}()"\
            .format(self.__class__.__name__)

    def generate(self, number_dimacs):
        print(time.time())
        for i in range(number_dimacs):
            print  ("start")
            Formula = CliqueColoring(10, 10, 10)
            # print(Formula.dimacs())
            print(Formula.is_satisfiable(cmd="cryptominisat5", sameas="cryptominisat"))

KCliqueGenerator().generate(number_dimacs=3)

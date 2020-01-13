import os
import re

from B_SAT_to_graph_converter.wrapper.SAT_problem import SATProblem


from utils import logger


class DimacLoader:

    def __init__(self, dir_path):
        self.dir_path = os.getcwd() + "/" + dir_path

    def load_sat_problems(self):

        SAT_problems = []
        for (dirpath, dirnames, filenames) in os.walk(self.dir_path):
            for i in range(len(filenames)):
                logger.get().debug("{:.1f}% SAT problems loaded\r".format(i / len(filenames) * 100))
                file_path = os.path.join(self.dir_path, filenames[i])
                SAT_problems.append(self.__load_dimac(file_path))

        return SAT_problems

    def __load_dimac(self, file_path):
        with open(file_path) as f:
            content = f.readlines()
            clauses = [list(map(int, clause.split())) for clause in content[1:]]

        return SATProblem(clauses, self.__is_sat(file_path))

    def __is_sat(self, file_path):
        m = re.search("sat=(\\d+)_", file_path)
        is_sat = m.group(1)
        return int(is_sat) == 1


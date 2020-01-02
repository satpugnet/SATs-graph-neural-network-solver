from A_data_generator.abstract_data_generator import AbstractDataGenerator

# TODO: implement
class GraphColoringGenerator(AbstractDataGenerator):
    def __init__(self, percentage_sat=0.50, seed=None, min_max_n_vars=(None, None), min_max_n_clauses=(None, None)):
        '''
        Generate SATs based on classical distributions.
        :param percentage_sat: The percentage of SAT to UNSAT problems.
        :param seed: The seed used if any.
        :param min_max_n_vars: The min and max number of variable in the problems.
        :param min_max_n_clauses: The min and max number of clauses in the problems.
        '''
        super().__init__(percentage_sat, seed, min_max_n_vars, min_max_n_clauses)

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(), **{}}

    def _generate_CNF(self):
        pass

    def _make_filename(self, n_vars, n_clause, is_sat, iter_num):
        pass



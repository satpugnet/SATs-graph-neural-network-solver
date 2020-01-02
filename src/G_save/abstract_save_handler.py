from collections import OrderedDict
from abc import ABC, abstractmethod

from utils.abstract_repr import AbstractRepr


class AbstractSaveHandler(ABC, AbstractRepr):
    def __init__(self, config, experiment_results):
        '''
        The save handler for saving experiments.
        :param config: The config used for the experiment
        '''
        self._config = config
        self._all_data = OrderedDict(list(experiment_results.items()) + list(config.items()))

    def _get_fields_for_repr(self):
        return {}

    @abstractmethod
    def save(self):
        pass

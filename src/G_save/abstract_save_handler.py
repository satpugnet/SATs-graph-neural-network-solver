import collections
from abc import ABC, abstractmethod



class AbstractSaveHandler(ABC):
    def __init__(self, config, experiment_results):
        '''
        The save handler for saving experiments.
        :param config: The config used for the experiment
        '''
        self._config = config
        self._all_data = collections.OrderedDict(list(experiment_results.items()) + list(config.items()))


    @abstractmethod
    def save(self):
        pass

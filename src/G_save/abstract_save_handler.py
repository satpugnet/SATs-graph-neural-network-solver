from abc import ABC, abstractmethod



class AbstractSaveHandler(ABC):
    def __init__(self, config):
        self._config = config

    @abstractmethod
    def save(self):
        pass

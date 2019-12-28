from abc import ABC, abstractmethod


class AbstractSATToGraphConverter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def convert_all(self, SAT_problems):
        pass

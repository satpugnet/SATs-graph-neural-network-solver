from abc import ABC, abstractmethod


class AbstractDataGenerator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, number_dimacs):
        pass

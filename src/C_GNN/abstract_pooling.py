from abc import ABC, abstractmethod


class AbstractPooling(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def pool(self, x, batch):
        pass

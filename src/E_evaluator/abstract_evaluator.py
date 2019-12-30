from abc import ABC, abstractmethod


class AbstractEvaluator(ABC):
    def __init__(self, device):
        pass

    @abstractmethod
    def eval(self, model, train_loss=None, do_print=True, time=None):
        pass
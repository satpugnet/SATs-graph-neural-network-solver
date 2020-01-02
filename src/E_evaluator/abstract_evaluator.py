from abc import ABC, abstractmethod


class AbstractEvaluator(ABC):
    def __init__(self, device):
        '''
        The evaluator to evaluate the experiment.
        :param device: The device to use for pytorch.
        '''
        self._device = device

    @abstractmethod
    def eval(self, model, train_loss=None, do_print=True, time=None):
        pass
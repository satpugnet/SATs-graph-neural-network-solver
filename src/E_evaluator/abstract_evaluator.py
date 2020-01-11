from abc import ABC, abstractmethod

from utils.abstract_repr import AbstractRepr


class AbstractEvaluator(ABC, AbstractRepr):
    def __init__(self, device, bce_loss):
        '''
        The evaluator to evaluate the experiment.
        :param device: The device to use for pytorch.
        '''
        self._device = device
        self._bce_loss = bce_loss

    def _get_fields_for_repr(self):
        return {}

    @abstractmethod
    def eval(self, model, train_loss=None, do_print=True, time=None):
        pass

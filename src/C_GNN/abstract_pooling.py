from abc import ABC, abstractmethod

import torch

from utils.abstract_repr import AbstractRepr


class AbstractPooling(torch.nn.Module, ABC, AbstractRepr):
    def __init__(self):
        super().__init__()

    def _get_fields_for_repr(self):
        return {}

    def _initialise(self, in_channels, out_channels):
        return out_channels

    @abstractmethod
    def pool(self, x, batch):
        pass

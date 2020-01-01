import torch

from D_trainer.abstract_trainer import AbstractTrainer
from utils import logger


class AdamTrainer(AbstractTrainer):

    def __init__(self, learning_rate, weight_decay, device, num_epoch_before_halving_lr):
        super().__init__(learning_rate, weight_decay, device)
        self._num_epoch_before_halving_lr = num_epoch_before_halving_lr

    def __repr__(self):
        return "{}(learning_rate({}), weight_decay({}), num_epoch_before_halving_lr({}))".format(
            self.__class__.__name__,
            self._learning_rate,
            self._weight_decay,
            self._num_epoch_before_halving_lr
        )

    def _create_optimizer(self, parameters, learning_rate, weight_decay):
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)

    def _set_learning_rate(self, epoch, learning_rate, optimizer):
        if epoch != 0 and epoch % self._num_epoch_before_halving_lr == 0:
            logger.get().warning("Halving the learning rate")
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2 # TODO: check that this is doing the right thing

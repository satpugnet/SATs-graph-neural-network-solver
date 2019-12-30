import torch

from D_trainer.abstract_trainer import AbstractTrainer


class AdamTrainer(AbstractTrainer):

    def __init__(self, learning_rate, weight_decay, device):
        super().__init__(learning_rate, weight_decay, device)

    def __repr__(self):
        return "{}(learning_rate({}), weight_decay({}))".format(self.__class__.__name__, self._learning_rate, self._weight_decay)

    def _create_optimizer(self, parameters, learning_rate, weight_decay):
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)

    def _set_learning_rate(self, epoch, learning_rate, optimizer):
        if epoch == 150:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate / 2

        if epoch == 300:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate / 4

        if epoch == 400:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate / 10

import torch

from D_trainer.abstract_trainer import AbstractTrainer


class AdamTrainer(AbstractTrainer):

    def __init__(self, model, train_loader, test_loader, device, model_evaluator, learning_rate, weight_decay):
        super().__init__(model, train_loader, test_loader, device, model_evaluator, learning_rate, weight_decay)

    def _create_optimizer(self, parameters, learning_rate, weight_decay):
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)

    def _set_learning_rate(self, epoch, learning_rate):
        if epoch == 200:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = learning_rate / 2

        if epoch == 300:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = learning_rate / 4

        if epoch == 400:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = learning_rate / 10

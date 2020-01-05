import torch

from D_trainer.abstract_trainer import AbstractTrainer
from utils import logger


class AdamTrainer(AbstractTrainer):

    def __init__(self, learning_rate, weight_decay, device, num_epoch_before_halving_lr, activate_amp, bce_loss):
        '''
        An adam trainer for the network.
        :param learning_rate: The learning rate.
        :param weight_decay: The weight decay.
        :param device: The device used.
        '''
        super().__init__(learning_rate, weight_decay, device, activate_amp, bce_loss)
        self._num_epoch_before_halving_lr = num_epoch_before_halving_lr

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(),
                **{
                    "num_epoch_before_halving_lr": self._num_epoch_before_halving_lr
                }}

    def _training_step(self, model, train_loader, optimizer):
        return super()._training_step(model, train_loader, optimizer)

    def _testing_step(self, model_evaluator, current_train_loss, time, model, epoch):
        return super()._testing_step(model_evaluator, current_train_loss, time, model, epoch)

    def _create_optimizer(self, parameters, learning_rate, weight_decay):
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)

    def _set_learning_rate(self, epoch, learning_rate, optimizer):
        if epoch != 0 and epoch % self._num_epoch_before_halving_lr == 0:
            logger.get().warning("Halving the learning rate")
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2 # TODO: check that this is doing the right thing

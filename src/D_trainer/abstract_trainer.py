import time
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCELoss

from utils import logger
import time
from console_progressbar import ProgressBar


class AbstractTrainer(ABC):
    def __init__(self, learning_rate, weight_decay, device):
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._device = device

    def __repr__(self):
        return "{}(learning_rate({}), weight_decay({}))".format(
            self.__class__.__name__,
            self._learning_rate,
            self._weight_decay
        )

    @abstractmethod
    def _create_optimizer(self, parameters, learning_rate, weight_decay):
        pass

    def train(self, number_of_epochs, model, train_loader, model_evaluator):
        logger.get().info("Starting the training")

        optimizer = self._create_optimizer(model.parameters(), self._learning_rate, self._weight_decay)

        train_loss = []
        test_loss = []
        accuracy = []
        start_time = time.time()

        for epoch in range(number_of_epochs):
            model.train()

            self._set_learning_rate(epoch, self._learning_rate, optimizer)

            current_train_loss = self.training_step(model, train_loader, optimizer)
            train_loss.append(current_train_loss)

            model.eval()
            current_test_loss, current_accuracy, _ = self.testing_step(model_evaluator, current_train_loss,
                                                                    time.time() - start_time, model, epoch)
            test_loss.append(current_test_loss)
            accuracy.append(current_accuracy)

        logger.get().info("Training completed")
        return train_loss, test_loss, accuracy, time.time() - start_time

    def training_step(self, model, train_loader, optimizer):
        train_error = 0

        pb = ProgressBar(total=len(train_loader)) # TODO: add a loading bar
        progress = 0
        for batch in train_loader:
            progress += 1
            pb.print_progress_bar(progress)

            batch = batch.to(self._device)

            optimizer.zero_grad()

            out = model(batch)

            loss = nn.BCELoss()(out, batch.y.view(-1, 1))
            # loss = F.mse_loss(out, batch.y.view(-1, 1))  # F.nll_loss(out, batch.y)
            train_error += loss

            loss.backward()
            optimizer.step()

        return train_error / len(train_loader)

    def testing_step(self, model_evaluator, current_train_loss, time, model, epoch):
        with torch.no_grad():
            return model_evaluator.eval(model, current_train_loss, do_print=True, time=time, epoch=epoch)

    @abstractmethod
    def _set_learning_rate(self, epoch, learning_rate, optimizer):
        pass

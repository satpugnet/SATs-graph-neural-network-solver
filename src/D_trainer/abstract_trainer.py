import time
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class AbstractTrainer(ABC):
    def __init__(self, learning_rate, weight_decay):
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay

    def __repr__(self):
        return "{}(learning_rate({}), weight_decay({}))".format(
            self.__class__.__name__,
            self._learning_rate,
            self._weight_decay
        )

    @abstractmethod
    def _create_optimizer(self, parameters, learning_rate, weight_decay):
        pass

    def train(self, number_of_epochs, model, train_loader, device, model_evaluator):
        optimizer = self._create_optimizer(model.parameters(), self._learning_rate, self._weight_decay)

        train_loss = []
        test_loss = []
        accuracy = []
        start_time = time.time()

        model.train()
        for epoch in range(number_of_epochs):
            print("Epoch: " + str(epoch))

            self._set_learning_rate(epoch, self._learning_rate, optimizer)

            current_train_loss = self.training_step(model, train_loader, device, optimizer)
            train_loss.append(current_train_loss)

            current_test_loss, current_accuracy, _ = self.testing_step(model_evaluator, current_train_loss,
                                                                    time.time() - start_time, model)
            test_loss.append(current_test_loss)
            accuracy.append(current_accuracy)

        return train_loss, test_loss, accuracy

    def training_step(self, model, train_loader, device, optimizer):
        train_error = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            out = model(batch)

            loss = F.mse_loss(out, batch.y.view(-1, 1))  # F.nll_loss(out, batch.y)
            train_error += loss

            loss.backward()
            optimizer.step()

        return train_error / len(train_loader)

    def testing_step(self, model_evaluator, current_train_loss, time, model):
        with torch.no_grad():
            return model_evaluator.eval(model, current_train_loss, do_print=True, time=time)

    @abstractmethod
    def _set_learning_rate(self, epoch, learning_rate, optimizer):
        pass

import time
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class AbstractTrainer(ABC):
    def __init__(self, model, train_loader, test_loader, device, model_evaluator, learning_rate, weight_decay):
        self.__model = model
        self.__train_loader = train_loader
        self.__test_loader = test_loader
        self.__device = device
        self.__model_evaluator = model_evaluator
        self.__learning_rate = learning_rate
        self.__weight_decay = weight_decay

        self._optimizer = self._create_optimizer(self.__model.parameters(), self.__learning_rate, self.__weight_decay)

    @abstractmethod
    def _create_optimizer(self, parameters, learning_rate, weight_decay):
        pass

    def train(self, number_of_epochs):
        train_loss = []
        test_loss = []
        accuracy = []
        start_time = time.time()

        self.__model.train()
        for epoch in range(number_of_epochs):
            print("Epoch: " + str(epoch))

            self._set_learning_rate(epoch, self.__learning_rate)

            current_train_loss = self.training_step()
            train_loss.append(current_train_loss)

            current_test_loss, current_accuracy, _ = self.testing_step(self.__model_evaluator, current_train_loss,
                                                                    time.time() - start_time)
            test_loss.append(current_test_loss)
            accuracy.append(current_accuracy)

        return train_loss, test_loss, accuracy

    def training_step(self):
        train_error = 0

        for batch in self.__train_loader:
            batch = batch.to(self.__device)

            self._optimizer.zero_grad()

            out = self.__model(batch)
            loss = F.mse_loss(out, batch.y.view(-1, 1))  # F.nll_loss(out, batch.y)
            train_error += loss

            loss.backward()
            self._optimizer.step()

        return train_error / len(self.__train_loader)

    def testing_step(self, model_evaluator, current_train_loss, time):
        with torch.no_grad():
            return model_evaluator.eval(self.__model, current_train_loss, do_print=True, time=time)

    @abstractmethod
    def _set_learning_rate(self, epoch, learning_rate):
        pass

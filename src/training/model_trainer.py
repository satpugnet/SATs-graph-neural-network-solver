import torch

import torch.nn.functional as F


class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, device, model_evaluator):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model_evaluator = model_evaluator

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)

    def train(self, number_of_epochs):
        train_loss = []
        test_loss = []
        accuracy = []

        self.model.train()
        for epoch in range(number_of_epochs):
            print("Epoch: " + str(epoch))
            current_train_loss = self.training_step()
            train_loss.append(current_train_loss)

            current_test_loss, current_accuracy = self.testing_step(self.model_evaluator, current_train_loss)
            test_loss.append(current_test_loss)
            accuracy.append(current_accuracy)

        return train_loss, test_loss, accuracy

    def testing_step(self, model_evaluator, current_train_loss):
        with torch.no_grad():
            return model_evaluator.eval(self.model, current_train_loss, do_print=True)

    def training_step(self):
        train_error = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = F.mse_loss(out, batch.y.view(-1, 1))  # F.nll_loss(out, batch.y)
            # print(out)
            # print(batch.y.view(-1, 1))
            # print(loss)
            train_error += loss

            loss.backward()
            self.optimizer.step()

        return train_error / len(self.train_loader)
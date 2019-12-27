import torch.nn.functional as F


class ModelEvaluator:

    def __init__(self, test_loader, device):
        self.test_loader = test_loader
        self.device = device

    def eval(self, model, train_loss=None, do_print=True):
        test_error = 0
        correct = 0

        for batch in self.test_loader:
            batch = batch.to(self.device)
            pred = model(batch)
            test_error += F.mse_loss(pred, batch.y.view(-1, 1))
            correct += ((pred > 0.5).int() == batch.y.view(-1, 1)).sum().item()  # float(pred.eq(data.y).sum().item())
            # print(pred)
            # print(batch.y.view(-1, 1))
            # print(correct)

        test_loss = test_error / len(self.test_loader)
        accuracy = correct / len(self.test_loader.dataset)

        if do_print:
            self.__perform_printing(accuracy, test_loss, train_loss)

        return test_loss, accuracy

    def __perform_printing(self, accuracy, test_loss, train_loss):
        text_start = 'train loss: {:.4f}, '.format(train_loss) if train_loss else ""
        print(text_start + 'test error: {:.4f}, '.format(test_loss) + 'accuracy: {:.4f}'.format(accuracy))

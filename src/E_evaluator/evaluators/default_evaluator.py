import torch
import torch.nn.functional as F

from E_evaluator.abstract_evaluator import AbstractEvaluator


class DefaultEvaluator(AbstractEvaluator):

    def __init__(self, device):
        super().__init__(device)
        self._test_loader = None
        self._device = device

    @property
    def test_loader(self):
        return self._test_loader

    @test_loader.setter
    def test_loader(self, new_test_loader):
        self._test_loader = new_test_loader

    def eval(self, model, train_loss=None, do_print=True, time=None):
        all_pred, all_truth, accuracy, test_loss = self.__eval_model( model)
        confusion_matrix = self.__confusion_matrix(all_pred, all_truth)

        if do_print:
            self.__perform_printing(accuracy, test_loss, train_loss, confusion_matrix, time)

        return test_loss, accuracy, confusion_matrix

    def __eval_model(self, model):
        all_pred = torch.tensor([])
        all_truth = torch.tensor([])

        test_error = 0
        correct = 0
        for batch in self.test_loader:
            batch = batch.to(self._device)
            pred = model(batch)
            test_error += F.mse_loss(pred, batch.y.view(-1, 1))

            pred_adjusted = (pred > 0.5).float()
            truth = batch.y.view(-1, 1)
            correct += (pred_adjusted == truth).sum().item()

            all_pred = torch.cat([all_pred, pred_adjusted]) if all_pred is not None else pred_adjusted
            all_truth = torch.cat([all_truth, truth]) if all_truth is not None else truth

        test_loss = test_error / len(self.test_loader)
        accuracy = correct / len(self.test_loader.dataset)

        return all_pred, all_truth, accuracy, test_loss

    def __confusion_matrix(self, prediction, truth):
        confusion_vector = prediction / truth

        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        return true_positives, false_positives, true_negatives, false_negatives

    def __perform_printing(self, accuracy, test_loss, train_loss, confusion_matrix, time):
        train_loss_text = 'train loss: {:.4f}, '.format(train_loss) if train_loss is not None else ""
        time_text = 'time: {:.1f}, '.format(time) if time is not None else ""

        print(time_text + train_loss_text + 'test loss: {:.4f}, '.format(test_loss) + 'accuracy: {:.4f}'.format(accuracy) +
              ", confusion matrix (TP, FP, TN, FN): " + str(confusion_matrix))

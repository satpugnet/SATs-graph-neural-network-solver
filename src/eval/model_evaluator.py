import torch
import torch.nn.functional as F


class ModelEvaluator:

    def __init__(self, test_loader, device):
        self.test_loader = test_loader
        self.device = device

    def eval(self, model, train_loss=None, do_print=True):
        test_error = 0
        correct = 0

        all_pred = torch.tensor([])
        all_truth = torch.tensor([])
        for batch in self.test_loader:
            batch = batch.to(self.device)
            pred = model(batch)
            test_error += F.mse_loss(pred, batch.y.view(-1, 1))

            pred_adjusted = (pred > 0.5).float()
            truth = batch.y.view(-1, 1)
            correct += (pred_adjusted == truth).sum().item()
            # print(pred)
            # print(batch.y.view(-1, 1))
            # print(correct)
            all_pred = torch.cat([all_pred, pred_adjusted]) if all_pred is not None else pred_adjusted
            all_truth = torch.cat([all_truth, truth]) if all_truth is not None else truth

        test_loss = test_error / len(self.test_loader)
        accuracy = correct / len(self.test_loader.dataset)

        if do_print:
            self.__perform_printing(accuracy, test_loss, train_loss, self.__confusion_matrix(all_pred, all_truth))

        return test_loss, accuracy

    def __confusion_matrix(self, prediction, truth):
        confusion_vector = prediction / truth

        true_positives = torch.sum(confusion_vector == 1).item()
        false_positives = torch.sum(confusion_vector == float('inf')).item()
        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
        false_negatives = torch.sum(confusion_vector == 0).item()

        return true_positives, false_positives, true_negatives, false_negatives

    def __perform_printing(self, accuracy, test_loss, train_loss, confusion_matrix):
        text_start = 'train loss: {:.4f}, '.format(train_loss) if train_loss else ""
        print(text_start + 'test error: {:.4f}, '.format(test_loss) + 'accuracy: {:.4f}'.format(accuracy) +
              ", confusion matrix (TP, FP, TN, FN): " + str(confusion_matrix))

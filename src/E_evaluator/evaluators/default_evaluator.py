import torch
import torch.nn.functional as F
from torch import nn

from E_evaluator.abstract_evaluator import AbstractEvaluator
from utils import logger
from torch_geometric.nn import DataParallel


class DefaultEvaluator(AbstractEvaluator):

    def __init__(self, device, bce_loss):
        '''
        The default evaluator to evaluate the experiment.
        :param device: The device to use for pytorch.
        '''
        super().__init__(device, bce_loss)
        self._test_loader = None

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(), **{}}

    @property
    def test_loader(self):
        return self._test_loader

    @test_loader.setter
    def test_loader(self, new_test_loader):
        self._test_loader = new_test_loader

    def eval(self, model, train_loss=None, do_print=True, time=None, epoch=None, training_pred=None, training_truth=None):
        all_pred, all_truth, accuracy, test_loss = self.__eval_model(model)
        confusion_matrix = self.__confusion_matrix(all_pred, all_truth)
        training_confusion_matrix = self.__confusion_matrix(training_pred, training_truth)

        if do_print:
            self.__perform_printing(accuracy, test_loss, train_loss, confusion_matrix, time, epoch, training_confusion_matrix)

        return test_loss, accuracy, confusion_matrix
    
    # TODO: remove duplication with the trainer
    def __eval_model(self, model):
        all_pred = torch.tensor([]).to(self._device)
        all_truth = torch.tensor([]).to(self._device)

        test_error = 0
        correct = 0
        progress = 0
        for batch in self.test_loader:
            progress += 1
            logger.get().debug("Testing at: {:.1f}%\r".format(progress/len(self.test_loader) * 100))
            
            if not isinstance(model, DataParallel): # If we are not using multi-gpu
                batch = batch.to(self._device)
            
            pred = model(batch)
            
            if isinstance(model, DataParallel): # If we are not using multi-gpu
                y = torch.cat([data.y for data in batch]).view(-1, 1).to(pred.device)
            else:
                y = batch.y.view(-1, 1)
            
            if self._bce_loss:
                test_error += nn.BCELoss()(pred, y)
            else:
                test_error = F.mse_loss(pred, y)  # F.nll_loss(out, batch.y)

            pred_adjusted = (pred > 0.5).float()
            correct += (pred_adjusted == y).sum().item()

            all_pred = torch.cat([all_pred, pred_adjusted]) if all_pred is not None else pred_adjusted
            all_truth = torch.cat([all_truth, y]) if all_truth is not None else y

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

    def __perform_printing(self, accuracy, test_loss, train_loss, confusion_matrix, time, epoch, training_confusion_matrix):
        epoch_text = "epoch: {}, ".format(epoch) if epoch is not None else ""
        time_text = 'time: {:.1f}, '.format(time) if time is not None else ""
        train_loss_text = 'train loss: {:.4f}, '.format(train_loss) if train_loss is not None else ""

        text = epoch_text + time_text + train_loss_text + 'test loss: {:.4f}, '.format(test_loss) + \
               'accuracy: {:.4f}'.format(accuracy) + ", confusion matrix (TP, FP, TN, FN): train " + str(training_confusion_matrix) + \
               ", test " + str(confusion_matrix)

        logger.get().info(text)

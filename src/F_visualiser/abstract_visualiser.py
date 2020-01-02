from abc import ABC, abstractmethod

from utils import logger


class AbstractVisualiser(ABC):
    def __init__(self):
        '''
        The visualiser to display interesting information about the experiment
        '''
        pass

    def visualise(self, train_loss, test_loss, accuracy, dirname, save=True):
        logger.get().info("Starting the visualisation")

        filename = self._perform_visualisation(train_loss, test_loss, accuracy, dirname, save)

        logger.get().info("Visualisation completed")
        return filename

    @abstractmethod
    def _perform_visualisation(self, train_loss, test_loss, accuracy, dirname, save):
        pass
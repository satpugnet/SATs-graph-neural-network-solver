import glob
import os
import re

import matplotlib.pyplot as plt

from F_visualiser.abstract_visualiser import AbstractVisualiser
from utils import logger


class DefaultVisualiser(AbstractVisualiser):
    def __init__(self):
        '''
        The default visualiser to display interesting information about the experiment
        '''
        super().__init__()

    def _get_fields_for_repr(self):
        return {**super()._get_fields_for_repr(), **{}}

    def _perform_visualisation(self, train_loss, test_loss, accuracy, dirname, save=True):
        fig, ax1 = plt.subplots()
        logger.get().error("1")

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (mse)')
        ax1.plot(train_loss, label="train_loss")
        ax1.plot(test_loss, label="test_loss")
        plt.ylim(top=1.0)
        plt.ylim(bottom=0)
        logger.get().error("2")

        ax1.legend(loc='lower left')
        logger.get().error("3")

        ax2 = ax1.twinx()
        logger.get().error("4")

        ax2.set_ylabel('accuracy (%)')
        plt.plot(accuracy, label="accuracy", color="green")
        logger.get().error("5")

        ax2.legend(loc='lower center')
        logger.get().error("6")

        filename = ""
        logger.get().error("7")

        if save:
            logger.get().error("looooo")
            self.__create_if_not_exist(dirname)
            logger.get().error("laaaaa")
            filename = self.__compute_file_name(dirname)
            logger.get().error("luuuuu")
            logger.set_debug_min_level(False)  # To prevent printing of debugging from plot function
            plt.savefig(dirname + "/" + filename)
            logger.set_debug_min_level(True)
            logger.get().error("liiiii")
            logger.get().error("lyyyyy")

        logger.get().error("8")

        logger.set_debug_min_level(False)  # To prevent printing of debugging from plot function
        plt.show()
        logger.set_debug_min_level(True)

        logger.get().error("9")

        return filename

    def __compute_file_name(self, dirname):
        all_files = [f for f in glob.glob(dirname + "/*.png")]

        max_number = 0
        for file in all_files:
            m = re.search(dirname + "/(\\d+).png", file)
            max_number = max(max_number, int(m.group(1)))

        return str(max_number + 1) + ".png"

    def __create_if_not_exist(self, dirname):
        if not os.path.exists(dirname):
            os.mkdir(dirname)


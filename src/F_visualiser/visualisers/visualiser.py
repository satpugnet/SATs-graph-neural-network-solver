import glob
import os
import re

import matplotlib.pyplot as plt


class DefaultVisualiser:
    def __init__(self):
        pass

    def visualise(self, train_loss, test_loss, accuracy, dirname, save=True):
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (mse)')
        ax1.plot(train_loss, label="train_loss")
        ax1.plot(test_loss, label="test_loss")

        ax1.legend(loc='lower left')

        ax2 = ax1.twinx()

        ax2.set_ylabel('accuracy (%)')
        plt.plot(accuracy, label="accuracy", color="green")

        ax2.legend(loc='lower center')

        filename = ""
        if save:
            self.__create_if_not_exist(dirname)
            filename = self.__compute_file_name(dirname)
            plt.savefig(dirname + "/" + filename)

        fig.tight_layout()
        plt.show()

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


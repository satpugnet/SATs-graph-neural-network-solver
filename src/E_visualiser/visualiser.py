import matplotlib.pyplot as plt


class Visualiser:
    def __init__(self):
        pass

    def visualise(self, train_loss, test_loss, accuracy):
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

        fig.tight_layout()
        plt.show()

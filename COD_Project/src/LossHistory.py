
import keras
import matplotlib.pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.sm_losses = []
        self.im_losses = []
        

    def on_batch_end(self, batch, logs={}):
        if batch % 10 == 0:
            self.losses.append(logs.get('loss'))
            self.sm_losses.append(logs.get('SM_loss'))
            self.im_losses.append(logs.get('IM_loss'))

    def plot(self):

        f = plt.figure()
        f.set_figwidth(15)
        f.set_figheight(4)


        plt.subplot(1, 3, 1)
        plt.xlabel("Batches")
        plt.ylabel('Total loss')
        plt.plot(self.losses)

        plt.subplot(1, 3, 2)
        plt.xlabel("Batches")
        plt.ylabel('SM loss')
        plt.plot(self.sm_losses)


        plt.subplot(1, 3, 3)
        plt.xlabel("Batches")
        plt.ylabel('IM loss')
        plt.plot(self.im_losses)

        plt.tight_layout()
        plt.show()
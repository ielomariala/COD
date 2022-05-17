from gc import callbacks
from SINet import SINet_ResNet50
from LossHistory import LossHistory
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, UpSampling2D
import matplotlib.pyplot as plt
from random import randrange
import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.nn import sigmoid_cross_entropy_with_logits
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf

class SINetModel:
    def __init__(self, DataLoader, weights_path=None, search_attention=None, resnet_weights=None) -> None:
        self.DataLoader = DataLoader
        self.search_attention = search_attention
        self.model = self.build()
        self.loss_history = LossHistory()
        
        if weights_path is None and resnet_weights is not None:
            print("Loading ResNet weights to our model ...")
            print("Weights before loading from ResNet50 = ", self.model.layers[0].weights)
            resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(352, 352, 3))

            resnet_model.save_weights(resnet_weights)

            self.model.load_weights(resnet_weights, by_name=True, skip_mismatch=True)

            print("Weights after loading from ResNet50 = ", self.model.layers[0].weights)
            print("Weights have not changed so Loading failed\nWe will train our model from scratch then ...")

 
        if weights_path is not None:
            self.model.load_weights(weights_path)
        self.STEP_SIZE_TRAIN, self.STEP_SIZE_VALID, self.STEP_SIZE_TEST = [None] * 3
        self.train_generator, self.val_generator, self.test_generator = [None] * 3
        
        

    def build(self):
        return SINet_ResNet50(search_attention=self.search_attention)
    
    def __load_train_data(self):
        train_images_generator, val_images_generator, train_labels_generator, val_labels_generator = self.DataLoader.load_train_data()
        if self.DataLoader.use_generator:
            self.train_generator = zip(train_images_generator, train_labels_generator)
            self.val_generator = zip(val_images_generator, val_labels_generator)

            self.batch_size = train_images_generator.batch_size
            
            self.STEP_SIZE_TRAIN=train_images_generator.n//train_images_generator.batch_size
            self.STEP_SIZE_VALID=val_images_generator.n//val_images_generator.batch_size
            print(f'Nb training data = {train_images_generator.n}, training generator batch size = {train_images_generator.batch_size}, training steps per epoch = {self.STEP_SIZE_TRAIN}')
            print(f'Nb validation data = {val_images_generator.n}, validation generator batch size = {val_images_generator.batch_size}, validation steps per epoch = {self.STEP_SIZE_VALID}')

        else:
            self.train_generator = (train_images_generator, train_labels_generator)
            self.val_generator = (val_images_generator, val_labels_generator)

    def train(self, epochs=40, save_weigths=None, batch_size=12, loss=None):
        if loss == 'bce':
            loss = BinaryCrossentropy()
        elif loss == 'bce_logits':
            loss = BinaryCrossentropy(from_logits=True)
        else:
            loss = lambda y_true, y_pred: sigmoid_cross_entropy_with_logits(y_true, y_pred)

        if self.train_generator is None:
            print("Loading train data ...")
            self.__load_train_data()

        opt = Adam(learning_rate=1e-4)

        self.model.compile(optimizer=opt, loss=[loss, loss])

        print("Training model ...")
        if self.DataLoader.use_generator:
                h = self.model.fit(self.train_generator, steps_per_epoch=self.STEP_SIZE_TRAIN, validation_data=self.val_generator, validation_steps=self.STEP_SIZE_VALID, epochs=epochs, callbacks=[self.loss_history])
        else:
            X, y = self.train_generator
            h = self.model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_data=self.val_generator, callbacks=[self.loss_history])

        if save_weigths:
            self.model.save_weights(save_weigths)
        return h

    def plot(self, h=None):
        if h is not None:
            print("Plotting history of total loss with respect to epochs ...")
            plt.plot(h.history["loss"], label="loss")
        print("Plotting history of total loss with respect to batches ...")
        self.loss_history.plot()
    
    def evaluate(self):
        if self.X_test is None:
            print("Loading test data ...")
            self.X_test, self.y_test = self.DataLoader.load_test_data()
        print("Evaluating model ...")
        return self.model.evaluate(self.X_test, self.y_test)

    def test(self, set='test'):
        if set == 'train':
            if self.train_generator is None:
                print("Loading train and validation data ...")
                self.__load_train_data()
            X, y = next(self.train_generator) if self.DataLoader.use_generator else self.train_generator

        elif set == 'valid':
            if self.val_generator is None:
                print("Loading train and validation data ...")
                self.__load_train_data()
            X, y = next(self.val_generator) if self.DataLoader.use_generator else self.val_generator

        else:
            if self.test_generator is None:
                print("Loading test data ...")
                self.test_generator = self.DataLoader.load_test_data()
            X, y = next(self.test_generator) if self.DataLoader.use_generator else self.test_generator
        print("Testing model visually ...")
        k = randrange(len(X))

        _, cam = self.model.predict(np.array([X[k]]))

        cam = UpSampling2D(size=3, interpolation='bilinear')(cam)
        cam = Activation('sigmoid')(cam)
        cam = cam.numpy().squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        f = plt.figure()
        f.set_figwidth(15)
        f.set_figheight(4)

        plt.subplot(1, 3, 1)
        plt.imshow(X[k])

        plt.subplot(1, 3, 2)
        plt.imshow(cam, cmap="gray")

        plt.subplot(1, 3, 3)
        plt.imshow(tf.squeeze(y[k]), cmap="gray")

        plt.tight_layout()


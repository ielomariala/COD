from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
# from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, image_path, label_path, image_path_test, label_path_test, use_generator=True, hyper_batch_size=32):
        self.image_path = image_path
        self.label_path = label_path
        self.hyper_batch_size = hyper_batch_size
        self.image_path_test = image_path_test
        self.label_path_test = label_path_test
        self.use_generator = use_generator

    def load_train_data(self):
        if self.use_generator:
            datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

            train_images_generator = datagen.flow_from_directory(self.image_path, class_mode=None, target_size=(352, 352), batch_size=self.hyper_batch_size, shuffle=False, subset="training")
            val_images_generator = datagen.flow_from_directory(self.image_path, class_mode=None, target_size=(352, 352), batch_size=self.hyper_batch_size, shuffle=False, subset="validation")

            train_labels_generator = datagen.flow_from_directory(self.label_path, class_mode=None, target_size=(352, 352), batch_size=self.hyper_batch_size, shuffle=False, subset="training", color_mode="grayscale")
            val_labels_generator = datagen.flow_from_directory(self.label_path, class_mode=None, target_size=(352, 352), batch_size=self.hyper_batch_size, shuffle=False, subset="validation", color_mode="grayscale")

            
            return train_images_generator, val_images_generator, train_labels_generator, val_labels_generator
        else:
            datagen = ImageDataGenerator(rescale=1./255)
            train_images_generator = datagen.flow_from_directory(self.image_path, class_mode=None, target_size=(352, 352), batch_size=self.hyper_batch_size, shuffle=False)
            train_labels_generator = datagen.flow_from_directory(self.label_path, class_mode=None, target_size=(352, 352), batch_size=self.hyper_batch_size, shuffle=False, color_mode="grayscale")

            X = train_images_generator[0]
            Y = train_labels_generator[0]

            shuffler = np.random.permutation(len(X))

            X = X[shuffler]
            Y = Y[shuffler]
            
            thresh = self.hyper_batch_size*3//4

            X_train, y_train = X[:thresh], Y[:thresh]
            X_val, y_val = X[thresh:], Y[thresh:]

            return X_train, X_val, y_train, y_val

    def load_test_data(self):
        
        datagen = ImageDataGenerator(rescale=1./255)
        test_images_generator = datagen.flow_from_directory(self.image_path_test, class_mode=None, target_size=(352, 352), batch_size=self.hyper_batch_size, shuffle=False)
        test_labels_generator = datagen.flow_from_directory(self.label_path_test, class_mode=None, target_size=(352, 352), batch_size=self.hyper_batch_size, shuffle=False, color_mode="grayscale")
        if self.use_generator:
            return zip(test_images_generator, test_labels_generator)
        else:
            return (test_images_generator[0], test_labels_generator[0])
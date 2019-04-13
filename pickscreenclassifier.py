''' Module contains Pick Screen Classifier Model '''

import os
import glob
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.models import Sequential
import keras
import numpy as np
import matplotlib.pylab as plt
import math


class PickScreenClassifier:
    ''' Picking Screen image classifier model '''

    def __init__(self):
        self.model = None
        self.datagen = None
        self.train_path: str = ''
        self.gen_weights_path: str = ''

    def init_model(self, train_path='./resources/pick_screen_training/', generator_weights_path='./resources/models/'):
        self.train_path = train_path
        self.gen_weights_path = generator_weights_path

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(
            224, 224, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # this converts our 3D feature maps to 1D feature vectors
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=1e-3),
                      metrics=['accuracy'])

        self.model = model

    def classify(self, image, preprocess=True):
        if preprocess:
            image = self.get_train_datagen().standardize(image)

        return np.argmax(self.model.predict(image))

    def get_train_datagen(self):
        if not self.datagen:
            self.datagen = ImageDataGenerator(
                validation_split=0.3,
                featurewise_center=True,
                featurewise_std_normalization=True,
                rotation_range=10,
                shear_range=.1,
                zoom_range=[0.7, 1.5],
                horizontal_flip=True,
                brightness_range=[0.25, 1.7]
            )

            mean, std = self.load_gen_weights()

            if mean is None or std is None:
                images = []
                images_dir = os.path.join(self.train_path, "**/*.jpg")
                filenames = glob.glob(images_dir, recursive=True)

                for path in filenames:
                    img = load_img(path, color_mode="rgb").resize((224, 224))
                    images.append(img_to_array(img))

                self.datagen.fit(np.stack(images, axis=0))
                self.save_gen_weights()
                print('saving weights')
            else:
                print('loading weights')
                self.datagen.mean = mean
                self.datagen.std = std

        return self.datagen

    def train(self, epochs=50):
        # this is the augmentation configuration we will use for training
        train_datagen = self.get_train_datagen()

        train_generator = train_datagen.flow_from_directory(directory=self.train_path,
                                                            target_size=(
                                                                224, 224),
                                                            color_mode="rgb",
                                                            batch_size=32,
                                                            class_mode="categorical",
                                                            shuffle=True,
                                                            subset='training')

        validation_generator = train_datagen.flow_from_directory(directory=self.train_path,
                                                                 target_size=(
                                                                     224, 224),
                                                                 color_mode="rgb",
                                                                 batch_size=32,
                                                                 class_mode="categorical",
                                                                 shuffle=True,
                                                                 subset='validation')

        history = self.model.fit_generator(
            train_generator,
            steps_per_epoch=math.ceil(
                train_generator.samples / train_generator.batch_size),
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=math.ceil(
                validation_generator.samples / validation_generator.batch_size),
            verbose=1
        )

        plt.plot(history.history['acc'], label='train')
        plt.plot(history.history['val_acc'], label='test')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left')
        plt.show()

    def load_gen_weights(self):
        meanpath = os.path.join(self.gen_weights_path, 'pickscreen_mean.npy')
        stdpath = os.path.join(self.gen_weights_path, 'pickscreen_std.npy')
        mean = None
        std = None

        if os.path.isfile(meanpath):
            mean = np.load(meanpath)

        if os.path.isfile(stdpath):
            std = np.load(stdpath)

        return mean, std

    def save_gen_weights(self):
        if not self.datagen:
            return

        meanpath = os.path.join(self.gen_weights_path, 'pickscreen_mean.npy')
        stdpath = os.path.join(self.gen_weights_path, 'pickscreen_std.npy')

        np.save(meanpath, self.datagen.mean)
        np.save(stdpath, self.datagen.std)

    def load(self, file_name):
        self.model.load_weights(file_name)

    def save(self, file_name):
        self.model.save_weights(file_name)

''' Module contains Hero Classifier Model '''

import os
import glob
import math
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.models import Sequential
import keras
import numpy as np
import matplotlib.pylab as plt


class HeroClassifier:
    ''' Hero image classifier model '''

    def __init__(self):
        self.model: Sequential = None
        self.datagen: ImageDataGenerator = None
        self.train_path: str
        self.gen_weights_path: str

    def init_model(self, num_classes=117, target_size=96, train_path='./resources/hero_training/', generator_weights_path='./resources/models/'):
        self.train_path = train_path
        self.gen_weights_path = generator_weights_path

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(
            target_size, target_size, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # this converts our 3D feature maps to 1D feature vectors
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=1e-4),
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
                rotation_range=5,
                shear_range=.05,
                zoom_range=[0.9, 1.35],
                horizontal_flip=True,
                brightness_range=[0.25, 1.7]
            )

            mean, std = self.load_gen_weights()

            if mean is None or std is None:
                images = []
                images_dir = os.path.join(self.train_path, "**/*.png")
                filenames = glob.glob(images_dir, recursive=True)

                for path in filenames:
                    images.append(img_to_array(
                        load_img(path, color_mode="grayscale")))

                self.datagen.fit(np.stack(images, axis=0))
                self.save_gen_weights()
            else:
                self.datagen.mean = mean
                self.datagen.std = std

        return self.datagen

    def train(self, epochs=50):
        # this is the augmentation configuration we will use for training
        train_datagen = self.get_train_datagen()

        train_generator = train_datagen.flow_from_directory(directory=self.train_path,
                                                            target_size=(
                                                                96, 96),
                                                            color_mode="grayscale",
                                                            batch_size=32,
                                                            class_mode="categorical",
                                                            shuffle=True,
                                                            subset='training')

        validation_generator = train_datagen.flow_from_directory(directory=self.train_path,
                                                                 target_size=(
                                                                     96, 96),
                                                                 color_mode="grayscale",
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
        meanpath = os.path.join(self.gen_weights_path, 'hero_mean.npy')
        stdpath = os.path.join(self.gen_weights_path, 'hero_std.npy')
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

        meanpath = os.path.join(self.gen_weights_path, 'hero_mean.npy')
        stdpath = os.path.join(self.gen_weights_path, 'hero_std.npy')

        np.save(meanpath, self.datagen.mean)
        np.save(stdpath, self.datagen.std)

    def load(self, file_name):
        self.model.load_weights(file_name)

    def save(self, file_name):
        self.model.save_weights(file_name)

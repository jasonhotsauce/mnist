import argparse
from math import ceil

import numpy as np
from keras.datasets import mnist
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

num_classes = 10
verbose = 0


def load_data():
    num_rows = num_columns = 28
    num_channels = 1

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], num_rows, num_columns, num_channels).astype(np.float32) / 255
    X_test = X_test.reshape(X_test.shape[0], num_rows, num_columns, num_channels).astype(np.float32) / 255

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    return (X_train, y_train), (X_test, y_test)


def build_model():
    model = Sequential()
    model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3, decay=1e-4), metrics=['accuracy'])
    return model


def fit(model, data_augmentation=False, batch_size=32):
    if verbose:
        print("Loading dataset...")

    (X_train, y_train), (X_test, y_test) = load_data()
    if data_augmentation:
        if verbose:
            print("Augmenting images...")

        data_gen = ImageDataGenerator(width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      zoom_range=0.25,
                                      rotation_range=30,
                                      fill_mode='nearest')
        data_gen.fit(X_train)
        train_data_generator = data_gen.flow(X_train, y_train, batch_size)
        model.fit_generator(train_data_generator,
                            steps_per_epoch=int(ceil(len(X_train)/float(batch_size))),
                            epochs=4,
                            validation_data=(X_test, y_test), verbose=verbose)
    else:
        model.fit(X_train, y_train, batch_size=32, epochs=4, validation_data=(X_test, y_test), verbose=verbose)


def save(model, path="minstCNN.h5"):
    for k in model.layers:
        if type(k) is Dropout:
            model.layers.remove(k)
    model.save(path)


def main():
    parser = argparse.ArgumentParser(description="Build a MNIST CNN")
    parser.add_argument('-t', '--train', help="start traing the model", action="store_true")
    parser.add_argument('-o', '--output', action="store_true", help="Output the model")
    parser.add_argument('-v', '--verbose', action="count", help="output verbose")
    parser.add_argument('-a', '--augment', help="enable data augmentation", action="store_true")

    args = parser.parse_args()

    model = build_model()
    global verbose
    verbose = args.verbose

    if args.train:
        fit(model, data_augmentation=args.augment)
        if args.output:
            save(model)


if __name__ == "__main__":
    main()

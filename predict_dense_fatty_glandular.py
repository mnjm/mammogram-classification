from __future__ import print_function

import os
# import glob
from glob import glob

import cv2 as cv
import numpy as np

from datetime import datetime
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
# from keras import backend as Kback

def build_model(hight, weight, num_classes):
    model = Sequential()

    # Layer 1
    model.add(Conv2D(4, (3,3), padding="same", input_shape = (hight, weight, 1)))
    model.add(LeakyReLU(alpha=0.03))

    # Layer 2
    model.add(Conv2D(4, (3,3), padding="same"))
    model.add(LeakyReLU(alpha=0.03))
    model.add(AveragePooling2D())
    

    # Layer 3
    model.add(Conv2D(4, (3,3), padding="same"))
    model.add(LeakyReLU(alpha=0.03))
    model.add(AveragePooling2D())

    # Layer 4
    model.add(Conv2D(4, (3,3), padding="same"))
    model.add(LeakyReLU(alpha=0.03))
    model.add(AveragePooling2D())

    # Layer 5
    model.add(Conv2D(4, (3,3), padding="same"))
    model.add(LeakyReLU(alpha=0.03))

    # Layer 6
    model.add(Conv2D(4, (3,3), padding="same"))
    model.add(LeakyReLU(alpha=0.03))
    model.add(AveragePooling2D())


    # Fully Connected Layer
    # Layer 1 fully connected
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    # Layer 2 fully connected
    model.add(Dense(100))
    model.add(Activation('relu'))
    # Layer 3 fully connected
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


# Hyperparameters
imageShape = (224, 224)

def test(weights_file):
    start = datetime.now()
    ipath = glob(os.path.join("test", "*.pgm"))
    arrList = []

    for path in ipath:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, imageShape).reshape((imageShape[0], imageShape[1], 1))
        arrList.append(img)

    x_test = np.array(arrList)

    x_test = x_test.astype('float32')
    x_test /= 255

    print(x_test.shape)

    model = build_model(imageShape[0], imageShape[1], 3)
    model.load_weights(weights_file)

    output = model.predict_classes(x_test, verbose=1)
    print(output)

    # for out in output:
    #     if out == 0:

    for out, name in zip(output, ipath):
        name = name.split("/")[1]
        if out == 0:
            # print(name, "Normal")
            print("{0:5} Fat".format(name))
        elif out == 1:
            print("{0:5} Fat Gland".format(name))
        else:
            print("{0:5} Dense Gland".format(name))

# train()
test('fat_dense_glandular.hdf5')

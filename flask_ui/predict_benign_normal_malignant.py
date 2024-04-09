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
weights_file = "weights-img-0.99.hdf5"
model = None

def load():
    global model
    model = build_model(imageShape[0], imageShape[1], 3)
    model.load_weights(weights_file)

def test(filepath):
    global model
    start = datetime.now()
    arrList = []

    ipath = [filepath]

    for path in ipath:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        # print(img.shape)
        img = cv.resize(img, imageShape).reshape((imageShape[0], imageShape[1], 1))
        arrList.append(img)

    x_test = np.array(arrList)

    x_test = x_test.astype('float32')
    x_test /= 255

    # print(x_test.shape)

    # output = model.predict_classes(x_test, verbose=0)
    output = model.predict(x_test, verbose=0)
    # print(output)
    max = np.max(output)
    output = np.where(output==max)
    # print(output)
    out = output[1][0]
    # print(out)

    label = ["Normal", "Benign", "Malignant"]
    # print(label[out], max)
    print("{},{}".format(label[out], max))


    # for out, name in zip(output, ipath):
    #     name = name.split("/")[1]
    #     if out == 0:
    #         # print(name, "Normal")
    #         # print("{0:5} Normal".format(name))
    #         print("Normal")
    #     elif out == 1:
    #         # print("{0:5} Benign".format(name))
    #         print("Benign")
    #     else:
    #         print("Malignant")
    #         # print("{0:5} Malignant".format(name))

# train()
# test('weights-img-0.99.hdf5')


import sys
if len(sys.argv) > 1: 
    imgNo = sys.argv[1]
    load()
    test('static/segmented/{}.png'.format(imgNo))
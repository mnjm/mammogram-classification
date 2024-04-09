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

def build_model(height, weight, num_classes):
    model = Sequential()

    # Layer 1
    model.add(Conv2D(4, (3,3), padding="same", input_shape = (height, weight, 1)))
    model.add(LeakyReLU(alpha=0.03))

    # Layer 2
    model.add(Conv2D(4, (3,3), padding="same"))
    model.add(LeakyReLU(alpha=0.03))
    model.add(MaxPooling2D())
    

    # Layer 3
    model.add(Conv2D(4, (3,3), padding="same"))
    model.add(LeakyReLU(alpha=0.03))
    model.add(MaxPooling2D())

    # Layer 4
    model.add(Conv2D(4, (3,3), padding="same"))
    model.add(LeakyReLU(alpha=0.03))
    model.add(MaxPooling2D())

    # Layer 5
    model.add(Conv2D(4, (3,3), padding="same"))
    model.add(LeakyReLU(alpha=0.03))

    # Layer 6
    model.add(Conv2D(4, (3,3), padding="same"))
    model.add(LeakyReLU(alpha=0.03))
    model.add(MaxPooling2D())

    # Layer 7
    model.add(Conv2D(4, (3,3), padding="same"))
    model.add(LeakyReLU(alpha=0.03))
    model.add(MaxPooling2D())


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

    return  model


# Hyperparameters
imageShape = (224, 224)
weights_file = "seven_stages.hdf5" 
model = None

def load():
    global model
    model = build_model(imageShape[0], imageShape[1], 7)
    model.load_weights(weights_file)

def test(filepath):
    global model
    start = datetime.now()
    arrList = []

    ipath = [filepath]

    for path in ipath:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, imageShape).reshape((imageShape[0], imageShape[1], 1))
        arrList.append(img)

    x_test = np.array(arrList)

    x_test = x_test.astype('float32')
    x_test /= 255

    # print(x_test.shape)

    # model = build_model(imageShape[0], imageShape[1], 7)
    # model.load_weights(weights_file)

    # output = model.predict_classes(x_test, verbose=0)
    # print(output)
    
    output = model.predict(x_test, verbose=0)
    max = np.max(output)
    out = np.where(output==max)[1][0]
    label = ["Arch", "Asym", "Calc", "Circ", "Norm", "Misc", "Spic"]
    # print label[out], max
    print "{},{}".format(label[out], max)

    # for out, name in zip(output, ipath):
    #     name = name.split("/")[1]
    #     if out == 0:
    #         # print("{0:5} arch".format(name))
    #         print "Arch"
    #     elif out == 1:
    #         # print("{0:5} asym".format(name))
    #         print "Asym"
    #     elif out == 2:
    #         # print("{0:5} calc".format(name))
    #         print "Calc"
    #     elif out == 3:
    #         # print("{0:5} circ".format(name))
    #         print "Circ"
    #     elif out == 4:
    #         # print("{0:5} norm".format(name))
    #         print "Norm"
    #     elif out == 5:
    #         # print("{0:5} misc".format(name))
    #         print "Misc"
    #     else:
    #         # print("{0:5} spic".format(name))
    #         print "Spic"

# train()
# test('seven_stages.hdf5')

import sys
if len(sys.argv) > 1: 
    imgNo = sys.argv[1]
    load()
    test('static/segmented/{}.png'.format(imgNo))

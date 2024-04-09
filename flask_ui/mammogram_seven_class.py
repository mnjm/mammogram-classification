from __future__ import print_function

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

    return model


def shuffle(x, y):
    # seed = np.random.randint(512, 1024, size=1)[0]
    seed = 1024
    rand_state = np.random.RandomState(seed)
    rand_state.shuffle(x)
    rand_state.seed(seed)
    rand_state.shuffle(y)

def get_data(imageShape):

    def read_from(filenames, label):
        imgs = []
        labels = []
        for filename in filenames:
            img = cv.imread('dataset/{}.pgm'.format(filename), cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, imageShape).reshape((imageShape[0], imageShape[1], 1))
            img_1 = cv.imread('dataset/{}_1.pgm'.format(filename), cv.IMREAD_GRAYSCALE)
            img_1 = cv.resize(img_1, imageShape).reshape((imageShape[0], imageShape[1], 1))
            img_2 = cv.imread('dataset/{}_2.pgm'.format(filename), cv.IMREAD_GRAYSCALE)
            img_2 = cv.resize(img_2, imageShape).reshape((imageShape[0], imageShape[1], 1))
            img_3 = cv.imread('dataset/{}_3.pgm'.format(filename), cv.IMREAD_GRAYSCALE)
            img_3 = cv.resize(img_3, imageShape).reshape((imageShape[0], imageShape[1], 1))
            imgs.extend([img, img_1, img_2, img_3])
            temp = [ label for _ in xrange(4) ]
            labels.extend(temp)

        return imgs, labels

    arch_file = open('dataset/ARCH.txt')
    asym_file = open('dataset/ASYM.txt')
    calc_file = open('dataset/CALC.txt')
    circ_file = open('dataset/CIRC.txt')
    norm_file = open('dataset/NORM.txt')
    misc_file = open('dataset/MISC.txt')
    spic_file = open('dataset/SPIC.txt')

    arch = [line.strip() for line in arch_file.readlines()]
    asym = [line.strip() for line in asym_file.readlines()]
    calc = [line.strip() for line in calc_file.readlines()]
    circ = [line.strip() for line in circ_file.readlines()]
    norm = [line.strip() for line in norm_file.readlines()]
    misc = [line.strip() for line in misc_file.readlines()]
    spic = [line.strip() for line in spic_file.readlines()]

    arch_img, arch_label = read_from(arch, [1,0,0,0,0,0,0])
    asym_img, asym_label = read_from(asym, [0,1,0,0,0,0,0])
    calc_img, calc_label = read_from(calc, [0,0,1,0,0,0,0])
    circ_img, circ_label = read_from(circ, [0,0,0,1,0,0,0])
    norm_img, norm_label = read_from(norm, [0,0,0,0,1,0,0])
    misc_img, misc_label = read_from(misc, [0,0,0,0,0,1,0])
    spic_img, spic_label = read_from(spic, [0,0,0,0,0,0,1])

    x = []
    x.extend(arch_img)
    x.extend(asym_img)
    x.extend(calc_img)
    x.extend(circ_img)
    x.extend(norm_img)
    x.extend(misc_img)
    x.extend(spic_img)

    y = []
    y.extend(arch_label)
    y.extend(asym_label)
    y.extend(calc_label)
    y.extend(circ_label)
    y.extend(norm_label)
    y.extend(misc_label)
    y.extend(spic_label)

    shuffle(x, y)

    train_size = int(len(x) * 0.99)
    train = True

    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    x_train = np.array(x_train, dtype = np.float32) / 255
    y_train = np.array(y_train, dtype = np.float32)
    x_test = np.array(x_test, dtype = np.float32) / 255
    y_test = np.array(y_test, dtype = np.float32)

    return (x_train, y_train), (x_test, y_test)

# ---------------- Training -------------------------

# Hyperparameters
epochs = 50
batch_size = 32
imageShape = (224, 224)

# Model
model = build_model(imageShape[0], imageShape[1], 7)
# Compiling Model
print('Done Building Model...')

(x_train, y_train), (x_test, y_test) = get_data(imageShape)
model.compile(  loss = keras.losses.categorical_crossentropy,
                optimizer = keras.optimizers.Adadelta(),
                metrics=['accuracy'])

print('Done Compiling Model...')

def train():
    start = datetime.now()
    print('Training...')

    # Storing best save only weights
    checkpointer = ModelCheckpoint(filepath='seven_stages.hdf5', verbose=1, save_best_only=True)

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks= [ checkpointer ],
          validation_data= (x_test, y_test))
    print('Training Completed')

    print('Testing with train data:')
    score_train = model.evaluate(x_train, y_train, verbose=1)

    print('Testing with test data:')
    score_test = model.evaluate(x_test, y_test, verbose=1)

    end = datetime.now()

    print()
    print('Time Took for training and testing: {}'.format(str(end - start)))
    print('Test Loss: {}'.format(score_test[0]))
    print('Test Accuracy: {}'.format(score_test[1]))
    print()
    print('Train Loss: {}'.format(score_train[0]))
    print('Train Accuracy: {}'.format(score_train[1]))

def test(weights_file):
    start = datetime.now()
    model.load_weights(weights_file)

    print('Testing with train data:')
    score_train = model.evaluate(x_train, y_train, verbose=1)

    print('Testing with test data:')
    score_test = model.evaluate(x_test, y_test, verbose=1)
    end = datetime.now()


    print('Time Took for training and testing: {}'.format(str(end - start)))
    print('Test Loss: {}'.format(score_test[0]))
    print('Test Accuracy: {}'.format(score_test[1]))
    print()
    print('Train Loss: {}'.format(score_train[0]))
    print('Train Accuracy: {}'.format(score_train[1]))

train()
# test('Weight-0.85.hdf5')
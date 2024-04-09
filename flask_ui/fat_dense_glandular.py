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

def build_model(hight, weight, num_classes):
    model = Sequential()

    # Layer 1
    model.add(Conv2D(4, (3,3), padding="same", input_shape = (hight, weight, 1)))
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

    fatty_file = open('dataset/Fatty.txt')
    fatty_gland_file = open('dataset/Fatty-Gand.txt')
    dense_gland_file = open('dataset/Dense-Gand.txt')

    fatty = [line.strip() for line in fatty_file.readlines()]

    fatty_gland = [line.strip() for line in fatty_gland_file.readlines()]

    dense_gland = [line.strip() for line in dense_gland_file.readlines()]

    fatty_img, fatty_label = read_from(fatty, [0,1,0])

    fatty_gland_img, fatty_gland_label = read_from(fatty_gland, [0,0,1])

    dense_gland, dense_gland_label = read_from(dense_gland, [1,0,0])

    x = []
    x.extend(fatty_img)
    x.extend(fatty_gland_img)
    x.extend(dense_gland)

    y = []
    y.extend(fatty_label)
    y.extend(fatty_gland_label)
    y.extend(dense_gland_label)
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
model = build_model(imageShape[0], imageShape[1], 3)
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
    checkpointer = ModelCheckpoint(filepath='fat_dense_glandular.hdf5', verbose=1, save_best_only=True)

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
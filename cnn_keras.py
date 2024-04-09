from __future__ import print_function

import os

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

    benign_file = open('dataset/benign.txt')
    malignant_file = open('dataset/malignant.txt')
    normal_file = open('dataset/normal.txt')

    benign = [line.strip() for line in benign_file.readlines()]

    malignant = [line.strip() for line in malignant_file.readlines()]

    normal = [line.strip() for line in normal_file.readlines()]

    # benign_img = [ cv.imread('dataset/{}.pgm'.format(no)) for no in benign ]
    # benign_label = [ [0,1,0] for _ in xrange(len(benign)) ]
    benign_img, benign_label = read_from(benign, [0,1,0])

    # malignant_img = [ cv.imread('dataset/{}.pgm'.format(no)) for no in malignant ]
    # malignant_label = [ [0,0,1] for _ in xrange(len(malignant)) ]
    malignant_img, malignant_label = read_from(malignant, [0,0,1])

    # normal_img = [ cv.imread('dataset/{}.pgm'.format(no)) for no in normal ]
    # normal_label = [ [1,0,0] for _ in xrange(len(normal)) ]
    normal_img, normal_label = read_from(normal, [1,0,0])

    x = []
    x.extend(benign_img)
    x.extend(malignant_img)
    x.extend(normal_img)

    y = []
    y.extend(benign_label)
    y.extend(malignant_label)
    y.extend(normal_label)
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
    checkpointer = ModelCheckpoint(filepath='benign_malignant_normal.hdf5', verbose=1, save_best_only=True)

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
    wpath = "weights_try6.hdf5"
    ipath = glob(os.path.join("dataset", "*.pgm"))
    arrList = []

    for path in ipath:
        img = cv.imread(path)
        img = cv.resize(img, (imageShape))
        arrList.append(img)

    x_test = np.array(arrList)

    x_test = x_test.astype('float32')
    x_test /= 255

    model.load_weights(weights_file)

    output = model.predict_classes(x_test, verbose=1)
    print(output)


train()
# test('Weight-0.85.hdf5')
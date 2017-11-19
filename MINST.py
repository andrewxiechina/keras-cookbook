from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape
from keras.optimizers import RMSprop
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
import os

np.random.seed(98105)

# Constants
EPOCHS = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3
IMG_SHAPE = (28,28)
INPUT_SHAPE = (784,)
PATH = "../input/mnist.npy"

# Data
if not os.path.isfile(PATH):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Save data to save time
    np.save(PATH, [(X_train, y_train), (X_test, y_test)])
else:
    # Load data if saved
    (X_train, y_train), (X_test, y_test) = np.load(PATH)

# One hot encodimg
y_train_o = np_utils.to_categorical(y_train, NB_CLASSES)
y_test_o = np_utils.to_categorical(y_test, NB_CLASSES)

# Model
model = Sequential()
model.add(Reshape(INPUT_SHAPE, input_shape=IMG_SHAPE))
model.add(Dense(N_HIDDEN))
model.add(Activation("relu"))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation("relu"))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation("softmax"))
print(model.summary())

model.compile(
    optimizer=RMSprop(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Fit
history = model.fit(X_train, y_train_o,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT
)

score = model.evaluate(X_test, y_test_o, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])



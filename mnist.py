import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from activations import Tanh
from losses import mse, mse_prime

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data: 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 28 * 28, 1)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)
y_train = y_train.reshape(y_train.shape[0], 10, 1)

# same for test data: 10000 samples
x_test = x_test.reshape(x_test.shape[0], 28 * 28, 1)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)
y_test = y_test.reshape(y_test.shape[0], 10, 1)

# neural network
network = [
    Dense(28 * 28, 40),
    Tanh(),
    Dense(40, 10),
    Tanh()
]

epochs = 100
learning_rate = 0.1

# train
for e in range(epochs):
    error = 0
    # train on 1000 samples, since we're not training on GPU...
    for x, y in zip(x_train[:1000], y_train[:1000]):
        # forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # error
        error += mse(y, output)

        # backward
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)

    error /= 1000
    print('%d/%d, error=%f' % (e + 1, epochs, error))

# test on 20 samples
for x, y in zip(x_test[:20], y_test[:20]):
    output = x
    for layer in network:
        output = layer.forward(output)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))

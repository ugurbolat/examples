# REF: https://ml-cheatsheet.readthedocs.io/en/latest/forwardpropagation.html

import numpy as np

np.random.seed(42)

INPUT_LAYER_SIZE = 4
HIDDEN_LAYER_SIZE = 16
OUTPUT_LAYER_SIZE = 2

def init_weights():
    Wh = np.random.randn(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE) * \
                np.sqrt(2.0/INPUT_LAYER_SIZE)
    Wo = np.random.randn(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE) * \
                np.sqrt(2.0/HIDDEN_LAYER_SIZE)
    return Wh, Wo


def init_bias():
    Bh = np.full((1, HIDDEN_LAYER_SIZE), 0.1)
    Bo = np.full((1, OUTPUT_LAYER_SIZE), 0.1)
    return Bh, Bo

def relu(Z):
    return np.maximum(0, Z)

# def relu_prime(Z):
#     '''
#     Z - weighted input matrix

#     Returns gradient of Z where all
#     negative values are set to 0 and
#     all positive values set to 1
#     '''
#     Z[Z < 0] = 0
#     Z[Z > 0] = 1
#     return Z

def cost(yHat, y):
    cost = np.sum((yHat - y)**2) / 2.0
    return cost

# def cost_prime(yHat, y):
#     return yHat - y

def feed_forward(X, Wh, Wo, Bh, Bo):
    '''
    X    - input matrix
    Zh   - hidden layer weighted input
    Zo   - output layer weighted input
    H    - hidden layer activation
    y    - output layer
    yHat - output layer predictions
    '''

    # Hidden layer
    Zh = np.dot(X, Wh) + Bh
    H = relu(Zh)

    # Output layer
    Zo = np.dot(H, Wo) + Bo
    yHat = relu(Zo)
    
    return yHat


x = np.random.randn(1, INPUT_LAYER_SIZE)
print("x.shape: ", x.shape)
print("x: ", x)
y = np.random.randn(1, OUTPUT_LAYER_SIZE)
print("y.shape: ", y.shape)
print("y: ", y)

Wh_1, Wo_1 = init_weights()
Bh_1, Bo_1 = init_bias()
print("Wh_1: ", Wh_1)
print("Wh_1.shape: ", Wh_1.shape)
print("Wo_1: ", Wo_1)
print("Wo_1.shape: ", Wo_1.shape)
print("Bh_1: ", Bh_1)
print("Bh_1.shape: ", Bh_1.shape)
print("Bo_1: ", Bo_1)
print("Bo_1.shape: ", Bo_1.shape)

for i in range(10):
    print("--------- Iteration: ", i)

    y_hat = feed_forward(x, Wh_1, Wo_1, Bh_1, Bo_1)
    print("y_hat: ", y_hat)

    loss = cost(y_hat, y)
    print("loss: ", loss)
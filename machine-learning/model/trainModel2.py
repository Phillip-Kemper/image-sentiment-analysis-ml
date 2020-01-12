import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import model.neuralNetworkBackpropagation as ml
import dataRetrieval.dataRetrieval as getData

CSV_FILE = "../dataRetrieval/sources/fer2013.csv"

trainingData, evalData = getData.load_fer(CSV_FILE)
# print(trainingData)
y = trainingData["emotion"].values
# getting labels for training
# y = pd.get_dummies(trainingData['emotion']).values

width, height = 48, 48
datapoints = trainingData['pixels'].tolist()

# getting features for training
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    X.append(xx)

X = np.asarray(X)
# X = np.expand_dims(X, axis=0)
# X = np.expand_dims(X, -1)

# print('X_TRAINING')
# print(X)
# print('test')
# print(X.shape)

X = X[:100, :]
y = y[:100]


def sigmoid(z):
    """
    return the sigmoid of z
    """

    return 1 / (1 + np.exp(-z))


def predict(Theta1, Theta2, X):
    """
    Predict the label of an input given a trained neural network
    """
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))

    a1 = sigmoid(X @ Theta1.T)
    a1 = np.hstack((np.ones((m, 1)), a1))  # hidden layer
    a2 = sigmoid(a1 @ Theta2.T)  # output layer

    return np.argmax(a2, axis=1) + 1


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    """
    nn_params contains the parameters unrolled into a vector

    compute the cost and gradient of the neural network
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2
    Theta1 = nn_params[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)

    m = X.shape[0]
    J = 0
    X = np.hstack((np.ones((m, 1)), X))
    y10 = np.zeros((m, num_labels))
    print(np.shape(y10))
    y_exp = np.expand_dims(y, axis=1)

    a1 = sigmoid(X @ Theta1.T)
    a1 = np.hstack((np.ones((m, 1)), a1))  # hidden layer
    a2 = sigmoid(a1 @ Theta2.T)  # output layer

    for i in range(1, num_labels + 1):
        y10[:, i - 1][:, np.newaxis] = np.where(y_exp == i, 1, 0)
    for j in range(num_labels):
        J = J + sum(-y10[:, j] * np.log(a2[:, j]) - (1 - y10[:, j]) * np.log(1 - a2[:, j]))

    cost = 1 / m * J
    reg_J = cost + Lambda / (2 * m) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))

    # Implement the backpropagation algorithm to compute the gradients

    grad1 = np.zeros((Theta1.shape))
    grad2 = np.zeros((Theta2.shape))

    for i in range(m):
        xi = X[i, :]  # 1 X 401
        a1i = a1[i, :]  # 1 X 26
        a2i = a2[i, :]  # 1 X 10
        d2 = a2i - y10[i, :]
        d1 = Theta2.T @ d2.T * sigmoidGradient(np.hstack((1, xi @ Theta1.T)))
        grad1 = grad1 + d1[1:][:, np.newaxis] @ xi[:, np.newaxis].T
        grad2 = grad2 + d2.T[:, np.newaxis] @ a1i[:, np.newaxis].T

    grad1 = 1 / m * grad1
    grad2 = 1 / m * grad2

    grad1_reg = grad1 + (Lambda / m) * np.hstack((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]))
    grad2_reg = grad2 + (Lambda / m) * np.hstack((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]))

    return cost, grad1, grad2, reg_J, grad1_reg, grad2_reg


def sigmoidGradient(z):
    """
    computes the gradient of the sigmoid function
    """
    sigmoid = 1 / (1 + np.exp(-z))

    return sigmoid * (1 - sigmoid)


input_layer_size = 2304
hidden_layer_size = 25
num_labels = 3


def randInitializeWeights(L_in, L_out):
    """
    randomly initializes the weights of a layer with L_in incoming connections and L_out outgoing connections.
    """

    epi = (6 ** 1 / 2) / (L_in + L_out) ** 1 / 2

    W = np.random.rand(L_out, L_in + 1) * (2 * epi) - epi

    return W


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
initial_nn_params = np.append(initial_Theta1.flatten(), initial_Theta2.flatten())


def gradientDescentnn(X, y, initial_nn_params, alpha, num_iters, Lambda, input_layer_size, hidden_layer_size,
                      num_labels):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha

    return theta and the list of the cost of theta during each iteration
    """
    Theta1 = initial_nn_params[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size,
                                                                                      input_layer_size + 1)
    Theta2 = initial_nn_params[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)

    m = len(y)
    J_history = []

    for i in range(num_iters):
        nn_params = np.append(Theta1.flatten(), Theta2.flatten())
        cost, grad1, grad2 = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[
                             3:]
        Theta1 = Theta1 - (alpha * grad1)
        Theta2 = Theta2 - (alpha * grad2)
        J_history.append(cost)

    nn_paramsFinal = np.append(Theta1.flatten(), Theta2.flatten())
    return nn_paramsFinal, J_history


EPOCH_NUMBER = 40
for i in range(EPOCH_NUMBER):
    print('currently at epoch')
    print(EPOCH_NUMBER)

    nnTheta, nnJ_history = gradientDescentnn(X, y, initial_nn_params, 0.8, 800, 1, input_layer_size, hidden_layer_size,
                                             num_labels)
    Theta1 = nnTheta[:((input_layer_size + 1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nnTheta[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size + 1)

    pred = ml.predict(Theta1, Theta2, X, y)

    with open("out_trainModel1.txt", "a") as myfile:
        myfile.write("Epoch Number:")
        myfile.write(str(i))
        myfile.write(np.mean(pred == y.flatten()) * 100, "%")

        if i == EPOCH_NUMBER:
            myfile.write("END")
            np.save('FinalModel1.npy', initial_nn_params)



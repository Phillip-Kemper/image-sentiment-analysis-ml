import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt
import checkNNGradients
import displayData


# deprecated
def reverseY(yM, num_entradas, num_etiquetasM):
    rev = np.zeros((5000, 1))
    num_iter = 5000
    for i in range(num_iter):
        j = yM[i]
        test = np.where(j == j.max())
        test = (test[0][0]) + 1
        #    print(test)
        rev[i] = test
    return rev


def onlyGrad(nn_params, input_layer_size, hidden_layer_size, num_labelsM, X, yM, lmbda):
    return gradient(nn_params, input_layer_size, hidden_layer_size, num_labelsM, X, yM, lmbda)[1]


def reverseY(yM, num_entradas, num_etiquetasM):
    rev = np.zeros((5000, 1))
    num_iter = 5000
    for i in range(num_iter):
        j = yM[i]
        test = np.where(j == j.max())
        test = (test[0][0]) + 1
        #    print(test)
        rev[i] = test
    return rev


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradientCheck(params_rm, num_entradas, num_ocultas, num_etiquetas, X, y, reg, epsilon):
    cost1Vector = params_rm + epsilon
    cost1 = costFunction(cost1Vector, num_entradas, num_ocultas, num_etiquetas, X, y, reg)

    cost2Vector = params_rm - epsilon
    cost2 = costFunction(cost2Vector, num_entradas, num_ocultas, num_etiquetas, X, y, reg)

    aprox = (cost1 + cost2) / (2 * epsilon)
    return aprox


def costFunction(nn_params, num_entradas, num_ocultas, num_etiquetas, X, y, reg):
    m = len(y)
    theta1 = np.reshape(nn_params[:num_ocultas * (num_entradas + 1)], (num_ocultas, num_entradas + 1), 'F')
    theta2 = np.reshape(nn_params[num_ocultas * (num_entradas + 1):], (num_etiquetas, num_ocultas + 1), 'F')

    a1 = np.hstack((np.ones((m, 1)), X))
    a2 = sigmoid(np.dot(a1, theta1.T))
    a2 = np.hstack((np.ones((m, 1)), a2))
    h = sigmoid(np.dot(a2, theta2.T))

    y = pd.get_dummies(y.flatten()).to_numpy()

    temp = np.sum(np.multiply(y, np.log(h)) + np.multiply(1 - y, np.log(1 - h)))

    firstS = np.sum(np.sum(np.power(theta1[:, 1:], 2), axis=1))
    secondS = np.sum(np.sum(np.power(theta2[:, 1:], 2), axis=1))

    cost = np.sum(temp / (-m)) + (firstS + secondS) * reg / (2 * m)
    return cost


def pesosAleatorios(L_in, L_out):
    epsilon = 0.12
    rand = np.random.rand(L_out, L_in + 1) * 2 * epsilon - epsilon
    return rand


def gradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
    # print("loading/training...")
    initial_theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                                (hidden_layer_size, input_layer_size + 1), 'F')
    initial_theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                                (num_labels, hidden_layer_size + 1), 'F')
    y_d = pd.get_dummies(y.flatten())
    delta1 = np.zeros(initial_theta1.shape)
    delta2 = np.zeros(initial_theta2.shape)
    m = len(y)

    for i in range(X.shape[0]):
        ones = np.ones(1)
        a1 = np.hstack((ones, X[i]))
        z2 = np.dot(a1,initial_theta1)
        a2 = np.hstack((ones, sigmoid(z2)))
        z3 = np.dot(a2,initial_theta2.T)
        a3 = sigmoid(z3)

        d3 = a3 - y_d.iloc[i, :][np.newaxis, :]
        z2 = np.hstack((ones, z2))
        d2 = np.multiply(np.dot(initial_theta2.T,d3.T), (np.multiply(sigmoid(z2), 1 - sigmoid(z2))).T[:, np.newaxis])
        delta1 = delta1 + np.dot(d2[1:, :] , a1[np.newaxis, :])
        delta2 = delta2 + np.dot(d3.T ,a2[np.newaxis, :])

    delta1 /= m
    delta2 /= m
    # print(delta1.shape, delta2.shape)
    delta1[:, 1:] = delta1[:, 1:] + initial_theta1[:, 1:] * lmbda / m
    delta2[:, 1:] = delta2[:, 1:] + initial_theta2[:, 1:] * lmbda / m

    grad = np.hstack((delta1.ravel(order='F'), delta2.ravel(order='F')))
    cost = costFunction(grad, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
    return cost, grad


def predict(theta1, theta2, X, y):
    m = len(y)
    ones = np.ones((m, 1))
    a1 = np.hstack((ones, X))
    a2 = sigmoid(np.dot(a1, theta1.T))
    a2 = np.hstack((ones, a2))
    h = sigmoid(np.dot(a2, theta2.T))
    return np.argmax(h, axis=1) + 1


data = loadmat('ex4data1.mat')
y = data['y']
X = data['X']
num_etiquetas = 10
num_entradas = 400
num_ocultas = 25
_lambda = 1
epsilon = np.math.pow(10, -4)

# visualize data 100 times
for i in range(1, 10):
    figure = plt.subplot(10, 1, i)
    sample = np.random.choice(X.shape[0], 10)
    plt.imshow(X[sample, :].reshape(-1, 20).T)
    plt.axis('off')
plt.show
weights = loadmat('ex4weights.mat')

theta1, theta2 = weights['Theta1'], weights['Theta2']
# Theta1 es de dimension 25 x 401
# Theta2 es de dimension 10 x 26
thetaVector = np.append(np.ravel(theta1, order='F'), np.ravel(theta2, order='F'))

# cF = costFunction(thetaVector, num_entradas, num_ocultas, num_etiquetas, X, y, _lambda)

# gradientCheckResult = gradientCheck(thetaVector, num_entradas, num_ocultas, num_etiquetas, X, y, _lambda, epsilon)

initialTheta1 = pesosAleatorios(num_entradas, num_ocultas)
initialTheta2 = pesosAleatorios(num_ocultas, num_etiquetas)

initialThetaVector = np.append(np.ravel(initialTheta1, order='F'), np.ravel(initialTheta2, order='F'))

for i in range(11):
    lamb = i / 10

    thetaOpt = opt.fmin_cg(maxiter=70, f=costFunction, x0=initialThetaVector, fprime=onlyGrad,
                           args=(num_entradas, num_ocultas, num_etiquetas, X, y.flatten(), lamb))

    theta1 = np.reshape(thetaOpt[:num_ocultas * (num_entradas + 1)], (num_ocultas, num_entradas + 1), 'F')
    theta2 = np.reshape(thetaOpt[num_ocultas * (num_entradas + 1):], (num_etiquetas, num_ocultas + 1), 'F')

    pred = predict(theta1, theta2, X, y)
    #    print(pred)
    print(lamb)
    print(np.mean(pred == y.flatten()) * 100, "%")

print("end")

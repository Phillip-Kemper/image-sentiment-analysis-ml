import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt
import dataRetrieval.dataRetrieval as getData
import model.neuralNetworkBackpropagation as ml
import sys

CSV_FILE = "../dataRetrieval/sources/fer2013.csv"


trainingData, evalData = getData.load_fer(CSV_FILE)
# print(trainingData)
y = trainingData["emotion"].values
print(len(y))
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


X = np.array([ml.min_max_norm(x) for x in X.T]).T


num_labels = 3
num_input = 2304
num_hidden = 1539          # first tried 25 neurons in hidden layer, now applying rule of thumb
                          # The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
_lambda = 1
epsilon = np.math.pow(10, -4)

initialTheta1 = ml.randomTheta(num_input, num_hidden)
initialTheta2 = ml.randomTheta(num_hidden, num_labels)

initialThetaVector = np.append(np.ravel(initialTheta1, order='F'), np.ravel(initialTheta2, order='F'))


EPOCH_NUMBER = 60
for i in range(EPOCH_NUMBER):
    print('currently at epoch')
    print(i)
    thetaOpt = initialThetaVector

    thetaOpt = opt.fmin_cg(maxiter=200, f=ml.costFunction, x0=thetaOpt, fprime=ml.onlyGrad,
                           args=(num_input, num_hidden, num_labels, X, y.flatten(), _lambda))

    theta1 = np.reshape(thetaOpt[:num_hidden * (num_input + 1)], (num_hidden, num_input + 1), 'F')
    theta2 = np.reshape(thetaOpt[num_hidden * (num_input + 1):], (num_labels, num_hidden + 1), 'F')

    initialThetaVector = np.append(np.ravel(initialTheta1, order='F'), np.ravel(initialTheta2, order='F'))

    pred = ml.predict(theta1, theta2, X, y)
    with open("out_trainModel1.txt", "a") as myfile:
        myfile.write("Epoch Number:")
        myfile.write("\n")
        myfile.write(str(i))
        myfile.write("\n")
        temp = np.mean(pred == y.flatten()) * 100
        myfile.write(str(temp))
        myfile.write("\n")

        if i == EPOCH_NUMBER:
            myfile.write("END")
            np.save('FinalModel1.npy', thetaOpt)



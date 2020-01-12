import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt
import dataRetrieval.dataRetrieval as getData
import model.neuralNetworkBackpropagation as ml

CSV_FILE = "../dataRetrieval/sources/fer2013.csv"

def main():
    trainingData, evalData = getData.load_fer(CSV_FILE)
    # print(trainingData)
    y = trainingData["emotion"].values
    # getting labels for training
    #y = pd.get_dummies(trainingData['emotion']).values


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

    #print('X_TRAINING')
    #print(X)
    #print('test')
    #print(X.shape)

    return X, y


X, y = main()
num_labels = 3
num_input = 2304
num_hidden = 25
_lambda = 1
epsilon = np.math.pow(10, -4)

initialTheta1 = ml.randomTheta(num_input, num_hidden)
initialTheta2 = ml.randomTheta(num_hidden, num_labels)

initialThetaVector = np.append(np.ravel(initialTheta1, order='F'), np.ravel(initialTheta2, order='F'))

print(np.shape(initialThetaVector))

thetaOpt = opt.fmin_cg(maxiter=70, f=ml.costFunction, x0=initialThetaVector, fprime=ml.onlyGrad,
                       args=(num_input, num_hidden, num_labels, X, y.flatten(), _lambda))

theta1 = np.reshape(thetaOpt[:num_hidden * (num_input + 1)], (num_hidden, num_input + 1), 'F')
theta2 = np.reshape(thetaOpt[num_hidden * (num_input + 1):], (num_labels, num_hidden + 1), 'F')

pred = ml.predict(theta1, theta2, X, y)
print(pred)
print(np.mean(pred == y.flatten()) * 100, "%")



#print(res)

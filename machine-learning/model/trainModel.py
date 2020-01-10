import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt
import dataRetrieval.dataRetrieval as getData

CSV_FILE = "../dataRetrieval/sources/fer2013.csv"

def main():
    trainingData, evalData = getData.load_fer(CSV_FILE)
    #print(trainingData)
    y = trainingData["emotion"]
    # TODO @PK: write pixel data to X matrix
    X_data = trainingData["pixels"].values.reshape((17010, 1))
    X = np.zeros((17010, 2304))

    width, height = 48, 48

    datapoints = trainingData['pixels'].tolist()

    # getting features for training
    X = []
    for xseq in datapoints:
        xx = [int(xp) for xp in xseq.split(' ')]
#        xx = np.asarray(xx).reshape(1, 48*48)
#        X.append(xx.astype('float64'))
        X.append(xx)

    X = np.asarray(X)
    X = np.expand_dims(X, axis=0)
    #X = np.expand_dims(X, -1)

    print('X_TRAINING')
    print(X)

    # getting labels for training
    y = pd.get_dummies(trainingData['emotion']).values

    return X


res = main()

#print(res)

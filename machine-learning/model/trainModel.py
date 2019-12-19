import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as opt
import dataRetrieval.dataRetrieval as getData

CSV_FILE = "../dataRetrieval/sources/fer2013.csv"

def main():
    trainingData, evalData = getData.load_fer(CSV_FILE)
    print(trainingData)
    y = trainingData["emotion"]

    # TODO @PK: write pixel data to X matrix
    X = trainingData["pixels"]
    return X


res = main()


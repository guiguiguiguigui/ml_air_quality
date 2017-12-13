import sklearn.kernel_ridge as skr
import sklearn
import numpy as np
import pylab as pl
import math
import air_vecs as av
import random
import data_merge as dr

FEATURES = ["temp", "humidity"] + ["d" + str(i) for i in range(16)]
OUTPUT = ["no2", "so2", "no"]

DEGREE = 5

def process(input_indices, output_indices, data_point):
    input = []
    output = []

    for i in sorted(list(input_indices)):
        input.append(float(data_point[i]))
    for i in sorted(list(output_indices)):
        output.append(float(data_point[i]))

    return (input, output)


if __name__ == "__main__":
    
    data = av.get_data()
    train_X, train_Y = data["train"]
    test_X, test_Y = data["test"]

    poly = sklearn.preprocessing.PolynomialFeatures(degree=DEGREE)

    train_X = train_X[:15000]
    train_Y = np.array(map(lambda x: x[0], train_Y[:15000]))
    test_X = test_X[:5000]
    test_Y = np.array(map(lambda x: x[0], test_Y[:5000]))

    degrees = [2,3,4,5,6,7]
    gammas = [None]
    
    best_model = None
    best_score = None
    best_params = None

    model = sklearn.linear_model.LinearRegression()
    model.fit(np.array(train_X), np.array(train_Y))
    score = model.score(np.array(test_X), np.array(test_Y))
    print (score)

    total = 0
    for i in range(len(test_X)):
        total += ((model.predict([test_X[i]]) - test_Y[i])**2)

    print (total/float(len(test_X)))

    total = 0
    for i in range(len(train_X)):
        total += ((model.predict([train_X[i]]) - train_Y[i])**2)

    print (total/float(len(train_X)))

    print (np.sum((train_Y-np.average(train_Y))**2))/len(train_Y)


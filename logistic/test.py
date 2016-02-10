import numpy as np
import numpy as numpy
import pandas as pd
from math import exp, expm1
from sklearn.metrics import roc_auc_score

train = pd.read_csv('data-logistic.csv', header=None)
X_train = train.ix[:, 1:2].values
y_train = train[0].values

y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])


# print roc_auc_score(y_true, y_scores)

def gradientStep(w, k, X, y):

    C=1
    l = len(X)
    sum = 0.0
    for i in l:
        sum += X[i]*y[i]*(1-1/(1+exp(-1*y[i]*(w1*x[i]+w2))))

    return w + (k*1/l)*sum - k*C*w

def evclidDistance(arr):
    norm = numpy.linalg.norm(arr)
    norm_sqr = norm ** 2
    return norm_sqr


print("evclidDistance: {0}".format(evclidDistance([1, 1])))

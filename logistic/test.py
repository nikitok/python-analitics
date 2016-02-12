# -*- coding: utf-8 -*-
import math
import numpy as np
import numpy as numpy
import pandas as pd
from math import exp, expm1
from sklearn.metrics import roc_auc_score

train = pd.read_csv('data-logistic.csv', header=None)
X = train.ix[:, 1:2].values
y = train[0].values

print y[0]
print X[0][0]
print X[0][1]
print train[:3]

def evclidDistance(arr):
    norm = numpy.linalg.norm(arr)
    norm_sqr = norm ** 2
    return norm_sqr

#РАСЧИТЫВАЕМ ВЕСА
C = 10
evcl = 10
cnt = 0
w1 = 0.0
w2 = 0.0
l = len(X)
k = 0.1
evcl_delta = 1.0 #evcl_delta > 0.00001 and
delta = 10
delta_old = 11

while delta > 10**-5 and cnt < 500:
    s1 = 0
    s2 = 0
    v1 = w1
    v2 = w2
    delta_old = delta
    delta = 0
    for i in range(l):
        s1 += y[i]*X[i][0]*(1-1/(1+exp(-y[i]*(w1*X[i][0]+w2*X[i][1]))))
        s2 += y[i]*X[i][1]*(1-1/(1+exp(-y[i]*(w1*X[i][0]+w2*X[i][1]))))
        delta += (math.log(1 + exp(-y[i] * (w1 * X[i][0] + w2 * X[i][1]))))
    delta = delta/l
    w1 += (k/l)*s1 - k*C*w1 #k*C*w1
    w2 += (k/l)*s2 - k*C*w2
    cnt = cnt + 1
    delta = math.sqrt((w1-v1)**2 + (w2-v2)**2)

    #print cnt, delta
print cnt, w1, w2, (delta-delta_old)

yt = np.zeros(l)
for i in range(l):
    yt[i] = 1/(1 + math.exp(-w1*X[i][0] - w2*X[i][1]))
# без регуляризации и при ее использовании
print "ver: {0}".format(roc_auc_score(y, yt))

# 244 итерации, с регуляризацией - 8
#for i in range(10000):
#   w1+=((k*(1.0/(len(y)))) y[i]*X[i][0]*( (1 - (1 / (1 + math.exp( -y[i] * (w1 * X[i][0] + w2 * X[i][1]) )))))- k*C*w1
#    i+=1




#print("evclidDistance: {0}".format(evclidDistance([1, 1])))

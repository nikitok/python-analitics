# -*- coding: utf-8 -*-

#выбрать оптимальное число соседей
import pandas as pd
import numpy as np
import sys

import sklearn

data = pd.read_csv('wine.data', header = None)
print data.head()
print '---\ntypes\n---\n'
print data.dtypes

#кросс-валидацию по блокам
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
X = data.ix[:,1:13].values
y = data[0].values
kf = KFold(178, n_folds=5, shuffle=True, random_state=42)
len(kf)
print(kf)

X = data.ix[:,1:13]
y = data[0]


#точность классификации на кросс-валидации для метода k ближайших соседей
X = sklearn.preprocessing.scale(X)
max_val = 0.0
max_step = 0
for k in range(1, 50):
    #print k
    fit = KNeighborsClassifier(n_neighbors=k)
    mean = cross_val_score(fit, X=X, y=y, cv=kf).mean()
    if(max_val < mean):
        max_val = mean
        max_step = k
    #print mean
print "max_val %.3f max_step %d" % (max_val,max_step)

#random
X = np.random.normal(loc=1, scale=10, size=(50, 50))
y = X[...,0]
y =y.astype(int)
print "test1"
print len(y)

kf = KFold(len(y), n_folds=5, shuffle=True, random_state=42)

max_val = 0.0
max_step = 0
for k in range(1, 10):
    #print k
    fit = KNeighborsClassifier(n_neighbors=k)
    mean = cross_val_score(fit, X=X, y=y, cv=kf).mean()
    if(max_val < mean):
        max_val = mean
        max_step = k
        #print mean
print "max_val %.3f max_step %d" % (max_val,max_step)
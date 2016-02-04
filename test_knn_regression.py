# -*- coding: utf-8 -*-
#выбирать оптимальную метрику из параметрического семейства
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
boston = load_boston()

X = boston.data
y = boston.target

X = sklearn.preprocessing.scale(X)
kf = KFold(len(y), n_folds=5, shuffle=True, random_state=42)

max_val = -1000.0
max_step = 0
for k in np.linspace(1,10, num=200):
    print k
    fit = KNeighborsRegressor(n_neighbors=5, weights='distance', p=k)
    mean = cross_val_score(fit, X=X, y=y, scoring='mean_squared_error', cv = kf).max()
    if(max_val < mean):
        max_val = mean
        max_step = k
    print mean.mean()
print "max_val %.3f max_step %d" % (max_val,max_step)
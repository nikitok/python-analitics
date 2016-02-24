# -*- coding: utf-8 -*-
import math
import numpy as np
import numpy as numpy
import pandas as pd
import sklearn
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('abalone.csv')

#Преобразуйте признак Sex в числовой: значение F должно перейти в -1, I — в 0, M — в 1.
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
y = data['Rings'].values
X = data.drop(['Rings'], axis=1).values

#Обучите случайный лес (sklearn.ensemble.RandomForestRegressor) с различным числом деревьев: от 1 до 50
#from sklearn.ensemble import RandomForestRegressor
#clf = RandomForestRegressor(n_estimators=100, random_state=1, shuffle=True)
#clf.fit(X, y)
#print clf.predict(X)


kf = KFold(len(y), n_folds=5, shuffle=True, random_state=1)
for k in range(1, 50):
    fit = RandomForestRegressor(n_estimators=k, random_state=1)
    mean = cross_val_score(fit, X=X, y=y, cv=kf, scoring = 'r2').mean()
    print "step %.3f val %.3f " % (k, mean)


for k in range(1, 50):
    kf = KFold(len(y), n_folds=5, shuffle=True, random_state=1)
    max = 0
    for train_index, test_index in kf:
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print train_index, test_index
        #print X_train, X_test, y_train, y_test
        clf = RandomForestRegressor(n_estimators=k, random_state=1)
        clf.fit(X_train, y_train)
        mean = sklearn.metrics.r2_score(y_test, clf.predict(X_test)).mean()
        if(mean > max):
            max = mean
    print "%.2f %.3f" % (k,max)

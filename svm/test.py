# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

train = pd.read_csv('vm-data.csv', header = None)
X_train = train.ix[:,1:2].values
y_train = train[0].values


from sklearn.svm import SVC
clf = SVC(C = 100000, random_state=241, kernel='linear')
clf.fit(X_train, y_train)
#опорные объекты
print clf.support_

print(clf.predict([[-0.8, -1]]))
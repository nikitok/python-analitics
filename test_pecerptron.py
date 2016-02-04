# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import Perceptron


X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = Perceptron()
clf.fit(X, y)
#построение прогнозов
predictions = clf.predict(X)
#print predictions #[0 0 0]


from sklearn.metrics import accuracy_score
y_true = [0, 0, 0]
#print accuracy_score(y_true, predictions, normalize=False)


#Для стандартизации признаков удобно воспользоваться классом
from sklearn.preprocessing import StandardScaler
X_train = np.array([[100.0, 2.0], [50.0, 4.0], [70.0, 6.0]])
X_test = np.array([[90.0, 1], [40.0, 3], [60.0, 4]])
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#print X_train_scaled
#print X_test_scaled


#-------------
import pandas as pd
test = pd.read_csv('perceptron-test.csv', header = None)
train = pd.read_csv('perceptron-train.csv', header = None)

X_train = train.ix[:,1:2].values
y_train = train[0].values

X_test = test.ix[:,1:2].values
y_test = test[0].values



#Обучите персептрон со стандартными параметрами и random_state=241.
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)
print clf

#Подсчитайте качество (долю правильно классифицированных объектов, accuracy) полученного классификатора на тестовой выборке.
predictions = clf.predict(X_test)
#print predictions
#качество
print clf.score(X_test, y_test) #0.665
print accuracy_score(y_test, predictions) #0.665


#Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Обучите персептрон на новых выборках. Найдите долю правильных ответов на тестовой выборке.
clf = Perceptron(random_state=241)
clf.fit(X_train_scaled, y_train)
predictions = clf.predict(X_test_scaled)
#print predictions
#качество
print accuracy_score(y_test, predictions) #0.665
print clf.score(X_test_scaled, y_test) #0.665
print 0.36 - 0.925
#Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее. Это число и будет ответом на задание.
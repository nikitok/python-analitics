# -*- coding: utf-8 -*-
import numpy as np

#loc: среднее нормального распределения (в нашем случае 1)
#scale: стандартное отклонение нормального распределения (в нашем случае 10)
#size: размер матрицы (в нашем случае (1000, 50))


X = np.random.normal(loc=1, scale=10, size=(50, 50))
y = X[...,0]
print y

for x in range(10):
    print x

#r = np.sum(X, axis = 1)
#print np.nonzero(r > 10)

#print '-------'
#print np.mean(X, axis=0)
#print '-------'

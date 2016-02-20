# -*- coding: utf-8 -*-
import math
import numpy as np
import numpy as numpy
import pandas as pd

data = pd.read_csv('close_prices.csv')
X = data.ix[:,1:30].values

from sklearn.decomposition import  PCA

pca = PCA(n_components=10)
pca.fit(X)

print pca.explained_variance_ratio_
print pca.components_

print "---"

pca = PCA(n_components=0.9)
pca.fit(X)
print pca.n_components_
print pca.explained_variance_ratio_

print '---'
prices_red = pca.transform(X)
print prices_red[:, 0]
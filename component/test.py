# -*- coding: utf-8 -*-
import math
import numpy as np
import numpy as numpy
import pandas as pd
import sklearn

data = pd.read_csv('close_prices.csv')
djia = pd.read_csv('djia_index.csv')
X = data.ix[:,1:31].values
pricesOnly = data.drop("date", axis=1) # (374, 30)

from sklearn.decomposition import  PCA

pca = sklearn.decomposition.PCA(n_components=10)
pca.fit(X)
price = pca.transform(X)
#print price[::,0]
print np.corrcoef(price[::,0], djia['^DJI'].values)

components = pca.components_
components2 = pca.explained_variance_ratio_
t = list(zip(pca.components_[0], pricesOnly.columns))
print(max(t))
#print len(components2)
#print len(djia['^DJI'].values)




#print djia['^DJI'].values
#print np.corrcoef(components[:,0], djia['^DJI'].values)

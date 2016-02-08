# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
#from sklearn.datasets import fetch_20newsgroups
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

newsgroups = datasets.fetch_20newsgroups(
        subset='all',
        categories=['alt.atheism', 'sci.space']
)

X = newsgroups.data
y = newsgroups.target

#print X

#Одна из сложностей работы с текстовыми данными состоит в том, что для них нужно построить числовое представление.
#Одним из способов нахождения такого представления является вычисление TF-IDF. В Scikit-Learn это реализовано
#в классе sklearn.feature_extraction.text.TfidfVectorizer.
#Преобразование обучающей выборки нужно делать с помощью функции fit_transform, тестовой — с помощью transform.


from  sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer = 'word')
X_ = vectorizer.fit_transform(X)
X = vectorizer.transform(X)

#print X_
#print vectorizer.get_feature_names()

#print X
#print vectorizer.get_feature_names()

clf = SVC(kernel='linear', random_state=241, C=4)
clf.fit(X, y)
resNumber=np.argsort(np.abs(clf.coef_.data))[-10:]
print clf.coef_
print clf.coef_.data
print resNumber


for i in range(10):
    print vectorizer.get_feature_names()[resNumber[i]]

print 'end'
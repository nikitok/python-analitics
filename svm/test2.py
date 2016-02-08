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
#Одна из сложностей работы с текстовыми данными состоит в том, что для них нужно построить числовое представление.
#Одним из способов нахождения такого представления является вычисление TF-IDF. В Scikit-Learn это реализовано
#в классе sklearn.feature_extraction.text.TfidfVectorizer.
#Преобразование обучающей выборки нужно делать с помощью функции fit_transform, тестовой — с помощью transform.


from  sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer = 'word')
X = vectorizer.fit_transform(X)

#print X
#print vectorizer.get_feature_names()[7135]

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)
for a in gs.grid_scores_:
    mean = a.mean_validation_score #— оценка качества по кросс-валидации
    parameters = a.parameters #— значения параметров
    best_params_ = a.best_params_ #— значения параметров
    print mean
    print parameters
    print best_params_

clf = SVC(kernel='linear', random_state=241, C=10.0)
clf.fit(X, y)
resNumber=np.argsort(np.abs(clf.coef_.data))[-10:]
#print resNumber


for i in range(10):
    print vectorizer.get_feature_names()[resNumber[i]]

print 'end'
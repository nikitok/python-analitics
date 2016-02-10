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
X_1 = vectorizer.fit_transform(X)
X_2 = vectorizer.transform(X)

#print X_
#print vectorizer.get_feature_names()

#print X
#print vectorizer.get_feature_names()

clf = SVC(kernel='linear', random_state=241, C=1)
clf.fit(X_1, y)


#resNumber=np.argsort(np.abs(clf.coef_.data))[-10:]
#dens_f = clf.coef_.todense()
#sort_f = np.absolute(np.asarray(dens_f)).reshape(-1)
#bst_f = np.argsort(sort_f)[-10:]
#print clf.coef_.todense()
#bst_f = abs(clf.coef_.todense()).argpartition(-10)[-10:]
#print bst_f

bst_f = np.argsort(np.absolute(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]
print bst_f
#a,a,b,g,k,m,n,r,s,s
#print pd.Series(clf.coef_.toarray().reshape(-1)).abs().nlargest(10).index

result = []
for i in range(10):
    res = vectorizer.get_feature_names()[bst_f[i]]
    result.append(res)

#atheism atheists bible god keith moon nick religion sky space
print np.sort(result)
print vectorizer.get_feature_names()[bst_f]

print 'end'
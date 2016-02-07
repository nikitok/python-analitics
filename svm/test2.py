# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(
        subset='all',
        categories=['alt.atheism', 'sci.space']
)

X = newsgroups.data
y = newsgroups.target

#Одна из сложностей работы с текстовыми данными состоит в том, что для них нужно построить числовое представление.
# Одним из способов нахождения такого представления является вычисление TF-IDF. В Scikit-Learn это реализовано
#в классе sklearn.feature_extraction.text.TfidfVectorizer.
#Преобразование обучающей выборки нужно делать с помощью функции fit_transform, тестовой — с помощью transform.


from  sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer = 'word')
X = vectorizer.fit_transform(X)
X
print X
print vectorizer.get_feature_names()[7135]
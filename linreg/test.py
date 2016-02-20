# -*- coding: utf-8 -*-
import math
import numpy as np
import numpy as numpy
import pandas as pd

data_train = pd.read_csv('salary-train.csv')
data_test = pd.read_csv('salary-test-mini.csv')
from sklearn.feature_extraction.text import TfidfVectorizer



print '---\ntypes\n---\n'
print data_train.dtypes
print '----'


data_train['FullDescription'] = data_train['FullDescription'].apply(lambda x: x.lower())
data_train['FullDescription'] = data_train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

#Примените TfidfVectorizer для преобразования текстов в векторы признаков.
# Оставьте только те слова, которые встречаются хотя бы в 5 объектах

vectorizer = TfidfVectorizer(min_df = 5)
X_train_words = vectorizer.fit_transform(data_train['FullDescription'].values)
X_test_words = vectorizer.transform(data_test['FullDescription'].values)

#Замените пропуски в столбцах LocationNormalized и ContractTime на специальную строку 'nan'. Код для этого был приведен выше.
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)

#Примените DictVectorizer для получения one-hot-кодирования признаков LocationNormalized и ContractTime.
from sklearn.feature_extraction import DictVectorizer
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))


from scipy.sparse import hstack
Y = hstack([X_train_words,X_train_categ])
Y_test = hstack([X_test_words,X_test_categ])

#3. Обучите гребневую регрессию с параметром alpha=1. Целевая переменная записана в столбце SalaryNormalized.
from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0, fit_intercept=False, solver='lsqr')
clf.fit(Y, data_train['SalaryNormalized'])


print clf.predict(Y_test)
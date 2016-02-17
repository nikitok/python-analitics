# -*- coding: utf-8 -*-
import math
import numpy as np
import numpy as numpy
import pandas as pd

#data = pd.read_csv('salary-train.csv')
data = pd.read_csv('salary-test-mini.csv')

print '---\ntypes\n---\n'
print data.dtypes
print '----'


data['FullDescription'] = data['FullDescription'].apply(lambda x: x.lower())

from sklearn.feature_extraction import DictVectorizer
#data['LocationNormalized'].fillna('nan', inplace=True)
#data['ContractTime'].fillna('nan', inplace=True)
enc = DictVectorizer()
X_train_categ = enc.transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))

from  sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=5)
X_train_words = vectorizer.fit_transform(data['FullDescription'])
X_test_words = vectorizer.transform(data['FullDescription'])

from scipy.sparse import hstack
new_test_data = hstack([data['FullDescription'], X_test_categ])

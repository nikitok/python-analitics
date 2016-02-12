# -*- coding: utf-8 -*-
import math
import numpy as np
import numpy as numpy
import pandas as pd
from math import exp, expm1
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


data = pd.read_csv('classification.csv')

#Actual Positive	Actual Negative
#Predicted Positive	 TP	 FP
#Predicted Negative	 FN	 TN

#TP=43, FP=34, FN=59, TN
conf_matrix = confusion_matrix(data['true'], data['pred'], labels=[1, 0])

print conf_matrix.T

#TP, FP, FN, TN.


print roc_auc_score(data['true'], data['pred']), precision_score(data['true'], data['pred']),recall_score(data['true'], data['pred']),f1_score(data['true'], data['pred'])
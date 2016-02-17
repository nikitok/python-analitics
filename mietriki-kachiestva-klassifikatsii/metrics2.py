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
from sklearn.metrics import precision_recall_curve


#для логистической регрессии — вероятность положительного класса (колонка score_logreg),
#для SVM — отступ от разделяющей поверхности (колонка score_svm),
#для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
#для решающего дерева — доля положительных объектов в листе (колонка score_tree).
data = pd.read_csv('scores.csv')


score_logreg = roc_auc_score(data['true'],data['score_logreg'])
score_svm = roc_auc_score(data['true'],data['score_svm'])
score_knn = roc_auc_score(data['true'],data['score_knn'])
score_tree = roc_auc_score(data['true'],data['score_tree'])

print "%.2f %.2f %.2f %.2f" % (score_logreg,score_svm,score_knn,score_tree)

#Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?
#Какое значение точности при этом получается?
#Чтобы получить ответ на этот вопрос, найдите все точки precision-recall-кривой с помощью функции
#sklearn.metrics.precision_recall_curve. Она возвращает три массива: precision, recall, thresholds.
#В них записаны точность и полнота при определенных порогах,
# указанных в массиве thresholds. Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7

#percision = точность, насколько можно доверять коасификатору
#recall = полнота, как много объектов находит класификатор

#PR - левая точка

recall_curve_logreg = precision_recall_curve(data['true'],data['score_logreg'])
recall_curve_svm = precision_recall_curve(data['true'],data['score_svm'])
recall_curve_knn = precision_recall_curve(data['true'],data['score_knn'])
recall_curve_tree = precision_recall_curve(data['true'],data['score_tree'])


def maxPercission(recall_curve):
    l = len(recall_curve[0])
    max = 0.0
    for i in range(l):
        if recall_curve[1][i] >= 0.7:
            if max < recall_curve[0][i]:
                max = recall_curve[0][i]
                print max
    print "------"
    return max

print "%.2f %.2f %.2f %.2f" % (maxPercission(recall_curve_logreg),
                               maxPercission(recall_curve_svm),
                               maxPercission(recall_curve_knn),
                               maxPercission(recall_curve_tree)
                               )
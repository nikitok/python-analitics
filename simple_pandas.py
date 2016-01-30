import pandas as pd


import matplotlib.pyplot as plt
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number

data = pd.read_csv('titanic.csv', index_col='PassengerId')

print data[:3]
print '------'
print 'dtypes'
print data.dtypes
print '------'
print 'value_counts'
print data['Survived'].value_counts()
print '------'
data['Survived'].plot()
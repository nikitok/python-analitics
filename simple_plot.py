import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as np
import sys
import matplotlib

data = pd.read_csv('tree/titanic.csv', index_col='PassengerId')
print data.dtypes
print type(data)


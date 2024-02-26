import pandas as pd
import numpy as np
import math
import csv
import re 
import matplotlib.pyplot as plt

from NettoyageDonnees import Nettoyage 

import seaborn as sns

n = Nettoyage()

data=n.RemoveNA(n.train)
data.drop('Id', axis=1, inplace=True)

numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
numerical_corr_matrix = data.corr()
corrmat = data[numeric_columns].corr()
k = 36
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
print("les colonnes à garder dans nos données:", cols)
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#https://www.kaggle.com/code/sherifkhaledosman/house-prices-ml
import pandas as pd
import numpy as np
import math
import csv
import re 
import matplotlib.pyplot as plt

from NettoyageDonnees import Nettoyage 


n = Nettoyage()

data=n.RemoveNA(n.train)
corrmat = data.corr()
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


plt.show()

#https://www.kaggle.com/code/sherifkhaledosman/house-prices-ml
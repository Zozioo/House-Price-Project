import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#histogramme avant 
data = pd.read_csv("data/train.csv")

sns.displot(data['SalePrice'])
plt.show()


data_log = pd.read_csv("data/CleanTrain.csv")
sns.displot(data_log['SalePrice'])
plt.show()
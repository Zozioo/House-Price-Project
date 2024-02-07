import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz


myPath= '/home/40008593/Documents/M1/S2/Projet-HP/CleanTrain.csv'
house = pd.read_csv(myPath, encoding='utf-8')


y = house['SalePrice']
X = house.drop('SalePrice', axis=1)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred = rf.predict(X_validation)
accuracy = accuracy_score(y_validation, y_pred)
print("Accuracy:", accuracy)


# Export the first three decision trees from the forest

for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,  
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
    graph = graphviz.Source(dot_data)
    graph.render(f"tree_{i}", format="png", cleanup=True)
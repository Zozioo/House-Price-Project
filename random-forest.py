import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_val_score, GridSearchCV

from scipy.stats import randint

from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz



myPath = "data/CleanTrain.csv"
house = pd.read_csv(myPath, encoding='utf-8')

#découpage des données
y = house['SalePrice']
X = house.drop('SalePrice', axis=1)

np.random.seed(40)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



# Validation croisée:
param_dist = {'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]}

rf = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf, param_grid=param_dist, cv=5)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print('Best hyperparameters:',  grid_search.best_params_)

y_pred = best_rf.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)



# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar()

# Export the first three decision trees from the forest
'''
for i in range(3):
    tree = best_rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,  
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
    graph = graphviz.Source(dot_data)
    graph.render(f"tree_{i}", format="png", cleanup=True)'''
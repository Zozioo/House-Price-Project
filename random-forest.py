import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split


from sklearn.tree import export_graphviz
import graphviz



myPath = "data/CleanTrain.csv"
house = pd.read_csv(myPath, encoding='utf-8')

#découpage des données
y = house['SalePrice']
X = house.drop('SalePrice', axis=1)

np.random.seed(1118)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Validation croisée:
param_dist = {"max_depth":[None,range(15,30)], "n_estimators":[27,30,33,40,50,60,70,80,90,100,110,120,130]}

np.random.seed(1118)
rf = RandomForestRegressor()
grid_search = RandomizedSearchCV(rf, param_dist, cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print('Best hyperparameters:',  grid_search.best_params_)

y_pred = best_rf.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mea = mean_absolute_error(y_test,y_pred)
y_bar = np.mean(y_test)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error", rmse, "Mean SalePrice", y_bar)
print("Mean Average Error", mea)
print("Mediane SalesPrice", np.median(y_test))



#Prédiction des données Test avec la méthode des rabdoms forest:

Xtest = pd.read_csv("data/CleanTest")
yTest_pred = best_rf.predict(Xtest)
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt




Dtrain = pd.read_csv('data/CleanTrain.csv')
Dtest = pd.read_csv('data/CleanTest.csv')


np.random.seed(300)


pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('knn', KNeighborsRegressor())  
])


param_grid = {
    'knn__n_neighbors': range(1, 201) 
}

y = Dtrain['SalePrice']
X = Dtrain.drop('SalePrice', axis=1)

np.random.seed(1118)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

 
grid_search = GridSearchCV(pipeline, param_grid, cv=10)
grid_search.fit(X_train,y_train)



np.random.seed(300)

results = grid_search.cv_results_
param_values = [params['knn__n_neighbors'] for params in results['params']]
mean_scores = results['mean_test_score']

best_knn= grid_search.best_estimator_
y_pred = best_knn.predict(X_test)

plt.plot(param_values, mean_scores, marker='o')
plt.xlabel('Nombre de voisins (k)')
plt.ylabel('Score moyen de validation croisée')
plt.title('Validation croisée pour k-NN')
plt.grid(True)
plt.show()



print("\nMeilleurs paramètres :")
print(grid_search.best_params_)

mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error:", mse)

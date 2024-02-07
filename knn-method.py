import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt




Dtrain = pd.read_csv('/Users/madina/Desktop/Projet py/CleanTrain.csv', sep =';')
Dtest = pd.read_csv('/Users/madina/Desktop/Projet py/CleanTest.csv', sep = ';')


np.random.seed(300)


pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('knn', KNeighborsRegressor())  
])


param_grid = {
    'knn__n_neighbors': range(1, 201) 
}


cv = 10  
grid_search = GridSearchCV(pipeline, param_grid, cv=cv)
grid_search.fit(Dtrain.drop(columns=['SalePrice']), Dtrain['SalePrice'])

np.random.seed(300)


results = grid_search.cv_results_


param_values = [params['knn__n_neighbors'] for params in results['params']]


mean_scores = results['mean_test_score']


plt.plot(param_values, mean_scores, marker='o')
plt.xlabel('Nombre de voisins (k)')
plt.ylabel('Score moyen de validation croisée')
plt.title('Validation croisée pour k-NN')
plt.grid(True)
plt.show()


print("Résultats de la validation croisée :")
print(grid_search.cv_results_)


print("\nMeilleurs paramètres :")
print(grid_search.best_params_)
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


np.random.seed(100)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor())  
])

df = pd.read_csv('data/CleanTrain.csv')
#recherche de k optimal
param_grid = {
    'knn__n_neighbors': range(1, 201)  
}


grid_search = GridSearchCV(pipeline, param_grid, cv=10) 
grid_search.fit(df.drop(columns=['SalePrice']), df['SalePrice'])

results = grid_search.cv_results_

param_values = [params['knn__n_neighbors'] for params in results['params']]

mean_scores = results['mean_test_score']

plt.plot(param_values, mean_scores, marker='o')
plt.xlabel('Nombre de voisins (k)')
plt.ylabel('Score moyen de validation croisée')
plt.title('Validation croisée pour k-NN')
plt.grid(True)
plt.show()


best_k = grid_search.best_params_['knn__n_neighbors']
print("\nMeilleurs paramètres :")
print(best_k) #k=19

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
X_train,X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

knn_regressor = KNeighborsRegressor(n_neighbors=best_k)

knn_regressor.fit(X_train, Y_train)

predictions = knn_regressor.predict(X_test)


mae_knn = metrics.mean_absolute_error(Y_test, predictions).round(2)
mse_knn = metrics.mean_squared_error(Y_test, predictions).round(2)
rmse_knn = np.sqrt(metrics.mean_squared_error(Y_test, predictions)).round(2)
mean_sale_price = np.mean(Y_test).round(2)

print('MAE_knn: ', mae_knn)
print('MSE_knn: ', mse_knn)
print('RMSE_knn:', rmse_knn)  
print('Moyenne des valeurs de la variable cible:', mean_sale_price)

if rmse_knn < mean_sale_price:
    print("Le RMSE_knn est plus petit que la moyenne des valeurs de la variable cible.")
    print("Cela suggère que le modèle a une performance relative acceptable.")
else:
    print("Le RMSE_knn est plus grand ou égal à la moyenne des valeurs de la variable cible.")
    print("Cela suggère que le modèle pourrait avoir des performances insuffisantes.")
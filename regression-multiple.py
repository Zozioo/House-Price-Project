import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


df = pd.read_csv('data/CleanTrain.csv')
df = df.drop('Id', axis=1)
variables = df[['GrLivArea', 'OverallQual', 'GarageCars']]


X = df[['GrLivArea', 'OverallQual', 'GarageCars']]
Y = df['SalePrice']

#Division des donnees
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

reg_model = linear_model.LinearRegression()

reg_model = LinearRegression().fit(X_train, Y_train)


print('Intercept: ', round(reg_model.intercept_, 2))
coefficients_arrondis = [(variable, round(coef, 2)) for variable, coef in zip(X.columns, reg_model.coef_)]
print(coefficients_arrondis)


y_pred= reg_model.predict(X_test)  
x_pred= reg_model.predict(X_train) 


#Valeur actuelle and la valeur prédite
reg_model_diff = pd.DataFrame({'Valuer actuelle': Y_test, 'Valeur predite': y_pred})
reg_model_diff

mae_regression = metrics.mean_absolute_error(Y_test, y_pred).round(2)
mse_regression = metrics.mean_squared_error(Y_test, y_pred).round(2)
rsme_regression = np.sqrt(metrics.mean_squared_error(Y_test, y_pred)).round(2)

print('MAE_regression:', mae_regression)
print('MSE_regression:', mse_regression)
print('RMSE_regression:', rsme_regression)

# 'y_test' contient les vraies valeurs de la variable cible et y_pred contient les valeurs prédites par votre modèle.

# Calculer la moyenne des valeurs de la variable cible
mean_sale_price = np.mean(Y_test).round(2)


# Comparer le RMSE avec la moyenne des valeurs de la variable cible
print("Moyenne des valeurs de la variable cible:", mean_sale_price)


if rsme_regression < mean_sale_price:
    print("Le RSME_regression est plus petit que la moyenne des valeurs de la variable cible.")
    print("Cela suggère que le modèle a une performance relative acceptable.")
else:
    print("Le RSME_regression est plus grand ou égal à la moyenne des valeurs de la variable cible.")
    print("Cela suggère que le modèle pourrait avoir des performances insuffisantes.")
    
    
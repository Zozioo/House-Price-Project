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
print(len(y_pred), len(X_test), len(Y_test)) # me retourne 438 438 438

#Valeur actuelle and la valeur prédite
reg_model_diff = pd.DataFrame({'Valeur actuelle': Y_test, 'Valeur predite': y_pred})
reg_model_diff

mae_regression = metrics.mean_absolute_error(Y_test, y_pred).round(2)
mse_regression = metrics.mean_squared_error(Y_test, y_pred).round(2)
rsme_regression = np.sqrt(metrics.mean_squared_error(Y_test, y_pred)).round(2)

print('MAE_regression:', mae_regression)
print('MSE_regression:', mse_regression)
print('RMSE_regression:', rsme_regression)



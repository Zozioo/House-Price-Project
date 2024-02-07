import pandas as pd
import numpy as np
import math
import csv
import re 



train = '/home/40008593/Documents/M1/S2/Projet-HP/train.csv'
new_train = '/home/40008593/Documents/M1/S2/Projet-HP/CleanTrain.csv'
test = '/home/40008593/Documents/M1/S2/Projet-HP/test.csv'
new_test= '/home/40008593/Documents/M1/S2/Projet-HP/CleanTest.csv'

variables_a_supprimer = ['Id','MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                         'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                         'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                         'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
                         'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
                         'MiscFeature', 'SaleType', 'SaleCondition']


def RemoveNA(initialFilePath):
    string_columns=[] 
    donnees = pd.read_csv(initialFilePath, encoding='utf-8')
    for d in donnees:
        if donnees[d].dtype in ("int64","float64"):
            donnees[d].replace(np.nan, 0, inplace=True)
        else:
            donnees[d].replace(np.nan, "NA", inplace=True)
            string_columns.append(d)

    donnees = pd.get_dummies(donnees,columns=string_columns)
    return donnees

 

def RemoveColumns(data,columnsToRemove, finalPath):
    exactColumns = [col for col in data.columns if col in columnsToRemove]
    data.drop(columns=exactColumns, inplace=True) 
    # Maintenant, utiliser une expression régulière pour supprimer les colonnes commençant par l'un des fragments
    pattern = re.compile(r'\b(?:' + '|'.join(columnsToRemove) + r')_\w+\b')
    #Cela signifie que le motif doit commencer par l'un des mots spécifiés dans variables_a_supprimer, 
    #suivis d'un tiret bas, puis suivi par un ou plusieurs caractères alphanumériques.
    realColumnsList = [col for col in data.columns if pattern.match(col)]
    data.drop(columns=realColumnsList, inplace=True)
    data.to_csv(finalPath, index=False, encoding='utf-8')
    


RemoveColumns(RemoveNA(train),variables_a_supprimer,new_train)
RemoveColumns(RemoveNA(test),variables_a_supprimer,new_test)


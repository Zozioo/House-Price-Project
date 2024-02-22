import pandas as pd
import numpy as np

import csv
import re 

class Nettoyage:
    
    def __init__(self):

        #self.train = '/home/40008593/Documents/M1/S2/Projet-HP/train.csv'
        self.train = "C:/Users/zoero/OneDrive/Documents/perso/Python/données/train.csv"
        #self.new_train = '/home/40008593/Documents/M1/S2/Projet-HP/CleanTrain.csv'
        self.new_train ="C:/Users/zoero/OneDrive/Documents/perso/Python/données/CleanTrain.csv"
        #self.test = '/home/40008593/Documents/M1/S2/Projet-HP/test.csv'
        self.test ="C:/Users/zoero/OneDrive/Documents/perso/Python/données/test.csv"
        #self.new_test= '/home/40008593/Documents/M1/S2/Projet-HP/CleanTest.csv'
        self.new_test ="C:/Users/zoero/OneDrive/Documents/perso/Python/données/CleanTest.csv"

        self.variables_a_supprimer = ['Id','MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                            'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                            'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                            'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
                            'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
                            'MiscFeature', 'SaleType', 'SaleCondition']


    def RemoveNA(self, initialPath): #pour les string uniquement
        string_columns=[] 
        donnees = pd.read_csv(initialPath, encoding='utf-8')
        for d in donnees:
            if donnees[d].dtype in ("int64","float64"):
                donnees[d].replace(np.nan, 0, inplace=True)
            else:
                donnees[d].replace(np.nan, "NA", inplace=True)
                string_columns.append(d)

        donnees = pd.get_dummies(donnees,columns=string_columns)
        return donnees

    

    def RemoveColumns(seld,data,columnsToRemove, finalPath):
        exactColumns = [col for col in data.columns if col in columnsToRemove]
        data.drop(columns=exactColumns, inplace=True) 
        # Maintenant, utiliser une expression régulière pour supprimer les colonnes commençant par l'un des fragments
        pattern = re.compile(r'\b(?:' + '|'.join(columnsToRemove) + r')_\w+\b')
        #Cela signifie que le motif doit commencer par l'un des mots spécifiés dans variables_a_supprimer, 
        #suivis d'un tiret bas, puis suivi par un ou plusieurs caractères alphanumériques.
        realColumnsList = [col for col in data.columns if pattern.match(col)]
        data.drop(columns=realColumnsList, inplace=True)
        data.to_csv(finalPath, index=False, encoding='utf-8')
        
def executer():
    n = Nettoyage()
    n.RemoveColumns(n.RemoveNA(n.train),n.variables_a_supprimer, n.new_train)
    n.RemoveColumns(n.RemoveNA(n.test),n.variables_a_supprimer,n.new_test)

executer()
import pandas as pd
import numpy as np
import matplotlib as math
import statistics as stat
import csv
import re 

class Nettoyage:
    
    def __init__(self, trainPath, testPath, variables_a_supprimer):

        self.train = trainPath
        self.new_train ="data/CleanTrain.csv"
        self.test = testPath
        self.new_test ="data/CleanTest.csv"

        self.variables_a_supprimer = variables_a_supprimer


    def RemoveNA(self, initialPath): #pour les non strings uniquement
        string_columns=[] 
        donnees = pd.read_csv(initialPath, encoding='utf-8')
        for d in donnees:
            if donnees[d].dtype in ("int64","float64"):
                donnees[d].replace(np.nan, 0, inplace=True)
            else:
                donnees[d].replace(np.nan, "NA", inplace=True)
                string_columns.append(d)

        donnees = pd.get_dummies(donnees,columns=string_columns,dtype=int)
        
        #conversion logarithme
        if 'SalePrice' in donnees.columns:
            donnees['SalePrice'] = np.log10(donnees['SalePrice'])
            
        return donnees

    

    def RemoveColumns(self,data, finalPath):
        exactColumns = [col for col in data.columns if col in self.variables_a_supprimer]
        data.drop(columns=exactColumns, inplace=True) 

        pattern = re.compile(r'\b(?:' + '|'.join(self.variables_a_supprimer) + r')_\w+\b')
        #Cela signifie que le motif doit commencer par l'un des mots spécifiés dans variables_a_supprimer, 
        #suivis d'un tiret bas, puis suivi par un ou plusieurs caractères alphanumériques.
        realColumnsList = [col for col in data.columns if pattern.match(col)]
        data.drop(columns=realColumnsList, inplace=True)
        data.to_csv(finalPath, index=False, encoding='utf-8')


variables_a_supprimer=['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 
                                      'LotConfig', 'LandSlope',
                            'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                            'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                            'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
                            'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence',
                            'MiscFeature', 'SaleType', 'SaleCondition']
        
def executer():
    n = Nettoyage("data/train.csv","data/test.csv",variables_a_supprimer)
    n.RemoveColumns(n.RemoveNA(n.train), n.new_train)
    n.RemoveColumns(n.RemoveNA(n.test),n.new_test)

   


executer()
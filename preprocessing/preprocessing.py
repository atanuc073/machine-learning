#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:37:31 2020

@author: atanuc73
"""
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing datasets
dataset=pd.read_csv('Data.csv')
X=dataset.iloc[:, :-1].values  # it includes all the columns except the last one
y=dataset.iloc[:,3].values      # y is the dependent variable

# handling the missing data 
#a missing data can b ereplaced by the mean of other data sets.

from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, 1:3])
X[:, 1:3]=missingvalues.transform(X[:, 1:3])

#categorical data we need to encode the categorical data into integer for ease of calculation.

# Encoding categorical data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,0]=labelencoder.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)


#splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)





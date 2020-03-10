#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:49:27 2020

@author: atanuc73
"""

# multiple Linear Regression


# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing datasets
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:, :-1].values  # it includes all the columns except the last one
y=dataset.iloc[:,4].values      # y is the dependent variable


#Encoding the independent variable state

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy variable Trap

X=X[:,1:]

#splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# fitting multiple linear regression to the training set

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# predicting result for X_test values
y_pred=regressor.predict(X_test)


#Building the optimal model using backward elimination
# first we need to add a matrix of 1 because of the constant b0
import statsmodels.api as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5]]      # initially we take the whole array with all columns
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,1,3,4,5]]      # initially we take the whole array with all columns
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,3,4,5]]      # initially we take the whole array with all columns
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
X_opt=X[:,[0,3,5]]      # initially we take the whole array with all columns
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()



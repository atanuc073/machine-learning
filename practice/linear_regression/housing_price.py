#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 18:36:13 2020

@author: atanuc73
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import datasets
dataset=pd.read_csv('USA_Housing.csv')
X=dataset.iloc[:,0:5].values
y=dataset.iloc[:,5].values

#spliting into training and test datasets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting the training dataset with multiple_linear_regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting new results
y_pred=regressor.predict(X_test)

#visualizing the result
new=X_train[:,0]

plt.scatter(new,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.show()



























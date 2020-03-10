#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:47:38 2020

@author: atanuc73
"""
#simple linear regression
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing datasets
dataset=pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:, :-1].values  # it includes all the columns except the last one
y=dataset.iloc[:,1].values      # y is the dependent variable



#splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
'''
#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
'''


# fitting simple linear regression to the training dataset

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set results
y_pred=regressor.predict(X_test)￼


#visualizing the training set results

plt.scatter(X_train,y_train, color='red')
plt.plot(X_train,regressor.predict(X_train))
plt.title('Salary vs Experience (Training data)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#visualizing the test set results

plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train))
plt.title("Salary vs Experience (Test data)")
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


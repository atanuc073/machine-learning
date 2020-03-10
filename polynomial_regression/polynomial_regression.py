#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:53:11 2020

@author: atanuc73
"""

#polynomial_regression
# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing datasets
position_salary=pd.read_csv('Position_Salaries.csv')
X=position_salary.iloc[:, 1:2].values  # it includes all the columns except the last one
y=position_salary.iloc[:,2].values      # y is the dependent variable
'''
#splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

'''

#Fitting Linear Regression to the dataset

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)
# Fitting Polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=6)
X_poly=poly_reg.fit_transform(X)     # poly_reg will transform the  X matrix to the X_poly matrix
# polynomial regrewssion model will automatically include the 1's matrix to the X_poly matrix
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)


#visualizing the Linear Regression Model
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Position vs Salary (Linear Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#visualizing the Polynomial Regression Model

plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)))
plt.title('Position vs Salary (Polynomial Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


#Predicting a new result with Linear Regression Model
lin_reg.predict([[6.5]])

#Predicting a new result with Polynomial Regression model
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))











#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:36:02 2020

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

# Fitting the Regression models to the dataset


#Predicting a new result with Polynomial Regression model
y_pred=regressor.predict([[6.5]])



#visualizing the  Regression Model

plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X))
plt.title('Position vs Salary (Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#visualizing the  Regression Model

X_grid=np.arrange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid))
plt.title('Position vs Salary (Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()




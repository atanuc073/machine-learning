#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:15:10 2020

@author: atanuc73
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#importing the datasets
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

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


#fitting the decision tree regression to the dataset

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

# Predict a new result

y_pred=regressor.predict(np.array([[6.5]]))


#Visualizing the decision_tree_regression result

plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Level vs Salary (Decision Tree regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the decision tree regression result (for higher resolution)
#decision tree regression model is not continuous
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
#X_grid=np.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')

plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Level vs Salary (Decisiuon tree regression)')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()





















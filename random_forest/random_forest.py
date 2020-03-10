#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:43:15 2020

@author: atanuc73
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the datasets
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


#Fitting the random forest regression model to the dataset

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(X,y)


#predicting a new result

y_pred=regressor.predict([[6.5]])

# visualizing the randm forest result

X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Level vs Salary (Random Forest regression)')
plt.xlabel('level')
plt.ylabel('Salary')
plt.show()

























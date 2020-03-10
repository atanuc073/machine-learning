#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:57:35 2020

@author: atanuc73
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset=pd.read_csv('Position_Salaries.csv')
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
y=y.reshape(-1,1)




'''
#splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
'''
#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)



#Fitting svr to the dataset
# SVR class do not have predefined feature scalling feature we have to do it by our self

from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)


#predicting a new result
y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#visualizing the SVR result

plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('LEVEL VS SALARY (SVR MODEL)')
plt.xlabel('LEVEL')
plt.ylabel('SALARY')
plt.show()






















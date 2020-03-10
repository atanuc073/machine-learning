#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:24:25 2020

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

#splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

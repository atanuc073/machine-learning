#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:53:32 2020

@author: atanuc73
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import the datasets
dataset= pd.read_csv('Market_Basket_Optimisation.csv',header=None)

# Apriori expects a list of list as a input
transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
#Training Apriori on the dataset
from apyori import apriori
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)

#Visualizing the results

results=list(rules)    
    

    
    
    

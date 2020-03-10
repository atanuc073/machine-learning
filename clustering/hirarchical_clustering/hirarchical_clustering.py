#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:01:20 2020

@author: atanuc73
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,3:5].values

#Use the dendogram to find the OPtimal number of Clusters

import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#Fitting the hierarchycal clustering to the dataset

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

#visualizing the Clusters

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='cluster 1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='cluster 2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='cluster 3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='cluster 4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='cluster 5')
plt.title('Clusters Of Clients')
plt.xlabel('Anual Income (k$)')
plt.ylabel('Spending Scores (1-100)')
plt.lagend()
plt.show()




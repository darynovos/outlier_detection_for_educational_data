#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score


# In[11]:


def plot_ec_silhouette(X,random_state):
    plt.style.use('seaborn')
    Sum_of_squared_distances = []
    silhouete_s = []
    silhouete_negative = []
    K = range(2,21)

    for k in K:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state = random_state).fit(X)
        Sum_of_squared_distances.append(kmeans.inertia_)
        silhouete_s.append(silhouette_score(X, kmeans.labels_))
        ss = np.where(silhouette_samples(X, kmeans.labels_)<0)
        silhouete_negative.append(len(ss[0]))

    plt.figure(figsize = (15,5))
    plt.subplot(1,2,1)
    plt.plot(K,Sum_of_squared_distances,'o-')   
    plt.title('Elbow curve')
    plt.xlabel('k')
    plt.ylabel('Sum of squared distances')
    plt.xticks(np.arange(min(K), max(K)+1, 1.0))

    plt.subplot(1,2,2)
    plt.plot(K,silhouete_s)
    plt.xticks(np.arange(2,21,1))
    plt.title('Silhouette score')


# In[8]:


def intersections(outliers_results):
    N = len(outliers_results.columns)
    matrix = np.zeros((N,N))

    for i in range(0,N):
        col_1 = outliers_results.columns[i]
        left = outliers_results[outliers_results[col_1] == 1].index
        for j in range(i,N):
            col_2 = outliers_results.columns[j]
            right = outliers_results[outliers_results[col_2] == 1].index
            matrix[i][j] = len(left.intersection(right))
            
    intersection_m = pd.DataFrame(matrix)
    intersection_m.columns = outliers_results.columns
    intersection_m.index = outliers_results.columns
    return   intersection_m


# In[10]:


def clusters(X, class_o):
    max_values = pd.Series({'Etmath':5, 'Etph':20, 'Etch':24, 'AT1':15, 'AT2':5, 'Exam':5})
    
    tab_ = X.groupby(class_o).mean().round(2)
    tab_ = tab_.iloc[:,:6]
    tab_ = (tab_/max_values).round(2)
    tab_['N'] = X.groupby(class_o).count().iloc[:,[1]]
    return tab_


# In[ ]:





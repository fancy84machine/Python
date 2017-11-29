# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:14:02 2017

@author: Ling Sang 
This program uses K-Nearest Neighbor algorithm to predict a congressman's party based on his voting record. 
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv ('house-votes-84.data', delimiter = ',', header=None, names = ['party', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])


for i in range (1, 17):
    df[i].replace ( to_replace = ['?'], value = df[i].mode().iloc[0], inplace = True)
    df[i] = LabelEncoder().fit_transform (df[i])
    
# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier (n_neighbors = 6)

# Fit the classifier to the data
knn.fit (X, y)


# Predict the labels for the training data X: y_pred
y_pred = knn.predict(X)




# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv ('diabetes.csv')
df.info()


# In[6]:


X = df.drop('diabetes', axis=1)
y = df['diabetes']


# In[7]:


#Imputing within a pipeline
imp = Imputer (missing_values = 'NaN', strategy = 'mean', axis=0)
logreg = LogisticRegression()

steps = [('imputation', imp), 
        ('logistic_regression', logreg)]

pipeline = Pipeline (steps)
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.3, random_state = 42)
pipeline.fit (X_train, y_train)
y_pred = pipeline.predict (X_test)
pipeline.score (X_test, y_test)



# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv ('diabetes.csv')
df.info()


# In[16]:


df.insulin.replace (0, np.nan, inplace = True)
df.triceps.replace (0, np.nan, inplace = True)
df.bmi.replace (0, np.nan, inplace = True)

#not applicable here - df = df.dropna ()

X = df.drop('diabetes', axis=1)
y = df['diabetes']

imp = Imputer (missing_values = 'NaN', strategy = 'mean', axis = 0)
imp.fit (X)
X = imp.transform (X)


# In[17]:


X[0, :]


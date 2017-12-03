
# coding: utf-8

# In[22]:


from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

plt.style.use ('ggplot')
iris = datasets.load_iris()
print (type (iris))
print (iris.keys())


# In[6]:


type(iris.data), type (iris.target)


# In[7]:


iris.data.shape


# In[8]:


iris.target_names


# In[10]:


X=iris.data
y= iris.target
df = pd.DataFrame (X, columns = iris.feature_names)
print (df.head())


# In[14]:


_=pd.scatter_matrix (df, c=y, figsize = [8,8], s=150, marker = 'D')
plt.show()


# In[18]:


knn = KNeighborsClassifier (n_neighbors = 6)
knn.fit (iris['data'], iris ['target'])


# In[20]:


'''
prediction = knn.predict (X_new)
X_new.shape
print ('Prediction {}'.format (prediction))
'''


# In[23]:


X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.3, random_state = 21, stratify = y)
knn=KNeighborsClassifier (n_neighbors = 8)
knn.fit (X_train, y_train)
y_pred = knn.predict (X_test)
print ("Test set prediction: \n {}".format (y_pred))


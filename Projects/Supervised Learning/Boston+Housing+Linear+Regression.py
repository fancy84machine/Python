
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

boston = pd.read_csv ('boston.csv')

print (boston.head())


# In[22]:


X=boston.drop ('MEDV', axis = 1).values
y= boston ['MEDV'].values
X_rooms = X[:, 5]
type (X_rooms), type (y)


# In[23]:


y=y.reshape (-1, 1)
y


# In[24]:


X_rooms = X_rooms.reshape (-1, 1)


# In[25]:


X_rooms


# In[26]:


plt.scatter(X_rooms, y)
plt.ylabel ('Value of house /1000 ($)')
plt.xlabel ('Number of rooms')
plt.show()


# In[27]:


reg = linear_model.LinearRegression ()
reg.fit (X_rooms, y)
prediction_space = np.linspace (min(X_rooms), max (X_rooms)).reshape (-1, 1)
prediction_space


# In[28]:


plt.scatter (X_rooms, y, color = 'blue')
plt.plot (prediction_space, reg.predict (prediction_space), color = 'black', linewidth = 3)
plt.show()


# In[30]:


X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.3, random_state = 42)
reg_all = linear_model.LinearRegression()
reg_all.fit (X_train, y_train)
y_pred = reg_all.predict (X_test)
reg_all.score (X_test, y_test)


# In[32]:


#K-fold Cross-Validation
reg = linear_model.LinearRegression ()

#cv = number of folds
cv_results = cross_val_score (reg, X, y, cv=5)
print (cv_results)
np.mean (cv_results)


# In[34]:


X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.3, random_state = 42)
ridge = Ridge (alpha = 0.1, normalize = True)
ridge.fit (X_train, y_train)
ridge_pred = ridge.predict (X_test)
ridge.score (X_test, y_test)


# In[37]:


X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.3, random_state = 42)
lasso = Lasso (alpha = 0.1, normalize = True)
lasso.fit (X_train, y_train)
lasso_pred = lasso.predict (X_test)
lasso.score (X_test, y_test)


# In[38]:


names = boston.drop ('MEDV', axis = 1).columns
lasso = Lasso (alpha = 0.1)
lasso_coef = lasso.fit (X, y).coef_
_= plt.plot (range (len(names)), lasso_coef)
_=plt.xticks (range(len(names)), names, rotation = 60)
_=plt.ylabel ('Coefficients')
plt.show()


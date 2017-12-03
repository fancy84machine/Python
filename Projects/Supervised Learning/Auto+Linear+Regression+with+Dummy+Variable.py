
# coding: utf-8

# In[1]:



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


# In[4]:


#Encoding dummy variables
df = pd.read_csv ('auto.csv')
df_origin = pd.get_dummies (df)
print (df_origin.head())


# In[5]:


df_origin = df_origin.drop ('origin_Asia', axis = 1)
print (df_origin.head())


# In[6]:


X = df_origin.drop ('mpg', axis = 1)
y = df_origin ['mpg']


# In[8]:



#Linear regression with dummy variables
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.3, random_state = 42)
ridge = Ridge (alpha = 0.5, normalize = True).fit (X_train, y_train)
ridge.score (X_test, y_test)


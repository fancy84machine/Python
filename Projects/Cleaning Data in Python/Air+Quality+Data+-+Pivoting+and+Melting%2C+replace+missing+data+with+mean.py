
# coding: utf-8

# In[6]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


airquality = pd.read_csv ('airquality.csv')


# In[7]:


airquality.info()
airquality.head()


# In[8]:


# Print the head of airquality
print(airquality.head())

# Melt airquality: airquality_melt
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'])

# Print the head of airquality_melt
print(airquality_melt.head())


# In[10]:


#Customizing melted data
# Print the head of airquality
print(airquality.head())

# Melt airquality: airquality_melt with variables renamed
airquality_melt = pd.melt(airquality, id_vars=['Month', 'Day'], var_name='measurement', value_name='reading')

# Print the head of airquality_melt
print(airquality_melt.head())


# In[11]:


#Pivot data
# Print the head of airquality_melt
print(airquality_melt.head())

# Pivot airquality_melt: airquality_pivot
airquality_pivot = airquality_melt.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading')

# Print the head of airquality_pivot
print(airquality_pivot.head())


# In[13]:


airquality


# In[14]:


#Resetting the index of a DataFrame
# Print the index of airquality_pivot
print(airquality_pivot.index)

# Reset the index of airquality_pivot: airquality_pivot
airquality_pivot = airquality_pivot.reset_index()

# Print the new index of airquality_pivot
print(airquality_pivot.index)

# Print the head of airquality_pivot
print(airquality_pivot.head())


# In[15]:


'''
#Pivoting duplicate values
# Pivot airquality_dup: airquality_pivot
airquality_pivot = airquality_dup.pivot_table(index=['Month', 'Day'], columns='measurement', values='reading', aggfunc=np.mean)

# Reset the index of airquality_pivot
airquality_pivot = airquality_pivot.reset_index()

# Print the head of airquality_pivot
print(airquality_pivot.head())

# Print the head of airquality
print(airquality.head())
'''



# In[16]:


#Filling missing data
# Calculate the mean of the Ozone column: oz_mean
oz_mean = airquality['Ozone'].mean()

# Replace all the missing values in the Ozone column with the mean
airquality['Ozone'] = airquality['Ozone'].fillna (oz_mean)

# Print the info of airquality
print(airquality.info())


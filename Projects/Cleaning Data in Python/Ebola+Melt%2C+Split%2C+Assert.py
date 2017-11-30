
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ebola = pd.read_csv ('ebola.csv', header=0)


# In[2]:


ebola.columns


# In[3]:


#Splitting a column with .split() and .get()
# Melt ebola: ebola_melt
ebola_melt = pd.melt(ebola, id_vars=['Date', 'Day'], var_name='type_country', value_name='counts')

# Create the 'str_split' column
ebola_melt['str_split'] = ebola_melt.type_country.str.split('_')

# Create the 'type' column
ebola_melt['type'] = ebola_melt['str_split'].str.get(0)

# Create the 'country' column
ebola_melt['country'] = ebola_melt['str_split'].str.get(1)

# Print the head of ebola_melt
print(ebola_melt.head())


# In[6]:


'''
# Concatenate ebola_melt and status_country column-wise: ebola_tidy
ebola_tidy = pd.concat ([ebola_melt, status_country], axis = 1)

# Print the shape of ebola_tidy
print(ebola_tidy.shape)

# Print the head of ebola_tidy
print(ebola_tidy.head())

'''


# In[9]:


'''
all() method together with the .notnull() DataFrame method to check for missing values in a column. 
The .all() method returns True if all values are True. When used on a DataFrame, it returns a Series of Booleans - 
one for each column in the DataFrame. So if you are using it on a DataFrame, like in this exercise, you need to chain 
another .all() method so that you return only one True or False value. When using these within an assert statement, 
nothing will be returned if the assert statement is true: This is how you can confirm that the data you are checking are valid.

Note: You can use pd.notnull(df) as an alternative to df.notnull().



# Assert that there are no missing values
assert pd.notnull (ebola).all().all()

# Assert that all values are >= 0
assert (ebola>=0).all().all()
'''





# coding: utf-8

# In[21]:


# Import pandas
import pandas as pd
import matplotlib.pyplot as plt

# Read the file into a DataFrame: df
df = pd.read_csv ('dob_job_application_filings_subset.csv')

# Print the head of df
print(df.head())

# Print the tail of df
print(df.tail())

# Print the shape of df
print(df.shape)

# Print the columns of df
print(df.columns)

# Print the head and tail of df_subset
print(df.head())
print(df.tail())



# In[6]:


df.shape


# In[7]:


#Further diagnosis
# Print the info of df
print(df.info())



# In[8]:


df.describe()


# In[10]:


#Frequency counts for categorical data
# Print the value counts for 'Borough'
print(df['Borough'].value_counts(dropna=False))


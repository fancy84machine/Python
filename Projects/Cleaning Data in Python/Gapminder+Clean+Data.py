
# coding: utf-8

# In[63]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
g1800s = pd.read_csv ('gapminder.csv')
gapminder = pd.read_csv ('gapminder.csv')


# In[64]:


g1800s.head()


# In[65]:


g1800s.info()


# In[66]:


g1800s.describe()


# In[67]:


g1800s.columns
g1800s.shape


# In[68]:


#Visualizing your data
# Create the scatter plot
g1800s.plot(kind='scatter', x='1800', y='1899')

# Specify axis labels
plt.xlabel('Life Expectancy by Country in 1800')
plt.ylabel('Life Expectancy by Country in 1899')

# Specify axis limits
plt.xlim(20, 55)
plt.ylim(20, 55)

# Display the plot
plt.show()


# In[69]:


#Thinking about the question at hand
def check_null_or_valid(row_data):
    """Function that takes a row of data,
    drops all missing values,
    and checks if all remaining values are greater than or equal to 0
    """
    no_na = row_data.dropna()[1:-1]
    numeric = pd.to_numeric (no_na)
    ge0 = numeric >= 0
    return ge0

# Check whether the first column is 'Life expectancy'
assert g1800s.columns[0] == 'Life expectancy'

# Check whether the values in the row are valid
assert g1800s.iloc[:, 1:].apply(check_null_or_valid, axis=1).all().all()

# Check that there is only one instance of each country
assert g1800s['Life expectancy'].value_counts()[0] == 1


# In[70]:


'''
# Concatenate the DataFrames row-wise
gapminder = pd.concat ([g1800s, g1900s, g2000s])

# Print the shape of gapminder
print(gapminder.shape)

# Print the head of gapminder
print(gapminder.head())
'''


# In[71]:


# Melt gapminder: gapminder_melt
gapminder_melt = pd.melt(g1800s, id_vars='Life expectancy')

# Rename the columns
gapminder_melt.columns = ['country', 'year', 'life_expectancy']

# Print the head of gapminder_melt
print(gapminder_melt.head())



# In[74]:


gapminder = gapminder_melt
import numpy as np

# Convert the year column to numeric
gapminder.year = pd.to_numeric (gapminder['year'])

# Test if country is of type object
assert gapminder.country.dtypes == np.object

# Test if year is of type int64
assert gapminder.year.dtypes == np.int64

# Test if life_expectancy is of type float64
assert gapminder.life_expectancy.dtypes == np.float64


# In[76]:


# Create the series of countries: countries
countries = gapminder ['country']

# Drop all the duplicates from countries
countries = countries.drop_duplicates()

# Write the regular expression: pattern
# Anchor the pattern to match exactly what you want by placing a ^ in the beginning and $ in the end
# Use A-Za-z to match the set of lower and upper case letters, \. to match periods, and \s to match whitespace between words.
pattern = '^[A-Za-z\.\s]*$'

# Create the Boolean vector: mask
mask = countries.str.contains(pattern)

# Invert the mask: mask_inverse
mask_inverse = ~mask

# Subset countries using mask_inverse: invalid_countries
invalid_countries =  countries.loc[mask_inverse]

# Print invalid_countries
print(invalid_countries)


# In[77]:


# Assert that country does not contain any missing values
assert pd.notnull(gapminder.country).all()

# Assert that year does not contain any missing values
assert pd.notnull (gapminder.year).all()

# Drop the missing values
gapminder = gapminder.dropna (how='any')

# Print the shape of gapminder
print(gapminder.shape)


# In[78]:


# Add first subplot
plt.subplot(2, 1, 1) 

# Create a histogram of life_expectancy
gapminder.life_expectancy.plot (kind= 'hist')

# Group gapminder: gapminder_agg
gapminder_agg = gapminder.groupby('year')['life_expectancy'].mean ()

# Print the head of gapminder_agg
print(gapminder_agg.head())

# Print the tail of gapminder_agg
print(gapminder_agg.tail())

# Add second subplot
plt.subplot(2, 1, 2)

# Create a line plot of life expectancy per year
gapminder_agg.plot ()

# Add title and specify axis labels
plt.title('Life expectancy over the years')
plt.ylabel('Life expectancy')
plt.xlabel('Year')

# Display the plots
plt.tight_layout()
plt.show()

# Save both DataFrames to csv files
gapminder.to_csv ('gapminder.csv')
gapminder_agg.to_csv ('gapminder_agg.csv')


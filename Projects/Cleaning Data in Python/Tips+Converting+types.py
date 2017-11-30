
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


tips = pd.read_csv ('tips.csv', header=0)


# In[2]:


tips.info()


# In[3]:


# Convert the sex column to type 'category'
tips.sex = tips.sex.astype('category')

# Convert the smoker column to type 'category'
tips.smoker = tips.smoker.astype('category')

# Print the info of tips
print(tips.info())


# In[4]:


#Convert 'total_bill' to a numeric dtype
#Coerce the errors to NaN by specifying the keyword argument errors='coerce'
tips['total_bill'] = pd.to_numeric (tips['total_bill'], errors='coerce')

# Convert 'tip' to a numeric dtype
tips['tip'] = pd.to_numeric (tips['tip'], errors='coerce')

# Print the info of tips  to confirm that the data types of 'total_bill' and 'tips' are numeric.
print(tips.info())


# In[5]:


# Import the regular expression module
import re

# Compile the pattern: prog
prog = re.compile('\d{3}-\d{3}-\d{4}')

# See if the pattern matches
result = prog.match('123-456-7890')
print(bool(result))

# See if the pattern matches
result = prog.match('1123-456-7890')
print(bool(result))


# In[6]:


# Import the regular expression module
import re

# Find the numeric values: matches
matches = re.findall('\d+', 'the recipe calls for 10 strawberries and 1 banana')

# Print the matches
print(matches)


# In[7]:


# A telephone number of the format xxx-xxx-xxxx
pattern1 = bool(re.match(pattern='\d{3}-\d{3}-\d{4}', string='123-456-7890'))
print(pattern1)

# A string of the format: A dollar sign, an arbitrary number of digits, a decimal point, 2 digits.
pattern2 = bool(re.match(pattern='\$\d*\.\d{2}', string='$123.45'))
print(pattern2)

#A capital letter, followed by an arbitrary number of alphanumeric characters.
pattern3 = bool(re.match(pattern='[A-Z]\w*', string='Australia'))
print(pattern3)


# In[8]:


# Define recode_sex()
def recode_sex(sex_value):

    # Return 1 if sex_value is 'Male'
    if sex_value == 'Male':
        return 1
    
    # Return 0 if sex_value is 'Female'    
    elif sex_value == 'Female':
        return 0
    
    # Return np.nan    
    else:
        return np.nan

# Apply the function to the sex column
tips['sex_recode'] = tips.sex.apply (recode_sex)

# Print the first five rows of tips
print(tips.head())


# In[17]:


tips['total_dollar'] = tips.total_bill.apply (lambda x: '$' + str(x))
tips.head()


# In[18]:


# Write the lambda function using replace
tips['total_dollar_replace'] = tips.total_dollar.apply (lambda x: x.replace('$', ''))

# Write the lambda function using regular expressions
tips['total_dollar_re'] = tips.total_dollar.apply (lambda x: re.findall('\d+\.\d+', x)[0])

# Print the head of tips
print(tips.head())


# In[ ]:


'''
# Create the new DataFrame: tracks
tracks = billboard[['year', 'artist', 'track', 'time']]

# Print info of tracks
print(tracks.info())

# Drop the duplicates: tracks_no_duplicates
#drop all duplicate rows.
tracks_no_duplicates = tracks.drop_duplicates()

# Print info of tracks_no_duplicates
print(tracks_no_duplicates.info())

'''


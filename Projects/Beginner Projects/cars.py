# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 20:51:27 2017

@author: Ling Sang 2
"""

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out country column as Pandas Series
print (cars['country'])

# Print out country column as Pandas DataFrame
print (cars [['country']])

# Print out first 3 observations
print (cars [0:3])

# Print out fourth, fifth and sixth observation
print (cars [3:6])

# Print out observation for Japan
print (cars.loc ['JAP'])

# Print out observations for Australia and Egypt
print (cars.loc [['AUS', 'EG']])


#loc and iloc (2)
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out drives_right value of Morocco
print (cars.loc['MOR'] ['drives_right'])

# Print sub-DataFrame
print (cars.loc [['RU', 'MOR'], ['country', 'drives_right']])

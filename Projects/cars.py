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



# Create car_maniac: observations that have a cars_per_cap over 500
cpc = cars ['cars_per_cap']
many_cars = cpc > 500
car_maniac = cars [many_cars]

# Print car_maniac
print (car_maniac)


# Create medium: observations with cars_per_cap between 100 and 500
cpc = cars['cars_per_cap']
between = np.logical_and(cpc > 100, cpc < 500)
medium = cars[between]

# Print medium
print ( medium)


# Adapt for loop
for lab, row in cars.iterrows() :
    print(lab + ": " + str (row ['cars_per_cap']))

# Use .apply(str.upper) - don't use the for loop
#for lab, row in cars.iterrows() :
#    cars.loc[lab, "COUNTRY"] = row["country"].upper()
cars ['COUNTRY'] = cars ['country'].apply (str.upper)

print (cars)



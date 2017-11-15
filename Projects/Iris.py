# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:30:16 2017

@author: Ling Sang 2
"""
from sklearn import datasets
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use ('ggplot')

iris = datasets.load_iris()


#Computing the ECDF
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1)*1.00000 / n

    return x, y




X= iris.data
y=iris.target

df = pd.DataFrame (X, columns = iris.feature_names)

petal_length = df ['petal length (cm)']

setosa_petal_length = petal_length [y==0]
versicolor_petal_length = petal_length [y==1]
virginica_petal_length = petal_length [y==2]

# Plot histogram of versicolor petal lengths
_ = plt.hist (versicolor_petal_length)

# Label axes
_ = plt.xlabel ('petal length (cm)')
_ = plt.ylabel ('count')

# Show histogram
plt.show ()



#Plotting the ECDF
# Compute ECDF for versicolor data: x_vers, y_vers
x_vers, y_vers = ecdf(versicolor_petal_length)

# Generate plot
_ = plt.plot (x_vers, y_vers, marker = '.', linestyle = 'none')


# Make the margins nice
plt.margins (0.02)

# Label the axes
_= plt.ylabel ('ECDF')
_= plt.xlabel ('petal length (cm)')

# Display the plot
plt.show ()






#Comparison of ECDFs
# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf (virginica_petal_length)

# Plot all ECDFs on the same plot
_ = plt.plot (x_set, y_set, marker = ".", linestyle = 'none')
_ = plt.plot (x_vers, y_vers, marker = ".", linestyle = 'none')
_ = plt.plot (x_virg, y_virg, marker = ".", linestyle = 'none')

# Make nice margins
plt.margins (0.02)

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()








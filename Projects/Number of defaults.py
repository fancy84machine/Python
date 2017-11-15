# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:23:04 2017

@author: Ling Sang 2
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Computing the ECDF
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1)*1.000000 /n

    return x, y



#The np.random module and Bernoulli trials
def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0


    # Perform trials
    for i in range (n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random ()

        # If less than p, it's a success so add one to n_success
        if (random_number < p):
            n_success += 1

    return n_success




#Number of defaults simulator 
np.random.seed(42)


# Initialize the number of defaults: n_defaults
n_defaults = np.empty (1000)

#p = probability of defaults
p = 0.05

#n = 100 loans
n = 100

# Compute the number of defaults
for i in range (1000):
    n_defaults[i] = perform_bernoulli_trials(n, p)




# Compute ECDF: x, y
x, y = ecdf (n_defaults)


# Plot the ECDF with labeled axes
_ = plt.plot (x, y, marker = ".", linestyle = 'none')
_ = plt.xlabel ('number of defaults')
_ = plt.ylabel ('CDF')

# Show the plot
plt.show ()



#Plotting the Binomial PMF
# Compute bin edges: bins
bins = np.arange(0, max(n_defaults) + 2) - 0.5

# Generate histogram
plt.hist (n_defaults, normed = True, bins = bins)

# Set margins
plt.margins (0.02)

# Label axes
plt.xlabel ('Number of Defaults')
plt.ylabel ('PMF')

# Show the plot
plt.show ()




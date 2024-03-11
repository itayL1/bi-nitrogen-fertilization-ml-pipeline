from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import chisquare


def calc_gini_coefficient(array: np.array):
    # Mean absolute difference
    mad = np.abs(np.subtract.outer(array, array)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(array)

    gini_coefficient = 0.5 * rmad
    return gini_coefficient


years_series = pd.Series([1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])

gini = calc_gini_coefficient(years_series.to_numpy())
print(f"Gini Coefficient: {gini}")

# Perform the chi-squared test
chi2_stat, p_value = chisquare(f_obs=years_series.value_counts().sort_index())

# Print the results
print(f"Chi-squared statistic: {chi2_stat}")
print(f"P-value: {p_value}")

# Check the significance level (e.g., 0.05)
alpha = 0.05
if p_value < alpha:
    print("The distribution is not balanced (reject null hypothesis)")
else:
    print("The distribution is relatively balanced (fail to reject null hypothesis)")

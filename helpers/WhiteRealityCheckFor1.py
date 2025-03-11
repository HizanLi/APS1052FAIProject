#!/usr/bin/env python
"""
WhiteRealityCheckFor1 module

This module performs a bootstrap analysis on a given time series. It:
  - Computes basic statistics from the input series.
  - Plots a histogram using a modified Freedman–Diaconis rule to determine
    the number of bins.
  - Performs bootstrap resampling to estimate a 95% confidence interval for
    the mean.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bootstrap(ser, num_bootstrap=1000):
    """
    Perform bootstrap analysis on a given time series.

    Parameters:
      ser (pandas.Series or array-like): The input time series data.
      num_bootstrap (int): Number of bootstrap iterations (default: 1000).

    Returns:
      None. Displays a histogram and prints the 95% confidence interval.
    """
    # Ensure input is a pandas Series
    if not isinstance(ser, pd.Series):
        ser = pd.Series(ser)
    
    # Remove NaN values
    ser = ser.dropna()
    if ser.empty:
        raise ValueError("Input series is empty after removing NaN values.")

    # Calculate descriptive statistics using .describe()
    desc = ser.describe()
    count = desc.iloc[0]       # 'count'
    mean_val = desc.loc['mean']
    std = desc.iloc[2]         # 'std'
    minim = desc.iloc[3]       # 'min'
    maxim = desc.iloc[-1]      # 'max'

    # Calculate the range
    R = maxim - minim

    # Check for invalid standard deviation
    if std == 0 or np.isnan(std):
        raise ValueError("Standard deviation is zero or NaN, cannot compute histogram bins.")

    # Calculate the number of histogram bins using a modified Freedman-Diaconis rule:
    # bins = round( R * (n^(1/3)) / (3.49 * std) )
    bins_value = int(round(R * (count ** (1/3)) / (3.49 * std), 0))
    if bins_value < 1:
        bins_value = 1

    # Plot histogram with computed bins
    plt.figure()
    plt.hist(ser, bins=bins_value, edgecolor='black')
    plt.axvline(x=mean_val, color='b', label='Mean')
    plt.title("Bootstrap Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Bootstrap resampling to compute the 95% confidence interval for the mean
    bootstrap_means = []
    for i in range(num_bootstrap):
        sample = ser.sample(n=int(count), replace=True)
        bootstrap_means.append(sample.mean())

    lower_bound = np.percentile(bootstrap_means, 2.5)
    upper_bound = np.percentile(bootstrap_means, 97.5)
    print("Bootstrap 95% CI for the mean: [{:.4f}, {:.4f}]".format(lower_bound, upper_bound))


if __name__ == "__main__":
    # Example usage:
    np.random.seed(0)
    # Generate synthetic data from a normal distribution
    data = np.random.normal(loc=0, scale=1, size=1000)
    bootstrap(pd.Series(data))

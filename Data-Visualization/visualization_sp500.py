# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12

@author: Jonathan Arriaga

"""

#%% Import 3rd party libs
import pandas as pd
import matplotlib.pyplot as plt

#%% Time Series using tow years of the S&P 500

sp500 = pd.read_csv('data_SP500_1year.csv')
sp500.index = sp500['Date'] # Set the index as the date
sp500.sort_index(inplace=True)

# Compute moving average, returns, and rolling standard deviation
sp500['MA(20)'] = pd.rolling_mean(sp500['Adj Close'], window=20)
sp500['Returns'] = sp500['Adj Close'].diff() / sp500['Adj Close']
sp500['Roll Std.(20)'] = pd.rolling_std(sp500['Returns'], window=20)

# Plot transformations
fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12,6))
ax1.set_title('S&P 500 - March 2015 to March 2016')
sp500[['Adj Close','MA(20)']].plot(ax=ax1)
sp500[['Returns', 'Roll Std.(20)']].plot(ax=ax2)

ax1.grid(True)
ax2.grid(True)

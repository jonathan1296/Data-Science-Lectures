# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 12:55:47 2016

@author: 212415731
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 10:50:12 2016

@author: Jonathan Arriaga

Dataset used: Occupancy Detection Data Set 
http://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+

Abstract: Experimental data used for binary classification (room occupancy) 
from Temperature,Humidity,Light and CO2. Ground-truth occupancy was obtained 
from time stamped pictures that were taken every minute.

Data Set Information: Three data sets are submitted, for training and testing. 
Ground-truth occupancy was obtained from time stamped pictures that were taken 
every minute. 

Attribute Information:

date time year-month-day hour:minute:second 
- Temperature, in Celsius 
- Relative Humidity, % 
- Light, in Lux 
- CO2, in ppm 
- Humidity Ratio, Derived quantity from temperature and relative humidity, in 
kgwater-vapor/kg-air 
- Occupancy, 0 or 1, 0 for not occupied, 1 for occupied status

"""

#%% 3-rd party libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix

#%% Defined functions

def missing_data_by_cols(df):
    """
    Returns a pandas data frame with the columns and the percentage of missing data for each column.
    """
    missing = 100 - np.array([df[c].notnull().sum()*100.0 / df.shape[0] for c in df.columns])
    return pd.DataFrame({'Column':df.columns, 'Missing %':missing})

#%% Read the data

df = pd.read_csv('data_bike_rental_day.csv')
df.index = df['dteday']

print df.dtypes
print missing_data_by_cols(df)

#%% Scatter Plot

df['Day of the year'] = df.groupby('yr').cumcount()
grouped = df.groupby('yr')
year0 = grouped.get_group(0)
year1 = grouped.get_group(1)

fig, ax = plt.subplots(figsize=(9,6))
year0.plot.scatter(x='Day of the year', y='cnt', ax=ax, c='g', label='2011')
year1.plot.scatter(x='Day of the year', y='cnt', ax=ax, c='b', label='2012')
ax.set_xlim(0,366)
ax.set_ylabel('Total Count')

#%% Scatter Matrix

numerical_features = ['mnth', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']
scatter_matrix(df[numerical_features ], alpha=0.2, figsize=(8,8), diagonal='hist')


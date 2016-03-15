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

#%% Aggregate the data

grouped = df.groupby('yr')
year0 = grouped.get_group(0)
year1 = grouped.get_group(1)

fig, ax = plt.subplots(figsize=(12,6))
for i,dataset in enumerate((year0, year1)):
    weekday_aggregated = dataset.groupby('weekday').agg({'cnt':'mean',
                                                        'casual':'mean',
                                                        'registered':'mean'})
    ls = '-' if i == 0 else '-.'
    weekday_aggregated['cnt'].plot(color='b', ax=ax, label='Total Count year '+str(2011+i), ls=ls)
    weekday_aggregated['casual'].plot(color='g', ax=ax, label='Casual year '+str(2011+i), ls=ls)
    weekday_aggregated['registered'].plot(color='r', ax=ax, label='Registered year '+str(2011+i), ls=ls)
    ax.grid(True)
    ax.legend(loc='best')
ax = ax.set_ylim(0,9000)


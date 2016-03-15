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
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
from sklearn.decomposition import PCA
# import pydot_ng
import matplotlib.pyplot as plt

#%% Defined functions

def missing_data_by_cols(df):
    """
    Returns a pandas data frame with the columns and the percentage of missing data for each column.
    """
    missing = 100 - np.array([df[c].notnull().sum()*100.0 / df.shape[0] for c in df.columns])
    return pd.DataFrame({'Column':df.columns, 'Missing %':missing})

#%% Read the data

df = pd.read_csv('data_occupancy.csv')

print df.dtypes
print missing_data_by_cols(df)

#%% Data Preprocessing

# Define numerical features
numerical_features = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
scaled_features = [f+'_scaled' for f in numerical_features]

for f in scaled_features: df[f] = None

# Normalize
scaler = StandardScaler()
df.loc[:,scaled_features] = scaler.fit_transform(df[numerical_features])

# Softmax normalization
df.loc[:,scaled_features] = expit(df[scaled_features].values)*2.0 - 1.0

#%% Separate the data by groups

groups = df.groupby('Occupancy')

negatives = groups.get_group(0) 
positives = groups.get_group(1) 

#%% Box plots for multiple features

fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5,8))

negatives[scaled_features].plot(kind='box', vert=False, ax=ax1)
ax1.set_title('Not Occupied')
ax1.grid(True)

positives[scaled_features].plot(kind='box', vert=False, ax=ax2)
ax2.set_title('Occupied')
ax2.grid(True)

#%% Histograms example

fig, ax = plt.subplots(figsize=(8,5))

# Plot density of non-survived
negatives['Temperature'].plot(kind='hist', ax=ax, color='r', label='Not Occupied', alpha=0.5)
# Plot density of survived
positives['Temperature'].plot(kind='hist', ax=ax, color='g', label='Occupied', alpha=0.5)
ax.set_title('Temperature')
ax.grid(True)
ax.legend(loc='best')

#%% Density plots for multiple features

fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(17,7))

for f,ax in zip(scaled_features ,(ax1,ax2,ax3,ax4,ax5)):
    # Plot density of non-survived
    negatives[f].plot(kind='kde', ax=ax, color='r', label='Not Occupied')
    ax.set_title(f)
    ax.grid(True)
    # Plot density of survived
    positives[f].plot(kind='kde', ax=ax, color='g', label='Occupied')
    ax.set_title(f)
    ax.grid(True)
    ax.legend(loc='best')

df[scaled_features].plot(kind='kde', ax=ax6)
ax6.set_title('All Features')
ax6.grid(True)

#%% PCA Visualization

pca_features = scaled_features
pca_features = ['Temperature_scaled','Light_scaled','CO2_scaled']

# Fit PCA and check how much variance is explained with each component
pca = PCA(n_components=None)
pca.fit(df[pca_features])
print '% Explained variance: ', pca.explained_variance_ratio_.cumsum()

# Make data frame with principal components transformations
df_prin = pd.DataFrame(pca.transform(df[pca_features])[:,[0,1]], 
                       columns=['Prin1', 'Prin2'], index=df.index)
df = pd.concat((df, df_prin), axis=1)

groups = df.groupby('Occupancy')
negatives = groups.get_group(0) 
positives = groups.get_group(1) 

# Plot first two principal components
fig, ax = plt.subplots(figsize=(9,7))
ax.scatter(positives['Prin1'], positives['Prin2'], color='g', label='Occupied')
ax.scatter(negatives['Prin1'], negatives['Prin2'], color='r', label='Not Occupied')
ax.set_title('PCA of Occupancy Dataset Features')
ax.legend(loc='best')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')



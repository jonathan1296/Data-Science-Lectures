# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 17:29:29 2016

@author: Jonathan Arriaga
"""

#%% 3rd party libs

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

#%% Defined functions

def missing_data_by_cols(df):
    """
    Returns a pandas data frame with the columns and the percentage of missing data for each column.
    """
    missing = 100 - np.array([df[c].notnull().sum()*100.0 / df.shape[0] for c in df.columns])
    return pd.DataFrame({'Column':df.columns, 'Missing %':missing})


def binarize_label_columns(df, columns, two_classes_as='single'):
    '''
    Inputs:
        df: Pandas dataframe object.
        columns: Columns to binarize.
        tow_classes_as: How to handle two classes, as 'single' or 'multiple' columns.
    Returns a tuple with the following items:
        df: Pandas dataframe object with new columns.
        binlabel_names: Names of the newly created binary variables.
        lb_objects: a dictionary with columns as keys and sklear.LabelBinarizer 
        objects as values.
    '''
    binlabel_names = []
    lb_objects = {}
    for col in columns:
        if len(df[col].unique()) > 1: 
            rows_notnull = df[col].notnull() # Use only valid feature observations
            lb = LabelBinarizer()
            binclass = lb.fit_transform(df[col][rows_notnull]) # Fit & transform on valid observations
            if len(lb.classes_) == 2 and two_classes_as == 'multiple':
                binclass = np.hstack((1 - binclass, binclass))
            lb_objects[col] = lb
            if len(lb.classes_) > 2 or two_classes_as == 'multiple':
                col_binlabel_names = [col+'_'+str(c) for c in lb.classes_]
                binlabel_names += col_binlabel_names # Names for the binarized classes
                for n in col_binlabel_names: df[n] = np.NaN # Initialize columns
                df.loc[rows_notnull, col_binlabel_names] = binclass # Merge binarized data
            elif two_classes_as == 'single': 
                binlabel_names.append(col+'_bin') # Names for the binarized classes
                df[col+'_bin'] = np.NaN # Initialize columns
                df.loc[rows_notnull, col+'_bin'] = binclass # Merge binarized data
    return df, binlabel_names, lb_objects


#%% Read the data

df = pd.read_csv('engineering_data - blade_damage_assessment.csv')

#%% Visualize a chunk of the data

print df.head(10)

#%% Visualize data types

print df.dtypes

print '''
NOTE: Variables t_3, t_4, vibrations_2, vibrations_4, and core_speed should be 
numeric, but its data type is 'object', they need to be converted to numeric.
'''

#%% How much data is missing?

print missing_data_by_cols(df)

#%% Convert columns to numeric one by one 
# NOTE: An alternative is to use pandas.DataFrame.convert_objects, but is deprecated

object_cols = ['t_3', 't_4', 'vibrations_2', 'vibrations_4', 'core_speed']
for col in object_cols: 
    df.loc[:, col] = pd.to_numeric(df.loc[:, col], errors='coerce')

print df.dtypes

'''
How much data is missing after converting to numeric? WHY? Strings were 
converted to NaN thanks to the coerce method.
'''

print missing_data_by_cols(df)

#%% Handle categorical columns

categorical_cols = ['customer', 'engine_type']
df, binlabel_names, lb_objects = binarize_label_columns(df, categorical_cols, 
                                                        two_classes_as='multiple')

#%% Aggregate the data
# NOTE: The mean of string columns can't be computed, so we get only the first value.

grouped = df.groupby('engine_id')
df_agg = grouped.agg({'customer': 'first',
                     'engine_type': 'first',
                     'category':'first',
                     'damage': 'first',
                     't_1':'mean',
                     't_2':'mean',
                     't_3':'mean',
                     't_4':'mean',
                     't_oil':'mean',
                     'p_oil':'mean',
                     'vibrations_2':'mean',
                     'vibrations_4':'mean',
                     'core_speed':'mean',
                     'fan_speed':'mean',
                     'thrust':'mean',
                     'customer_ACC':'first',
                     'customer_ASI':'first',
                     'customer_DME':'first',
                     'customer_FAR':'first',
                     'customer_SLA':'first',
                     'engine_type_EX-50A':'first',
                     'engine_type_EX-50B':'first'})
                     
df_agg['n_flights'] = df.groupby('engine_id')['engine_id'].count()

print '''
NOTE: You can save df_agg in a csv file and continue the analysis, either in Python 
or other data analysis software.
'''

#%% How much data is missing in the aggregated dataset?

print missing_data_by_cols(df_agg)

#%% Scale numerical features

numerical_features = ['t_1','t_2','t_3','t_4', 't_oil', 'p_oil', 'vibrations_2', 
                      'vibrations_4', 'core_speed', 'fan_speed', 'thrust']

scaler = StandardScaler()
df_agg.loc[:,numerical_features] = scaler.fit_transform(df_agg.loc[:,numerical_features])

#%% Boxplots grouping by category for each group of features
for v in numerical_features :
    df_agg[['category']+[v]].groupby('category').boxplot()


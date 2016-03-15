# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 16:29:26 2016

@author: 212415731
"""

#%% 3-rd party libraries
import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix

#%% Scatter matrix example with toy data

df = pd.DataFrame(np.random.randn(1000, 3), columns=['a', 'b', 'c'])
df['d'] = df['b'] + 0.3*np.random.randn(1000)
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='hist')
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 10:18:42 2018

@author: preston
"""

# This code looks into one file at a time (instead all 6)
# Goal:
#   - compare the shape of TD vs. CP kids. (TD - spheres; CP - ellipsoid)

import pandas as pd # for reading raw data'
import matplotlib.pyplot as plt

CPR = '/Users/preston/SCH-Local/Data in GTX/CIMT13_v2_RRAW.csv'
CPL = '/Users/preston/SCH-Local/Data in GTX/CIMT13_v2_LRAW.csv'

TDR = '/Users/preston/SCH-Local/Data in GTX/TD11_v1_RRAW.csv'
TDL = '/Users/preston/SCH-Local/Data in GTX/TD11_v1_LRAW.csv'

dfCPR = pd.read_csv(CPR, header = 10, usecols = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z'])
dfCPL = pd.read_csv(CPL, header = 10, usecols = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z'])
dfTDR = pd.read_csv(TDR, header = 10, usecols = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z'])
dfTDL = pd.read_csv(TDL, header = 10, usecols = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z'])
#%%


plt.close('all')
# let's plot 1000 points at a time
CPRsubset = dfCPR.iloc[0:10001].values
CPLsubset = dfCPL.iloc[0:10001].values
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(CPRsubset[:, 0], CPRsubset[:, 1], CPRsubset[:, 2])
ax.scatter(CPLsubset[:, 0], CPLsubset[:, 1], CPLsubset[:, 2])

TDRsubset = dfTDR.iloc[0:10001].values
TDLsubset = dfTDL.iloc[0:10001].values
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(TDRsubset[:, 0], TDRsubset[:, 1], TDRsubset[:, 2])
ax.scatter(TDLsubset[:, 0], TDLsubset[:, 1], TDLsubset[:, 2])

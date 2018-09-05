#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 14:15:15 2018

@author: preston
"""

import pandas as pd
import csv
import numpy as np
import time

def useCSV(file, activeRanges):
    start = time.time()
    UTV1 = [] # stores the entire, unsliced data
    UTV2 = np.empty((0, 3)) #stores the sliced data
    with open(file, 'r', encoding = 'utf-8') as csvFile:
        csvReader = csv.reader(csvFile) #basically the same as a scanner
        for i in range(11): #consumes through all of the header information
            next(csvReader) 
        for row in csvReader:
            UTV1.append(list(map(float, row[1:4])))
        for bounds in activeRanges:
            UTV2 = np.vstack((UTV2, UTV1[bounds[0] * 100 : bounds[1] * 100]))  
    end = time.time()
    print('total =', str(end - start))
    return UTV2
     
def usePD(file, activeRanges):     
    start = time.time()      
    df = pd.read_csv(file, header = 10, usecols = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z'])
    UTV2 = np.empty((0, 3))
    for bounds in activeRanges:
        UTV2 = np.vstack((UTV2, df.iloc[bounds[0] * 100 : bounds[1] * 100].values))
    end = time.time()
    print('total =', str(end - start))
    return UTV2

file = '/Users/Preston/Documents/MATLAB/CIMT04_v1_LRAW.csv'
activeRanges = [[3000, 259200]]

csvRead = useCSV(file, activeRanges)
pdRead = usePD(file, activeRanges)

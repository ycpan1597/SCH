#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 16:12:25 2018

@author: preston
"""
import csv
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

class jerk:
    def __init__(self, file, applyButter = False, timeRange = None):
        self.raw, self.Amag = self.readFile(file, applyButter, timeRange = timeRange)
        self.jerk, self.Jmag = self.findJerk()
        self.avg = np.mean(self.raw, axis = 0)
        self.std = np.std(self.raw, axis = 0)
        self.filtered = applyButter
    
    def matMag(self, mat): 
        mag = []
        for row in mat:
            mag.append(math.sqrt(row[0]**2 + row[1]**2+ row[2]**2))
        return mag
    
    def readFile(self, file, applyButter, timeRange = None):
    
        # takes in a 3D dataset, filters each component with the specified cutoff frequencies
        # and returns a filtered 3D dataset
        def butterworthFilt(data):
            # user input
            order = 4
            fsampling = 100 #in Hz
            wcLow = 0.25 #in Hz
            wcHigh = 2.5 #in Hz
            
            nyquist = fsampling/2 * 2 * np.pi #in rad/s
            wcLow = wcLow * 2 * np.pi #in rad/s
            wcHigh = wcHigh * 2 * np.pi #in rad/s
            b, a = signal.butter(order, [wcLow/nyquist, wcHigh/nyquist], 'bandpass')
    #            b, a = signal.butter(order, wcHigh/nyquist) # just a low pass
            filtedX = signal.filtfilt(b, a, data[:, 0])
            filtedY = signal.filtfilt(b, a, data[:, 1])
            filtedZ = signal.filtfilt(b, a, data[:, 2])
            return np.array([list(a) for a in zip(filtedX, filtedY, filtedZ)])
    
        UTV = []
        # file = '/Users/preston/Desktop/' + file
        with open(file, 'r', encoding = 'utf-8') as csvFile:
            csvReader = csv.reader(csvFile) #basically the same as a scanner
            for i in range(11): #consumes through all of the header information
                next(csvReader)
            for row in csvReader:
                UTV.append(list(map(float, row)))
        if applyButter:
            return butterworthFilt(np.array(UTV)), butterworthFilt(np.array(self.matMag(UTV)))
        else: 
            if timeRange is not None: 
                return np.array(UTV[timeRange[0]:timeRange[1]]), np.array(self.matMag(UTV[timeRange[0]:timeRange[1]]))
            else:
                return np.array(UTV), np.array(self.matMag(UTV))
    def findJerk(self):
        jerk = []
        for i in range(len(self.raw) - 1):
            jerk.append(np.subtract(self.raw[i + 1], self.raw[i]))
        jerk = np.multiply(jerk, 100)
        return jerk, np.array(self.matMag(jerk))


def butterworthFilt(data):
    # user input
    order = 4
    fsampling = 100 #in Hz
    wcLow = 0.25 #in Hz
    wcHigh = 2.5 #in Hz
    
    nyquist = fsampling/2 * 2 * np.pi #in rad/s
    wcLow = wcLow * 2 * np.pi #in rad/s
    wcHigh = wcHigh * 2 * np.pi #in rad/s
    b, a = signal.butter(order, [wcLow/nyquist, wcHigh/nyquist], 'bandpass')
#            b, a = signal.butter(order, wcHigh/nyquist) # just a low pass
    filtedX = signal.filtfilt(b, a, data[:, 0])
    filtedY = signal.filtfilt(b, a, data[:, 1])
    filtedZ = signal.filtfilt(b, a, data[:, 2])
    return np.array([list(a) for a in zip(filtedX, filtedY, filtedZ)])

def fftSNR(a, title = None, thresh = 1.5, axis = 'fixed'):
    plt.figure(figsize = (5, 5))
    for i, direction in zip(range(3), 'xyz'):
        afft = np.fft.fft(a[:, i])
        afft_new = abs(afft)[0:int(len(afft)/2)]
        freq = np.linspace(0, 50, int(len(afft)/2))
        threshIndex = int(np.where(freq > thresh)[0][0])
        signal = sum(afft_new[0:threshIndex])
        noise = sum(afft_new[threshIndex:])    
        plt.subplot(3, 1, i + 1)
        plt.plot(freq, afft_new, label = 'SNR = ' + str(round(signal/noise, 2)))
        plt.title(direction)
        plt.axvline(x = thresh, label = 'signal-noise cutoff = ' + str(thresh), color = 'r')
        if axis is 'fixed':
            plt.axis([0, 3, 0, 70])
        plt.xlabel('freq (Hz)')
        plt.ylabel('Power')
        plt.legend()
    plt.tight_layout()
        
    plt.suptitle(title)

# takes a 3D array and plots 3 subplots, one direction in each    
def plotSignal(exp, title, content, timeRange = None):
    # need to integrate these two if statements into one
    if content is 'accel':
        signal, mag = exp.raw, exp.Amag
    else:
        signal, mag = exp.jerk, exp.Jmag
    if timeRange is not None:
        signal = signal[timeRange[0]:timeRange[1]]
        mag = mag[timeRange[0]:timeRange[1]]
    plt.figure()
    for i, c, direction in zip(range(3), 'gkr', 'xyz'):
        plt.subplot(4, 1, i + 1)
        plt.plot(signal[:, i], color = c)
        plt.title(direction)
    plt.subplot(4, 1, 4)
    plt.plot(mag)
    plt.suptitle(title)

def embedSubplots(a, title = None, thresh = 1.5, axis = 'fixed'): 
    fig = plt.figure(figsize = (10, 8))
    outer = gridspec.GridSpec(1, len(a), wspace = 0.5, hspace = 0.2)
    for i in range(len(a)):
        inner = gridspec.GridSpecFromSubplotSpec(a[0].shape[1], 1, subplot_spec = outer[i], wspace = 0.1, hspace = 0.1)
        
#        for j in range(3):
#            ax = plt.Subplot(fig, inner[j])
#            t = ax.text(0.5, 0.5, 'outer = %d, inner = %d' % (i, j))
#            t.set_ha('center')
#            ax.set_xticks([])
#            ax.set_yticks([])
#            fig.add_subplot(ax)
            
        for j, direction in zip(range(a[0].shape[1]), 'xyz'):
            afft = np.fft.fft(a[i][:, j])
            afft_new = abs(afft)[0:int(len(afft)/2)]
            freq = np.linspace(0, 50, int(len(afft)/2))
            threshIndex = int(np.where(freq > thresh)[0][0])
            signal = sum(afft_new[0:threshIndex])
            noise = sum(afft_new[threshIndex:]) 
            
            #turn into subplots
            ax = plt.Subplot(fig, inner[j])
            ax.plot(freq, afft_new, label = 'SNR = ' + str(round(signal/noise, 2)))
            ax.set_title(direction)
            ax.axvline(x = thresh, label = 'signal-noise cutoff = ' + str(thresh), color = 'r')
            ax.set_xlabel('freq (Hz)')
            ax.set_ylabel('Power')
            ax.set_xlim([0, 3])
            ax.set_ylim([0, 70])
            ax.legend()
            fig.add_subplot(ax)
            
def onsetAnalysis(exp, title, timeRange = None, levels = [0, 30, 130]):
    if timeRange is not None:
        Amag = exp.Amag[timeRange[0]:timeRange[1]]
        Jmag = exp.Jmag[timeRange[0]:timeRange[1]]
    else:
        Amag = exp.Amag
        Jmag = exp.Jmag
        
    def plot1(content, stat, std, levels = [0, 30, 130], colors = 'rgby'):
        while len(levels) > len(colors):
            colors += 'k'
        plt.plot(content, label = 'magnitude')
        if levels is not 'auto':
            for factor, c in zip(levels, colors[:len(levels)]):
                plt.axhline(y = stat + factor * std, color = c, ls = '--', label = 'avg + ' + str(factor) + ' * std')
        else:
            plt.axhline(y = np.mean(content), label = 'average is ' + str(int(np.mean(content)/std)) + 'stds from resting', color = 'r')
        plt.legend(loc = 'upper right')
            
    
    statA, stdA = np.mean(exp.Amag[3000:]), np.std(exp.Amag[3000:])
    statJ, stdJ = np.mean(exp.Jmag[3000:]), np.std(exp.Jmag[3000:])
    
    plt.figure(figsize = (10, 4))
    plt.subplot(1, 2, 1)
    plt.title('Acceleration magnitude')
    plot1(Amag, statA, stdA, levels = levels)
    plt.subplot(1, 2, 2)
    plt.title('Jerk magnitude')
    plot1(Jmag, statJ, stdJ, levels = levels)
    plt.suptitle(title)

def rotAnalysis(exp, title, content):
    if content is 'accel':  
        data, mu, sigma = exp.Amag, np.mean(exp.Amag), np.std(exp.Amag)
    else:
        data, mu, sigma = exp.Jmag, np.mean(exp.Jmag), np.std(exp.Jmag)
    plt.plot(data)
    plt.axhline(y = mu, label = 'mu = ' + str(round(mu, 2)), color = 'r')
    plt.axhline(y = mu + sigma, label = 'mu + sigma = ' + str(round(mu + sigma, 2)), color = 'g')
    plt.axhline(y = mu - sigma, label = 'mu - sigma = ' + str(round(mu - sigma, 2)), color = 'g')
    plt.legend(loc = 'upper right')
    plt.title(title)
    print(title, content, mu, sigma)
    
def magAnalysis(exp, content, title):
    windowLength = 30
    
    if content is 'accel':
        data = exp.Amag
        yFiltBounds = [1.02, 1.07]
        yStds = 1.069
        yRatio = yStds - 0.003
        
    else:
        data = exp.Jmag
        yFiltBounds = [0, 7]
        yStds = 6.9
        yRatio = yStds - 0.4
    filtered = runningAvg(data, windowLength)
    
    
#    plt.figure(figsize= (9, 7))
#    plt.subplot(4, 1, 1)
#    plt.plot(data)
#    plt.title('(a) ' + title)
#    
#    filtered = runningAvg(data, windowLength)
#    plt.subplot(4, 1, 2)
#    plt.plot(filtered)
#    plt.title('(b), winLength = ' + str(windowLength))
#    
#    plt.subplot(4, 1, 3)
#    plt.plot(filtered)
#    cutoffs = [110, 600, 810, 1100, 1300, len(filtered)]
#    for item in cutoffs:
#        plt.axvline(x = item, ls = '--')
#    plt.title('(c)')
#    stds = []
#    newData = np.ones(len(filtered)) * np.mean(filtered)
#    for i in np.arange(0, len(cutoffs) - 1, 2):
#        activityRange = filtered[cutoffs[i]: cutoffs[i + 1]]
#        newData[cutoffs[i]:cutoffs[i+1]] = activityRange
#        stds.append(np.std(activityRange))
#    
#    plt.subplot(4, 1, 4)
#    plt.plot(newData)
#    x = 330
#    for item in stds:
#        plt.text(x, yStds, 'std = ' + str(round(item, 4)))
#        plt.text(x, yRatio, 'ratio = ' + str(round(item/stds[0], 4)))
#        x += 500
#    plt.title('(d)')
#    plt.tight_layout()
    
    plt.figure(figsize = (9, 7))
    plt.subplot(2, 1, 1)
    plt.plot(data)
    plt.xlabel('time (10ms)')
    plt.ylabel(content + ' magnitude')
#    plt.title('(a) ' + title)
    
    plt.subplot(2, 1, 2)
    plt.plot(filtered)
    plt.ylim(yFiltBounds)
    cutoffs = [110, 600, 810, 1100, 1300, len(filtered)]
    for item in cutoffs:
        plt.axvline(x = item, ls = '--')
    stds = []
    cutoffAvg = []
    newData = np.ones(len(filtered)) * np.mean(filtered)
    for i in np.arange(0, len(cutoffs) - 1, 2):
        activityRange = filtered[cutoffs[i] : cutoffs[i + 1]]
        newData[cutoffs[i] : cutoffs[i + 1]] = activityRange
        cutoffAvg.append(np.mean([cutoffs[i], cutoffs[i + 1]]))
        stds.append(np.std(activityRange))
#    xLoc = 330
    for item, xLoc in zip(stds, cutoffAvg):
        plt.text(xLoc, yStds, 'std: ' + str(round(item, 4)), horizontalalignment = 'center', verticalalignment = 'top')
        plt.text(xLoc, yRatio, 'ratio: ' + str(round(item/stds[0], 4)), horizontalalignment = 'center', verticalalignment = 'top')
        xLoc += 500
    plt.xlabel('time (10ms)')
    plt.ylabel(content + ' magnitude')
    plt.tight_layout()

def runningAvg(data, winLength):
    newData = []
    i = 0
    while i + winLength < len(data):
        newData.append(np.mean(data[i : i + winLength]))
        i += 1
    return newData

def schmittOne(data, trig):
    output = np.zeros(len(data))
    for i, item in enumerate(data):
        if item > trig:
            output[i] = 1
    return output

# I think I got it?! 
def schmittTwo(data, trigLow, trigHigh, offset = 0, factor = 1):
    plt.figure()
    output = np.zeros(len(data))
    trigged = False 
    for i, item in enumerate(data):
        if not trigged and item > trigHigh:
            trigged = True
#            plt.axvline(x = i, color = 'g')
        if trigged and item < trigLow:
            trigged = False
#            plt.axvline(x = i, color = 'b')
        if trigged:
            output[i] = 1

    plt.plot(data)
    plt.plot(output* factor + offset)
    plt.axhline(y = trigLow, color = 'r')
    plt.axhline(y = trigHigh, color = 'r')
            
    return output

plt.close('all')
exp1 = jerk('Linear2Raw.csv')
exp2 = jerk('Linear3Raw.csv')
exp3 = jerk('Linear4Raw.csv')
slowRotation = jerk('slowRotationRaw.csv')
fastRotation = jerk('fastRotationRaw.csv')
onset = jerk('onsetRaw.csv')
mag = jerk('differentMagRaw.csv', timeRange = [2000, 3500])
veryslowRotation = jerk('veryslowRotationRaw.csv')


#SNR comparison
#fftSNR(exp1.raw, 'Exp1 raw accel (a)')
#fftSNR(exp1.jerk, 'Exp1 raw jerk (b)')

#onsetAnalysis(onset, 'Onset comparison')
#onsetAnalysis(onset, 'Onset comparison, slow', timeRange = [200, 800], levels = 'auto')



#plt.figure(figsize= (10, 4))
#plt.subplot(1, 2, 1)
#plt.plot(mag.Amag)
#plt.title('Acceleration at 3 speeds')
#plt.subplot(1, 2, 2)
#plt.plot(mag.Jmag)
#plt.title('Jerk at 3 speeds')

#finds the running average of a one-D data set

def magRatio(exp, content, trigLow, trigHigh, winLength = 172, offset = 0, factor = 1):
    if content is 'accel':
        data = exp.Amag
    else:
        data = exp.Jmag
    runAvg = runningAvg(data, winLength)
    event = schmittTwo(runAvg, trigLow, trigHigh, offset = offset, factor = factor)
    
    interval = []
    found = False
    for i, x in enumerate(event):
        if not found and x == 1:
            found = True
            interval.append(i)
        if found and x == 0:
            found = False
            interval.append(i - 1)
    if event[-1] == 1:
        interval.append(len(event))
            
    output = []        
    for i in np.arange(0, len(interval) - 1, 2):
        output.append(sum(runAvg[interval[i]: interval[i+ 1]]))
    return interval, output, [output[0]/item for item in output]



magAnalysis(mag, 'accel', 'Acceleration at 3 magnitudes')
magAnalysis(mag, 'jerk', 'Jerk at 3 magnitudes')
    

#plt.figure()
#total = 8
#lengths = np.linspace(10, 200, total)
#for i in range(total):
#    plt.subplot(4, 2, i + 1)
#    plt.plot(runningAvg(mag.Jmag, int(lengths[i])))
#    plt.title('length = ' + str(int(lengths[i])))
#plt.suptitle('Filtered Jerk at different window lengths')
#plt.tight_layout()


#plt.figure(figsize = (10, 8))
#plt.subplot(2, 2, 1)
#rotAnalysis(slowRotation, 'slow rotation accel', 'accel')
#plt.subplot(2, 2, 2)
#rotAnalysis(slowRotation, 'slow rotation jerk', 'jerk')
#plt.subplot(2, 2, 3)
#rotAnalysis(fastRotation, 'fast rotation accel', 'accel')
#plt.subplot(2, 2, 4)
#rotAnalysis(fastRotation, 'fast rotation jerk', 'jerk')
#plt.suptitle('Rotation comparison \n slow vs. fast, accel vs. jerk')




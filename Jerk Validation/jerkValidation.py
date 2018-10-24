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

class jerk:
    def __init__(self, file, applyButter = False):
        self.raw = self.readFile(file, applyButter)
        self.jerk = self.findJerk()
        self.avg = np.mean(self.raw, axis = 0)
        self.std = np.std(self.raw, axis = 0)
        self.filtered = applyButter
    def readFile(self, file, applyButter):
            
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
            return butterworthFilt(np.array(UTV))
        else: 
            return np.array(UTV)
    def findJerk(self):
        jerk = []
        for i in range(len(self.raw) - 1):
            jerk.append(np.subtract(self.raw[i + 1], self.raw[i]))
        jerk = np.multiply(jerk, 100)
        return jerk


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

def snr(a, axis = 0, ddof = 0):
    #This is not a very good way to find SNR. 
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    SNR = np.where(sd == 0, 0, m/sd)
    plt.bar(['x', 'y', 'z'], abs(SNR))
    return SNR

def fftSNR(a, title, thresh = 1.5):
    plt.figure()
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
        plt.axvline(x = thresh, label = 'signal-noise cutoff', color = 'r')
        plt.axis([0, 3, 0, 70])
        plt.xlabel('freq (Hz)')
        plt.ylabel('Power')
        plt.legend()
        
    plt.suptitle(title)
    
plt.close('all')
horG = jerk('Horizontal_gRaw.csv')
slaG = jerk('Slanted_gRaw.csv')
linear = jerk('LinearRaw.csv')


#plt.figure()
#plt.subplot(2, 1, 1)
#for i, direction, oneC in zip(range(3), 'xyz', 'gkr'):
#    plt.plot(linear.raw[:, i], label = direction, alpha = 0.5, color = oneC)
#    plt.plot(slaG.raw[:, i], '-', color = oneC)
#plt.legend(loc = 'best')
#plt.subplot(2, 1, 2)
#plt.title('SNR of unfiltered accelerometry')
#snr(linear.raw)
#plt.suptitle('Unfiltered - linear motion vs. gravity')

#    
filtered = butterworthFilt(linear.raw)
#plt.figure()
#plt.subplot(2, 1, 1)
#for i, direction, oneC in zip(range(3), 'xyz', 'gkr'):
#    plt.plot(filtered[:, i], label = direction, alpha = 0.5, color = oneC)
#plt.legend(loc = 'best')
#plt.subplot(2, 1, 2)
#plt.title('SNR of filtered accelerometry')
#snr(filtered)
#plt.suptitle('BPFed(0.25~2) - linear motion vs. gravity')


#
subtracted = np.subtract(linear.raw, slaG.avg)
#plt.figure()
#for i, direction, oneC in zip(range(3), 'xyz', 'gkr'):
#    plt.subplot(4, 1, i + 1)
#    plt.title(direction)
#    plt.plot(subtracted[:, i], alpha = 0.5, color = oneC)
#plt.subplot(4, 1, 4)
#snr(subtracted)
#plt.title('SNR of linear motion mius gravity')
#plt.suptitle('unfiltered - linear motion minus avg. gravity')


#
#plt.figure()
#for i, direction, oneC in zip(range(3), 'xyz', 'gkr'):
#    plt.subplot(3, 1, i + 1)
#    plt.title(direction)
#    plt.plot(linear.jerk[:, i], alpha = 0.5, color = oneC)
#    plt.ylim(-20, 20)
#plt.suptitle('jerk')
#
filteredJerk = butterworthFilt(linear.jerk)
#plt.figure()
#for i, direction, oneC in zip(range(3), 'xyz', 'gkr'):
#    plt.subplot(3, 1, i + 1)
#    plt.title(direction)
#    plt.plot(filteredJerk[:, i], alpha = 0.5, color = oneC)
#    plt.ylim(-1, 1)
#plt.suptitle('filtered jerk')

#SNR comparison
fftSNR(linear.raw, 'raw accel')
fftSNR(filtered, 'filtered accel')
fftSNR(subtracted, 'linear - gravity')
fftSNR(linear.jerk, 'raw jerk')
fftSNR(filteredJerk, 'filtered jerk')

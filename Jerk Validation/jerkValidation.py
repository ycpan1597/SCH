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
    def __init__(self, file, applyButter = False):
        self.raw, self.Amag = self.readFile(file, applyButter)
        self.jerk, self.Jmag = self.findJerk()
        self.avg = np.mean(self.raw, axis = 0)
        self.std = np.std(self.raw, axis = 0)
        self.filtered = applyButter
    
    def matMag(self, mat): 
        mag = []
        for row in mat:
            mag.append(math.sqrt(row[0]**2 + row[1]**2+ row[2]**2))
        return mag
    
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
            return butterworthFilt(np.array(UTV)), butterworthFilt(np.array(self.matMag(UTV)))
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
        plt.axvline(x = thresh, label = 'signal-noise cutoff = ' + str(thresh), color = 'r')
        if axis is 'fixed':
            plt.axis([0, 3, 0, 70])
        plt.xlabel('freq (Hz)')
        plt.ylabel('Power')
        plt.legend()
        
    plt.suptitle(title)

# takes a 3D array and plots 3 subplots, one direction in each    
def plotSignal(signal, title):
    plt.figure()
    for i, c, direction in zip(range(3), 'gkr', 'xyz'):
        plt.subplot(3, 1, i + 1)
        plt.plot(signal[:, i], color = c)
        plt.title(direction)
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
            
def AvsJ(exp, title, timeRange = None):
    if timeRange is not None:
        Amag = exp.Amag[timeRange[0]:timeRange[1]]
        Jmag = exp.Jmag[timeRange[0]:timeRange[1]]
    else:
        Amag = exp.Amag
        Jmag = exp.Jmag
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(Amag, label = 'std = ' + str(round(np.std(Amag), 2)))
    plt.axhline(y = np.mean(Amag), color = 'r', label = 'Accel Avg')
    plt.legend()
    plt.title('Acceleration magnitude')
    plt.subplot(2, 1, 2)
    plt.plot(Jmag, label = 'std = ' + str(round(np.std(Jmag), 2)))
    plt.axhline(y = np.mean(Jmag), color = 'r', label = 'Accel Avg')
    plt.legend()
    plt.title('Jerk magnitude')
    plt.suptitle(title)

    
plt.close('all')
horG = jerk('Horizontal_gRaw.csv')
slaG = jerk('Slanted_gRaw.csv')
linear = jerk('LinearRaw.csv')
exp1 = jerk('Linear2Raw.csv')
exp2 = jerk('Linear3Raw.csv')
exp3 = jerk('Linear4Raw.csv')
rotation = jerk('rotationRaw.csv')
onset = jerk('onsetRaw.csv')
mag = jerk('magnitudeRaw.csv')

G = horG.avg
Gs = slaG.avg #slanged gravity

#plotSignal(exp1.raw, 'exp1 raw accel time domain')
#plotSignal(exp1.jerk, 'jerk')
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


#SNR comparison
#fftSNR(exp1.raw, 'Exp1 raw accel (a)')
#fftSNR(exp1.jerk, 'Exp1 raw jerk (b)')


AvsJ(onset, 'Onset comparison')
#AvsJ(rotation, 'Rotation comparison')
#AvsJ(mag, 'Moderate motion', [0, 3000])
#AvsJ(mag, 'Vigorous motion', [4000, -1])

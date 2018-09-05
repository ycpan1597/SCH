#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:34:44 2018

@author: Preston Pan
"""
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
Fs = 100               # Sampling Freq (in Hz)
T = 1/Fs               # Sampling Period (in sec)
L = 5                  # Length of Signal (in sec)
nfft = L * Fs          # Number of Points
t = np.arange(0, L, T) # Time Vector
freq = 3               # Signal Freq (in Hz)

x = np.sin(freq * 2 * np.pi * t) + 0.5 * np.sin(5 * 2 * np.pi * t + 1.3 * np.pi)
X = np.fft.fft(x, nfft, norm = 'ortho')
X_half = X[0: int(nfft/2 - 1)]
f = np.arange(0, nfft/2 - 1, 1) * Fs/nfft
plt.figure()
plt.stem(f, abs(X_half))

plt.figure()
plt.plot(t, x)
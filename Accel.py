# Work in progress
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.decomposition import PCA
from scipy import signal

class Accel:
# Abbreviation list:
    # UTV = UntrimOnemed Vector, UTM = UntrimOnemed Magnitude
    # DA = Dominant Arm

    FIRST_LINE = 11
    DURATION = '1sec'
    EXT = '.csv' #file extension
    def __init__(self, filename, os, filetype, epochLength = 60, applyButter = True, status = 'Awake'):
        self.os = os
        self.filetype = filetype
        self.status = status
        if self.canRun():
            start = time.time()
            self.filename = filename
            self.filenameList = self.makeNameList(filename) #by doing so, you have created the filenameList array
            self.titles = self.makeTitleList()
            self.UTV, self.UTM, self.DA = self.readAll(applyButter)
            self.jerk = self.findJerk()
            self.dp, self.cov, self.weightedDots, self.AI, self.VAF = self.findPCMetrics(epochLength) #cov = coefficient of variationv
            end = time.time()
            print('total time to read ' + self.filename + ' (' + status + ')' + ' = ' + str(end - start))        
        else:
            print('Sorry; this program can\'t run ' + self.filetype + ' on ' + self.os)
#        
    def __str__(self):
        return self.filename + ' (' + self.status + ')'
    def __repr__(self):
        return self.__str__()
    
    def canRun(self):
        if self.os == 'Mac' and self.filetype == 'Raw':
            return False
        else:
            return True
    
    def makeSlash(self):
        if self.os == 'Baker':
            return '\\'
        else:    
            return '/'
    
    def makeNameList(self, filename): 
        if self.os == 'Baker':
            if self.filetype == 'Raw': 
                directory = 'E:\\Projects\\Brianna\\' #can only be run on Baker
                baseList = ['_v1_LRAW', '_v1_RRAW', '_v2_LRAW', '_v2_RRAW', '_v3_LRAW', '_v3_RRAW']
                baseList = [directory + filename + item + Accel.EXT for item in baseList]
            else: # Baker Epoch
                directory = 'C:\\Users\\SCH CIMT Study\\SCH\\' # for Baker
                baseList = ['_v1_L', '_v1_R', '_v2_L', '_v2_R', '_v3_L', '_v3_R']
                baseList = [directory + Accel.DURATION + self.makeSlash() + filename + item + Accel.DURATION + Accel.EXT for item in baseList]
            if self.status == 'Sleep':
                baseList.append('C:\\Users\\SCH CIMT Study\\SCH\\Timing File\\' + filename + '_Sleep' + Accel.EXT)
            else: # Baker Epoch Awake
                baseList.append('C:\\Users\\SCH CIMT Study\\SCH\\Timing File\\' + filename + Accel.EXT)
        else: # Mac
            # Mac Raw has been excluded
            # Mac Epoch (Sleep or Awake)
            directory = '/Users/preston/SCH/' # for running on my laptop
            baseList = ['_v1_L', '_v1_R', '_v2_L', '_v2_R', '_v3_L', '_v3_R']
            baseList = [directory + Accel.DURATION + self.makeSlash() + filename + item + Accel.DURATION + Accel.EXT for item in baseList]
            if self.status == 'Sleep': # Mac Epoch Sleep
                baseList.append('/Users/preston/SCH/Timing File/' + filename + '_Sleep' + Accel.EXT)
            else: # Mac Epoch Awake
                baseList.append('/Users/preston/SCH/Timing File/' + filename + Accel.EXT)
        return baseList
    
    def makeTitleList(self):
        return ['Pre Left', 'Pre Right', 'During Left', 'During Right', 'Post Left', 'Post Right']
    
    def makeSuperTitle(self):
        if (self.DA == 1):
            pareticArm = 'left'
        else:
            pareticArm = 'right'
        return 'Subject: ' + self.filename + '\nparetic/nondominant arm = ' + pareticArm  
    
    def readAll(self, applyButter):
        # Define all relevant subfunctions
        def readTiming():
            timing = []
            with open(self.filenameList[-1], 'r', encoding = 'utf-8') as timingFile:
                timingReader = csv.reader(timingFile)
                for row in timingReader:
                    timing.append([int(i) for i in row[1:] if i != ''])
            return timing
        
        def readOne(file, activeRanges):
            
            #takes in a 1 by X by 3 matrix of signal and coverts it into an 1 by X array of magnitude
            def magnitude(mat): 
                mag = []
                for row in mat:
                    mag.append(math.sqrt(row[0]**2 + row[1]**2+ row[2]**2))
                return mag
            
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
                filtedX = signal.filtfilt(b, a, data[:, 0])
                filtedY = signal.filtfilt(b, a, data[:, 1])
                filtedZ = signal.filtfilt(b, a, data[:, 2])
                return np.array([list(a) for a in zip(filtedX, filtedY, filtedZ)])
            
            UTV2 = np.empty((0, 3)) #stores the sliced data
            if self.filetype == 'Epoch':
                df = pd.read_csv(file, header = 10, usecols = ['Axis1', 'Axis2', 'Axis3'])
                for bounds in activeRanges:
                    UTV2 = np.vstack((UTV2, df.iloc[bounds[0] : bounds[1]].values))
            else:
                df = pd.read_csv(file, header = 10, usecols = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z'])
                for bounds in activeRanges:
                    UTV2 = np.vstack((UTV2, df.iloc[bounds[0] * 100 : bounds[1] * 100].values))
            
            if applyButter:
                return butterworthFilt(np.array(UTV2)), np.array(magnitude(UTV2)) #return it as an numpy array at the end
            else:
                return np.array(UTV2), np.array(magnitude(UTV2))
        UTV = []
        UTM = []
        infoArray = readTiming()
        DA = infoArray[-1][0]
        
        for file in self.filenameList[0:6]:
            if 'v1' in file:
                activeRanges = np.array(infoArray[0])
            elif 'v2' in file:
                activeRanges = np.array(infoArray[1])
            else: #v3
                activeRanges = np.array(infoArray[2])
            activeRanges = activeRanges.reshape(int(len(activeRanges) / 2), 2)
            oneUTV, oneUTM = readOne(file, activeRanges)
            UTV.append(oneUTV)
            UTM.append(oneUTM)
        return np.array(UTV), np.array(UTM), DA #not going to trim it further

    def findJerk(self):
        jerk = [[] for i in range(len(self.UTV))]
        for i in range(len(self.UTV)):
            for j in range(len(self.UTV[i]) - 1):
#                jerk[i] = np.append(jerk[i], np.subtract(self.UTV[i][j + 1], self.UTV[i][j]))
                jerk[i].append(np.subtract(self.UTV[i][j + 1], self.UTV[i][j]).tolist())
            jerk[i] = np.array(jerk[i])
        if self.filetype == 'Raw':
            jerk = np.multiply(jerk, 100) # Raw samples at 100Hz
        return np.array(jerk)
    # Finds PC1 within each epoch which is of length "window" for both left and right arm,
    # finds the dot product and generates AI (Alignment Index) and COV (Coefficient of Variation)          
    def findPCMetrics(self, window = 60, VAFonly = False):
        if self.filetype == 'Raw':
            window = window * 100
        pca = PCA(1)
        dotProducts = []
        endpointsLens = []
        AI = [] 
        std = []
        mu = []
        VAF = [] #Variance Accounted For
        for i in np.arange(0, 5, 2):
            endpoints = np.arange(0, len(self.UTV[i]) + window, window)
            endpointsLens = len(endpoints)
            curDot = []
            curVAF = []
            curDotSum = 0
            for j in range(len(endpoints) - 1):
                pca.fit_transform(self.UTV[i][endpoints[j]: endpoints[j + 1]])
                leftPC1 = pca.components_[0]
                leftPC1VAF = pca.explained_variance_ratio_[0]
                pca.fit_transform(self.UTV[i + 1][endpoints[j]: endpoints[j + 1]])
                rightPC1 = pca.components_[0]
                rightPC1VAF = pca.explained_variance_ratio_[0]
                curVAF.append(leftPC1VAF)
                curVAF.append(rightPC1VAF)
                absDot = abs(np.dot(leftPC1, rightPC1))
                curDot.append(absDot)
                curDotSum += absDot
            AI.append(np.divide(curDotSum, endpointsLens))
            std.append(np.std(curDot))
            mu.append(np.mean(curDot))
            dotProducts.append(curDot)
            VAF.append(curVAF)
        if VAFonly:
            return VAF
        else: 
            weightedDots = []
            for dot in dotProducts:
                freqArr, endpointArr = np.histogram(dot, bins = 10)
                
                avgEndpoint = []
                for i in range(len(endpointArr) - 1):
                    avgEndpoint.append(np.mean((endpointArr[i], endpointArr[i + 1])))
                weightedDot = 0
                for oneFreq, oneEndpoint in zip(freqArr, avgEndpoint):
                    weightedDot += oneFreq/sum(freqArr) * oneEndpoint
                weightedDots.append(weightedDot)    
            return np.array(dotProducts), np.divide(std, mu), weightedDots, AI, VAF
    # ECDF: Empirical cumulative distribution function
    # goal is to create a CDF for each of the 6 dataset. I don't exactly know how
    # to use this yet but I think creating these graphs will help - I did it but now what?
    def ECDF(self, n = 30, kind = 'mag', inverse = False, threshold = 0.9):
        
        if kind == 'mag':
            plt.figure()
            plt.title(self.__str__())
            for i in range(len(self.UTM)):
                freq, bins = np.histogram(self.UTM[i], bins = n)
                cumulativeFreq = [0] * n
                for k in range(n):
                    cumulativeFreq[k] = sum(freq[0:k])/sum(freq)
                if inverse:    
                    plt.plot(cumulativeFreq, np.linspace(min(self.UTM[i]), max(self.UTM[i]), n), label = self.titles[i])
                    plt.xlabel('Probability')
                    plt.ylabel('Magnitude')
                    plt.axvline(x = threshold)
                else:
                    plt.plot(np.linspace(min(self.UTM[i]), max(self.UTM[i]), n), cumulativeFreq, label = self.titles[i])
                    plt.xlabel('Magnitude')
                    plt.ylabel('Probability')
                    plt.axhline(y = threshold)
            plt.legend()
            plt.grid()
            
        elif kind == 'vector':
            for i in range(len(self.UTV)):
                plt.figure()
                plt.title('Cumulative Mass Fuction')
                for j in range(3): 
                    freq, bins = np.histogram(self.UTV[i][:, j], bins = n)
                    cumulativeFreq = [0] * n
                    for k in range(n):
                        cumulativeFreq[k] = sum(freq[0:k])/sum(freq)
                    if inverse:
                        plt.plot(cumulativeFreq, np.linspace(min(self.UTV[i][:, j]), max(self.UTV[i][:, j]), n))
                        plt.axis([-0.2, 1.2, -200, 200])       
                        plt.xlabel('Probability')
                        plt.ylabel('Magnitude')
                    else:
                        plt.plot(np.linspace(min(self.UTV[i][:, j]), max(self.UTV[i][:, j]), n), cumulativeFreq)
                        plt.axis([-200, 200, -0.2, 1.2])
                        plt.xlabel('Magnitude')
                        plt.ylabel('Probability')
                    plt.legend(['x', 'y', 'z'])
                    
        else:
            print('kind must be either mag or vector')
    # The function "consistency" compares all vectors to a reference vector and
    # quantifies how consistent a set of vectors is. This is used to compare
    # sleep vs. awake periods (sleep is expected to have higher consistency)
    def consistency(self, ref = [1, 0, 0]):
        
        def findMag(vec):
            return math.sqrt(vec[0]**2 + vec[1]**2+ vec[2]**2)
        
        dotProducts = [[] for i in range(6)]
        
        plt.figure()
        plt.title(self.__str__() + ' ref = [' + ''.join(str(e) + ',' for e in ref)[:-1] + ']') 
        #I used [:-1] to exclude the very last comma (a hack to solve the light-post problem)
        for i in range(len(self.UTV)):
            data = self.UTV[i]
            for entry in data:
                dotProducts[i].append(np.divide(np.dot(ref, entry), findMag(entry)))
            plt.hist(dotProducts[i], bins = 30, alpha = 0.3)
            plt.xlabel('Range of normalized dot product')
            plt.ylabel('Number of Occurences')
                
                
        
    
    
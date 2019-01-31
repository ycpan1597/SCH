# Work in progress
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy import signal

class Accel:
# Abbreviation list:
    # UTV = UntrimOnemed Vector, UTM = UntrimOnemed Magnitude
    # DA = Dominant Arm

    FIRST_LINE = 11
    DURATION = '1sec'
    EXT = '.csv' #file extension
<<<<<<< HEAD
    def __init__(self, filename, OS, filetype, epochLength = 60, applyButter = True, status = 'awake', numFiles = 6, selectRange = True):
        self.OS = OS
        self.filetype = filetype
        self.status = status
        self.numFiles = numFiles
        if self.canRun():
            start = time.time()
            self.filename = filename
            self.filenameList = self.makeNameList(filename, numFiles) #by doing so, you have created the filenameList array
            self.titles = self.makeTitleList(numFiles)
            self.UTV, self.UTM, self.DA, self.age = self.readAll(applyButter, numFiles, selectRange)
            self.jerk, self.jerkMag = self.findJerk()
            self.sumVec, self.binAvg, self.mass = self.jerkRatio(cutoff = 3)
            self.UR, self.activeVec = self.findActiveDuration()
#            self.dp, self.cov, self.weightedDots, self.AI, self.VAF = self.findPCMetrics(epochLength) #cov = coefficient of variationv
            end = time.time()
            print('total time to read ' + self.filename + ' (' + status + ')' + ' = ' + str(end - start))        
=======
    def __init__(self, filename, filetype, OS = 'Baker', applyButter = True, numFiles = 6, status = 'awake', direct = False):
        if (direct):
            self.UTV, self.UTM = self.directRead(filename)
>>>>>>> 9e2b5af4d783a8ad44dc764a5cab10c1973d2107
        else:
            self.OS = OS
            self.filetype = filetype
            self.status = status
            self.numFiles = numFiles
            if self.canRun():
                start = time.time()
                self.filename = filename
                self.filenameList = self.makeNameList(filename, numFiles) #by doing so, you have created the filenameList array
                self.titles = self.makeTitleList(numFiles)
                self.UTV, self.UTM, self.DA, self.age = self.readAll(applyButter, numFiles)
                self.jerk, self.jerkMag = self.findJerk()
                self.sumVec, self.binAvg, self.mass = self.jerkRatio(cutoff = 3)
                self.UR = self.findUseRatio()
                end = time.time()
                print('total time to read ' + self.filename + ' (' + status + ')' + ' = ' + str(end - start))        
            else:
                print('Sorry; this program can\'t run ' + self.filetype + ' on ' + self.OS)
    def directRead(self, filename):
        UTV2 = []
        df = pd.read_csv(filename, header = 10, usecols = ['x', 'y', 'z'])
        UTV2.append(df.iloc[0:].values)
        UTV2 = UTV2[0]
        
    def __str__(self):
        return self.filename + ' (' + self.status + ')'
    def __repr__(self):
        return self.__str__()
    
    def canRun(self):
        if self.OS == 'Mac' and self.filetype == 'Raw':
            return False
        else:
            return True
    
    def makeSlash(self):
        if self.OS == 'Baker':
            return '\\'
        else:    
            return '/'
    
    def makeNameList(self, filename, numFiles): 
        if self.OS == 'Baker':
            if self.filetype == 'Raw': 
                directory = 'E:\\Projects\\Brianna\\' #can only be run on Baker
                baseList = ['_v1_LRAW', '_v1_RRAW', '_v2_LRAW', '_v2_RRAW', '_v3_LRAW', '_v3_RRAW']
#                baseList = [directory + filename + item + Accel.EXT for item in baseList]
                baseList = [directory + filename + baseList[i] + Accel.EXT for i in range(numFiles)]
            else: # Baker Epoch
                directory = 'C:\\Users\\SCH CIMT Study\\SCH\\' # for Baker
                baseList = ['_v1_L', '_v1_R', '_v2_L', '_v2_R', '_v3_L', '_v3_R']
#                baseList = [directory + Accel.DURATION + self.makeSlash() + filename + item + Accel.DURATION + Accel.EXT for item in baseList]
                baseList = [directory + Accel.DURATION + self.makeSlash() + filename + baseList[i] + Accel.DURATION + Accel.EXT for i in range(numFiles)]
            if self.status == 'asleep':
                baseList.append('C:\\Users\\SCH CIMT Study\\SCH\\Timing File\\' + filename + '_asleep' + Accel.EXT)
            else: # Baker Epoch Awake
                baseList.append('C:\\Users\\SCH CIMT Study\\SCH\\Timing File\\' + filename + Accel.EXT)
        else: # Mac
            # Mac Raw has been excluded
            # Mac Epoch (asleep or Awake)
            directory = '/Users/preston/SCH/' # for running on my laptop
            baseList = ['_v1_L', '_v1_R', '_v2_L', '_v2_R', '_v3_L', '_v3_R']
#            baseList = [directory + Accel.DURATION + self.makeSlash() + filename + item + Accel.DURATION + Accel.EXT for item in baseList]
            baseList = [directory + Accel.DURATION + self.makeSlash() + filename + baseList[i] + Accel.DURATION + Accel.EXT for i in range(numFiles)]
            if self.status == 'asleep': # Mac Epoch asleep
                baseList.append('/Users/preston/SCH/Timing File/' + filename + '_asleep' + Accel.EXT)
            else: # Mac Epoch Awake
                baseList.append('/Users/preston/SCH/Timing File/' + filename + Accel.EXT)
        return baseList
    
    def makeTitleList(self, numFiles):
        base = ['Pre Left', 'Pre Right', 'During Left', 'During Right', 'Post Left', 'Post Right']
        return base[:numFiles]
    def makeSuperTitle(self):
        if (self.DA == 1):
            pareticArm = 'left'
        else:
            pareticArm = 'right'
        return 'Subject: ' + self.filename + '\nparetic/nondominant arm = ' + pareticArm  
    
#takes in a 1 by X by 3 matrix of signal and coverts it into an 1 by X array of magnitude
    def matMag(self, mat): 
        mag = []
        for row in mat:
            mag.append(math.sqrt(row[0]**2 + row[1]**2+ row[2]**2))
        return mag
    
    def readAll(self, applyButter, numFiles, selectRange):
        # Define all relevant subfunctions
        def readTiming():
            timing = []
            with open(self.filenameList[-1], 'r', encoding = 'utf-8') as timingFile:
                timingReader = csv.reader(timingFile)
                for row in timingReader:
                    timing.append([int(i) for i in row[1:] if i != ''])
            return timing
        
        def readOne(file, activeRanges):
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
            
            UTV2 = []
            if self.filetype == 'Epoch':
                df = pd.read_csv(file, header = 10, usecols = ['Axis1', 'Axis2', 'Axis3'])
                if selectRange:
                    for bounds in activeRanges:
                        UTV2.append(df.iloc[bounds[0] : bounds[1]].values)
##           Issue: without selecting the range, the total sample duration of the left and right might be a little off 
#                else:
#                        UTV2.append(df.iloc[0:].values)
                    
            else:
                df = pd.read_csv(file, header = 10, usecols = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z'])
                for bounds in activeRanges:
                    UTV2.append(df.iloc[bounds[0] * 100 : bounds[1] * 100].values)
            UTV2 = UTV2[0]
            
            if applyButter:
                return butterworthFilt(np.array(UTV2)), np.array(self.matMag(UTV2)) #return it as an numpy array at the end
            else:
                return np.array(UTV2), np.array(self.matMag(UTV2))
        UTV = []
        UTM = []
        infoArray = readTiming()
        DA = infoArray[-2][0]
        age = infoArray[-1][0]
        
        for file in self.filenameList[0:numFiles]:
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
        return np.array(UTV), np.array(UTM), DA, age #not going to trim it further

    def findJerk(self):
        jerk = [[] for i in range(len(self.UTV))]
        jerkMag = [[] for i in range(len(self.UTV))]
        for i in range(len(self.UTV)):
            for j in range(len(self.UTV[i]) - 1):
#                jerk[i] = np.append(jerk[i], np.subtract(self.UTV[i][j + 1], self.UTV[i][j]))
                jerk[i].append(np.subtract(self.UTV[i][j + 1], self.UTV[i][j]).tolist())
            jerkMag[i] = np.array(self.matMag(jerk[i]))
            jerk[i] = np.array(jerk[i])
        if self.filetype == 'Raw':
            jerk = np.multiply(jerk, 100) # Raw samples at 100Hz
        return np.array(jerk), np.array(jerkMag)
    
    def jerkHist(self):

        	for i in range(len(self.jerkMag)):
        		plt.figure()
        		plt.title(self.titles[i])
        		plt.hist(self.jerkMag[i], bins = 30, label = self.titles[i], alpha = 0.3)
        		plt.axis([0, 10, 0, 150000])
        	plt.show()
    
    def compareJerk(self):
        result = [[] for i in range(3)]
        j = 0
        for i in np.arange(0, 5, 2):
            left = self.jerkMag[i]
            right = self.jerkMag[i + 1]
            if self.DA == 1:
                ratio = np.divide(left, right)
            else: 
                ratio = np.divide(right, left)
            result[j].append(ratio)
            j += 1
        plt.figure()
        plt.title(self.__str__())
        j = 0
        for collections in result:
            plt.hist(collections, bins = np.linspace(0, 3, 50), alpha = 0.5, density = True, label = self.titles[j] + '/' + self.titles[j + 1])
            plt.ylim(0, 2.0)
            plt.legend()
            j += 2
        return result
    '''
    Michael's Ratio (MR) is explained as the following:
       D = Dominant, N = Non-dominant
       MR = abs(N) / (abs(N) + abs(D))
       
       Several cases:
           {1/2, abs(N) == abs(D) != 0}
           {(0, 1/2), abs(N) < abs(D)}
    MR:    {(1/2, 1), abs(N) > abs(D)}
           {NaN, abs(N) == abs(D) == 0}
           {0, abs(N) = 0}
           
       Comment: there are still about 10^4~10^5 NaNs when using activity count
       Comment: if the raw signal is NOT filtered, there are also some NaNs
    '''
    def jerkRatio(self, variable = 'Jerk', saveFig = False, numFiles = None, showPlot = False, cutoff = 3):
        

        def findMass(sumVec, binEdges, threshold = 0.5):
            #threshold should be between 0 and 1
            diff = binEdges[1] - binEdges[0]
            result = []
            for item in sumVec:
                mass = 0
                i = 0
                while binEdges[i] < threshold: 
                    mass += item[i] * diff
                    i += 1
                result.append(mass)
            return result
        def butterworthFilt(data, cutoff):
            filteredData =[]
            # user input
            order = 4
            fsampling = 100 #in Hz
            
            nyquist = fsampling/2 * 2 * np.pi #in rad/s
            cutoff = cutoff * 2 * np.pi #in rad/s
            b, a = signal.butter(order, cutoff/nyquist, 'lowpass')
            for item in data:
                filteredData.append(signal.filtfilt(b, a, item))
            return filteredData
        
        if numFiles is None:
            numFiles = self.numFiles
        MR = [[] for j in range(int(numFiles / 2))]
        
        if variable == 'ENMO':
            content = self.UTM
                
        elif variable == 'Jerk':
            content = self.jerkMag
            
        j = 0
        for i in np.arange(0, numFiles - 1, 2):
            if self.DA == 1:
                N = content[i] #left 
                D = content[i + 1] #right
            else:
                N = content[i + 1]
                D = content[i]
            MR[j] = np.divide(N, np.add(N, D))
            j += 1
        
#        sumvec = []
        histBins = np.linspace(0.1, 0.9, 200)
        sumVec = []
        if numFiles == 6:
            for oneSet in MR:
                counts = np.histogram(oneSet, bins = histBins, density = True)[0]
#                sumVec.append(np.divide(counts, sum(counts)))
                sumVec.append(counts)
        elif numFiles == 4:
            preN, binEdges = np.histogram(MR[0], histBins, density = True)
            plt.plot(0.5*(binEdges[1:] + binEdges[:-1]), preN, label = 'Pre', color = 'g')
            durN, binEdges = np.histogram(MR[1], histBins, density = True)
            plt.plot(0.5*(binEdges[1:] + binEdges[:-1]), durN, label = 'During', color = 'k')
            sumVec = [preN, durN]
        else:
            preN, binEdges = np.histogram(MR[0], histBins, density = True)
            plt.plot(0.5*(binEdges[1:] + binEdges[:-1]), preN, label = 'Pre', color = 'g')
            sumVec = [preN]
            
        binAvg = 0.5*(histBins[1:] + histBins[:-1])
        if cutoff != 0:  
            sumVec = butterworthFilt(sumVec, cutoff)
        mass = findMass(sumVec, binAvg)
        
        if showPlot:
            plt.figure()
            graphTitle = self.__str__() + ' - ' + self.filetype + ' - ' + variable
            plt.title(graphTitle)
            for oneColor, trialType, prob, oneMass in zip('gkr', ['Pre', 'During', 'Post'], sumVec, mass):
                plt.plot(binAvg, prob, label = trialType + ': %.3f' % oneMass + '% dominant arm', color = oneColor)         
                plt.fill_between(binAvg, prob * 100, where = binAvg < 0.5, alpha = 0.5, color = oneColor)
            plt.ylabel('Probability (%)')
            plt.xlabel(variable + " Ratio")
            plt.axvline(x = 0.5, ls = '--', label = '0.5 Bimanual', color = 'b')
            plt.legend(loc = 'upper right')
            
        if saveFig:
            location = 'C:\\Users\\SCH CIMT Study\\Desktop\\Jerk'
            plt.savefig(location + '\\' + graphTitle)
        return sumVec, binAvg, mass
<<<<<<< HEAD
    def findActiveDuration(self):
        def findActiveDurationPerFile(file):
            active = 0 
            for item in file:
                if item > 0:
                    active += 1
            return active
        activeVec = []
        for oneSet in self.UTM:
            curActive = findActiveDurationPerFile(oneSet);
            activeVec.append(curActive)
        UR = []
        for i in np.arange(0, len(activeVec) - 1, 2):
            if self.DA == 1:
                UR.append(activeVec[i]/activeVec[i + 1])
            else:
                UR.append(activeVec[i + 1]/activeVec[i])
        return np.array(UR), np.array(activeVec)
        
=======
    def findUseRatio(self): #dur is either sub or full
        activityVector = np.zeros(6)
        for i in range(len(self.UTM)):
            activityVector[i] += np.count_nonzero(self.UTM[i])
        UR = []
        for i in range(0, 5, 2):
            UR.append(activityVector[i] / activityVector[i + 1])
        if self.DA == -1:
            UR = [1/i for i in UR]
        return UR
    
    
>>>>>>> 9e2b5af4d783a8ad44dc764a5cab10c1973d2107
        
                
                
        
    
    
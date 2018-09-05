# Work in progress
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy import signal

class Accel:
# Abbreviation list:
    # UTV = UntrimOnemed Vector, UTM = UntrimOnemed Magnitude
    # DA = Dominant Arm
         
    CPU = 'Baker' # either Baker or Mac
    FILETYPE = 'Raw' #This should either be Raw or Epoch
    FIRST_LINE = 11 #class constant - stays the same 
    DURATION = '1sec'
    EXT = '.csv' #file extension
    N_COMP = 2 #number of components in PCA
    COLOR_MAP = 'Spectral'
    BINS = np.linspace(-1, 1, 20)
    APPLY_BUTTER = True
    
    def __init__(self, filename, epochLength = 60):
        if self.canRun():
            start = time.time()
            
            self.filename = filename
            self.filenameList = self.makeNameList(filename) #by doing so, you have created the filenameList array
            self.titles = self.makeTitleList()
            self.UTV, self.UTM, self.DA = self.readAll()
            self.AI = self.findPCMetrics(epochLength) #cov = coefficient of variation
            
            end = time.time()
            print('total time to read ' + self.filename + ' = ' + str(end - start))        
        else:
            print('Sorry but I can\'t run ' + Accel.FILETYPE + ' on ' + Accel.CPU)
        
    def __str__(self):
        return self.filename
    
    def canRun(self):
        if Accel.CPU == 'Mac' and Accel.FILETYPE == 'Raw':
            return False
        else:
            return True
    
    def makeSlash(self):
        if Accel.CPU == 'Baker':
            return '\\'
        else:    
            return '/'
    
    def makeNameList(self, filename): 
        if Accel.FILETYPE == 'Raw':
            directory = 'E:\\Projects\\Brianna\\' #can only be run on Baker
            baseList = ['_v1_LRAW', '_v1_RRAW', '_v2_LRAW', '_v2_RRAW', '_v3_LRAW', '_v3_RRAW']
            baseList = [directory + filename + item + Accel.EXT for item in baseList]
        else:
            if self.CPU == 'Baker':
                directory = 'C:\\Users\\SCH CIMT Study\\SCH\\' # for Baker
            else:
                directory = '/Users/preston/SCH/' # for running on my laptop
            baseList = ['_v1_L', '_v1_R', '_v2_L', '_v2_R', '_v3_L', '_v3_R']
            baseList = [directory + Accel.DURATION + self.makeSlash() + filename + item + Accel.DURATION + Accel.EXT for item in baseList]
        if self.CPU == 'Baker':
            baseList.append('C:\\Users\\SCH CIMT Study\\SCH\\Timing File\\' + filename + Accel.EXT)
        else:
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
    
    def readAll(self):
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
    #            b, a = signal.butter(order, wcHigh/nyquist) # just a low pass
                filtedX = signal.filtfilt(b, a, data[:, 0])
                filtedY = signal.filtfilt(b, a, data[:, 1])
                filtedZ = signal.filtfilt(b, a, data[:, 2])
                return np.array([list(a) for a in zip(filtedX, filtedY, filtedZ)])
            
            UTV2 = np.empty((0, 3)) #stores the sliced data
            
            df = pd.read_csv(file, header = 10, usecols = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z'])
            if Accel.FILETYPE == 'Epoch':
                for bounds in activeRanges:
                    UTV2 = np.vstack((UTV2, df.iloc[bounds[0] : bounds[1]].values))
            else:
                for bounds in activeRanges:
                    UTV2 = np.vstack((UTV2, df.iloc[bounds[0] * 100 : bounds[1] * 100].values))

# Working
#            UTV1 = [] # stores the entire, unsliced data
#            UTV2 = np.empty((0, 3)) #stores the sliced data
#            with open(file, 'r', encoding = 'utf-8') as csvFile:
#                csvReader = csv.reader(csvFile) #basically the same as a scanner
#                for i in range(Accel.FIRST_LINE): #consumes through all of the header information
#                    next(csvReader)
#                
#                if Accel.FILETYPE == 'Epoch':
#                    for row in csvReader:
#                        UTV1.append(list(map(int, row[0:3])))
#                    for bounds in activeRanges:
#                        UTV2 = np.vstack((UTV2, UTV1[bounds[0] : bounds[1]]))                    
#                else:
#                    firstLine = csvFile.readline()
#                    if ':' in firstLine: # this checks if there are time stamps
#                        for row in csvReader:
#                            UTV1.append(list(map(float, row[1:4])))
#                    else:
#                        for row in csvReader:    
#                            UTV1.append(list(map(float, row[0:3])))
#                    for bounds in activeRanges:
#                        UTV2 = np.vstack((UTV2, UTV1[bounds[0] * 100: bounds[1] * 100])) #grabbing all the periods of activity
            
            
            if Accel.APPLY_BUTTER:
                return butterworthFilt(np.array(UTV2)),  np.array(magnitude(UTV2)) #return it as an numpy array at the end
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
        
    def plotPCs(self):
        
        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
                FancyArrowPatch.draw(self, renderer)
              
        fig = plt.figure(figsize = (15,15))
        ax = fig.gca(projection = '3d')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        colors = ['red', 'blue', 'salmon', 'deepskyblue', 'peachpuff', 'aqua']
        for i in range(len(self.pcs)):  
            pc1 = self.pcs[i][0]
            pc2 = self.pcs[i][1]
            firstPC = Arrow3D([0, pc1[0]], [0, pc1[1]], [0, pc1[2]], mutation_scale = 20, arrowstyle = '-|>', color = colors[i])
            secondPC = Arrow3D([0, pc2[0]], [0, pc2[1]], [0, pc2[2]], mutation_scale = 20, arrowstyle = '->', color = colors[i])
            ax.add_artist(firstPC)
            ax.add_artist(secondPC)
        ax.set_title(self.makeSuperTitle(), fontsize = 20)
        plt.show()
        
    def findLRAvg(self):
        lrAvg = []
        for i in np.arange(0, 5, 2):
            curLRAvg = []
            for oneLeft, oneRight in zip(self.UTV[i], self.UTV[i + 1]):
                curLRAvg.append(np.mean((oneLeft, oneRight), axis = 0))
            lrAvg.append(curLRAvg)
        return np.array(lrAvg)
    
    def findAccelMetrics(self, window= 60):
        avgSim = []
        index = 0
        for i in np.arange(0, 5, 2):
            
            endpoints = np.arange(0, len(self.UTV[i]) + window, window)
            oneAvgSim = []
            for j in range(len(endpoints) - 1):
                ref = self.lrAvg[index][endpoints[j]]
                ref = np.divide(ref, self.findMag(ref))
                subset = self.lrAvg[index][endpoints[j] : endpoints[j + 1]]
                for row in subset:
                    oneAvgSim.append(np.dot(np.divide(row, self.findMag(row)), ref))
            avgSim.append(oneAvgSim)
            index += 1
        return np.array(avgSim)
    
    # Finds PC1 within each epoch which is of length "window" for both left and right arm,
    # finds the dot product and generates AI (Alignment Index) and COV (Coefficient of Variation)          
    def findPCMetrics(self, epochLength, thresh = 0.8):
        if Accel.FILETYPE == 'Raw':
            epochLength = epochLength * 100
        pca = PCA(1)
        AI = []   
        for i in np.arange(0, 5, 2):
            endpoints = np.arange(0, len(self.UTV[i]) + epochLength, epochLength)
            curDotSum = 0
            length = 0
            for j in range(len(endpoints) - 1):
                pca.fit_transform(self.UTV[i][endpoints[j]: endpoints[j + 1]])
                leftPC1 = pca.components_[0]
                leftPC1EVR = pca.explained_variance_ratio_[0]
                pca.fit_transform(self.UTV[i + 1][endpoints[j]: endpoints[j + 1]])
                rightPC1 = pca.components_[0]
                rightPC1EVR = pca.explained_variance_ratio_[0]
                if leftPC1EVR > thresh and rightPC1EVR > thresh:
                    length += 1
                    avgEVR = np.mean(leftPC1EVR, rightPC1EVR)
                    absDot = abs(np.dot(leftPC1, rightPC1))
                    curDotSum += absDot * avgEVR      
            AI.append(curDotSum / length)   
        return AI
    
    def angDiffInPairs(self, vectors):
        diff = []
        for i in np.arange(0, 5, 2):
            diff.append(np.arccos(np.dot(vectors[i], vectors[i + 1]) / (self.PCMag[i] * self.PCMag[i + 1])) * 180 / np.pi)
        return diff                
    
    def plotPCOne(self, data):
        
        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
                FancyArrowPatch.draw(self, renderer)
              
        fig = plt.figure(figsize = (15,15))
        ax = fig.gca(projection = '3d')
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        colors = ['red', 'blue', 'salmon', 'deepskyblue', 'peachpuff', 'aqua']
        for i in range(len(data)):  
            pc1 = data[i]
            firstPC = Arrow3D([0, pc1[0]], [0, pc1[1]], [0, pc1[2]], mutation_scale = 20, arrowstyle = '-|>', color = colors[i])
            ax.add_artist(firstPC)
        ax.set_title(self.makeSuperTitle(), fontsize = 20)
        plt.show()
    
    @staticmethod
    def findMag(vec):
        return math.sqrt(vec[0]**2 + vec[1]**2+ vec[2]**2)

    # Epoch length optimization
    def epochLengthOpt(self, timeVec):
        start = time.time()
        if Accel.FILETYPE == 'Raw':
            timeVec = [item * 100 for item in timeVec]
        
        #define subfunctions
        def findValues(EVR):
            mu = []
            COV = []
            for item in EVR:
                mean = np.nanmean(item)
                mu.append(mean)
                COV.append(np.nanstd(item)/mean)
            return min(mu), max(COV)
        
        def plotFunc(mu, COV, fileName, timeVec):
            plt.figure(figsize = (10, 10))
            plt.title(fileName)
            plt.plot(timeVec, mu, label = 'min avg. EVR for each epoch length')
            plt.plot(timeVec, COV, label = 'COV of EVR for each epoch length')
            plt.xlabel('time(s)')
            plt.legend(loc = 'center right')
            
        minMu = []
        maxCOV = []
        for epochLength in timeVec:
            oneMu, oneCOV = findValues(self.findPCMetrics(window = epochLength, EVRonly = True))
            minMu.append(oneMu)
            maxCOV.append(oneCOV)
        plotFunc(minMu, maxCOV, self.filename, timeVec)
        end = time.time()
        print('This method took ' + str(end - start) + 'seconds')
        return minMu, maxCOV
    
def plotAllAngleDiff(arr):
    titles = ['Pre', 'During', 'Post']
    xpos = np.arange(3)
    plt.figure(figsize = (20, 10))
    plt.tight_layout()
    j = 0
    for i in np.arange(0, len(arr), 2):
        j += 1
        plt.subplot(len(arr)/2, 2, j)
        plt.bar(xpos - 0.2, arr[i], width = 0.4, label = 'CIMT kid')
        plt.bar(xpos + 0.2, arr[i + 1], width = 0.4, label = 'TD kid')
        plt.legend()
        plt.xticks(xpos, titles)
        plt.ylabel('diff in angle between arms')
    plt.show()
    


#%%
plt.close('all') 

timeVec = np.arange(100, 7200, 400)
start = time.time()
CIMT03 = Accel('CIMT03')
CIMT04 = Accel('CIMT04') 
CIMT06 = Accel('CIMT06') 
CIMT08 = Accel('CIMT08')
CIMT09 = Accel('CIMT09') #dp score doesn't work
TD01 = Accel('TD01') 
TD02 = Accel('TD02')
TD03 = Accel('TD03')
TD04 = Accel('TD04') 
TD05 = Accel('TD05')
TD06 = Accel('TD06')
TD07 = Accel('TD07')
# Make sure to do the epochLengthOpt with Raw data
l = [CIMT03, CIMT04, CIMT06, CIMT08, CIMT09, TD01, TD02, TD03, TD04, TD05, TD06, TD07]
#for item in l:
#    item.epochLengthOpt(timeVec)
    
#for item in l:
#    print(item)
#    print('Epoch Dot Scores:', item.scores)
#    print('Epoch Weighted Dots:', item.weightedDots)
#    print('Dot C:', item.dotC)
#    print()
end = time.time()
print('grand total = ' + str(end - start))

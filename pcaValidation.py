import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import signal
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class PCACheck:
    DIR = '/Users/preston/SCH/1sec/'
    L_EXT = '_LRAW.csv'
    R_EXT = '_RRAW.csv'
    
    def __init__(self, file, start = 0, end = 60, step = 1, applyButter = False, fileNum = 2):
        if fileNum == 1:
            self.file = file
            self.data = self.readFile(PCACheck.DIR + file + 'RAW.csv', applyButter)
        else:
            self.file = file
            self.filename = [PCACheck.DIR + file + PCACheck.L_EXT, PCACheck.DIR + file + PCACheck.R_EXT]
            self.fileType = 'RAW' in self.filename[0] and 'RAW' in self.filename[1]# else file is False
            self.endpoints = np.arange(start, end + step, step)
            
            if self.fileType: 
                self.endpoints = np.multiply(100, np.array(self.endpoints))
                
            self.left = self.readFile(self.filename[0], applyButter)
            self.slicedLeft = self.sliceIntoEpisodes(self.left)
            self.right = self.readFile(self.filename[1], applyButter)
            self.slicedRight = self.sliceIntoEpisodes(self.right)
            self.leftPCs, self.leftEVR, self.leftIndices = self.pcaAnalysis(self.slicedLeft) #these two aren't actually used but are necessary 
            self.rightPCs, self.rightEVR, self.rightIndices = self.pcaAnalysis(self.slicedRight) #these two aren't actually used but are necessary 
            self.pcs = np.array([self.leftPCs, self.rightPCs])
            self.EVR = np.mean((self.leftEVR, self.rightEVR), axis = 0) #average between left and right
            self.commonIndices = np.intersect1d(self.leftIndices, self.rightIndices)
            
            self.pcSim, self.AI = self.findPCSim()
#            self.avgEVR = np.mean(self.EVR)
#            self.EVR_COV = np.std(self.EVR) / np.mean(self.EVR)
#            self.AI = self.avgDot * self.avgEVR
        
    def __str__(self):
        return(self.file)        
    # this reads the entire file
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
    
        
    
    def sliceIntoEpisodes(self, data):
        slicedData = []
        for i in range(len(self.endpoints) - 1):
            slicedData.append(data[int(self.endpoints[i]): int(self.endpoints[i + 1])])
        return np.array(slicedData)

        
    def findLRAvg(self, content):
        if content == 'Accel':
            left, right = self.left, self.right
        else:
            left, right = self.leftPCs, self.rightPCs
            
        if len(left) != len(right):
            print('Sorry, left and right vectors need to be the same length')
        else:
            result = []
            sim = []
            for oneLeft, oneRight in zip(left, right):
                result.append(np.mean((oneLeft, oneRight), axis = 0))
                sim.append(np.dot(np.divide(oneLeft, self.findMag(oneLeft)), np.divide(oneRight, self.findMag(oneRight))))
        return np.array(result)
    
    def findSimWRTAvg(self):
        avgSim = []
        reference = np.mean(self.lrAvg, axis = 0)
        reference = np.divide(reference, self.findMag(reference))
        for item in self.lrAvg:
            avgSim.append(abs(np.dot(reference, np.divide(item, self.findMag(item)))))
        return avgSim
    
    def findWeightedSim(self):
        freq, bins = np.histogram(self.avgSim)
        totalFreq = sum(freq)
        
        centerPts = []
        for i in range(len(bins) - 1):
            centerPts.append(np.mean((bins[i], bins[i + 1])))
            
        weightedSim = 0 
        for oneOccur, oneAvg in zip(freq, centerPts):
            weightedSim += oneOccur/totalFreq * oneAvg
        
        return weightedSim

    def pcaAnalysis(self, file, thresh = 0.8):
        allPCs = []
        EVR = []
        highEVRIndex = []
        if len(file.shape) == 3: #for 3D arrays (broken into epochs)
            for index, oneSlice in enumerate(file):
                pca = PCA(1)
                pca.fit_transform(oneSlice)
                curEVR = pca.explained_variance_ratio_[0]
                EVR.append(curEVR)
                if curEVR > thresh:
                    highEVRIndex.append(index)
                allPCs.append(pca.components_.tolist())
            return np.array(allPCs)[:, 0], EVR, highEVRIndex #removes one extra unnecessary dimension         
        else: #for 2D arrays (not broken into epochs)
            pca = PCA(1)
            pca.fit_transform(file)
            return pca.components_
        
    def plotPCOne(self, title = None, overlay = False):
        if title is None:
            title = self.file + ' Left and Right PCs'
        class Arrow3D(FancyArrowPatch):
            def __init__(self, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
                self._verts3d = xs, ys, zs

            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
                self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
                FancyArrowPatch.draw(self, renderer)
              
        fig = plt.figure(figsize = (10, 10))
        ax = fig.gca(projection = '3d')
        ax.set_aspect('equal')
        ax.set_xlabel('X', fontsize = 20)
        ax.set_ylabel('Y', fontsize = 20)
        ax.set_zlabel('Z', fontsize = 20)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        if len(self.pcs.shape) == 3:
            colors = ['b', 'r']
            for j in range(len(self.pcs)):
                dataSet = self.pcs[j]
                for i in range(len(dataSet)):  
                    pc1 = dataSet[i]
                    firstPC = Arrow3D([0, pc1[0]], [0, pc1[1]], [0, pc1[2]], mutation_scale = 20, arrowstyle = '-|>', color = colors[j])
                    ax.add_artist(firstPC)
            ax.set_title(title, fontsize = 24)
        else: 
            for i in range(len(self.pcs)):  
                pc1 = self.pcs[i]
                firstPC = Arrow3D([0, pc1[0]], [0, pc1[1]], [0, pc1[2]], mutation_scale = 20, arrowstyle = '-|>')
                ax.add_artist(firstPC)
            ax.set_title(title, fontsize = 18)
        
        if overlay:
            offset = 0.5 #so that the arros don't hide the scatter plot
            ax.scatter(self.left[:, 0], self.left[:, 1], np.add(self.left[:, 2], offset), label = 'Left Accel', color = colors[0], alpha = 0.5)
            ax.scatter(self.right[:, 0], self.right[:, 1], np.add(self.right[:, 2], offset), label = 'Right Accel', color = colors[1], alpha = 0.5)
            ax.legend(loc = 'center left', fontsize = 15)
            
        plt.show()
    
    def findPCSim(self):
        pcSim = []
        AI = 0
        for index in self.commonIndices:
            curSim = abs(np.dot(self.leftPCs[index], self.rightPCs[index]))
            pcSim.append(curSim)
            AI += curSim * self.EVR[index]
        AI = AI / len(self.commonIndices)
        return pcSim, AI
    
    def report(self):
        print(self.file, ':')
        print('AI:', "%4f" % self.AI)
        print()
            
    
    @staticmethod
    def findRunningAverage(data, window, fileType):
        runningAvg = np.empty((0, 3))
        if fileType == 'RAW':
            window = window * 100
        start = 0
        while start + window < len(data):
            runningAvg = np.vstack((runningAvg, np.mean(data[int(start): int(start + window)], axis = 0)))
            start += 1
        return np.array(runningAvg)
    
    @staticmethod    
    def findMag(vec): 
        return math.sqrt(vec[0]**2 + vec[1]**2+ vec[2]**2)
    
def epochLengthOpt(file, timeVec, EVRthreshold = 0.6):
    print('Running', file, '...')
    dataVec = [[], [], [], []]
    for epochLength in timeVec:
        iteration = PCACheck(file, step = epochLength)
        dataVec[0].append(iteration.AI)
        dataVec[1].append(iteration.COV)
        dataVec[2].append(iteration.avgEVR)
        dataVec[3].append(iteration.EVR_COV)
    passed = False
    for index, item in enumerate(dataVec[2]):
        if item > EVRthreshold:
            passed = True
            print('Threshold passed at epoch length =', timeVec[index])
    if not passed:
        print('No epoch length passed the threshold')
    print()
    plt.figure(figsize = (10, 10))
    plt.title(file)
    plt.plot(timeVec, dataVec[0], label = 'AI')
    plt.plot(timeVec, dataVec[1], label = 'COV')
    plt.plot(timeVec, dataVec[2], label = 'Avg. EVR')
    plt.plot(timeVec, dataVec[3], label = 'EVR_COV')
    plt.axhline(y = EVRthreshold, color='k', linestyle='-')
    plt.legend()
    
    return dataVec
    
def plotRaw(data, title = 'Raw Accel (g)'):
        #assumes that left and right are the same in length
    plt.figure(figsize = (10, 10))
    xAxis = np.arange(0, len(data), 1)
    plt.plot(xAxis, data[:, 0], label = 'x', color = 'r')
    plt.plot(xAxis, data[:, 1], label = 'y', color = 'b')
    plt.plot(xAxis, data[:, 2], label = 'z', color = 'k')
    plt.ylim([-1, 1])
    plt.title(title)
    plt.legend()
    plt.show()
    
# data: a 3D array of epochs
# titles: an array of strings with the same length as the number of epochs 
# suptitle: super title for all subplots
def plotEpochs(data, titles = None, suptitle = None):
    if titles is None:
        titles = ['Interval ' + str(i + 1) for i in range(len(data))]
    if suptitle is None:
        suptitle = 'Epochs'
    plt.figure(figsize = (13, 13))
    if len(data) > 4:
        'Too many epochs! Please change the step size or data range'
    for i in range(len(data)):
        plt.subplot(len(data), 1, i + 1)
        plt.title(titles[i])
        plt.plot(data[i][:, 0], label = 'x')
        plt.plot(data[i][:, 1], label = 'y')
        plt.plot(data[i][:, 2], label = 'z')
        plt.legend(loc = 'center right')
    plt.suptitle(suptitle, fontsize = 18)
    plt.show()
    
def overlay3DScatter(*args):
    #takes in unsliced datasets from different experiments and plots them all
    #on one 3D scatter plot
    fig = plt.figure(figsize = (10, 10))
    ax = fig.gca(projection = '3d')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    bounds = 3
    ax.set_xlim(-bounds, bounds)
    ax.set_ylim(-bounds, bounds)
    ax.set_zlim(-bounds, bounds)
    for item in args:
        ax.scatter(item[:, 0], item[:, 1], item[:, 2], alpha = 0.3)
    plt.show()

# takes in an array of PCACheck objects that are of the same activity, and returns
# macros that can then be used to compare across different activities
def createSummary(*arr):
    result = {} #empty dictionary
    for item in arr:
        result.setdefault('Titles', []).append(str(item))
        result.setdefault('AI', []).append(item.AI)
    return result

# takes an entire array of PCACheck objects and plots a bar graph scattered with data points to show variation
def plotScatterBar(PCACheckArr, content): #content dictates what to plot; it's either EVR or AI
    fig, ax = plt.subplots(figsize = (10, 10))
    xpos = np.arange(len(PCACheckArr))
    xNames = [str(item) for item in PCACheckArr]
    ax.set_title(content + ' for several activities', fontsize = 22)
    if content == 'EVR':
        for i in range(len(PCACheckArr)):
            ax.bar(xpos[i], PCACheckArr[i].avgEVR, alpha = 0.5)
            ax.scatter([xpos[i]] * len(PCACheckArr[i].EVR), PCACheckArr[i].EVR, color = 'k')
        ax.legend(['different epochs'], fontsize = 14)
    elif content == 'avgDot':
        for i in range(len(PCACheckArr)):
            ax.bar(xpos[i], PCACheckArr[i].avgDot, alpha = 0.5)
            ax.scatter([xpos[i]] * len(PCACheckArr[i].pcSim), PCACheckArr[i].pcSim, color = 'k')
        ax.legend(['different epochs'], fontsize = 14)
    else:
        for i in range(len(PCACheckArr)):
            ax.bar(xpos[i], PCACheckArr[i].AI, alpha = 0.5)  
    ax.set_xticks(xpos)
    ax.set_xticklabels(xNames, fontsize = 14) #doesn't show all tick marks for some reason
    ax.tick_params(axis = 'y', labelsize = 24)
    plt.show()
    
    
def plotAll(l):
    fig, ax = plt.subplots(figsize = (10, 10))
    xpos = np.arange(len(l))
    xNames = [item['Titles'][0] for item in l]
    for i in xpos:
        ax.bar(xpos[i], np.mean(l[i]['AI']), alpha = 0.5)
        ax.errs = plt.errorbar(xpos[i], np.mean(l[i]['AI']), np.std(l[i]['AI']), alpha = 0.5)
#        ax.scatter([xpos[i]] * len(l[i]['AI']), l[i]['AI'], color = 'k')
    ax.set_xticks(xpos)
    ax.set_xticklabels(xNames, fontsize = 24)
    ax.tick_params(axis = 'y', labelsize = 24)
    ax.set_title('Average AI for Different Movements', fontsize = 24)
    ax.legend(['AI for each trial'], fontsize = 20)
    
def improvedPlotAll(l):
    fig, ax = plt.subplots(figsize = (10, 10))
    xpos = np.arange(len(l))
    xNames = [item['Titles'][0] for item in l]
    for i in xpos:
        ax.bar(xpos[i], np.mean(l[i]['AI']), alpha = 0.5)
        ax.scatter([xpos[i]] * len(l[i]['AI']), l[i]['AI'], color = 'k')
    ax.set_xticks(xpos)
    ax.set_xticklabels(xNames, fontsize = 24)
    ax.tick_params(axis = 'y', labelsize = 24)
    ax.set_title('Average AI for Different Movements', fontsize = 24)
    ax.legend(['AI for each trial'], fontsize = 20)

plt.close('all')
step = 1
applyButter = True
tray = PCACheck('Tray', applyButter = applyButter, step = step)   
tray2 = PCACheck('Tray2', applyButter = applyButter, step = step) 
tray3 = PCACheck('Tray3', applyButter = applyButter, step = step)
tray4 = PCACheck('Tray4', applyButter = applyButter, step = step)
tray5 = PCACheck('Tray5', applyButter = applyButter, step = step)
pants = PCACheck('Pants',applyButter = applyButter, step = step)
pants2 = PCACheck('Pants2', applyButter = applyButter, step = step)
pants3 = PCACheck('Pants3', applyButter = applyButter, step = step)
pants4 = PCACheck('Pants4', applyButter = applyButter, step = step)
pants5 = PCACheck('Pants5', applyButter = applyButter, step = step)
oneDX = PCACheck('X-axis', applyButter = applyButter, step = step)
oneDY = PCACheck('Y-axis', applyButter = applyButter, step = step)
random = PCACheck('Random', applyButter = applyButter, step = step)
random2 = PCACheck('Random2', applyButter = applyButter, step = step)
random3 = PCACheck('Random3', applyButter = applyButter, step = step)
random4 = PCACheck('Random4', applyButter = applyButter, step = step)
random5 = PCACheck('Random5', applyButter = applyButter, step = step)

summaryL = [createSummary(oneDX), createSummary(oneDY), createSummary(tray, tray2, tray3, tray4, tray5),
            createSummary(pants, pants2, pants3, pants4, pants5), createSummary(random, random2, random3, random4, random5)]



timeVec = [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]

l = [oneDX, oneDY, pants, pants2, pants3, tray, tray2, tray3, random, random2, random3]
l2 = [oneDX, oneDY, pants, tray3, random, random2, random3]
print('Apply Butterworth:', applyButter)
print('epochLength:', step)
print('*' * 20)
#for item in l:
#    resultVec = epochLengthOpt(str(item), timeVec, EVRthreshold = 0.7)
#    plt.figure(figsize = (10, 10))
#    plt.scatter(resultVec[2], resultVec[0])
    
#plotAll(summaryL)    

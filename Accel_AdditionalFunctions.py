import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

#self.proejcted, self.PCAHull, self.pcs = self.pcaProjectionHull()
#self.colorThresholds = Accel.COLOR_THRESHOLDS #if I were to specify the colorArray
#self.colorArray, self.totalDistribution = self.makeColorArray()
#5self.weights = self.findWeights()
#self.simVec, self.angVec = self.findSimilarity()
#self.re, self.err, self.errRatio = self.reconstruct()
#self.UR = self.findUseRatio()
            
def plotComponents(self):
        plt.figure(figsize = (20, 20))
        plt.tight_layout
        for i in range(len(self.TV)):
            plt.subplot(3, 2, i + 1)
            plt.plot(self.TV[i][:, 0], label = 'x direction', alpha = 0.5)
            plt.plot(self.TV[i][:, 1], label = 'y direction', alpha = 0.5)
            plt.plot(self.TV[i][:, 2], label = 'z direction', alpha = 0.5)
            plt.legend()
        plt.show()
        
def plotUTVScatter(self):
        plt.figure(figsize = (20, 20))
        plt.tight_layout()
        for i in range(len(self.UTV)):
            plt.subplot(3, 2, i + 1)
            plt.scatter(self.UTV[i][:, 0], self.UTV[i][:, 1], label = 'x vs. y')
            plt.scatter(self.UTV[i][:, 0], self.UTV[i][:, 2], label = 'x vs. z')
            plt.legend()
            plt.suptitle(self.makeSuperTitle(), fontsize = 10)
            plt.title(self.titles[i], fontsize = 8)
        plt.show()
        
# Can really be generated using np.histogram
def makeColorArray(self):
    colorArray = []
    totalDistribution = []
    #projectedPCA should have 2 components
    for file in self.projection:
        oneFileColorArray = []
        oneFileDistribution = [0] * (len(self.colorThresholds) + 1)
        for entry in file:
            curMag = math.sqrt((entry[0] ** 2 + entry[1] ** 2))
            if curMag > self.colorThresholds[-1]:
                    oneFileColorArray.append(len(self.colorThresholds) + 1 )
                    oneFileDistribution[-1] += 1
            elif curMag < self.colorThresholds[0]:
                    oneFileColorArray.append(0)
                    oneFileDistribution[0] += 1
            else: 
                for i in range(len(self.colorThresholds) - 1):
                    if curMag > self.colorThresholds[i] and curMag < self.colorThresholds[i + 1]:
                        oneFileColorArray.append(i + 1)
                        oneFileDistribution[i + 1] += 1
                        break
        colorArray.append(oneFileColorArray)
        totalDistribution.append(oneFileDistribution)
    return colorArray, totalDistribution

def trimOne(UTV, UTM, window, threshold): #Rate limiting; so slow
    TM = []
    TV = []
    tooHigh = False
    window = window * 3600 #convert window from hours into seconds
    currentIndex = 0 #must start at 0 for python
    while (currentIndex < len(UTM)):
        if UTM[currentIndex] < threshold and currentIndex + window < len(UTM):
            for y in range(window): #I feel like this for loop is causing the problem
                if UTM[currentIndex + y] > threshold:
                    tooHigh = True
                    break; 
            #the following statement happens after the previous for loop
            if (not tooHigh):
                currentIndex = currentIndex + window
        TM.append(UTM[currentIndex])
        TV.append(UTV[currentIndex])
        currentIndex = currentIndex + 1
        tooHigh = False
    return np.array(TV), np.array(TM)

def plotWindsizeMuSd(self, angleThresh = 20):
    plt.figure(figsize = (20, 20))
    plt.tight_layout()
    muList = [[], [], [], [], [], []]
    sdList = [[], [], [], [], [], []]
    windsize = np.arange(1, 21) * 2  
    for num in windsize:
        sd, mu, log, subsetPCs = self.findSubsetPCs(num, 100, False)
        for i in range(len(muList)):
            muList[i].append(mu[i])
            sdList[i].append(sd[i])
    for j in range(len(muList)):
        plt.subplot(3, 2, j + 1)
        plt.scatter(windsize, muList[j], label = 'mu')
        plt.scatter(windsize, sdList[j], label = 'sd')
        plt.axhline(y = angleThresh)
        plt.axis([0, windsize[-1], 10, 75])
        plt.title(self.makeTitleList()[j])
        plt.legend()
        plt.xlabel('window size')
        plt.ylabel('angle difference')
    plt.suptitle(self.makeSuperTitle())    
    plt.show()
    
def pcaProjectionHull(self):
    pca = PCA(Accel.N_COMP)
    projected = []
    pcaHull = []
    pcs = []
    for file in self.TV:
        currentSet = pca.fit_transform(file)
        projected.append(currentSet)
        pcaHull.append(ConvexHull(currentSet))
        pcs.append(pca.components_)
    return np.array(projected), np.array(pcaHull), pcs
def findSimilarity(self):
    # one reference vector, and one set of vectors to be compared
    ref = self.pcs[0][0] #use the first principal component of Pre Left
    simVector = [] #similarity vector
    angVector = [] #angles between the pcs and the baseline
    for treatment in self.pcs:
        simPC1 = float("%.5f" % np.dot(ref, treatment[0]))
        simPC2 = float("%.5f" % np.dot(ref, treatment[1]))
        simVector.append(simPC1)
        simVector.append(simPC2)
        angVector.append(np.arccos(simPC1) * 180 / np.pi)
        angVector.append(np.arccos(simPC2) * 180 / np.pi)
    return simVector, angVector
def plotPCAAndHull(self): 
    def polyArea2D(pts):
        lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
        area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
        return area
    
    plt.figure(figsize = (20, 20))
    plt.tight_layout()
    for i in range(len(self.projection)):
        PC1 = self.projection[i][:, 0]
        PC2 = self.projection[i][:, 1]
        oneFilePC = self.pcs[i]
        traceX = []
        traceY = []
        plt.subplot(3, 2, i + 1)
        plt.scatter(PC1, PC2, s = 0.25, c = self.colorArray[i], cmap = plt.cm.get_cmap(Accel.COLOR_MAP, len(self.colorThresholds) + 1))
        plt.xlabel('PC1 = ' + str("%.2f" % oneFilePC[0][0]) + 'x + ' + str("%.2f" % oneFilePC[0][1]) + 'y + ' + str("%.2f" % oneFilePC[0][2]) + 'z')
        plt.ylabel('PC2 = ' + str("%.2f" % oneFilePC[1][0]) + 'x + ' + str("%.2f" % oneFilePC[1][1]) + 'y + ' + str("%.2f" % oneFilePC[1][2]) + 'z')
        for point in self.pcaHull[i].vertices:
            traceX.append(PC1[point])
            traceY.append(PC2[point])
        traceX.append(PC1[self.pcaHull[i].vertices[0]])
        traceY.append(PC2[self.pcaHull[i].vertices[0]])
        plt.plot(traceX, traceY, 'k-')
        plt.title(self.titles[i] + ', area = ' + "%.2f" % polyArea2D(list(zip(traceX, traceY))), fontsize = 25)
        plt.axis([-100, 1300, -600, 600])
        plt.suptitle(self.makeSuperTitle(), fontsize = 30)
        plt.colorbar();
    plt.show()
    
def findWeights(self):
    totalWeight = []
    for i in range(len(self.totalDistribution)):
        oneFileWeight = []
        for entry in self.totalDistribution[i]:
            oneFileWeight.append(entry/len(self.projection[i]));
        totalWeight.append(oneFileWeight)
    return np.array(totalWeight)
    
def plotChangesInActivity(self):
    plt.figure(figsize = (15, 15))
    activityLevel = np.arange(len(self.weights[0])) + 1
    dataLength = self.weights.shape[1]
    for j in range(dataLength):
        if dataLength % 2 == 1:
            plt.subplot(int(dataLength/2) + 1 , 2, j + 1)
        else:
            plt.subplot(dataLength/2, 2, j + 1)
        plt.bar(self.titles, self.weights.T[j] * 100)
        plt.title('Activity Level = ' + str(activityLevel[j]))
        plt.ylabel('Percentage (%) of points at activity level ' + str(activityLevel[j]))
    plt.suptitle(self.makeSuperTitle(), fontsize = 30)
    plt.show()
def findWeightedPC1(self, bins = None): 
    def PCHistogram(bins):
        PCDis = []
        for i in range(len(self.subsetPCs)):
            curPCDis = []
            oneSet = np.array(self.subsetPCs[i])
            x = oneSet[:, 0]
            y = oneSet[:, 1]
            z = oneSet[:, 2]
            xDis = np.histogram(x, bins)
            yDis = np.histogram(y, bins)
            zDis = np.histogram(z, bins)       
            curPCDis.append(xDis[0])
            curPCDis.append(yDis[0])
            curPCDis.append(zDis[0])
            PCDis.append(curPCDis)
        return PCDis
    
    if bins is None:
        bins = Accel.BINS
    
    weightedPC = []
    for file in PCHistogram(bins):
        oneFilePC = []
        for direction in file:
            total = sum(direction)
            value = 0
            for i in range(len(direction)):
                value += direction[i]/total * (bins[i] + bins[i + 1]) / 2
            oneFilePC.append(value)
        weightedPC.append(oneFilePC)
    
    mag = []
    for row in weightedPC:
        mag.append(math.sqrt(row[0]**2 + row[1]**2+ row[2]**2))
    return weightedPC, mag

def reconstruct(self):
    reconstruction = []
    error = []
    totalError = np.zeros(6)
    
    totalActivity = np.zeros(6)
    for j in range(len(self.TV)):
        for direction in self.TV[j]:
            totalActivity[j] += sum(direction) #computes the total activity sum
            
    for i in range(len(self.pcs)):
        mean = [np.mean(self.TV[i][:, 0]), np.mean(self.TV[i][:, 1]), np.mean(self.TV[i][:, 2])]
        curRecon = np.dot(self.projection[i][:,0:Accel.N_COMP], self.pcs[i][0:Accel.N_COMP, :]) + mean
        curError = abs(self.TV[i] - curRecon)
        reconstruction.append(curRecon)
        error.append(curError)
        for row in curError:
            totalError[i] += sum(row)
        
    errRatio = totalError / totalActivity
    return np.array(reconstruction), np.array(error), np.array(errRatio)
def findUseRatio(self, inputVec = None): #dur is either sub or full
    if inputVec == None:
        inputVec = self.TM

    activityVector = np.zeros(6)
    for i in range(len(inputVec)):
        activityVector[i] += np.count_nonzero(inputVec[i])
    UR = []
    for i in range(0, 5, 2):
        UR.append(activityVector[i] / activityVector[i + 1])
    if self.DA == -1:
        UR = [1/i for i in UR]
    return UR
    
def findSubsetURs(self, window = 30, N = 50, plot = False):
    allURs = []
    
    for i in np.arange(0, 5, 2):
        onePairURs = []
#            startPoints = np.random.randint(0, len(self.TM[i]) - window - 1, N)
        startPoints = np.linspace(0, len(self.TM[i]) - window - 1, N)
        for k in range(N):
            start = startPoints[k]
            left, right = self.TM[i][int(start): int(start + window)], self.TM[i + 1][int(start): int(start + window)]
            leftNonZero = np.count_nonzero(left)
            rightNonZero = np.count_nonzero(right)
            if leftNonZero == 0 or rightNonZero == 0:
                break   
            curUR = np.count_nonzero(left)/np.count_nonzero(right)
            if self.DA == 1:
                onePairURs.append(curUR)
            else: 
                onePairURs.append(1/curUR)
        allURs.append(onePairURs)
    
    if plot: 
        plt.figure(figsize = (20, 20))
        for i in range(len(allURs)): 
            plt.subplot(3, 1, i + 1)
            plt.hist(allURs[i], bins = np.linspace(0, 5, 50))
            plt.show()
        plt.suptitle(self.makeSuperTitle())
        
    return allURs
def findSubsetPCs(self, window, N=60, doSkip = True):
    allPCs = []
    skip = False
    sd = []
    mu = []
    angVector = []
    startLog = []
    pcaLeft = PCA(2)
    pcaRight = PCA(2)
    
    for i in np.arange(0, 5, 2):
        pcaLeft.fit_transform(self.TV[i])
        pcaRight.fit_transform(self.TV[i + 1])
        refLeft = pcaLeft.components_[0]
        refRight = pcaRight.components_[0]
        oneAngVec = []
        oneStartLog = []
        oneTreatmentPCs = []
        startPoints = np.random.randint(0, len(file) - window - 1, N) #does not go in order; different every time
        #startPoints = np.linspace(0, len(self.TV[i]) - window - 1, N) #goes in order; the same every time
        for k in range(N):
            start = startPoints[k]
            curDataLeft, curDataRight = self.TV[i][int(start): int(start + window)], self.TV[i + 1][int(start): int(start + window)]
            
            pcaLeft.fit_transform(curDataLeft)
            curPC1Left = pcaLeft.components_[0] #only interested in PC1 right now
            pcaRight.fit_transform(curDataRight)
            curPC1Right = pcaRight.components_[0]
            
            oneAngVec.append(np.arccos(np.dot(curPC1Left, curPC1Right)) * 180 / np.pi)
            oneStartLog.append("%.2f" % (start / 3600))
        allPCs.append(oneTreatmentPCs)
        sd.append(np.nanstd(oneAngVec))
        mu.append(np.nanmean(oneAngVec))
        angVector.append(oneAngVec)
        startLog.append(oneStartLog)
    counter = 0
    
    #We want to ignore [1, 0, 0], [0, 1, 0], and [0, 0, 1]
    if doSkip: 
        newPCs = []
        for file in allPCs:
            newOneTreatmentPCs = []
            for i in range(len(file)):
                if np.all(file[i] == [1, 0, 0]) or np.all(file[i] == [0, 1, 0]) or np.all(file[i] == [0, 0, 1]):
                    skip = True
                    counter += 1
                    continue
                if np.any(file[i] != [1, 0, 0]): 
                    counter = 0 
                    skip = False
                newOneTreatmentPCs.append(file[i + counter])
            newPCs.append(newOneTreatmentPCs)
        return sd, mu, np.array(startLog).T, np.array(newPCs)
    else:
        return sd, mu, np.array(startLog).T, np.array(allPCs), angVector
    
# Probably should never run this (takes way too much computing power)
def scatter3D(self, yesSubplots = False):
    colors = ['r', 'b', 'y', 'g', 'k', 'm']
    #data should have 6 sublists
    fig = plt.figure(figsize = (20, 20))
    if yesSubplots:
        for i in range(len(self.UTV)):
            ax = fig.add_subplot(3, 2, i + 1, projection = '3d')
#                ax.set_aspect('equal')
            x, y, z = np.array(self.UTV[i])[:, 0], np.array(self.UTV[i])[:, 1], np.array(self.UTV[i])[:, 2]
            ax.scatter(x, y, z, c = colors[i], marker = 'o', label = self.makeTitleList()[i])
            ax.legend()
        plt.suptitle(self.makeSuperTitle())
    else:
        ax = fig.add_subplot(1, 1, 1, projection = '3d')
        ax.set_aspect('equal')
        for i in range(len(self.UTV)):
            x, y, z = np.array(self.UTV[i])[:, 0], np.array(self.UTV[i])[:, 1], np.array(self.UTV[i])[:, 2]
            ax.scatter(x, y, z, c = colors[i], marker = 'o', label = self.makeTitleList()[i])
            ax.legend()
        plt.title(self.makeSuperTitle())
    plt.show()
def plotPCHistogram(self, bins = None):
    if bins is None:
        bins = Accel.BINS
    fig = plt.figure(figsize = (20, 20))
    formattedBins = np.empty(len(bins))
    for i in range(len(bins)):
        formattedBins[i] = "%.2f" % bins[i]
    formattedBins = np.append(formattedBins, '-')
    PCDis = [] #And array to store the distribution of x, y, z component of each PC
    #Bins is a 1D array that creates len(bins) + 1 bins used to create a histogram
    for i in range(len(self.subsetPCs)):
        curPCDis = []
        ax = fig.add_subplot(3, 2, i + 1)
        oneSet = np.array(self.subsetPCs[i])
        x = oneSet[:, 0]
        y = oneSet[:, 1]
        z = oneSet[:, 2]
        xDis, newBins, patches = ax.hist(x, bins, alpha = 0.5, color = 'k', edgecolor = 'k', linewidth = 2, linestyle = '-', label = 'x')
        yDis, newBins, patches = ax.hist(y, bins, alpha = 0.5, color = 'r', edgecolor = 'k', linewidth = 2, linestyle = '--', label = 'y')
        zDis, newBins, patches = ax.hist(z, bins, alpha = 0.5, color = 'g', edgecolor = 'k', linewidth = 2, linestyle = ':', label = 'z')
        ax.legend()
        
        curPCDis.append(xDis)
        curPCDis.append(yDis)
        curPCDis.append(zDis)
        PCDis.append(curPCDis)
        plt.xlabel('component strength')
        plt.ylabel('frequency')
    
    plt.suptitle(self.makeSuperTitle())
    plt.show()
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
    if Accel.FILETYPE == 'Raw':
        timeVec = [item * 100 for item in timeVec]
    
    #define subfunctions
    def findValues(VAF):
        mu = []
        COV = []
        for item in VAF:
            mean = np.nanmean(item)
            mu.append(mean)
            COV.append(np.nanstd(item)/mean)
        return min(mu), max(COV)
    
    def plotFunc(mu, COV, fileName, timeVec):
        plt.figure(figsize = (10, 10))
        plt.title(fileName)
        plt.plot(timeVec, mu, label = 'min avg. VAF for each epoch length')
        plt.plot(timeVec, COV, label = 'COV of VAF for each epoch length')
        plt.xlabel('time(s)')
        plt.legend(loc = 'center right')
def asleepVSawake(awake, asleep):
    from scipy import stats as st
    stats = [[] for i in range(4)]
    for a, b in zip(awake.UTM, asleep.UTM):
        stats[0].append(np.average(a))
        stats[1].append(np.std(a))
        stats[2].append(np.average(b))
        stats[3].append(np.std(b))
    stats = np.array(stats)
    for i in range(6):
        plt.figure()
        plt.title(awake.titles[i])
        mu1, std1, mu2, std2 = stats[:, i]
        x1 = np.linspace(mu1 - 3 * std1, mu1 + 3 * std1, 100)
        x2 = np.linspace(mu2 - 3 * std2, mu2 + 3 * std2, 100)
        plt.plot(x1, st.norm.pdf(x1, mu1, std1), alpha = 0.5)
        plt.plot(x2, st.norm.pdf(x2, mu2, std2), alpha = 0.5)
        plt.xlabel('magnitude')
        plt.ylabel('probability density')
        plt.legend(['awake', 'asleep'])
    return stats
        
    minMu = []
    maxCOV = []
    for epochLength in timeVec:
        oneMu, oneCOV = findValues(self.findPCMetrics(window = epochLength, VAFonly = True))
        minMu.append(oneMu)
        maxCOV.append(oneCOV)
    plotFunc(minMu, maxCOV, self.filename, timeVec)
    return minMu, maxCOV

# The function "consistency" compares all vectors to a reference vector and
# quantifies how consistent a set of vectors is. This is used to compare
# asleep vs. awake periods (asleep is expected to have higher consistency)
 def consistency(self, ref = [1, 0, 0], saveFile = False):
        
        
        def findMag(vec):
            return math.sqrt(vec[0]**2 + vec[1]**2+ vec[2]**2)
        
        dotProducts = [[] for i in range(6)]
        graphTitle = self.__str__() + ' ref = [' + ''.join(str(e) + ',' for e in ref)[:-1] + ']'
        plt.figure(graphTitle)
        plt.title(graphTitle) 
        #I used [:-1] to exclude the very last comma (a hack to solve the light-post problem)
        for i in range(len(self.UTV)):
            data = self.UTV[i]
            for entry in data:
                dotProducts[i].append(np.divide(np.dot(ref, entry), findMag(entry)))
            plt.hist(dotProducts[i], bins = 30, alpha = 0.3)
            plt.xlabel('Range of normalized dot product')
            plt.ylabel('Number of Occurences')
        plt.show()
        
        if saveFile:
            location = 'C:\\Users\\SCH CIMT Study\\Desktop\\Vector Histogram'
            plt.savefig(location + '\\' + graphTitle)
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
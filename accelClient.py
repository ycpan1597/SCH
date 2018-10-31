from Accel import Accel #now you're the client
import matplotlib.pyplot as plt
import numpy as np

#initializes multiple at a time
def initialize(subjects, OS, filetype, applyButter = True):
    l = {}
    for item in subjects:
        l[str(item)] = Accel(item, OS, filetype, applyButter = applyButter)
    return l

#def findSummary(dic, minAge = 7, maxAge = 9, showBox = True, content = 'DA Prob'):
#    if content == 'DA Prob':
#        correctColumn = 1
#    else:
#        correctColumn = 0
#    ageRange = np.arange(minAge, maxAge + 1)
#    TD, CIMT = [], []
#    TDnum, CIMTnum = 0, 0
#    for key, value in dic.items():
#        if 'TD' in key and value.age in ageRange:
#            TD.append(value.jerkRatio(showPlot = False)[correctColumn])
#            TDnum += 1
#        elif 'CIMT' in key and value.age in ageRange:
#            CIMT.append(value.jerkRatio(showPlot = False)[correctColumn])
#            CIMTnum += 1
#    TD, CIMT = np.array(TD), np.array(CIMT)

def boxPlot(TD, CIMT):
    TDavg = np.mean(TD)
    TDstd = np.std(TD)
    plt.figure()
    plt.boxplot([CIMT[:, 0], CIMT[:, 1], CIMT[:, 2]], sym = '')
    plt.axhline(y = 0.5, label = 'Bimanual, 0.500', color = 'r')
    plt.axhline(TDavg, label = 'TD Avg, ' + "%.3f" % TDavg, color = 'b')
    plt.axhline(TDavg + TDstd, color = 'b', ls = '--')
    plt.axhline(TDavg - TDstd, color = 'b', ls = '--')
    plt.legend()
    plt.xticks([1, 2, 3], ['Pre', 'During', 'Post']) #maps x labels
    plt.ylabel('Probability of Using Dominant Arm')
    plt.xlabel('Data Collection')

def jerkPlot(binAvg, probVec, mass, subplotTitle, shaded = False):
    for prob, c, oneLab, oneMass in zip(probVec, 'gkr', ['Pre', 'During', 'Post'], mass):
        if shaded:
            plt.plot(binAvg, np.divide(prob, max(prob)), color = c, label = oneLab + ', JDP = ' + str(round(oneMass, 2)))
            plt.fill_between(binAvg, np.divide(prob, max(prob)), where = binAvg < 0.5, alpha = 0.5, color = c)
        else:
            plt.plot(binAvg, np.divide(prob, max(prob)), color = c, label = oneLab)
    plt.legend(loc = 'lower right')
    plt.title(subplotTitle)
    plt.xlabel('jerk ratio')
    plt.ylabel('normalized probability')

def BP(massVec, subplotTitle):
    plt.boxplot([massVec[:, 0], massVec[:, 1], massVec[:, 2]], sym = '')
    for item in massVec:
        plt.scatter([1, 2, 3], item)
    plt.xticks([1, 2, 3], ['Pre', 'During', 'Post'])
    plt.ylim(0, 0.7)
    plt.title(subplotTitle)
    plt.xlabel('data collection')
    plt.ylabel('probability before 0.5 (%)')
    
def fftSNR(dic, thresh = 1.5):
    SNR = []
    for key, value in dic.items():
        oneSNR = []
        for oneMag in value.jerkMag:
            oneFFT = np.fft.fft(oneMag)
            oneFFT_new = abs(oneFFT)[0: int(len(oneFFT)/2)]
            freq = np.linspace(0, 50, int(len(oneFFT)/2))
            threshIndex = int(np.where(freq > thresh)[0][0])
            signal = sum(oneFFT_new[0:threshIndex])
            noise = sum(oneFFT_new[threshIndex:])
            oneSNR.append(signal/noise)
        SNR.append(oneSNR)
    return SNR
           
#def fftSNR(a, title, thresh = 1.5):
#    plt.figure()
#    for i, direction in zip(range(3), 'xyz'):
#        afft = np.fft.fft(a[:, i])
#        afft_new = abs(afft)[0:int(len(afft)/2)]
#        freq = np.linspace(0, 50, int(len(afft)/2))
#        threshIndex = int(np.where(freq > thresh)[0][0])
#        signal = sum(afft_new[0:threshIndex])
#        noise = sum(afft_new[threshIndex:])

plt.close('all')

#dic = initialize(['TD01', 'TD02', 'TD05', 'TD06', 'TD07',
#                  'CIMT03', 'CIMT04','CIMT08', 'CIMT09', 'CIMT13'], 'Baker', 'Raw')
#for key, value in dic.items():
#    value.jerkRatio(showPlot = True, cutoff = 3, saveFig = True)

#TD, CIMT, CIMTmedian = findSummary(dic)
TDrep = 'TD01'
CIMTrep = 'CIMT03' 
   
plt.close('all')
TDsumVec, TDbinAvg, TDmass = dic[TDrep].jerkRatio(cutoff = 3, variable = 'Jerk')
CIMTsumVec, CIMTbinAvg, CIMTmass = dic[CIMTrep].jerkRatio(cutoff = 3, variable = 'Jerk')

TD, CIMT = [], []
TDnum, CIMTnum = 0, 0
for key, value in dic.items():
    if 'TD' in key:
        TD.append(value.mass)
        TDnum += 1
    elif 'CIMT' in key:
        CIMT.append(value.mass)
        CIMTnum += 1
TD, CIMT = np.array(TD), np.array(CIMT)
#
plt.figure()

plt.subplot(2, 3, 1)
jerkPlot(TDbinAvg, TDsumVec, TDmass, '(a)')
    
plt.subplot(2, 3, 2)
jerkPlot(TDbinAvg, TDsumVec, TDmass, '(b)', shaded = True)

plt.subplot(2, 3, 3)
BP(TD, '(c)')

plt.subplot(2, 3, 4)
jerkPlot(CIMTbinAvg, CIMTsumVec, CIMTmass, '(d)')
    
plt.subplot(2, 3, 5)
jerkPlot(CIMTbinAvg, CIMTsumVec, CIMTmass, '(e)', shaded = True)

plt.subplot(2, 3, 6)
BP(CIMT, '(f)')



    


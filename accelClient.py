from Accel import Accel #now you're the client
import matplotlib.pyplot as plt
import numpy as np

palette = ['grey', 'grey', 'grey']

#initializes multiple Accel objects at a time
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

def boxPlot(TD, CIMT, bimanualLoc = 0.5, title = None):
    TDavg = np.mean(TD)
    TDstd = np.std(TD)
    plt.figure()
    plt.boxplot([CIMT[:, 0], CIMT[:, 1], CIMT[:, 2]], sym = '')
    plt.axhline(y = bimanualLoc, label = 'Bimanual ' + str(bimanualLoc), color = 'r')
    plt.axhline(TDavg, label = 'TD Avg, ' + "%.3f" % TDavg, color = 'b')
    plt.axhline(TDavg + TDstd, color = 'b', ls = '--')
    plt.axhline(TDavg - TDstd, color = 'b', ls = '--')
    plt.title(title)
# Takes in two arrays, TD and CIMT. Both of them are 1-D arrays that contain the "mass" from
# each of the collection periods
# Displays a box plot summarizing all trials
def summaryBoxPlot(TD, CIMT):
    TDavg = np.mean(TD)
    TDstd = np.std(TD)
    fig, axes = plt.subplots()
    bp = axes.boxplot([CIMT[:, 0], CIMT[:, 1], CIMT[:, 2]], sym = '', patch_artist = True)
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color)
    plt.setp(bp['medians'], color = 'k')
#    plt.figure()
#    plt.boxplot([CIMT[:, 0], CIMT[:, 1], CIMT[:, 2]], sym = '', patch_artist = True)
#    for patch, color in zip(bplot['boxes'], ['blue','red','black']):
#        patch.set_facecolor(color)
    plt.axhline(y = 0.5, label = 'Bimanual, 0.50', color = 'k')
    plt.axhline(TDavg, label = 'TD Avg, ' + "%.2f" % TDavg, color = 'grey')
    plt.axhline(TDavg + TDstd, color = 'grey', ls = '--')
    plt.axhline(TDavg - TDstd, color = 'grey', ls = '--')
    plt.legend()
    plt.xticks([1, 2, 3], ['Pre', 'During', 'Post']) #maps x labels
    plt.ylabel('Probability of Using Dominant Arm')
    plt.xlabel('Data Collection')
    

def jerkPlot(binAvg, probVec, mass, subplotTitle, shaded = False):
    for prob, c, oneLab, oneMass, oneLS in zip(probVec, palette, ['Pre', 'During', 'Post'], mass, [':', '--', '-']):
        if shaded:
            plt.plot(binAvg, np.divide(prob, max(prob)), color = c, label = oneLab + ', JDP = ' + str(round(oneMass * 100, 1)) + '%', ls = oneLS)
            plt.fill_between(binAvg, np.divide(prob, max(prob)), where = binAvg < 0.5, alpha = 0.5, color = c)
        else:
            plt.plot(binAvg, np.divide(prob, max(prob)), color = c, label = oneLab + ", %.2f" % oneMass, ls = oneLS)
    plt.axvline(x = 0.5, color = 'k', label = 'Bimanual, 0.50')
    plt.legend(loc = 'lower right')
    plt.title(subplotTitle)
    plt.xlabel('jerk ratio')
    plt.ylabel('normalized probability')

def BP(massVec, subplotTitle):
    
    colors = ['darkred', 'red', 'darkblue', 'blue', 'grey']
    plt.boxplot([massVec[:, 0], massVec[:, 1], massVec[:, 2]], sym = '')
    for oneMassVec, c in zip(massVec, colors):
        plt.scatter([1, 2, 3], oneMassVec, color = c)
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

def writeToExcel(fileName):
    from xlwt import Workbook
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    i = 0
    for key, value in dic.items():
        j = 0
        for item in value.UR:
            sheet1.write(i, j, item)
            j += 1
        i += 1
    wb.save(fileName)
    
#def plotSubDic(dic):  
#    
#    # for some reason the filtering doesn't work right now
#    plt.figure()
#    histBins = np.linspace(0.1, 0.9, 200)
#    binAvg = 0.5*(histBins[1:] + histBins[:-1]) #used to plot
#    labels = ['Pre', 'During', 'Post']
#    for ind, (key, value) in enumerate(dic.items()):
#        plt.subplot(len(dic), 1, ind + 1) #subplot starts at one
#        plt.title(key)
#        for i in range(3):
#    #        plt.subplot(3, 1, i + 1)
#            plt.plot(binAvg, np.histogram(value[i], bins = histBins)[0], label = labels[i])
#    plt.legend()

def plotIndividaulMR(dic):
    histBins = np.linspace(-5, 5, 200)
    binAvg = 0.5*(histBins[1:] + histBins[:-1])
    labels = ['Pre', 'During', 'Post']
    for key, value in dic.items():
        plt.figure(figsize = (6, 6))
        plt.title(key)
        for i in range(3):
            plt.plot(binAvg, np.divide(value.MR[i], max(value.MR[i])), label = labels[i])
    plt.xlabel('magnitude ratio')
    plt.ylabel('probability')
    plt.tight_layout()        
    plt.legend()

def plotIndividaulJR(dic):
    histBins = np.linspace(0.1, 0.9, 200)
    binAvg = 0.5*(histBins[1:] + histBins[:-1])
    labels = ['Pre', 'During', 'Post']
    for ind, (key, value) in enumerate(dic.items()):
        plt.figure(figsize = (6, 6))
        plt.title(key)
        for i in range(3):
            plt.plot(binAvg, np.divide(value.[i], max(value[i])), label = labels[i])
    plt.xlabel('jerk ratio')
    plt.ylabel('probability')
    plt.tight_layout()        
    plt.legend()
#%%
plt.close('all')

dic = initialize(['TD01', 'TD02', 'TD05', 'TD06', 'TD07',
                  'CIMT03', 'CIMT04','CIMT08', 'CIMT09', 'CIMT13'], 'Baker', 'Raw')

#dic = initialize(['TD01', 'TD02', 'TD05', 'TD06', 'TD07',
#                  'CIMT03', 'CIMT04','CIMT08', 'CIMT09', 'CIMT13'], 'Mac', 'Epoch')
#%%
for key, value in dic.items():
    plt.figure()
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.hist(value.MR[i], bins = np.linspace(-5, 5, 150))
    plt.suptitle(key)
#%%
    
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
    #for key, value in dic.items():
#    value.jerkRatio(showPlot = True, cutoff = 3, saveFig = True)
#%%
TDrep = 'TD01'
CIMTrep = 'CIMT03'

TDsumVec, TDbinAvg, TDmass = dic[TDrep].jerkRatio(cutoff = 3, variable = 'Jerk')
CIMTsumVec, CIMTbinAvg, CIMTmass = dic[CIMTrep].jerkRatio(cutoff = 3, variable = 'Jerk')
TD_ENMO_sumVec, TD_ENMO_binAvg, TD_ENMO_mass = dic[TDrep].jerkRatio(cutoff = 3, variable = 'ENMO')
CIMT_ENMO_sumVec, CIMT_ENMO_binAvg, CIMT_ENMO_mass = dic[CIMTrep].jerkRatio(cutoff = 3, variable = 'ENMO')

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

plt.figure()
plt.subplot(2, 3, 1)
jerkPlot(TDbinAvg, TDsumVec, TDmass, 'A')
plt.subplot(2, 3, 2)
BP(TD, 'B')
plt.subplot(2, 3, 3)
jerkPlot(TD_ENMO_binAvg, TD_ENMO_sumVec, TD_ENMO_mass, 'C')
plt.subplot(2, 3, 4)
jerkPlot(CIMTbinAvg, CIMTsumVec, CIMTmass, 'D')
plt.subplot(2, 3, 5)
BP(CIMT, 'E')
plt.subplot(2, 3, 6)
jerkPlot(CIMT_ENMO_binAvg, CIMT_ENMO_sumVec, CIMT_ENMO_mass, 'F')

summaryBoxPlot(TD, CIMT)
    


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
    
# Takes in two arrays, TD and CIMT. Both of them are 1-D arrays that contain the "mass" from
# each of the collection periods. Displays a box plot summarizing all trials
def summaryBoxPlot(TD, CIMT):
    TDavg = np.mean(TD)
    TDstd = np.std(TD)
    fig, axes = plt.subplots()
    bp = axes.boxplot([CIMT[:, 0], CIMT[:, 1], CIMT[:, 2]], sym = '', patch_artist = True)
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color)
    plt.setp(bp['medians'], color = 'k')
    plt.axhline(y = 0.5, label = 'Bimanual, 0.50', color = 'k')
    plt.axhline(TDavg, label = 'TD Avg, ' + "%.2f" % TDavg, color = 'grey')
    plt.axhline(TDavg + TDstd, color = 'grey', ls = '--')
    plt.axhline(TDavg - TDstd, color = 'grey', ls = '--')
    plt.legend()
    plt.xticks([1, 2, 3], ['Pre', 'During', 'Post']) #maps x labels
    plt.ylabel('Probability of Using Dominant Arm')
    plt.xlabel('Data Collection')

# Do not remove - this function generates publishable figures
# Takes in either TD or CIMT categories and scatters the individual data points
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

# Do not remove - this function generates publishable figures
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

#def pearsonCorr(dic):
#    allCorr = []
#    for key, value in dic.items():
#        allCorr.append(value.corrAvg)
#    print()
#    return np.average(allCorr), np.std(allCorr)

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

def plotIndividual(dic, content):
    plt.close('all')
    labels = ['Pre', 'During', 'Post']
    for key, value in dic.items():
        plt.figure(figsize = (6, 6))
        plt.title(key)
        for i in range(3):
            if content == 'JR':
                x = value.JRbinAvg
                y = np.divide(value.JRcounts[i], max(value.JRcounts[i]))
                xlab = 'jerk ratio'
            elif content == 'MR':
                x = value.MRbinAvg
                y = np.divide(value.MRcounts[i], max(value.MRcounts[i]))
                xlab = 'magnitude ratio'
            plt.plot(x, y, label = labels[i])
        plt.xlabel(xlab)
        plt.ylabel('normalized probability')
        plt.tight_layout()
        plt.legend()
#        plt.savefig('/Users/preston/Desktop/' + content + '_' + key + '.png')
        plt.savefig('C:\\Users\\SCH CIMT Study\\Desktop\\' + content + '_' + key + '.png')
#%%
plt.close('all')

dic = initialize(['TD01', 'TD02', 'TD05', 'TD06', 'TD07',
                  'CIMT03', 'CIMT04','CIMT08', 'CIMT09', 'CIMT13'], 'Baker', 'Raw')
plotIndividual(dic, 'JR')

#dic = initialize(['TD01', 'TD02', 'TD05', 'TD06', 'TD07',
#                  'CIMT03', 'CIMT04','CIMT08', 'CIMT09', 'CIMT13'], 'Mac', 'Epoch')
#%%
TD, CIMT = [], []
avgTDHistPre = []
avgTDHistDur = []
avgTDHistPost = []
TDcorr, CIMTcorr = [], []
TDnum, CIMTnum = 0, 0

CIMTpre = []
CIMTduring = []
for key, value in dic.items():
    if 'TD' in key:
        TD.append(value.JRsummary)
        TDcorr.append(value.corrAvg)
        avgTDHistPre.append(value.JRcounts[0]) # This is to create the final "representative" TD curve during the first collection
        avgTDHistDur.append(value.JRcounts[1])
        avgTDHistPost.append(value.JRcounts[2])
        TDnum += 1
    elif 'CIMT' in key:
        CIMT.append(value.JRsummary)
        CIMTcorr.append(value.corrAvg)
        CIMTnum += 1
        CIMTpre.append(value.JRsummary[0])
        CIMTduring.append(value.JRsummary[1])
        CIMTpost.append(value.JRsummary[2])
TD, CIMT = np.array(TD), np.array(CIMT)
TDjrAvg, TDjrStd = np.mean(TD), np.std(TD)
TDcorrAvg, TDcorrStd = np.mean(TDcorr), np.std(TDcorr)

#%% Publishable section

TDrep = 'TD01'
CIMTrep = 'CIMT03'

TDsumVec, TDbinAvg, TDmass = dic[TDrep].jerkRatio(cutoff = 3, variable = 'Jerk')
CIMTsumVec, CIMTbinAvg, CIMTmass = dic[CIMTrep].jerkRatio(cutoff = 3, variable = 'Jerk')
TD_ENMO_sumVec, TD_ENMO_binAvg, TD_ENMO_mass = dic[TDrep].jerkRatio(cutoff = 3, variable = 'ENMO')
CIMT_ENMO_sumVec, CIMT_ENMO_binAvg, CIMT_ENMO_mass = dic[CIMTrep].jerkRatio(cutoff = 3, variable = 'ENMO')

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
    


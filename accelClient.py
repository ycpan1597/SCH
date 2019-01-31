from Accel import Accel #now you're the client
import matplotlib.pyplot as plt
import numpy as np
import tkinter,tkinter.filedialog

palette = ['grey', 'grey', 'grey']

#initializes multiple Accel objects at a time
def initialize(subjects, filetype, applyButter = True):
    l = {}
    for item in subjects:
        l[str(item)] = Accel(item, filetype, applyButter = applyButter)
    return l

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
    print('TDstd = %.2f' % TDstd)

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

def userInput():
    root = tkinter.Tk()
    root.withdraw()
    more = 'y'
    fs = input('What is the sampling frequency (Hz)? ')
    allFiles = {'L':[], 'R':[]}
    while (more is 'y' or more is 'Y'):
        allFiles['L'].append(tkinter.filedialog.askopenfile(parent=root, title='Please select a left file'))
        allFiles['R'].append(tkinter.filedialog.askopenfile(parent=root, title='Please select a right file'))
        more = input('more files? (y/n) ')
    return int(fs), allFiles
#%%
fs, allFiles = userInput()
#rawDic = initialize(['TD01', 'TD02', 'TD05', 'TD06', 'TD07',
#                  'CIMT03', 'CIMT04','CIMT08', 'CIMT09', 'CIMT13'], 'Raw')
#
#epochDic = initialize(['TD01', 'TD02', 'TD05', 'TD06', 'TD07', #we're using the AC for Use Ratio
#                  'CIMT03', 'CIMT04','CIMT08', 'CIMT09', 'CIMT13'], 'Epoch')
#for key, value in dic.items():
#    value.jerkRatio(showPlot = True, cutoff = 3, saveFig = True)

#TD, CIMT, CIMTmedian = findSummary(dic)
#%%
TDrep = 'TD01'
CIMTrep = 'CIMT03'
 
   
plt.close('all')
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
    


from Accel import Accel #now you're the client
import matplotlib.pyplot as plt
import numpy as np

#initializes multiple at a time
def initialize(subjects, OS, filetype):
    l = {}
    for item in subjects:
        l[str(item)] = Accel(item, OS, filetype, applyButter = True)
    return l

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

def findSummary(dic, minAge = 7, maxAge = 9, showBox = True, content = 'DA Prob'):
    if content == 'DA Prob':
        correctColumn = 1
    else:
        correctColumn = 0
    ageRange = np.arange(minAge, maxAge + 1)
    TD, CIMT = [], []
    TDnum, CIMTnum = 0, 0
    for key, value in dic.items():
        if 'TD' in key and value.age in ageRange:
            TD.append(value.jerkRatio(showPlot = False)[correctColumn])
            TDnum += 1
        elif 'CIMT' in key and value.age in ageRange:
            CIMT.append(value.jerkRatio(showPlot = False)[correctColumn])
            CIMTnum += 1
    TD, CIMT = np.array(TD), np.array(CIMT)

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
        plt.title(str(minAge) + '~' + str(maxAge) + 'yrs old, ' + str(TDnum) + ' TD and ' + str(CIMTnum) + ' CP subjects')
        
    if showBox: 
        boxPlot(TD, CIMT)
    
    return TD, CIMT, np.median(CIMT, axis = 0)

plt.close('all')

dic = initialize(['TD01', 'TD02', 'TD05', 'TD06', 'TD07',
                  'CIMT03', 'CIMT04','CIMT08', 'CIMT09', 'CIMT13'], 'Baker', 'Raw')
#for key, value in dic.items():
#    value.jerkRatio(showPlot = True, cutoff = 3, saveFig = True)

#%%
#TD, CIMT, CIMTmedian = findSummary(dic)
TDrep = 'TD01'
CIMTrep = 'CIMT03' 
   
plt.close('all')
TDsumVec, TDbinAvg, TDmass = dic[TDrep].jerkRatio(cutoff = 3)
CIMTsumVec, CIMTbinAvg, CIMTmass = dic[CIMTrep].jerkRatio(cutoff = 3)

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

x = ['Pre', 'During', 'Post'] * 5
plt.figure()

plt.subplot(2, 3, 1)
for prob, oneColor in zip(TDsumVec, 'gkr'):
    plt.plot(TDbinAvg, np.divide(prob, max(prob)), color = oneColor)
plt.title('(a)')
plt.xlabel('jerk ratio')
plt.ylabel('normalized probability')
    
plt.subplot(2, 3, 2)
for prob, oneColor in zip(TDsumVec, 'gkr'):
    plt.plot(TDbinAvg, np.divide(prob, max(prob)), color = oneColor)
    plt.fill_between(TDbinAvg, np.divide(prob, max(prob)), where = TDbinAvg < 0.5, alpha = 0.5, color = oneColor)
plt.title('(b)')
plt.xlabel('jerk ratio')
plt.ylabel('normalized probability')

plt.subplot(2, 3, 3)
plt.boxplot([TD[:, 0], TD[:, 1], TD[:, 2]], sym = '')
for item in TD:
    plt.scatter([1, 2, 3], item)
plt.xticks([1, 2, 3], ['Pre', 'During', 'Post'])
plt.ylim(0, 0.7)
plt.title('(c)')
plt.xlabel('data collection')
plt.ylabel('probability before 0.5 (%)')

plt.subplot(2, 3, 4)
for prob, oneColor in zip(CIMTsumVec, 'gkr'):
    plt.plot(CIMTbinAvg, np.divide(prob, max(prob)), color = oneColor)
plt.title('(d)')
plt.xlabel('jerk ratio')
plt.ylabel('normalized probability')
    
plt.subplot(2, 3, 5)
for prob, oneColor in zip(CIMTsumVec, 'gkr'):
    plt.plot(CIMTbinAvg, np.divide(prob, max(prob)), color = oneColor)
    plt.fill_between(CIMTbinAvg, np.divide(prob, max(prob)), where = CIMTbinAvg < 0.5, alpha = 0.5, color = oneColor)
plt.title('(e)')
plt.xlabel('jerk ratio')
plt.ylabel('normalized probability')

plt.subplot(2, 3, 6)
plt.boxplot([CIMT[:, 0], CIMT[:, 1], CIMT[:, 2]], sym = '')
for item in CIMT:
    plt.scatter([1, 2, 3], item)
plt.xticks([1, 2, 3], ['Pre', 'During', 'Post'])
plt.ylim(0, 0.7)
plt.title('(f)')
plt.xlabel('data collection')
plt.ylabel('probability before 0.5 (%)')



    

